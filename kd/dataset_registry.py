"""Versioned dataset catalog with an optional private Git LFS mirror.

Every archive is pinned by the catalog's MD5 (and, when the registry lists it, a SHA-256),
so any source that serves the same bytes is interchangeable. Downloads rank the private
registry, any configured mirrors and the canonical upstream by a short speed test, use
the fastest, and move to another source at the same byte offset when one stalls, fails or
falls far behind. Mirrors are configured per machine, never in the catalog, because the
catalog's hash is part of every run's data identity.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import zipfile
import re


CATALOG_PATH = Path(__file__).with_name("catalog") / "v1.json"
PROFILES = {"balanced": 1.0, "lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}
MARKER = ".kd-dataset.json"
DEFAULT_REGISTRY_REF = "catalog-v1.2.0"
REGISTRY_CONFIG_ENV = "KD_DATASET_REGISTRY_CONFIG"
MIRRORS_ENV = "KD_DATASET_MIRRORS"
USER_AGENT = "kd-repair-dataset-fetch/1"
# Sources are ranked by a short ranged download; archives smaller than PROBE_MIN_BYTES skip
# the test and use preference order, because probing would cost about as much as fetching.
PROBE_BYTES = 4 * 1024 * 1024
PROBE_SECONDS = 8.0
PROBE_MIN_BYTES = 32 * 1024 * 1024
# A download moves to an untried source, resuming at the same byte, once its rate over the
# last SLOW_WINDOW seconds falls below SLOW_FRACTION of that source's probed rate.
SLOW_WINDOW = 30.0
SLOW_FRACTION = 0.25
READ_TIMEOUT = 60.0


class CatalogError(ValueError):
    pass


class RegistryUnavailable(RuntimeError):
    pass


def _progress_enabled() -> bool:
    return os.environ.get("KD_PROGRESS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _status(message: str) -> None:
    """Report preparation progress on stderr; stdout stays reserved for the JSON report."""
    if _progress_enabled():
        print(message, file=sys.stderr, flush=True)


def _human_bytes(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024:
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TiB"


def _progress_interval() -> float:
    """Redraw often on a terminal; keep piped supervisor logs readable."""
    return 0.5 if sys.stderr.isatty() else 5.0


def _report_transfer(label: str, copied: int, total: int, elapsed: float, *,
                     final: bool = False, rate: float | None = None) -> None:
    if not _progress_enabled():
        return
    if rate is None:
        rate = copied / elapsed if elapsed > 0 else 0.0
    parts = [_human_bytes(copied)]
    if total:
        parts.append(f"of {_human_bytes(total)} ({100 * copied / total:5.1f}%)")
    parts.append(f"at {_human_bytes(rate)}/s")
    if total and rate > 0 and not final:
        remaining = max(total - copied, 0) / rate
        parts.append(f"eta {int(remaining // 60)}m{int(remaining % 60):02d}s")
    line = f"    {label}: {' '.join(parts)}"
    if sys.stderr.isatty():
        print(f"\r\x1b[K{line}", end="\n" if final else "", file=sys.stderr, flush=True)
    else:
        print(line, file=sys.stderr, flush=True)


def registry_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(REGISTRY_CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "kd-repair" / "datasets.json"


def _validate_registry_settings(registry: str, ref: str) -> None:
    if not registry or not isinstance(registry, str):
        raise CatalogError("Dataset registry URL must be a nonempty string")
    if not isinstance(ref, str) or not re.fullmatch(r"catalog-v\d+\.\d+\.\d+", ref):
        raise CatalogError("Private registry ref must be an immutable catalog tag")
    if "://" in registry:
        parsed = urlsplit(registry)
        if not parsed.scheme or not parsed.hostname:
            raise CatalogError("Dataset registry URL is malformed")
        if parsed.password or (parsed.scheme in {"http", "https"} and parsed.username):
            raise CatalogError("Dataset registry URL must not contain credentials")


def _validate_mirror(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https", "file"} or (parsed.scheme != "file" and not parsed.hostname):
        raise CatalogError(f"Dataset mirror must be an http(s) or file URL: {url}")
    if parsed.username or parsed.password:
        raise CatalogError("Dataset mirror URL must not contain credentials")
    return url.rstrip("/")


def _configured_mirrors(location: Path) -> list:
    if not location.is_file():
        return []
    try:
        return json.loads(location.read_text()).get("mirrors", [])
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot load dataset registry configuration {location}: {error}") from error


def dataset_mirrors(path: str | Path | None = None) -> list[str]:
    """Base URLs that serve the registry tree (datasets/NAME/VERSION/archives/FILE).

    KD_DATASET_MIRRORS (space- or comma-separated) overrides the persisted list.
    """
    value = os.environ.get(MIRRORS_ENV)
    urls = value.replace(",", " ").split() if value is not None else _configured_mirrors(
        registry_config_path(path))
    return [_validate_mirror(url) for url in urls]


def configure_registry(registry: str | None = None, ref: str = DEFAULT_REGISTRY_REF, *,
                       path: str | Path | None = None, remove: bool = False,
                       mirrors: list[str] | None = None) -> dict:
    location = registry_config_path(path)
    if remove:
        if registry is not None:
            raise CatalogError("Do not provide a registry URL with --remove")
        location.unlink(missing_ok=True)
        return {"configured": False, "path": str(location)}
    if registry is None:
        raise CatalogError("Registry URL is required unless --remove is used")
    _validate_registry_settings(registry, ref)
    # Launchers rewrite the registry on every run; keep mirrors unless new ones are given.
    mirrors = [_validate_mirror(url) for url in
               (_configured_mirrors(location) if mirrors is None else mirrors)]
    payload = {"schema_version": 1, "registry": registry, "ref": ref, "mirrors": mirrors}
    location.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=location.parent, delete=False) as temporary:
        temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    os.replace(temporary_path, location)
    return {"configured": True, "registry": registry, "ref": ref, "mirrors": mirrors,
            "source": "config_file", "path": str(location)}


def registry_status(path: str | Path | None = None) -> dict:
    registry = os.environ.get("KD_DATASET_REGISTRY")
    if registry:
        ref = os.environ.get("KD_DATASET_REGISTRY_REF", DEFAULT_REGISTRY_REF)
        _validate_registry_settings(registry, ref)
        return {"configured": True, "registry": registry, "ref": ref,
                "mirrors": dataset_mirrors(path), "source": "environment", "path": None}
    location = registry_config_path(path)
    if not location.is_file():
        return {"configured": False, "mirrors": dataset_mirrors(path), "source": None,
                "path": str(location)}
    try:
        payload = json.loads(location.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot load dataset registry configuration {location}: {error}") from error
    if payload.get("schema_version") != 1:
        raise CatalogError("Unsupported dataset registry configuration schema")
    registry, ref = payload.get("registry"), payload.get("ref")
    _validate_registry_settings(registry, ref)
    return {"configured": True, "registry": registry, "ref": ref,
            "mirrors": dataset_mirrors(path), "source": "config_file", "path": str(location)}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def sha256_value(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: str | Path | None = None) -> dict:
    location = Path(path) if path else CATALOG_PATH
    try:
        catalog = json.loads(location.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot load dataset catalog {location}: {error}") from error
    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported dataset catalog schema")
    if set(catalog.get("profiles", ())) != set(PROFILES):
        raise CatalogError("Dataset catalog profiles differ from the supported protocol")
    datasets = catalog.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise CatalogError("Dataset catalog must contain datasets")
    for name, recipe in datasets.items():
        if Path(name).name != name or not isinstance(recipe.get("artifacts"), list):
            raise CatalogError(f"Malformed dataset recipe: {name}")
        if not recipe.get("version") or not recipe.get("license_url"):
            raise CatalogError(f"Dataset recipe lacks version/license: {name}")
        for artifact in recipe["artifacts"]:
            required = {"id", "url", "filename", "md5"}
            if not required.issubset(artifact) or Path(artifact["filename"]).name != artifact["filename"]:
                raise CatalogError(f"Malformed artifact in dataset recipe: {name}")
            for nested in artifact.get("nested_extract", []):
                if set(nested) != {"path", "extract_to"}:
                    raise CatalogError(f"Malformed nested archive in dataset recipe: {name}")
                for field in ("path", "extract_to"):
                    path = Path(nested[field])
                    if path.is_absolute() or ".." in path.parts:
                        raise CatalogError(f"Unsafe nested archive path in dataset recipe: {name}")
    return catalog


def dataset_recipe(name: str, version: str | None = None, *, catalog: dict | None = None) -> tuple[dict, dict]:
    catalog = load_catalog() if catalog is None else catalog
    try:
        recipe = catalog["datasets"][name]
    except KeyError as error:
        raise CatalogError(f"Unknown dataset: {name}") from error
    if version is not None and recipe["version"] != version:
        raise CatalogError(f"Unsupported {name} version {version}; expected {recipe['version']}")
    return catalog, recipe


def dataset_directory(root: str | Path, name: str, version: str) -> Path:
    return Path(root) / name / version


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None,
         label: str | None = None) -> None:
    """Run a registry command, streaming its output so long clones and LFS pulls stay visible."""
    if label:
        _status(f"  {label}")
    process = subprocess.Popen(command, cwd=cwd, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail: deque[str] = deque(maxlen=20)
    pending = ""
    last_report = 0.0
    interval = _progress_interval()

    def emit(segment: str) -> None:
        nonlocal last_report
        if not segment.strip():
            return
        tail.append(segment)
        now = time.monotonic()
        # Percentage counters repeat constantly; throttle them but never hide plain messages.
        if "%" in segment and now - last_report < interval:
            return
        last_report = now
        _status(f"    {segment.strip()}")

    with process:
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            pending += chunk.decode("utf-8", "replace")
            segments = re.split(r"[\r\n]", pending)
            pending = segments.pop()
            for segment in segments:
                emit(segment)
        emit(pending)
    if process.returncode:
        detail = "\n".join(tail).strip() or f"exit status {process.returncode}"
        raise RegistryUnavailable(f"Dataset registry command failed: {detail}")


def _registry_checkout(root: Path, registry: str, ref: str) -> Path:
    candidate = Path(registry).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if not re.fullmatch(r"catalog-v\d+\.\d+\.\d+", ref):
        raise RegistryUnavailable("Private registry ref must be an immutable catalog tag")
    checkout = root / ".registry" / "kd-repair"
    # Never block an unattended run on a credential or host-key prompt; failing fast
    # lets fetch_dataset fall back to the canonical upstream sources instead.
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_LFS_FORCE_PROGRESS": "1",
                   "GIT_TERMINAL_PROMPT": "0",
                   "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")}
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--progress", "--filter=blob:none", "--no-checkout",
              registry, str(checkout)], env=environment,
             label=f"cloning dataset registry into {checkout}")
    _run(["git", "fetch", "--progress", "--tags", "origin"], cwd=checkout, env=environment,
         label="fetching registry tags")
    _run(["git", "checkout", "--detach", ref], cwd=checkout, env=environment,
         label=f"checking out catalog {ref}")
    return checkout


def _manifest_records(manifest: dict, recipe: dict) -> dict[str, dict]:
    """Validate a registry manifest against the public recipe; return its artifact records."""
    if manifest.get("schema_version") != 1 or manifest.get("recipe_sha256") != sha256_value(recipe):
        raise RegistryUnavailable("Private mirror manifest does not match the public recipe")
    records = manifest.get("artifacts", {})
    for artifact in recipe["artifacts"]:
        relative = records.get(artifact["id"], {}).get("path")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RegistryUnavailable(f"Invalid private mirror path for {artifact['id']}")
    return records


def _registry_web_base(registry: str) -> str | None:
    """An HTTP(S) Gitea registry also serves raw files and LFS objects over plain HTTPS."""
    if urlsplit(registry).scheme not in {"http", "https"}:
        return None
    base = registry.rstrip("/")
    return base[:-4] if base.endswith(".git") else base


def _registry_manifest(web: str, ref: str, name: str, recipe: dict) -> dict[str, dict]:
    url = f"{web}/raw/tag/{quote(ref)}/manifests/{quote(name)}/{quote(recipe['version'])}.json"
    try:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=READ_TIMEOUT) as response:
            manifest = json.loads(response.read())
    except (OSError, ValueError, HTTPException) as error:
        raise RegistryUnavailable(f"Private mirror manifest unavailable: {error}") from error
    return _manifest_records(manifest, recipe)


def _mirror_files(root: Path, name: str, recipe: dict) -> dict[str, Path]:
    """Local or SSH registries: check out the tag and pull the recipe's LFS objects."""
    settings = registry_status()
    if not settings["configured"]:
        return {}
    checkout = _registry_checkout(root, settings["registry"], settings["ref"])
    manifest_path = checkout / "manifests" / name / f"{recipe['version']}.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryUnavailable(f"Private mirror manifest unavailable: {error}") from error
    records = _manifest_records(manifest, recipe)
    paths = [records[artifact["id"]]["path"] for artifact in recipe["artifacts"]]
    if (checkout / ".git").exists():
        _run(["git", "lfs", "pull", "--include", ",".join(paths), "--exclude", ""], cwd=checkout,
             env={**os.environ, "GIT_LFS_FORCE_PROGRESS": "1", "GIT_TERMINAL_PROMPT": "0",
                  "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")},
             label=f"pulling {len(paths)} mirrored archive(s) for {name}")
    result = {}
    for artifact, relative in zip(recipe["artifacts"], paths):
        path = checkout / relative
        record = records[artifact["id"]]
        if not path.is_file() or file_hash(path) != record.get("sha256"):
            raise RegistryUnavailable(f"Private mirror hash failed for {artifact['id']}")
        if path.stat().st_size != record.get("size"):
            raise RegistryUnavailable(f"Private mirror size failed for {artifact['id']}")
        result[artifact["id"]] = path
    return result


def _host(url: str) -> str:
    return urlsplit(url).netloc or url


def _probe(url: str) -> float:
    """Bytes per second over a short ranged download, or 0.0 when the source fails."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes=0-{PROBE_BYTES - 1}"})
    started, received = time.monotonic(), 0
    try:
        with urlopen(request, timeout=PROBE_SECONDS) as response:
            read = getattr(response, "read1", response.read)
            while received < PROBE_BYTES and time.monotonic() - started < PROBE_SECONDS:
                block = read(64 * 1024)
                if not block:
                    break
                received += len(block)
    except (OSError, ValueError, HTTPException):
        return 0.0
    return received / max(time.monotonic() - started, 1e-6)


def _rank_sources(sources: list[tuple[str, str]], size: int | None,
                  prefix: str) -> list[tuple[str, str, float | None]]:
    """Order sources fastest first; unreachable ones go last and are kept as a final resort."""
    if len(sources) < 2 or (size is not None and size < PROBE_MIN_BYTES):
        return [(label, url, None) for label, url in sources]
    _status(f"{prefix}: testing the speed of {len(sources)} sources")
    with ThreadPoolExecutor(len(sources)) as pool:
        speeds = list(pool.map(_probe, [url for _, url in sources]))
    for (label, url), speed in zip(sources, speeds):
        _status(f"      {_host(url)} ({label}): "
                + (f"{_human_bytes(speed)}/s" if speed else "unreachable"))
    # sorted() is stable, so equally fast sources keep the preference order.
    order = sorted(range(len(sources)), key=lambda index: -speeds[index])
    return [(*sources[index], speeds[index]) for index in order]


class _Slow(Exception):
    """The current source fell far behind an untried one."""


def _content_total(response, offset: int) -> int:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
        return int(content_range.rsplit("/", 1)[1])
    length = int(response.headers.get("Content-Length") or 0)
    return length + offset if length else 0


def _stream(url: str, partial: Path, label: str, total: int, switch_below: float) -> None:
    """Append url's bytes to partial, resuming at its current size when the server allows."""
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={offset}-"} if offset else {})}
    with urlopen(Request(url, headers=headers), timeout=READ_TIMEOUT) as response:
        if offset and getattr(response, "status", None) != 206:
            offset = 0  # The server ignored the range, so this source starts from byte zero.
        total = total or _content_total(response, offset)
        read = getattr(response, "read1", response.read)
        with partial.open("ab" if offset else "wb") as handle:
            started = last_report = time.monotonic()
            window, copied, interval = deque([(started, offset)]), offset, _progress_interval()
            rate = 0.0
            while True:
                block = read(256 * 1024)
                if not block:
                    break
                handle.write(block)
                copied += len(block)
                now = time.monotonic()
                window.append((now, copied))
                while len(window) > 2 and now - window[1][0] >= SLOW_WINDOW:
                    window.popleft()
                rate = (copied - window[0][1]) / max(now - window[0][0], 1e-6)
                if now - last_report >= interval:
                    last_report = now
                    _report_transfer(label, copied, total, now - started, rate=rate)
                if switch_below and now - started >= SLOW_WINDOW and rate < switch_below:
                    raise _Slow(f"{_human_bytes(rate)}/s at {_human_bytes(copied)}")
            _report_transfer(label, copied, total, time.monotonic() - started, final=True, rate=rate)


def _verified(path: Path, md5: str, record: dict) -> bool:
    if file_hash(path, "md5") != md5:
        return False
    return not record.get("sha256") or file_hash(path) == record["sha256"]


def _fetch_from_sources(sources: list[tuple[str, str]], destination: Path, md5: str,
                        record: dict, prefix: str) -> str:
    """Download destination from the fastest source, switching when one stalls, fails or lags.

    Bytes accumulate in a .part file, so a switch or a rerun resumes rather than restarts;
    the checksum decides whether the assembled file is kept. Returns the source's label.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    queue, tried, failures = _rank_sources(sources, record.get("size"), prefix), set(), []
    while queue:
        label, url, _ = queue.pop(0)
        tried.add(url)
        # Only an untried source that probed clearly faster can pull a download away.
        faster = [speed for _, other, speed in queue if other not in tried and speed]
        switch_below = SLOW_FRACTION * max(faster) if faster else 0.0
        done = partial.stat().st_size if partial.exists() else 0
        _status(f"{prefix}: {'resuming at ' + _human_bytes(done) if done else 'downloading'} "
                f"from {_host(url)} ({label})")
        try:
            _stream(url, partial, f"{destination.name} <- {_host(url)}", record.get("size", 0),
                    switch_below)
        except _Slow as slow:
            _status(f"{prefix}: {_host(url)} slowed to {slow}; switching to a faster source")
            queue.append((label, url, None))
            continue
        except (OSError, ValueError, HTTPException) as error:
            failures.append(f"{_host(url)}: {error}")
            _status(f"{prefix}: {_host(url)} failed ({error}); trying the next source")
            continue
        if _verified(partial, md5, record):
            os.replace(partial, destination)
            return label
        failures.append(f"{_host(url)}: checksum mismatch")
        _status(f"{prefix}: {_host(url)} delivered a file with the wrong checksum; discarding it")
        partial.unlink(missing_ok=True)
    raise CatalogError(f"No source delivered {destination.name}: {'; '.join(failures)}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def safe(name: str) -> bool:
        return (root / name).resolve().is_relative_to(root)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            if not all(safe(member.filename) for member in handle.infolist()):
                raise CatalogError(f"Archive contains an unsafe path: {archive}")
            handle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            if not all(safe(member.name) for member in handle.getmembers()):
                raise CatalogError(f"Archive contains an unsafe path: {archive}")
            handle.extractall(destination, filter="data")
    else:
        raise CatalogError(f"Unsupported archive format: {archive}")


_LAYOUT_LINKS = {
    "eurosat": (Path("eurosat/2750"), Path("EuroSAT_RGB")),
    "caltech101": (Path("caltech101/101_ObjectCategories"),
                   Path("../101_ObjectCategories")),
}


def _materialize_dataset_layout(name: str, target: Path) -> dict:
    """Bridge canonical archive layouts to paths expected by torchvision."""
    layout = _LAYOUT_LINKS.get(name)
    if layout is None:
        return {}
    link, relative_target = target / layout[0], layout[1]
    link.parent.mkdir(parents=True, exist_ok=True)
    source = link.parent / relative_target
    if not source.is_dir():
        raise CatalogError(f"Extracted {name} dataset directory is missing: {source}")
    if link.is_symlink():
        if Path(os.readlink(link)) != relative_target:
            raise CatalogError(
                f"Dataset layout symlink has unexpected target: {link} -> {os.readlink(link)}")
    elif link.exists():
        raise CatalogError(f"Refusing to replace existing dataset layout path: {link}")
    else:
        link.symlink_to(relative_target, target_is_directory=True)
    return {str(layout[0]): str(relative_target)}


def _loader_smoke_check(name: str, target: Path, expected_classes: int) -> dict:
    """Ensure prepared layouts are discoverable through their production loader."""
    if name not in _LAYOUT_LINKS:
        return {}
    from torchvision import datasets

    try:
        dataset = (datasets.EuroSAT(target, download=False) if name == "eurosat"
                   else datasets.Caltech101(target, download=False))
        classes = dataset.classes if name == "eurosat" else dataset.categories
        if len(classes) != expected_classes:
            raise CatalogError(
                f"Prepared {name} has {len(classes)} classes; expected {expected_classes}")
        if not dataset:
            raise CatalogError(f"Prepared {name} contains no samples")
        image, label = dataset[0]
        try:
            if image is None or not 0 <= int(label) < expected_classes:
                raise CatalogError(f"Prepared {name} returned an invalid first sample")
        finally:
            close = getattr(image, "close", None)
            if close is not None:
                close()
    except CatalogError:
        raise
    except Exception as error:
        raise CatalogError(f"Prepared {name} cannot be loaded: {error}") from error
    return {"class_count": len(classes), "sample_count": len(dataset),
            "first_label": int(label)}


def fetch_dataset(name: str, version: str | None = None, profile: str = "balanced",
                  root: str | Path = "data") -> dict:
    if profile not in PROFILES:
        raise CatalogError(f"Unknown dataset profile: {profile}")
    catalog, recipe = dataset_recipe(name, version)
    target = dataset_directory(root, name, recipe["version"])
    target.mkdir(parents=True, exist_ok=True)
    mirror, records, registry_base, mirror_error = {}, {}, None, None
    try:
        settings = registry_status()
        web = _registry_web_base(settings["registry"]) if settings["configured"] else None
        if web:
            records = _registry_manifest(web, settings["ref"], name, recipe)
            registry_base = f"{web}/media/tag/{quote(settings['ref'])}"
        else:
            mirror = _mirror_files(Path(root), name, recipe)
    except RegistryUnavailable as error:
        mirror_error = str(error)
    mirrors = dataset_mirrors()
    _status(f"[fetch] {name} {recipe['version']} (profile {profile}) -> {target}")
    if mirror_error:
        _status(f"  private registry unavailable ({mirror_error}); using canonical upstream")
    artifacts = []
    total_artifacts = len(recipe["artifacts"])
    for position, artifact in enumerate(recipe["artifacts"], start=1):
        destination = target / artifact["filename"]
        prefix = f"  [{position}/{total_artifacts}] {artifact['filename']}"
        if not destination.is_file() or file_hash(destination, "md5") != artifact["md5"]:
            if artifact["id"] in mirror:
                _status(f"{prefix}: copying from the private registry")
                shutil.copy2(mirror[artifact["id"]], destination)
                source = "private_registry"
            else:
                record = records.get(artifact["id"], {})
                relative = quote(record.get("path") or
                                 f"datasets/{name}/{recipe['version']}/archives/{artifact['filename']}")
                sources = [("private_registry", f"{registry_base}/{relative}")] if record else []
                sources += [("mirror", f"{base}/{relative}") for base in mirrors]
                sources.append(("upstream", artifact["url"]))
                source = _fetch_from_sources(sources, destination, artifact["md5"], record, prefix)
        else:
            _status(f"{prefix}: already present")
            source = "local"
        _status(f"{prefix}: verifying checksum")
        if file_hash(destination, "md5") != artifact["md5"]:
            destination.unlink(missing_ok=True)
            raise CatalogError(f"Checksum mismatch for {name}/{artifact['id']}")
        if artifact.get("extract_to") is not None:
            _status(f"{prefix}: extracting")
            extracted_root = target / artifact["extract_to"]
            _safe_extract(destination, extracted_root)
            for nested in artifact.get("nested_extract", []):
                _safe_extract(extracted_root / nested["path"], target / nested["extract_to"])
        artifacts.append({"id": artifact["id"], "filename": artifact["filename"],
                          "size": destination.stat().st_size,
                          "sha256": file_hash(destination), "source": source})
    layout = _materialize_dataset_layout(name, target)
    loader_smoke = _loader_smoke_check(name, target, recipe["classes"])
    marker = {"schema_version": 1, "catalog_version": catalog["catalog_version"],
              "catalog_sha256": sha256_value(catalog), "recipe_sha256": sha256_value(recipe),
              "dataset": name, "version": recipe["version"], "requested_profile": profile,
              "profiles": list(PROFILES),
              "split_seed": catalog["split_seed"], "artifacts": artifacts}
    if layout:
        marker["materialized_layout"] = layout
        marker["loader_smoke"] = loader_smoke
    if mirror_error:
        marker["mirror_fallback"] = "Private registry unavailable; fetched from canonical upstream"
    (target / MARKER).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    _status(f"[fetch] {name} ready ({_human_bytes(sum(r['size'] for r in artifacts))})")
    return marker


def verify_dataset(name: str, version: str | None = None, profile: str = "balanced",
                   root: str | Path = "data") -> dict:
    if profile not in PROFILES:
        raise CatalogError(f"Unknown dataset profile: {profile}")
    catalog, recipe = dataset_recipe(name, version)
    target = dataset_directory(root, name, recipe["version"])
    try:
        marker = json.loads((target / MARKER).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Dataset is not prepared: {name}: {error}") from error
    if marker.get("recipe_sha256") != sha256_value(recipe):
        raise CatalogError(f"Dataset recipe changed: {name}")
    if profile not in marker.get("profiles", []):
        raise CatalogError(f"Prepared dataset does not support profile {profile}")
    recorded = {record["id"]: record for record in marker.get("artifacts", [])}
    total = 0
    _status(f"[verify] {name} {recipe['version']} (profile {profile})")
    for artifact in recipe["artifacts"]:
        path = target / artifact["filename"]
        record = recorded.get(artifact["id"], {})
        if not path.is_file() or file_hash(path) != record.get("sha256"):
            raise CatalogError(f"Dataset artifact changed: {name}/{artifact['id']}")
        total += path.stat().st_size
    if total > catalog["storage_limit_bytes"]:
        raise CatalogError("Dataset exceeds the catalog storage limit")
    loader_smoke = _loader_smoke_check(name, target, recipe["classes"])
    return {**marker, "profile": profile, "verified": True, "total_bytes": total,
            **({"loader_smoke": loader_smoke} if loader_smoke else {})}


def list_datasets() -> dict:
    catalog = load_catalog()
    return {"catalog_version": catalog["catalog_version"], "profiles": list(PROFILES),
            "datasets": [{"name": name, "version": recipe["version"],
                          "classes": recipe["classes"], "image_size": recipe["image_size"],
                          "validation_policy": recipe["validation_policy"]}
                         for name, recipe in sorted(catalog["datasets"].items())]}


def initialize_registry(path: str | Path) -> dict:
    root = Path(path).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Registry destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir()
    (root / "manifests").mkdir()
    (root / "licenses").mkdir()
    (root / "splits").mkdir()
    workflow = root / ".gitea" / "workflows" / "validate.yml"
    workflow.parent.mkdir(parents=True)
    (root / ".gitattributes").write_text(
        "datasets/**/archives/** filter=lfs diff=lfs merge=lfs -text\n")
    (root / ".gitignore").write_text(".DS_Store\n*.tmp\n")
    (root / "README.md").write_text(
        "# Private kd-repair dataset cache\n\n"
        "Private Git LFS cache. Canonical recipes and executable code live in the public source repository.\n")
    workflow.write_text(
        "name: validate-datasets\n"
        "on: [push]\n"
        "jobs:\n"
        "  verify:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "        with:\n"
        "          lfs: true\n"
        "      - run: git lfs fsck\n"
        "      - run: git clone --depth 1 --branch source-v0.3.2 https://github.com/spozi/kd-repair.git /tmp/kd-repair\n"
        "      - run: python -m pip install -e /tmp/kd-repair --no-deps\n"
        "      - run: python -m kd dataset registry-verify --registry-root .\n")
    return {"initialized": True, "root": str(root), "storage_limit_bytes": load_catalog()["storage_limit_bytes"]}


def create_mirror_manifest(name: str, registry_root: str | Path, *,
                           version: str | None = None, terms_reviewed: bool = False) -> dict:
    if not terms_reviewed:
        raise CatalogError("--terms-reviewed is required before private archival")
    catalog, recipe = dataset_recipe(name, version)
    root = Path(registry_root).resolve()
    records = {}
    total = 0
    for artifact in recipe["artifacts"]:
        relative = Path("datasets") / name / recipe["version"] / "archives" / artifact["filename"]
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing regular mirror artifact: {path}")
        if file_hash(path, "md5") != artifact["md5"]:
            raise CatalogError(f"Upstream checksum mismatch: {relative}")
        size = path.stat().st_size
        total += size
        records[artifact["id"]] = {"path": relative.as_posix(), "size": size,
                                   "sha256": file_hash(path)}
    if total > catalog["storage_limit_bytes"]:
        raise CatalogError("Private mirror exceeds the 100 GB catalog limit")
    manifest = {"schema_version": 1, "dataset": name, "version": recipe["version"],
                "recipe_sha256": sha256_value(recipe), "terms_reviewed": True,
                "artifacts": records, "total_bytes": total}
    destination = root / "manifests" / name / f"{recipe['version']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest": str(destination)}


def validate_registry(registry_root: str | Path) -> dict:
    catalog = load_catalog()
    root = Path(registry_root).resolve()
    total, validated = 0, []
    for path in sorted((root / "manifests").glob("*/*.json")):
        manifest = json.loads(path.read_text())
        name, version = manifest.get("dataset"), manifest.get("version")
        _, recipe = dataset_recipe(name, version, catalog=catalog)
        if (manifest.get("schema_version") != 1 or not manifest.get("terms_reviewed")
                or manifest.get("recipe_sha256") != sha256_value(recipe)):
            raise CatalogError(f"Invalid private mirror manifest: {path}")
        expected = {artifact["id"] for artifact in recipe["artifacts"]}
        if set(manifest.get("artifacts", {})) != expected:
            raise CatalogError(f"Mirror artifact set differs from recipe: {path}")
        for record in manifest["artifacts"].values():
            relative = Path(record["path"])
            artifact_path = root / relative
            if (relative.is_absolute() or ".." in relative.parts or not artifact_path.is_file()
                    or artifact_path.is_symlink() or artifact_path.stat().st_size != record["size"]
                    or file_hash(artifact_path) != record["sha256"]):
                raise CatalogError(f"Invalid private mirror artifact: {relative}")
            total += record["size"]
        validated.append(name)
    if total > catalog["storage_limit_bytes"]:
        raise CatalogError("Private mirror exceeds the 100 GB catalog limit")
    return {"valid": True, "datasets": validated, "total_bytes": total,
            "storage_limit_bytes": catalog["storage_limit_bytes"]}
