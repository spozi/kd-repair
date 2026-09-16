"""Versioned dataset catalog with an optional private Git LFS mirror."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile
import re


CATALOG_PATH = Path(__file__).with_name("catalog") / "v1.json"
PROFILES = {"balanced": 1.0, "lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}
MARKER = ".kd-dataset.json"
DEFAULT_REGISTRY_REF = "catalog-v1.2.0"
REGISTRY_CONFIG_ENV = "KD_DATASET_REGISTRY_CONFIG"


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
                     final: bool = False) -> None:
    if not _progress_enabled():
        return
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


def _copy_with_progress(response, handle, label: str, total: int) -> int:
    copied = 0
    started = time.monotonic()
    last_report = started
    interval = _progress_interval()
    while True:
        block = response.read(256 * 1024)
        if not block:
            break
        handle.write(block)
        copied += len(block)
        now = time.monotonic()
        if now - last_report >= interval:
            last_report = now
            _report_transfer(label, copied, total, now - started)
    _report_transfer(label, copied, total, time.monotonic() - started, final=True)
    return copied


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


def configure_registry(registry: str | None = None, ref: str = DEFAULT_REGISTRY_REF, *,
                       path: str | Path | None = None, remove: bool = False) -> dict:
    location = registry_config_path(path)
    if remove:
        if registry is not None:
            raise CatalogError("Do not provide a registry URL with --remove")
        location.unlink(missing_ok=True)
        return {"configured": False, "path": str(location)}
    if registry is None:
        raise CatalogError("Registry URL is required unless --remove is used")
    _validate_registry_settings(registry, ref)
    payload = {"schema_version": 1, "registry": registry, "ref": ref}
    location.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=location.parent, delete=False) as temporary:
        temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    os.replace(temporary_path, location)
    return {"configured": True, "registry": registry, "ref": ref,
            "source": "config_file", "path": str(location)}


def registry_status(path: str | Path | None = None) -> dict:
    registry = os.environ.get("KD_DATASET_REGISTRY")
    if registry:
        ref = os.environ.get("KD_DATASET_REGISTRY_REF", DEFAULT_REGISTRY_REF)
        _validate_registry_settings(registry, ref)
        return {"configured": True, "registry": registry, "ref": ref,
                "source": "environment", "path": None}
    location = registry_config_path(path)
    if not location.is_file():
        return {"configured": False, "source": None, "path": str(location)}
    try:
        payload = json.loads(location.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot load dataset registry configuration {location}: {error}") from error
    if payload.get("schema_version") != 1:
        raise CatalogError("Unsupported dataset registry configuration schema")
    registry, ref = payload.get("registry"), payload.get("ref")
    _validate_registry_settings(registry, ref)
    return {"configured": True, "registry": registry, "ref": ref,
            "source": "config_file", "path": str(location)}


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
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_LFS_FORCE_PROGRESS": "1"}
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


def _mirror_files(root: Path, name: str, recipe: dict) -> dict[str, Path]:
    settings = registry_status()
    if not settings["configured"]:
        return {}
    checkout = _registry_checkout(root, settings["registry"], settings["ref"])
    manifest_path = checkout / "manifests" / name / f"{recipe['version']}.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryUnavailable(f"Private mirror manifest unavailable: {error}") from error
    if manifest.get("schema_version") != 1 or manifest.get("recipe_sha256") != sha256_value(recipe):
        raise RegistryUnavailable("Private mirror manifest does not match the public recipe")
    records = manifest.get("artifacts", {})
    paths = []
    for artifact in recipe["artifacts"]:
        record = records.get(artifact["id"], {})
        relative = record.get("path")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RegistryUnavailable(f"Invalid private mirror path for {artifact['id']}")
        paths.append(relative)
    if (checkout / ".git").exists():
        _run(["git", "lfs", "pull", "--include", ",".join(paths), "--exclude", ""], cwd=checkout,
             env={**os.environ, "GIT_LFS_FORCE_PROGRESS": "1"},
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


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "kd-repair-dataset-fetch/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request) as response:
                total = int(response.headers.get("Content-Length") or 0)
                _copy_with_progress(response, temporary, destination.name, total)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


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


def fetch_dataset(name: str, version: str | None = None, profile: str = "balanced",
                  root: str | Path = "data") -> dict:
    if profile not in PROFILES:
        raise CatalogError(f"Unknown dataset profile: {profile}")
    catalog, recipe = dataset_recipe(name, version)
    target = dataset_directory(root, name, recipe["version"])
    target.mkdir(parents=True, exist_ok=True)
    mirror, mirror_error = {}, None
    try:
        mirror = _mirror_files(Path(root), name, recipe)
    except RegistryUnavailable as error:
        mirror_error = str(error)
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
                _status(f"{prefix}: downloading from {urlsplit(artifact['url']).netloc}")
                _download(artifact["url"], destination)
                source = "upstream"
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
    marker = {"schema_version": 1, "catalog_version": catalog["catalog_version"],
              "catalog_sha256": sha256_value(catalog), "recipe_sha256": sha256_value(recipe),
              "dataset": name, "version": recipe["version"], "requested_profile": profile,
              "profiles": list(PROFILES),
              "split_seed": catalog["split_seed"], "artifacts": artifacts}
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
    return {**marker, "profile": profile, "verified": True, "total_bytes": total}


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
