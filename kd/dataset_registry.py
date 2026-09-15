"""Versioned dataset catalog with an optional private Git LFS mirror."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zipfile
import re


CATALOG_PATH = Path(__file__).with_name("catalog") / "v1.json"
PROFILES = {"balanced": 1.0, "lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}
MARKER = ".kd-dataset.json"


class CatalogError(ValueError):
    pass


class RegistryUnavailable(RuntimeError):
    pass


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


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RegistryUnavailable(f"Dataset registry command failed: {detail}")


def _registry_checkout(root: Path, registry: str, ref: str) -> Path:
    candidate = Path(registry).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if not re.fullmatch(r"catalog-v\d+\.\d+\.\d+", ref):
        raise RegistryUnavailable("Private registry ref must be an immutable catalog tag")
    checkout = root / ".registry" / "kd-repair"
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", registry, str(checkout)],
             env=environment)
    _run(["git", "fetch", "--tags", "origin"], cwd=checkout, env=environment)
    _run(["git", "checkout", "--detach", ref], cwd=checkout, env=environment)
    return checkout


def _mirror_files(root: Path, name: str, recipe: dict) -> dict[str, Path]:
    registry = os.environ.get("KD_DATASET_REGISTRY")
    if not registry:
        return {}
    checkout = _registry_checkout(root, registry,
                                  os.environ.get("KD_DATASET_REGISTRY_REF", "catalog-v1.0.0"))
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
        _run(["git", "lfs", "pull", "--include", ",".join(paths), "--exclude", ""], cwd=checkout)
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
                shutil.copyfileobj(response, temporary)
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
    artifacts = []
    for artifact in recipe["artifacts"]:
        destination = target / artifact["filename"]
        if not destination.is_file() or file_hash(destination, "md5") != artifact["md5"]:
            if artifact["id"] in mirror:
                shutil.copy2(mirror[artifact["id"]], destination)
                source = "private_registry"
            else:
                _download(artifact["url"], destination)
                source = "upstream"
        else:
            source = "local"
        if file_hash(destination, "md5") != artifact["md5"]:
            destination.unlink(missing_ok=True)
            raise CatalogError(f"Checksum mismatch for {name}/{artifact['id']}")
        if artifact.get("extract_to") is not None:
            _safe_extract(destination, target / artifact["extract_to"])
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
        "      - run: git clone --depth 1 --branch source-v0.2.0 https://github.com/spozi/kd-repair.git /tmp/kd-repair\n"
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
