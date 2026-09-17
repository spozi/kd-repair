import hashlib
import gzip
import json
import os
from pathlib import Path
import stat
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from kd.config import DataConfig, ExperimentConfig, TrainConfig
from kd.data import build_data
from kd.dataset_registry import (CatalogError, _loader_smoke_check,
                                 _materialize_dataset_layout, configure_registry,
                                 create_mirror_manifest, fetch_dataset, initialize_registry,
                                 load_catalog, registry_status, sha256_value,
                                 validate_registry, verify_dataset)


# Dataset preparation reports progress on stderr; unittest discovery may import this
# module outside the package, so silence it here as well as in tests/__init__.py.
os.environ.setdefault("KD_PROGRESS", "0")


def _digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def _catalog(payload: bytes, url: str) -> dict:
    return {
        "schema_version": 1, "catalog_version": "test", "split_seed": 2026,
        "storage_limit_bytes": 1024, "profiles": ["balanced", "lt-if10", "lt-if50", "lt-if100"],
        "datasets": {"toy": {"version": "1", "classes": 2, "image_size": 32,
                               "validation_policy": "official", "license_url": "https://example.test/license",
                               "artifacts": [{"id": "raw", "url": url, "filename": "payload.bin",
                                              "md5": _digest(payload, "md5")}]}},
    }


class DatasetRegistryTests(unittest.TestCase):
    def test_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"schema_version": 99}')
            with self.assertRaisesRegex(CatalogError, "Unsupported"):
                load_catalog(path)

    def test_private_mirror_fetch_and_tamper_detection(self):
        payload = b"private dataset bytes"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.bin"
            source.write_bytes(payload)
            catalog = _catalog(payload, source.as_uri())
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            registry = root / "registry"
            registry.mkdir()
            mirror = registry / "datasets/toy/1/archives/payload.bin"
            mirror.parent.mkdir(parents=True)
            mirror.write_bytes(payload)
            manifest = {"schema_version": 1, "dataset": "toy", "version": "1",
                        "recipe_sha256": sha256_value(catalog["datasets"]["toy"]),
                        "terms_reviewed": True, "total_bytes": len(payload),
                        "artifacts": {"raw": {"path": "datasets/toy/1/archives/payload.bin",
                                                "size": len(payload),
                                                "sha256": _digest(payload, "sha256")}}}
            manifest_path = registry / "manifests/toy/1.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest))
            data_root = root / "data"
            with patch("kd.dataset_registry.CATALOG_PATH", catalog_path), \
                    patch.dict(os.environ, {"KD_DATASET_REGISTRY": str(registry)}, clear=False):
                result = fetch_dataset("toy", root=data_root)
                self.assertEqual(result["artifacts"][0]["source"], "private_registry")
                self.assertTrue(verify_dataset("toy", root=data_root)["verified"])
                (data_root / "toy/1/payload.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(CatalogError, "changed"):
                    verify_dataset("toy", root=data_root)

    def test_persisted_registry_config_is_used_without_shell_exports(self):
        payload = b"persistent private dataset"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.bin"
            source.write_bytes(payload)
            catalog = _catalog(payload, source.as_uri())
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            registry = root / "registry"
            mirror = registry / "datasets/toy/1/archives/payload.bin"
            mirror.parent.mkdir(parents=True)
            mirror.write_bytes(payload)
            manifest = {"schema_version": 1, "dataset": "toy", "version": "1",
                        "recipe_sha256": sha256_value(catalog["datasets"]["toy"]),
                        "terms_reviewed": True, "total_bytes": len(payload),
                        "artifacts": {"raw": {"path": "datasets/toy/1/archives/payload.bin",
                                                "size": len(payload),
                                                "sha256": _digest(payload, "sha256")}}}
            manifest_path = registry / "manifests/toy/1.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest))
            config_path = root / "datasets.json"
            configured = configure_registry(str(registry), "catalog-v1.2.0", path=config_path)
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
            self.assertEqual(configured["source"], "config_file")
            environment = {"KD_DATASET_REGISTRY": "", "KD_DATASET_REGISTRY_CONFIG": str(config_path)}
            with patch("kd.dataset_registry.CATALOG_PATH", catalog_path), \
                    patch.dict(os.environ, environment, clear=False):
                self.assertEqual(registry_status()["source"], "config_file")
                result = fetch_dataset("toy", root=root / "data")
            self.assertEqual(result["artifacts"][0]["source"], "private_registry")

    def test_registry_config_rejects_embedded_http_credentials_and_can_be_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "datasets.json"
            with self.assertRaisesRegex(CatalogError, "credentials"):
                configure_registry("https://token@example.test/private.git", path=path)
            configure_registry("ssh://git@example.test:2222/private.git", path=path)
            self.assertTrue(path.is_file())
            self.assertFalse(configure_registry(path=path, remove=True)["configured"])
            self.assertFalse(path.exists())

    def test_registry_initialization_manifest_and_validation(self):
        payload = b"small archive"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.bin"
            source.write_bytes(payload)
            catalog = _catalog(payload, source.as_uri())
            catalog_path = base / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            registry = base / "registry"
            with patch("kd.dataset_registry.CATALOG_PATH", catalog_path):
                initialize_registry(registry)
                archive = registry / "datasets/toy/1/archives/payload.bin"
                archive.parent.mkdir(parents=True)
                archive.write_bytes(payload)
                with self.assertRaisesRegex(CatalogError, "terms-reviewed"):
                    create_mirror_manifest("toy", registry)
                create_mirror_manifest("toy", registry, terms_reviewed=True)
                self.assertEqual(validate_registry(registry)["datasets"], ["toy"])
            self.assertIn("filter=lfs", (registry / ".gitattributes").read_text())

    def test_nested_archive_extraction_is_recipe_driven(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "sample.txt"
            payload.write_text("sample")
            nested = root / "objects.tar.gz"
            with tarfile.open(nested, "w:gz") as handle:
                handle.add(payload, arcname="objects/sample.txt")
            archive = root / "outer.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(nested, "bundle/objects.tar.gz")
            catalog = _catalog(archive.read_bytes(), archive.as_uri())
            recipe = catalog["datasets"]["toy"]
            recipe["artifacts"][0].update(
                filename="outer.zip", extract_to=".",
                nested_extract=[{"path": "bundle/objects.tar.gz", "extract_to": "."}])
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            with patch("kd.dataset_registry.CATALOG_PATH", catalog_path):
                fetch_dataset("toy", root=root / "data")
            self.assertEqual((root / "data/toy/1/objects/sample.txt").read_text(), "sample")

    def test_torchvision_layout_materialization_is_idempotent_and_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eurosat = root / "eurosat"
            for label in range(10):
                path = eurosat / "eurosat" / "EuroSAT_RGB" / f"class_{label}" / "one.jpg"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(label, 0, 0)).save(path)
            expected = {"eurosat/2750": "EuroSAT_RGB"}
            self.assertEqual(_materialize_dataset_layout("eurosat", eurosat), expected)
            self.assertEqual(_materialize_dataset_layout("eurosat", eurosat), expected)
            self.assertEqual(os.readlink(eurosat / "eurosat/2750"), "EuroSAT_RGB")
            self.assertEqual(_loader_smoke_check("eurosat", eurosat, 10)["class_count"], 10)

            caltech = root / "caltech101"
            categories = ["BACKGROUND_Google", *(f"class_{label:03d}" for label in range(101))]
            for label, category in enumerate(categories):
                path = caltech / "101_ObjectCategories" / category / "image_0001.jpg"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(label % 255, 0, 0)).save(path)
            expected = {"caltech101/101_ObjectCategories": "../101_ObjectCategories"}
            self.assertEqual(_materialize_dataset_layout("caltech101", caltech), expected)
            self.assertEqual(_materialize_dataset_layout("caltech101", caltech), expected)
            self.assertEqual(os.readlink(caltech / "caltech101/101_ObjectCategories"),
                             "../101_ObjectCategories")
            self.assertEqual(
                _loader_smoke_check("caltech101", caltech, 101)["class_count"], 101)

    def test_eurosat_fetch_materializes_and_verifies_torchvision_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for label in range(10):
                path = source / "EuroSAT_RGB" / f"class_{label}" / "one.jpg"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(label, 0, 0)).save(path)
            archive = root / "eurosat.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for path in source.rglob("*.jpg"):
                    handle.write(path, path.relative_to(source))
            catalog = _catalog(archive.read_bytes(), archive.as_uri())
            recipe = catalog["datasets"].pop("toy")
            recipe.update(classes=10, artifacts=[{
                **recipe["artifacts"][0], "filename": "eurosat.zip",
                "extract_to": "eurosat",
            }])
            catalog["datasets"]["eurosat"] = recipe
            catalog["storage_limit_bytes"] = archive.stat().st_size + 1
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog))
            data_root = root / "data"
            with patch("kd.dataset_registry.CATALOG_PATH", catalog_path), \
                    patch("kd.dataset_registry._mirror_files", return_value={}):
                fetched = fetch_dataset("eurosat", root=data_root)
                verified = verify_dataset("eurosat", root=data_root)
            target = data_root / "eurosat/1"
            self.assertEqual(fetched["materialized_layout"],
                             {"eurosat/2750": "EuroSAT_RGB"})
            self.assertEqual(fetched["loader_smoke"]["class_count"], 10)
            self.assertEqual(verified["loader_smoke"]["class_count"], 10)
            self.assertEqual(os.readlink(target / "eurosat/2750"), "EuroSAT_RGB")

    def test_layout_materialization_rejects_conflicting_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "eurosat/EuroSAT_RGB"
            source.mkdir(parents=True)
            conflict = root / "eurosat/2750"
            conflict.mkdir()
            with self.assertRaisesRegex(CatalogError, "Refusing to replace"):
                _materialize_dataset_layout("eurosat", root)

            caltech = root / "caltech"
            (caltech / "101_ObjectCategories").mkdir(parents=True)
            link = caltech / "caltech101/101_ObjectCategories"
            link.parent.mkdir()
            link.symlink_to("../wrong-directory", target_is_directory=True)
            with self.assertRaisesRegex(CatalogError, "unexpected target"):
                _materialize_dataset_layout("caltech101", caltech)


class FakeSVHN(Dataset):
    def __init__(self, root, split, download, transform):
        self.labels = np.repeat(np.arange(10), 100 if split == "train" else 2)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), color=(index % 255, 0, 0))
        return self.transform(image), int(self.labels[index])


class FakeCIFAR100(Dataset):
    def __init__(self, root, train, download, transform):
        self.targets = np.repeat(np.arange(100), 10 if train else 1).tolist()
        self.classes = [f"class_{i}" for i in range(100)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.new("RGB", (32, 32))), self.targets[index]


class FakeGTSRB(Dataset):
    def __init__(self, root, split, download, transform):
        count = 10 if split == "train" else 1
        self._samples = [(f"{label}-{i}.ppm", label) for label in range(43) for i in range(count)]
        self.transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self.transform(Image.new("RGB", (40, 30))), self._samples[index][1]


class FakeCaltech101(Dataset):
    def __init__(self, root, download, transform):
        self.categories = [f"class_{i}" for i in range(101)]
        self.y = np.repeat(np.arange(101), 10).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.transform(Image.new("L", (40, 30))), self.y[index]


class FakeSTL10(Dataset):
    classes = [f"class_{i}" for i in range(10)]

    def __init__(self, root, split, download, transform):
        count = 20 if split == "train" else 3
        self.labels = np.repeat(np.arange(10), count)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.new("RGB", (96, 96))), int(self.labels[index])


class FakeEuroSAT(Dataset):
    classes = [f"class_{i}" for i in range(10)]

    def __init__(self, root, download, transform):
        self.targets = np.repeat(np.arange(10), 20).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.new("RGB", (64, 64))), self.targets[index]


class CatalogDataTests(unittest.TestCase):
    def test_named_profile_is_deterministic_and_test_is_untouched(self):
        config = DataConfig(source="svhn", num_classes=10, image_size=32,
                            dataset_profile="lt-if10")
        with patch("kd.data.datasets.SVHN", FakeSVHN):
            first = build_data(config, TrainConfig(seed=42))
            second = build_data(config, TrainConfig(seed=43))
        self.assertEqual(first.train.dataset.indices, second.train.dataset.indices)
        self.assertEqual(first.val.dataset.indices, second.val.dataset.indices)
        self.assertEqual(len(first.test.dataset), 20)
        self.assertEqual(first.provenance["imbalance_factor"], 10.0)

    def test_profile_and_legacy_factor_cannot_both_be_set(self):
        with self.assertRaisesRegex(ValueError, "either dataset_profile"):
            ExperimentConfig(data=DataConfig(source="svhn", num_classes=10,
                                             dataset_profile="lt-if10",
                                             imbalance_factor=10)).validate(require_teacher=False)

    def test_cifar100_and_gtsrb_adapters_preserve_official_tests(self):
        cases = (("cifar100", 100, "CIFAR100", FakeCIFAR100),
                 ("gtsrb", 43, "GTSRB", FakeGTSRB))
        for source, classes, attribute, fake in cases:
            with self.subTest(source=source), patch(f"kd.data.datasets.{attribute}", fake):
                bundle = build_data(DataConfig(source=source, num_classes=classes, image_size=32),
                                    TrainConfig())
                self.assertEqual(len(bundle.classes), classes)
                self.assertEqual(len(bundle.test.dataset), classes)
                self.assertEqual(len(bundle.provenance["train_per_class"]), classes)

    def test_cinic10_uses_official_validation_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "valid", "test"):
                for label in range(10):
                    target = root / split / f"class_{label}"
                    target.mkdir(parents=True)
                    Image.new("RGB", (32, 32), color=(label, 0, 0)).save(target / "one.png")
            bundle = build_data(DataConfig(source="cinic10", root=str(root), num_classes=10,
                                           image_size=32), TrainConfig())
            self.assertEqual(len(bundle.train.dataset), 10)
            self.assertEqual(len(bundle.val.dataset), 10)
            self.assertEqual(len(bundle.test.dataset), 10)
            self.assertEqual(bundle.provenance["validation_policy"], "Official validation split")

    def test_medmnist_preserves_official_validation_and_test_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {}
            for split, per_class in (("train", 10), ("val", 2), ("test", 1)):
                payload[f"{split}_images"] = np.zeros((9 * per_class, 28, 28, 3), dtype=np.uint8)
                payload[f"{split}_labels"] = np.repeat(np.arange(9), per_class)[:, None]
            np.savez_compressed(root / "pathmnist.npz", **payload)
            bundle = build_data(DataConfig(source="pathmnist", root=str(root), num_classes=9,
                                           image_size=32), TrainConfig())
            self.assertEqual((len(bundle.train.dataset), len(bundle.val.dataset),
                              len(bundle.test.dataset)), (90, 18, 9))
            self.assertEqual(bundle.provenance["validation_policy"], "Official validation split")

    def test_long_tailed_profile_keeps_rare_classes_in_small_official_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {}
            for split, per_class in (("train", 10), ("val", 2), ("test", 1)):
                payload[f"{split}_images"] = np.zeros(
                    (9 * per_class, 28, 28, 3), dtype=np.uint8)
                payload[f"{split}_labels"] = np.repeat(np.arange(9), per_class)[:, None]
            np.savez_compressed(root / "pathmnist.npz", **payload)
            bundle = build_data(
                DataConfig(source="pathmnist", root=str(root), num_classes=9,
                           image_size=32, dataset_profile="lt-if100"),
                TrainConfig())
            self.assertTrue(all(count >= 1 for count in bundle.provenance["train_per_class"]))
            self.assertTrue(all(count >= 1 for count in bundle.provenance["val_per_class"]))
            self.assertEqual(len(bundle.test.dataset), 9)
            self.assertEqual(bundle.provenance["imbalance_minimum_per_class"], 1)

    def test_caltech_protocol_split_is_deterministic_and_disjoint(self):
        config = DataConfig(source="caltech101", num_classes=101, image_size=32)
        with patch("kd.data.datasets.Caltech101", FakeCaltech101):
            first = build_data(config, TrainConfig(seed=42))
            second = build_data(config, TrainConfig(seed=43))
        train_indices = set(first.train.dataset.indices)
        val_indices = set(first.val.dataset.indices)
        test_indices = set(first.test.dataset.indices)
        self.assertFalse(train_indices & val_indices or train_indices & test_indices or val_indices & test_indices)
        self.assertEqual(len(train_indices | val_indices | test_indices), 1010)
        self.assertEqual(first.test.dataset.indices, second.test.dataset.indices)
        self.assertEqual(first.provenance["test_per_class"], [2] * 101)
        images, _ = next(iter(first.train))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))

    def test_stl10_loader_downsamples_without_touching_test(self):
        with patch("kd.data.datasets.STL10", FakeSTL10):
            bundle = build_data(DataConfig(source="stl10", num_classes=10, image_size=32),
                                TrainConfig(batch_size=4))
        images, _ = next(iter(bundle.train))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
        self.assertEqual(len(bundle.test.dataset), 30)

    def test_fashionmnist_reads_canonical_idx_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, count in (("train", 100), ("t10k", 10)):
                image_name = ("train-images-idx3-ubyte.gz" if prefix == "train"
                              else "t10k-images-idx3-ubyte.gz")
                label_name = ("train-labels-idx1-ubyte.gz" if prefix == "train"
                              else "t10k-labels-idx1-ubyte.gz")
                with gzip.open(root / image_name, "wb") as handle:
                    handle.write(struct.pack(">IIII", 2051, count, 28, 28))
                    handle.write(bytes(count * 28 * 28))
                with gzip.open(root / label_name, "wb") as handle:
                    handle.write(struct.pack(">II", 2049, count))
                    handle.write(bytes(i % 10 for i in range(count)))
            bundle = build_data(DataConfig(source="fashionmnist", root=str(root),
                                           num_classes=10, image_size=32),
                                TrainConfig(batch_size=2))
            images, _ = next(iter(bundle.train))
            self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
            self.assertEqual(len(bundle.test.dataset), 10)

    def test_eurosat_gets_a_pinned_protocol_test_split(self):
        with patch("kd.data.datasets.EuroSAT", FakeEuroSAT):
            bundle = build_data(DataConfig(source="eurosat", num_classes=10, image_size=32),
                                TrainConfig())
        self.assertEqual(len(bundle.test.dataset), 20)
        self.assertEqual(bundle.provenance["test_per_class"], [2] * 10)


if __name__ == "__main__":
    unittest.main()
