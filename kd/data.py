"""One augmentation per image, shared by teacher and student."""

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DataConfig, TrainConfig
from .dataset_registry import PROFILES, dataset_directory, dataset_recipe, sha256_value


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalization(source: str):
    # Fixed [-1, 1] scaling for CIFAR: no validation/test statistics are fitted.
    fixed = {
        "cifar10": ((0.5,) * 3, (0.5,) * 3),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "svhn": ((0.5,) * 3, (0.5,) * 3),
        "cinic10": ((0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835)),
    }
    return fixed.get(source, (MEAN, STD))


def image_transform(config: DataConfig, *, training: bool):
    size = config.image_size
    mean, std = normalization(config.source)
    if config.source in {"cifar10", "cifar100", "svhn", "cinic10"}:
        operations = [transforms.RandomCrop(32, padding=4)] if training else []
        if training and config.horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip())
        if training and config.augmentation == "strong":
            operations.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.02))
        return transforms.Compose([*operations, transforms.ToTensor(), transforms.Normalize(mean, std)])
    if training:
        scale = (0.6, 1.0) if config.augmentation == "strong" else (0.85, 1.0)
        operations = [transforms.RandomResizedCrop(size, scale=scale, ratio=(0.9, 1.1))]
        # Directional traffic signs need class remapping when mirrored.
        if config.horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip())
        if config.augmentation == "strong":
            operations.extend([transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
                               transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2)])
    else:
        operations = [transforms.Resize(size + max(2, size // 8)), transforms.CenterCrop(size)]
    return transforms.Compose([*operations, transforms.ToTensor(), transforms.Normalize(mean, std)])


class SyntheticImages(Dataset):
    """Deterministic, learnable colored patterns for offline checks, not a benchmark."""

    def __init__(self, count: int, classes: int, size: int, seed: int, transform):
        self.count, self.classes, self.size = count, classes, size
        self.seed, self.transform = seed, transform

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        label = index % self.classes
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((3, self.size, self.size), generator=generator) * 0.15
        image[label % 3] += 0.5
        period = 2 + label // 3
        image[:, ::period, :] += 0.25
        array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return self.transform(Image.fromarray(array)), label


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    classes: list[str]
    test: DataLoader | None = None
    provenance: dict = field(default_factory=dict)
    confirmation: DataLoader | None = None


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def stratified_split(targets, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Fixed class-balanced holdout, independent of model initialization seed."""
    labels = np.asarray(targets)
    generator = np.random.default_rng(seed)
    training, validation = [], []
    for label in np.unique(labels):
        indices = generator.permutation(np.flatnonzero(labels == label))
        count = round(len(indices) * fraction)
        if count < 1 or count >= len(indices):
            raise ValueError("Each class needs at least one training and validation example")
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    return sorted(training), sorted(validation)


def stratified_confirmation_split(targets, validation_fraction: float,
                                  confirmation_fraction: float,
                                  seed: int) -> tuple[list[int], list[int], list[int]]:
    """Create disjoint train/validation/confirmation partitions per class."""
    labels = np.asarray(targets)
    generator = np.random.default_rng(seed)
    training, validation, confirmation = [], [], []
    for label in np.unique(labels):
        indices = generator.permutation(np.flatnonzero(labels == label))
        val_count = round(len(indices) * validation_fraction)
        confirmation_count = round(len(indices) * confirmation_fraction)
        if min(val_count, confirmation_count) < 1 or val_count + confirmation_count >= len(indices):
            raise ValueError("Each class needs train, validation, and confirmation examples")
        validation.extend(indices[:val_count].tolist())
        confirmation.extend(indices[val_count:val_count + confirmation_count].tolist())
        training.extend(indices[val_count + confirmation_count:].tolist())
    return sorted(training), sorted(validation), sorted(confirmation)


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def long_tailed_subset(targets, indices: list[int], factor: float, seed: int) -> list[int]:
    """Exponential class-imbalance profile over an existing index list.

    Class k keeps `round(n_k * factor ** (-k / (K - 1)))` of its examples, so the
    first class keeps all of them and the last keeps roughly `1 / factor` of them.
    Class order is the dataset's own label order, not a difficulty ranking.

    A factor of exactly one returns the input unchanged, so balanced studies keep
    byte-identical splits. Selection is deterministic given the split seed and
    independent of the model seed, matching `stratified_split`.
    """
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("Imbalance factor must be finite and at least one")
    if factor == 1:
        return list(indices)
    labels = np.asarray(targets)
    present = np.unique(labels[np.asarray(indices, dtype=np.int64)])
    if len(present) < 2:
        raise ValueError("Long-tailed sampling needs at least two classes")
    generator = np.random.default_rng(seed)
    kept: list[int] = []
    for position, label in enumerate(present):
        available = np.asarray(indices, dtype=np.int64)
        available = available[labels[available] == label]
        share = factor ** (-position / (len(present) - 1))
        count = int(round(len(available) * share))
        if count < 1:
            raise ValueError(f"Imbalance factor {factor} leaves class {label} empty; lower it")
        kept.extend(generator.permutation(available)[:count].tolist())
    return sorted(kept)


def _effective_factor(data: DataConfig) -> float:
    return PROFILES[data.dataset_profile] if data.dataset_profile != "balanced" else data.imbalance_factor


def _dataset_root(data: DataConfig) -> Path:
    if data.dataset_version is None:
        return Path(data.root)
    return dataset_directory(data.root, data.source, data.dataset_version)


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    if hasattr(dataset, "_samples"):
        return np.asarray([label for _, label in dataset._samples], dtype=np.int64)
    raise TypeError(f"Dataset {type(dataset).__name__} does not expose classification targets")


def _native_dataset(source: str, root: Path, split: str, transform, download: bool):
    if source in {"cifar10", "cifar100"}:
        cls = datasets.CIFAR10 if source == "cifar10" else datasets.CIFAR100
        return cls(root, train=split == "train", download=download, transform=transform)
    if source == "svhn":
        return datasets.SVHN(root, split=split, download=download, transform=transform)
    if source == "gtsrb":
        return datasets.GTSRB(root, split=split, download=download, transform=transform)
    if source == "cinic10":
        directory = root / ({"train": "train", "val": "valid", "test": "test"}[split])
        if not directory.is_dir():
            raise ValueError(f"Missing CINIC-10 split: {directory}; run `kd dataset fetch cinic10`")
        return datasets.ImageFolder(directory, transform)
    raise ValueError(f"No native dataset adapter for {source}")


def _catalog_provenance(data: DataConfig) -> dict:
    if data.dataset_version is None:
        return {}
    catalog, recipe = dataset_recipe(data.source, data.dataset_version)
    marker = _dataset_root(data) / ".kd-dataset.json"
    result = {"dataset_version": data.dataset_version, "dataset_profile": data.dataset_profile,
              "catalog_version": catalog["catalog_version"],
              "catalog_sha256": sha256_value(catalog), "recipe_sha256": sha256_value(recipe)}
    if marker.is_file():
        import json
        prepared = json.loads(marker.read_text())
        if prepared.get("recipe_sha256") != result["recipe_sha256"]:
            raise ValueError("Prepared dataset belongs to another catalog recipe")
        result["artifact_sha256"] = {row["id"]: row["sha256"]
                                     for row in prepared.get("artifacts", [])}
    return result


def _freeze_split_indices(root: Path, profile: str, train_indices: list[int],
                          val_indices: list[int]) -> str:
    directory = root / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile}.npz"
    train_array = np.asarray(train_indices, dtype="<i8")
    val_array = np.asarray(val_indices, dtype="<i8")
    digest = hashlib.sha256(profile.encode("ascii") + train_array.tobytes()
                            + val_array.tobytes()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if (stored["sha256"].item() != digest
                    or not np.array_equal(stored["train"], train_array)
                    or not np.array_equal(stored["val"], val_array)):
                raise ValueError(f"Frozen dataset split changed: {path}")
    else:
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, train=train_array, val=val_array,
                                sha256=np.asarray(digest))
        temporary.replace(path)
    return digest


def build_data(data: DataConfig, train: TrainConfig, *, include_test: bool = True,
               include_confirmation: bool = False, diagnostic: bool = False) -> DataBundle:
    """Build stable splits; diagnostic mode uses canonical transforms and ordering."""
    splits = {}
    provenance = {"source": data.source, **_catalog_provenance(data)}
    if data.source == "synthetic":
        classes = [f"class_{i}" for i in range(data.num_classes)]
        for i, (split, count) in enumerate((("train", data.train_samples), ("val", data.val_samples),
                                           ("test", data.test_samples))):
            if split == "test" and not include_test:
                continue
            splits[split] = SyntheticImages(count, data.num_classes, data.image_size,
                                            train.seed + i * 1_000_000,
                                            image_transform(data, training=split == "train" and not diagnostic))
    elif data.source in {"cifar10", "cifar100", "svhn", "cinic10", "gtsrb"}:
        root = _dataset_root(data)
        training = _native_dataset(data.source, root, "train",
                                   image_transform(data, training=not diagnostic), data.download)
        factor = _effective_factor(data)
        confirmation_indices = None
        if data.source == "cinic10":
            validation = _native_dataset(data.source, root, "val",
                                         image_transform(data, training=False), False)
            train_targets, val_targets = _targets(training), _targets(validation)
            train_indices, val_indices = list(range(len(training))), list(range(len(validation)))
            validation_policy = "Official validation split"
        else:
            validation = _native_dataset(data.source, root, "train",
                                         image_transform(data, training=False), False)
            train_targets = val_targets = _targets(training)
            validation_policy = "Stratified holdout from official training split"
        if data.confirmation_fraction:
            train_indices, val_indices, confirmation_indices = stratified_confirmation_split(
                train_targets, data.validation_fraction, data.confirmation_fraction,
                data.split_seed)
        elif data.source != "cinic10":
            train_indices, val_indices = stratified_split(
                train_targets, data.validation_fraction, data.split_seed)
        if factor > 1:
            # Train and validation share the profile: the calibration set is as
            # skewed as training, which is the setting where a single global
            # temperature fitted on validation fails to transfer to a balanced
            # test split. The official test split is never resampled.
            train_indices = long_tailed_subset(train_targets, train_indices, factor, data.split_seed)
            val_indices = long_tailed_subset(val_targets, val_indices, factor, data.split_seed + 1)
        splits["train"] = Subset(training, train_indices)
        splits["val"] = Subset(validation, val_indices)
        if include_confirmation and confirmation_indices is not None:
            confirmation = _native_dataset(data.source, root, "train",
                                           image_transform(data, training=False), False)
            splits["confirmation"] = Subset(confirmation, confirmation_indices)
        if data.source == "svhn":
            classes = [str(i) for i in range(10)]
        elif data.source == "gtsrb":
            classes = [str(i) for i in range(43)]
        else:
            classes = training.classes
        if include_test:
            splits["test"] = _native_dataset(data.source, root, "test",
                                             image_transform(data, training=False), data.download)
        provenance.update(split_seed=data.split_seed, validation_fraction=data.validation_fraction,
                          train_index_sha256=index_hash(train_indices), val_index_sha256=index_hash(val_indices),
                          train_per_class=np.bincount(train_targets[train_indices], minlength=data.num_classes).tolist(),
                          val_per_class=np.bincount(val_targets[val_indices], minlength=data.num_classes).tolist(),
                          test_policy="Official test split is loaded only for explicit final evaluation")
        if data.source != "cifar10" or data.dataset_version is not None:
            provenance["validation_policy"] = validation_policy
        if factor > 1:
            # Recorded only when imbalanced, so balanced runs keep byte-identical
            # provenance and the completed studies still verify against it.
            provenance.update(imbalance_factor=factor,
                              imbalance_profile="Exponential over dataset label order; train and validation resampled, test untouched")
        if data.dataset_version is not None:
            provenance["split_indices_sha256"] = _freeze_split_indices(
                root, data.dataset_profile, train_indices, val_indices)
        if confirmation_indices is not None:
            provenance.update(
                confirmation_fraction=data.confirmation_fraction,
                confirmation_index_sha256=index_hash(confirmation_indices),
                confirmation_per_class=np.bincount(
                    train_targets[confirmation_indices], minlength=data.num_classes).tolist(),
                confirmation_policy=("Balanced sealed holdout excluded from training, validation, "
                                     "and checkpoint selection"))
    else:
        root = Path(data.root)
        for split in ("train", "val", "test"):
            if split == "test" and not include_test:
                continue
            directory = root / split
            if split == "test" and not directory.exists():
                continue
            if not directory.is_dir():
                raise ValueError(f"Missing dataset split: {directory}")
            splits[split] = datasets.ImageFolder(
                directory, image_transform(data, training=split == "train" and not diagnostic))
        classes = splits["train"].classes
        if len(classes) != data.num_classes:
            raise ValueError(f"Expected {data.num_classes} classes, found {len(classes)}: {classes}")
        for split, dataset in splits.items():
            if dataset.class_to_idx != splits["train"].class_to_idx:
                raise ValueError(f"Class names/order differ between train and {split}")
    loaders = {}
    for split, dataset in splits.items():
        loaders[split] = DataLoader(dataset, batch_size=train.batch_size,
                                    shuffle=split == "train" and not diagnostic,
                                    num_workers=train.workers, worker_init_fn=seed_worker,
                                    generator=torch.Generator().manual_seed(train.seed))
    provenance["split_sizes"] = {name: len(dataset) for name, dataset in splits.items()}
    return DataBundle(loaders["train"], loaders["val"], classes, loaders.get("test"), provenance,
                      loaders.get("confirmation"))
