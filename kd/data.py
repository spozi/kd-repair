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


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalization(source: str):
    # Fixed [-1, 1] scaling for CIFAR: no validation/test statistics are fitted.
    return ((0.5,) * 3, (0.5,) * 3) if source == "cifar10" else (MEAN, STD)


def image_transform(config: DataConfig, *, training: bool):
    size = config.image_size
    mean, std = normalization(config.source)
    if config.source == "cifar10":
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


def build_data(data: DataConfig, train: TrainConfig, *, include_test: bool = True,
               include_confirmation: bool = False, diagnostic: bool = False) -> DataBundle:
    """Build stable splits; diagnostic mode uses canonical transforms and ordering."""
    splits = {}
    provenance = {"source": data.source}
    if data.source == "synthetic":
        classes = [f"class_{i}" for i in range(data.num_classes)]
        for i, (split, count) in enumerate((("train", data.train_samples), ("val", data.val_samples),
                                           ("test", data.test_samples))):
            if split == "test" and not include_test:
                continue
            splits[split] = SyntheticImages(count, data.num_classes, data.image_size,
                                            train.seed + i * 1_000_000,
                                            image_transform(data, training=split == "train" and not diagnostic))
    elif data.source == "cifar10":
        training = datasets.CIFAR10(data.root, train=True, download=data.download,
                                   transform=image_transform(data, training=not diagnostic))
        validation = datasets.CIFAR10(data.root, train=True, download=False,
                                     transform=image_transform(data, training=False))
        confirmation_indices = None
        if data.confirmation_fraction:
            train_indices, val_indices, confirmation_indices = stratified_confirmation_split(
                training.targets, data.validation_fraction, data.confirmation_fraction,
                data.split_seed)
        else:
            train_indices, val_indices = stratified_split(
                training.targets, data.validation_fraction, data.split_seed)
        if data.imbalance_factor > 1:
            # Train and validation share the profile: the calibration set is as
            # skewed as training, which is the setting where a single global
            # temperature fitted on validation fails to transfer to a balanced
            # test split. The official test split is never resampled.
            train_indices = long_tailed_subset(training.targets, train_indices,
                                               data.imbalance_factor, data.split_seed)
            val_indices = long_tailed_subset(training.targets, val_indices,
                                             data.imbalance_factor, data.split_seed + 1)
        splits["train"] = Subset(training, train_indices)
        splits["val"] = Subset(validation, val_indices)
        if include_confirmation and confirmation_indices is not None:
            confirmation = datasets.CIFAR10(data.root, train=True, download=False,
                                            transform=image_transform(data, training=False))
            splits["confirmation"] = Subset(confirmation, confirmation_indices)
        classes = training.classes
        if include_test:
            splits["test"] = datasets.CIFAR10(data.root, train=False, download=False,
                                             transform=image_transform(data, training=False))
        provenance.update(split_seed=data.split_seed, validation_fraction=data.validation_fraction,
                          train_index_sha256=index_hash(train_indices), val_index_sha256=index_hash(val_indices),
                          train_per_class=np.bincount(np.asarray(training.targets)[train_indices]).tolist(),
                          val_per_class=np.bincount(np.asarray(training.targets)[val_indices]).tolist(),
                          test_policy="Official test split is loaded only for explicit final evaluation")
        if data.imbalance_factor > 1:
            # Recorded only when imbalanced, so balanced runs keep byte-identical
            # provenance and the completed studies still verify against it.
            provenance.update(imbalance_factor=data.imbalance_factor,
                              imbalance_profile="Exponential over dataset label order; train and validation resampled, test untouched")
        if confirmation_indices is not None:
            provenance.update(
                confirmation_fraction=data.confirmation_fraction,
                confirmation_index_sha256=index_hash(confirmation_indices),
                confirmation_per_class=np.bincount(
                    np.asarray(training.targets)[confirmation_indices]).tolist(),
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
