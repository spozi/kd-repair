"""Validated experiment configuration, with no mutable global settings."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    root: str = "data"
    num_classes: int = 4
    image_size: int = 32
    train_samples: int = 96
    val_samples: int = 48
    test_samples: int = 48
    augmentation: str = "basic"
    horizontal_flip: bool = False
    download: bool = False
    validation_fraction: float = 0.1
    # Optional balanced holdout drawn from the training archive and excluded
    # from both fitting and validation-driven selection.
    confirmation_fraction: float = 0.0
    split_seed: int = 2026
    # 1.0 keeps every class balanced; >1 applies an exponential long-tailed
    # profile to the training and validation splits only.
    imbalance_factor: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "tiny_small"
    checkpoint: str | None = None


@dataclass(frozen=True)
class DistillationConfig:
    method: str = "dkd"
    temperature: float = 4.0
    weight: float = 0.5
    alpha: float = 1.0
    beta: float = 8.0
    warmup_epochs: int = 5
    feature_weight: float = 0.0
    # Explicit semantic pairing, not positional pairing of arbitrary layers.
    stage_pairs: tuple[tuple[str, str], ...] = (("stage2", "stage2"), ("stage3", "stage3"))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    workers: int = 0
    seed: int = 42
    device: str = "auto"
    threads: int = 2


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = False
    max_weight: float = 1.0
    kernel_bandwidth: float = 0.4
    monitor_start_epoch: int = 5
    ece_threshold: float = 0.03
    overconfidence_threshold: float = 0.01
    patience: int = 3
    ramp_epochs: int = 3


@dataclass(frozen=True)
class CPCConfig:
    enabled: bool = False
    discrimination_weight: float = 0.1
    exclusion_weight: float = 0.1
    # Zero preserves the original full-weight-from-epoch-1 behavior exactly.
    warmup_epochs: int = 0

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("cpc.enabled must be a boolean")
        for name in ("discrimination_weight", "exclusion_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"cpc.{name} must be finite and nonnegative")
        if not isinstance(self.warmup_epochs, int) or isinstance(self.warmup_epochs, bool) or self.warmup_epochs < 0:
            raise ValueError("cpc.warmup_epochs must be a nonnegative integer")


@dataclass(frozen=True)
class SurgeryConfig:
    """Opt-in removal of samples selected by a versioned surgery plan."""

    action: str = "none"
    plan: str | None = None
    max_drop_fraction: float = 0.2

    def validate(self) -> None:
        if self.action not in {"none", "drop"}:
            raise ValueError("surgery.action must be none or drop")
        if self.action == "drop" and not self.plan:
            raise ValueError("surgery.plan is required when surgery.action is drop")
        if self.action == "none" and self.plan is not None:
            raise ValueError("Set surgery.action to drop when a surgery.plan is supplied")
        if (isinstance(self.max_drop_fraction, bool)
                or not math.isfinite(self.max_drop_fraction)
                or not 0 < self.max_drop_fraction < 1):
            raise ValueError("surgery.max_drop_fraction must be finite and in (0, 1)")


@dataclass(frozen=True)
class SupervisedLossConfig:
    method: str = "cross_entropy"
    bins: int = 15
    update_rate: float = 1.0
    gamma_min: float = -2.0
    gamma_max: float = 20.0
    switch_threshold: float = 0.2

    def validate(self) -> None:
        if self.method not in {"cross_entropy", "adafocal", "adadualfocal"}:
            raise ValueError("supervised_loss.method must be cross_entropy, adafocal, or adadualfocal")
        if not isinstance(self.bins, int) or isinstance(self.bins, bool) or self.bins < 1:
            raise ValueError("supervised_loss.bins must be a positive integer")
        for name in ("update_rate", "gamma_min", "gamma_max", "switch_threshold"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"supervised_loss.{name} must be finite")
        if self.update_rate <= 0 or not 0 < self.switch_threshold <= 1:
            raise ValueError("update_rate must be positive and switch_threshold in (0, 1]")
        if self.gamma_max < 1 or self.gamma_min > -self.switch_threshold:
            raise ValueError("Gamma bounds must contain initial gamma=1 and both switching thresholds")


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 5
    iterations: int = 20
    max_parameters: int | None = None
    max_flops: int | None = None
    max_latency_ms: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "runs"
    task: str = "classification"
    data: DataConfig = field(default_factory=DataConfig)
    student: ModelConfig = field(default_factory=ModelConfig)
    teacher: ModelConfig = field(default_factory=lambda: ModelConfig(name="tiny_medium"))
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    supervised_loss: SupervisedLossConfig = field(default_factory=SupervisedLossConfig)
    cpc: CPCConfig = field(default_factory=CPCConfig)
    surgery: SurgeryConfig = field(default_factory=SurgeryConfig)

    def validate(self, *, require_teacher: bool = True) -> None:
        if self.task != "classification":
            raise ValueError("Only classification is supported. Detection/segmentation require task-specific adapters and metrics.")
        if not self.name or self.name in {".", ".."} or Path(self.name).name != self.name:
            raise ValueError("name must be a single directory name")
        if self.data.source not in {"synthetic", "imagefolder", "cifar10"}:
            raise ValueError("data.source must be synthetic, imagefolder, or cifar10")
        if self.data.source == "cifar10" and (self.data.num_classes != 10 or self.data.image_size != 32):
            raise ValueError("CIFAR-10 requires num_classes=10 and image_size=32")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if (isinstance(self.data.confirmation_fraction, bool)
                or not math.isfinite(self.data.confirmation_fraction)
                or not 0 <= self.data.confirmation_fraction < 1):
            raise ValueError("confirmation_fraction must be finite and in [0, 1)")
        if self.data.validation_fraction + self.data.confirmation_fraction >= 1:
            raise ValueError("validation_fraction + confirmation_fraction must be below one")
        if self.data.confirmation_fraction and self.data.source != "cifar10":
            raise ValueError("A sealed confirmation split is implemented for cifar10 only")
        if (isinstance(self.data.imbalance_factor, bool) or not math.isfinite(self.data.imbalance_factor)
                or self.data.imbalance_factor < 1):
            raise ValueError("data.imbalance_factor must be finite and at least one")
        if self.data.imbalance_factor > 1 and self.data.source != "cifar10":
            raise ValueError("Long-tailed sampling is implemented for the cifar10 source only")
        if self.data.augmentation not in {"basic", "strong"}:
            raise ValueError("data.augmentation must be basic or strong")
        for name, value in {
            "num_classes": self.data.num_classes, "image_size": self.data.image_size,
            "train_samples": self.data.train_samples, "val_samples": self.data.val_samples,
            "test_samples": self.data.test_samples, "epochs": self.train.epochs,
            "batch_size": self.train.batch_size, "threads": self.train.threads,
            "iterations": self.benchmark.iterations,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.data.num_classes < 2 or self.data.image_size < 16:
            raise ValueError("At least two classes and image_size >= 16 are required")
        if self.data.source == "synthetic" and min(self.data.train_samples, self.data.val_samples, self.data.test_samples) < self.data.num_classes:
            raise ValueError("Each synthetic split must contain every class")
        for name, value in {"workers": self.train.workers, "seed": self.train.seed,
                            "warmup": self.benchmark.warmup, "split_seed": self.data.split_seed,
                            "warmup_epochs": self.distillation.warmup_epochs}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        d = self.distillation
        self.supervised_loss.validate()
        self.cpc.validate()
        self.surgery.validate()
        if self.cpc.enabled:
            if d.method != "kd" or self.supervised_loss.method != "cross_entropy":
                raise ValueError("CPC currently requires standard KD with cross-entropy supervision")
            if self.calibration.enabled or d.feature_weight:
                raise ValueError("CPC cannot be combined with MMCE or feature losses in this experiment")
        if self.supervised_loss.method != "cross_entropy" and d.method == "kd" and d.weight == 1:
            raise ValueError("Adaptive supervised losses require a nonzero supervised weight in KD")
        if d.method not in {"supervised", "kd", "dkd"}:
            raise ValueError("distillation.method must be supervised, kd, or dkd")
        for name, value in {"temperature": d.temperature, "learning_rate": self.train.learning_rate}.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {"alpha": d.alpha, "beta": d.beta, "feature_weight": d.feature_weight,
                            "weight_decay": self.train.weight_decay}.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= d.weight <= 1 or not 0 <= self.train.momentum < 1:
            raise ValueError("weight must be in [0, 1]; momentum must be in [0, 1)")
        if d.method == "supervised" and d.feature_weight:
            raise ValueError("Supervised training cannot include feature distillation")
        if d.feature_weight and (not d.stage_pairs or any(len(p) != 2 for p in d.stage_pairs)):
            raise ValueError("Feature distillation needs explicit [student_stage, teacher_stage] pairs")
        if require_teacher and d.method != "supervised" and not self.teacher.checkpoint:
            raise ValueError("KD requires a trained teacher.checkpoint; train a supervised teacher first")
        if self.train.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        c = self.calibration
        for name, value in {"max_weight": c.max_weight, "kernel_bandwidth": c.kernel_bandwidth}.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"calibration.{name} must be finite and positive")
        for name in ("monitor_start_epoch", "patience", "ramp_epochs"):
            value = getattr(c, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"calibration.{name} must be a positive integer")
        if not 0 <= c.ece_threshold <= 1 or not 0 <= c.overconfidence_threshold <= 1:
            raise ValueError("Calibration thresholds must be in [0, 1]")
        for name in ("max_parameters", "max_flops", "max_latency_ms"):
            value = getattr(self.benchmark, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

    def to_dict(self) -> dict:
        return asdict(self)


def from_dict(raw: dict) -> ExperimentConfig:
    raw = dict(raw)
    sections = {"data": DataConfig, "student": ModelConfig, "teacher": ModelConfig,
                "distillation": DistillationConfig, "train": TrainConfig, "benchmark": BenchmarkConfig,
                "calibration": CalibrationConfig, "supervised_loss": SupervisedLossConfig,
                "cpc": CPCConfig, "surgery": SurgeryConfig}
    for name, cls in sections.items():
        if name in raw:
            values = dict(raw[name])
            if name == "distillation" and "stage_pairs" in values:
                values["stage_pairs"] = tuple(tuple(p) for p in values["stage_pairs"])
            raw[name] = cls(**values)
    config = ExperimentConfig(**raw)
    config.validate(require_teacher=False)
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    # Paths inside configurations are relative to the working directory.
    with Path(path).open("rb") as handle:
        return from_dict(tomllib.load(handle))
