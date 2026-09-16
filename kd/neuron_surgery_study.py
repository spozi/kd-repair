"""Reproducible long-tailed image-classification neuron repair and KD study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from .calibration_metrics import extended_prediction_metrics
from .checkpoints import (fingerprint, load_model_checkpoint, metadata, save_checkpoint,
                          write_json)
from .config import SurgeryConfig, from_dict
from .data import build_data
from .engine import resolve_device, run_experiment, seed_everything
from .evaluation import (collect_predictions, paired_calibration_comparison,
                         paired_comparison, prediction_metrics)
from .models import VisionModel, create_model
from .neuron_surgery import (FORMAT_VERSION, TAIL_CLASSES, build_companion_records,
                             causal_channel_validation, collect_teacher_diagnostics,
                             consensus_channel_ranking, differential_channel_scores,
                             localization_stages,
                             select_hard_tail_targets, select_preservation_indices,
                             train_repair_candidate, validate_repair_provenance)
from .runtime import (accelerator_workers, configure_accelerator,
                      loader_performance_kwargs, prepare_model, runtime_metadata)


STUDY_VERSION = 4
DEFAULT_SEEDS = (42, 43, 44)
REPAIR_SEED = 2026
CHANNEL_BUDGETS = (2, 4, 8)
LEARNING_RATES = (0.001, 0.005)
EXPECTED_TRAIN_COUNTS = (4500, 2698, 1617, 969, 581, 348, 209, 125, 75, 45)
EXPECTED_TARGET_COUNTS = (70, 42, 25, 15, 9)
STATISTICS_SEED = 2026
BOOTSTRAP_REPETITIONS = 2000
COMPUTE_SOURCES = ("config.py", "data.py", "neuron_surgery.py",
                   "neuron_surgery_study.py", "losses.py", "models.py",
                   "runtime.py")


@dataclass(frozen=True)
class NeuronSurgerySpec:
    """Frozen inputs and interventions for one repair/KD study."""

    name: str = "cifar-cnn-f100-full"
    imbalance_factor: float = 100.0
    teacher_run: str = "teacher_f100"
    student_run_template: str = "kd_f100_seed{seed}"
    teacher_model: str = "cifar_teacher"
    student_model: str = "cifar_student"
    teacher_seed: int = 41
    student_seeds: tuple[int, ...] = DEFAULT_SEEDS
    split_seed: int = 2026
    confirmation_fraction: float = 0.0
    training_epochs: int = 81
    target_classes: tuple[int, ...] | None = TAIL_CLASSES
    target_class_policy: str = "explicit"
    target_fraction: float = 0.2
    companion_count: int = 5
    stages: tuple[str, ...] | None = None
    score_mode: str = "differential"
    causal_validation: bool = True
    repair_subject: str = "teacher"
    kd_weight: float = 1.0
    feature_weight: float = 0.25
    preservation_ce_weight: float = 0.0
    channel_budgets: tuple[int, ...] = CHANNEL_BUDGETS
    learning_rates: tuple[float, ...] = LEARNING_RATES
    repair_epochs: int = 20
    repair_samples_per_epoch: int = 640
    repair_batch_size: int = 64
    downstream_kd: bool = True
    evaluation_split: str = "test"
    expected_train_counts: tuple[int, ...] | None = EXPECTED_TRAIN_COUNTS
    expected_target_counts: tuple[int, ...] | None = EXPECTED_TARGET_COUNTS
    dataset_source: str = "cifar10"
    dataset_num_classes: int = 10
    dataset_image_size: int = 32
    dataset_version: str | None = None
    dataset_profile: str | None = None
    allow_sparse_class_fallback: bool = False

    def validate(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("Study name must be a single directory name")
        if not self.student_seeds or len(set(self.student_seeds)) != len(self.student_seeds):
            raise ValueError("Study student seeds must be nonempty and unique")
        if self.imbalance_factor < 1:
            raise ValueError("Study imbalance_factor must be at least one")
        if not 0 <= self.confirmation_fraction < 1:
            raise ValueError("Study confirmation_fraction must be in [0, 1)")
        if self.evaluation_split not in {"test", "confirmation"}:
            raise ValueError("Study evaluation_split must be test or confirmation")
        if self.evaluation_split == "confirmation" and not self.confirmation_fraction:
            raise ValueError("Confirmation evaluation requires confirmation_fraction > 0")
        if self.score_mode not in {"differential", "gradient_only"}:
            raise ValueError("Study score_mode must be differential or gradient_only")
        if self.repair_subject not in {"teacher", "student"}:
            raise ValueError("Study repair_subject must be teacher or student")
        if not isinstance(self.causal_validation, bool) or not isinstance(self.downstream_kd, bool):
            raise ValueError("Study causal_validation and downstream_kd must be booleans")
        if not 0 < self.target_fraction <= 1 or self.companion_count < 1:
            raise ValueError("Study target fraction and companion count must be positive")
        if min(self.training_epochs, self.repair_epochs, self.repair_samples_per_epoch,
               self.repair_batch_size) < 1:
            raise ValueError("Study epoch, sample, and batch counts must be positive")
        if min(self.kd_weight, self.feature_weight, self.preservation_ce_weight) < 0:
            raise ValueError("Study preservation weights must be nonnegative")
        if (not self.channel_budgets or any(value < 1 for value in self.channel_budgets)
                or tuple(sorted(set(self.channel_budgets))) != self.channel_budgets):
            raise ValueError("Study channel budgets must be unique increasing positive integers")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("Study learning rates must be positive")
        if self.target_class_policy not in {"explicit", "frequency_tail", "all"}:
            raise ValueError("Target class policy must be explicit, frequency_tail, or all")
        if self.target_class_policy == "explicit":
            if not self.target_classes or len(set(self.target_classes)) != len(self.target_classes):
                raise ValueError("Explicit target classes must be nonempty and unique")
        elif self.target_classes is not None:
            raise ValueError("Derived target class policies require target_classes=null")
        if self.target_class_policy == "frequency_tail" and self.imbalance_factor == 1:
            raise ValueError("frequency_tail requires an imbalanced dataset profile")
        if (self.dataset_source not in {"cifar10", "cifar100", "svhn", "cinic10", "gtsrb"}
                or self.dataset_num_classes < 2 or self.dataset_image_size != 32):
            raise ValueError("Study dataset must be a supported 32x32 classification source")
        if (self.target_classes is not None
                and any(label < 0 or label >= self.dataset_num_classes
                        for label in self.target_classes)):
            raise ValueError("Study target classes must belong to the configured dataset")
        profiles = {"balanced": 1.0, "lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}
        if self.dataset_profile is not None:
            if self.dataset_profile not in profiles:
                raise ValueError(f"Unknown dataset profile {self.dataset_profile}")
            if profiles[self.dataset_profile] != self.imbalance_factor:
                raise ValueError("Dataset profile and imbalance factor disagree")
        if not isinstance(self.allow_sparse_class_fallback, bool):
            raise ValueError("allow_sparse_class_fallback must be a boolean")
        for template in (self.teacher_run, self.student_run_template.format(seed=self.student_seeds[0])):
            if not template or Path(template).name != template:
                raise ValueError("Study baseline run names must be single directory names")

    def to_dict(self) -> dict:
        return asdict(self)


def neuron_surgery_spec_from_dict(raw: dict) -> NeuronSurgerySpec:
    values = dict(raw)
    for name in ("student_seeds", "target_classes", "stages", "channel_budgets",
                 "learning_rates", "expected_train_counts", "expected_target_counts"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    spec = NeuronSurgerySpec(**values)
    spec.validate()
    return spec


def load_neuron_surgery_spec(path: str | Path) -> NeuronSurgerySpec:
    return neuron_surgery_spec_from_dict(_read(path))


def resolved_dataset_profile(spec: NeuronSurgerySpec) -> str:
    """Return the catalog profile while preserving legacy imbalance-factor specs."""
    return spec.dataset_profile or "balanced"


def resolve_target_classes(spec: NeuronSurgerySpec, train_counts) -> tuple[int, ...]:
    """Freeze explicit, all-class, or frequency-defined target classes."""
    counts = tuple(int(value) for value in train_counts)
    if len(counts) != spec.dataset_num_classes or min(counts) < 1:
        raise ValueError("Training counts must contain every configured class")
    if spec.target_class_policy == "explicit":
        return tuple(spec.target_classes or ())
    if spec.target_class_policy == "all":
        return tuple(range(spec.dataset_num_classes))
    count = (spec.dataset_num_classes + 1) // 2
    ranked = sorted(range(spec.dataset_num_classes), key=lambda label: (counts[label], label))
    return tuple(ranked[:count])


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _read(path: str | Path):
    return json.loads(Path(path).read_text())


def _json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _training_provenance(value: dict) -> dict:
    """Compare train/validation identity while ignoring final evaluation loaders."""
    result = _json(value)
    result.get("split_sizes", {}).pop("test", None)
    result.get("split_sizes", {}).pop("confirmation", None)
    return result


def _freeze(path: Path, value) -> None:
    value = _json(value)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Recorded artifact differs: {path}; use a new study directory")
    else:
        write_json(path, value)


def _freeze_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise ValueError(f"Recorded report differs: {path}; use a new study directory")
    else:
        path.write_text(value)


def _write_report_manifest(directory: Path, protocol_sha256: str, names: list[str]) -> None:
    write_json(directory / "report_manifest.json",
               {"protocol_sha256": protocol_sha256,
                "files": {name: fingerprint(directory / name) for name in names}})


def _stage_json(path: Path, identity: dict, compute):
    identity = _json(identity)
    if path.exists():
        artifact = _read(path)
        if artifact.get("identity") != identity or "payload" not in artifact:
            raise ValueError(f"Stage identity differs: {path}; use a new study directory")
        return artifact["payload"]
    payload = _json(compute())
    write_json(path, {"identity": identity, "payload": payload})
    return payload


def _stage_npz(path: Path, identity: dict, compute) -> dict[str, np.ndarray]:
    sidecar = path.with_suffix(".json")
    identity = _json(identity)
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise ValueError(f"Incomplete cached stage: {path}")
        recorded = _read(sidecar)
        if recorded.get("identity") != identity or recorded.get("sha256") != fingerprint(path):
            raise ValueError(f"Cached stage identity or contents changed: {path}")
        with np.load(path, allow_pickle=False) as cached:
            return {name: cached[name] for name in cached.files}
    arrays = compute()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    write_json(sidecar, {"identity": identity, "sha256": fingerprint(path)})
    return arrays


def _data_available(config) -> None:
    root = Path(config.data.root)
    if config.data.dataset_version is not None:
        root = root / config.data.source / config.data.dataset_version
    if not root.exists():
        raise FileNotFoundError(f"Dataset must already be downloaded: {root}")


def _configured_stages(spec: NeuronSurgerySpec) -> tuple[str, ...]:
    if spec.stages is not None:
        return spec.stages
    if spec.teacher_model in {"cifar_teacher", "cifar_student"}:
        return ("stage2", "stage3")
    if spec.teacher_model in {"resnet18", "resnet34", "resnet50"}:
        return ("stage3", "stage4")
    raise ValueError("Study teacher must be cifar_teacher or a supported ResNet")


def _load_inputs(baseline: str | Path, seeds: tuple[int, ...],
                 spec: NeuronSurgerySpec):
    baseline = Path(baseline).resolve()
    spec.validate()
    if tuple(seeds) != spec.student_seeds:
        raise ValueError(f"Study requires student seeds {spec.student_seeds}")
    teacher_directory = baseline / spec.teacher_run
    teacher_config_path = teacher_directory / "config.json"
    teacher_checkpoint = teacher_directory / "student.pt"
    if not teacher_config_path.is_file() or not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"Missing repair teacher config or checkpoint: {teacher_directory}")
    teacher_config = from_dict(_read(teacher_config_path))
    teacher_config.validate(require_teacher=False)
    factor = ({"lt-if10": 10.0, "lt-if50": 50.0, "lt-if100": 100.0}
              .get(teacher_config.data.dataset_profile, teacher_config.data.imbalance_factor))
    teacher_recipe = (teacher_config.data.source, teacher_config.data.num_classes,
                      teacher_config.data.image_size, teacher_config.data.dataset_version,
                      teacher_config.data.dataset_profile, factor,
                      teacher_config.data.split_seed, teacher_config.data.confirmation_fraction,
                      teacher_config.student.name, teacher_config.distillation.method,
                      teacher_config.train.seed, teacher_config.train.epochs)
    expected_teacher = (spec.dataset_source, spec.dataset_num_classes, spec.dataset_image_size,
                        spec.dataset_version, resolved_dataset_profile(spec),
                        spec.imbalance_factor, spec.split_seed,
                        spec.confirmation_fraction, spec.teacher_model, "supervised",
                        spec.teacher_seed, spec.training_epochs)
    if teacher_recipe != expected_teacher or teacher_config.name != spec.teacher_run:
        raise ValueError("Teacher does not match the frozen neuron-surgery study recipe")

    students = {}
    teacher_sha256 = fingerprint(teacher_checkpoint)
    if not spec.downstream_kd:
        return baseline, teacher_config, teacher_checkpoint, students
    for seed in seeds:
        run_name = spec.student_run_template.format(seed=seed)
        path = baseline / run_name
        config_path, checkpoint = path / "config.json", path / "student.pt"
        summary_path = path / "summary.json"
        if not config_path.is_file() or not checkpoint.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing matched KD baseline artifacts for seed {seed}")
        config = from_dict(_read(config_path))
        config.validate()
        recipe = (config.data, config.student.name, config.teacher.name,
                  config.distillation.method, config.distillation.temperature,
                  config.distillation.weight, config.distillation.warmup_epochs,
                  config.distillation.feature_weight, config.train.epochs, config.train.batch_size,
                  config.train.learning_rate, config.train.momentum, config.train.weight_decay,
                  config.train.seed, config.surgery.action)
        expected = (teacher_config.data, spec.student_model, spec.teacher_model,
                    "kd", 4.0, 0.5, 5, 0.0, spec.training_epochs,
                    128, 0.05, 0.9, 0.0005, seed, "none")
        if (recipe != expected or config.student.checkpoint is not None
                or config.name != run_name):
            raise ValueError(f"Seed {seed} baseline differs from the predeclared KD recipe")
        if fingerprint(config.teacher.checkpoint) != teacher_sha256:
            raise ValueError("Student baseline teacher differs from the repair teacher")
        students[seed] = {"config": config, "checkpoint": checkpoint,
                          "summary": _read(summary_path),
                          "checkpoint_sha256": fingerprint(checkpoint)}
    return baseline, teacher_config, teacher_checkpoint, students


def _protocol(baseline: Path, output: Path, device: str, teacher_config,
              teacher_checkpoint: Path, students: dict, spec: NeuronSurgerySpec) -> dict:
    source_root = Path(__file__).parent
    stages = _configured_stages(spec)
    return _json({
        "version": STUDY_VERSION,
        "method": ("AI-Lancet-inspired same-class companion channel repair followed by KD"
                   if spec.downstream_kd else
                   f"AI-Lancet-inspired direct {spec.repair_subject} channel repair"),
        "scope": (f"{spec.teacher_model} on {resolved_dataset_profile(spec)} "
                  f"{spec.dataset_source}"),
        "study": spec.to_dict(),
        "baseline": str(baseline), "output": str(output), "device": device,
        "teacher": {"config": teacher_config.to_dict(),
                    "checkpoint": str(teacher_checkpoint),
                    "checkpoint_sha256": fingerprint(teacher_checkpoint)},
        "student_baselines": {str(seed): {"checkpoint": str(row["checkpoint"]),
                                           "checkpoint_sha256": row["checkpoint_sha256"],
                                           "initial_student_sha256": row["summary"]["initial_student_sha256"]}
                              for seed, row in students.items()},
        "target_selection": {"classes": (list(spec.target_classes)
                                                   if spec.target_classes is not None else None),
                             "class_policy": spec.target_class_policy,
                             "hard_fraction": spec.target_fraction,
                             "policy": "all mistakes plus highest-NLL correct rows to each class quota"},
        "companions": {"count": spec.companion_count, "metric": "cosine_distance",
                       "sparse_class_fallback": spec.allow_sparse_class_fallback,
                       "embedding": f"global-average-pooled normalized {stages[-1]}"},
        "localization": {"stages": list(stages), "score_mode": spec.score_mode,
                         "score": ("mean absolute companion feature difference times absolute error-margin gradient"
                                   if spec.score_mode == "differential"
                                   else "absolute error-margin gradient without companion differences"),
                         "bootstrap_repetitions": 20, "bootstrap_seed": REPAIR_SEED,
                         "top_fraction": 0.25, "stability_threshold": 0.7,
                         "causal_intervention": ("zero one output channel"
                                                 if spec.causal_validation else "disabled"),
                         "preservation_accuracy_guardrail": -0.005},
        "repair": {"channel_budgets": list(spec.channel_budgets),
                   "learning_rates": list(spec.learning_rates), "epochs": spec.repair_epochs,
                   "samples_per_stream_per_epoch": spec.repair_samples_per_epoch,
                   "batch_size": spec.repair_batch_size,
                   "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0,
                   "scheduler": "cosine", "temperature": 4.0,
                   "kd_weight": spec.kd_weight, "feature_weight": spec.feature_weight,
                   "preservation_ce_weight": spec.preservation_ce_weight,
                   "loss": "CE(target)+weighted CE(preservation)+weighted KD_T4(anchor,preservation)+weighted late-stage feature preservation"},
        "teacher_selection": {"split": "development validation",
                              "primary": "target_macro_recall",
                              "overall_accuracy_guardrail": -0.005,
                              "tie_breaks": ["target_balanced_nll", "fewer_channels", "lower_learning_rate"],
                              "requires_strict_target_improvement": True,
                              "on_failure": "stop before student KD and test construction"},
        "student_kd": {"enabled": spec.downstream_kd, "seeds": list(spec.student_seeds),
                       "epochs": spec.training_epochs, "method": "kd",
                       "temperature": 4.0, "weight": 0.5, "warmup_epochs": 5,
                       "feature_weight": 0.0, "sample_surgery": "none"},
        "success": {"mean_target_recall_delta": 0.01, "positive_seed_count": 2,
                    "mean_accuracy_delta_guardrail": -0.005},
        "statistics": {"paired_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                       "seed": STATISTICS_SEED},
        "evaluation_policy": ("Balanced sealed training-archive holdout is constructed only after teacher and student selection are fixed."
                              if spec.evaluation_split == "confirmation" else
                              "Official test data are constructed only after teacher selection is persisted; evidence is exploratory because prior workspace studies examined test."),
        "source_sha256": {name: fingerprint(source_root / name) for name in COMPUTE_SOURCES},
    })


def _records_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict]:
    return [{"index": index, "true_label": int(arrays["labels"][index]),
             "predicted_label": int(arrays["predictions"][index]),
             "correct": bool(arrays["labels"][index] == arrays["predictions"][index]),
             "confidence": float(arrays["confidence"][index]), "nll": float(arrays["nll"][index])}
            for index in range(len(arrays["labels"]))]


def _tail_metrics(labels, probabilities, target_classes=TAIL_CLASSES) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(1)
    recalls, nlls = {}, {}
    for label in target_classes:
        selected = labels == label
        if not selected.any():
            raise ValueError(f"Evaluation has no samples for target class {label}")
        recalls[str(label)] = float((predictions[selected] == label).mean())
        nlls[str(label)] = float(-np.log(np.maximum(probabilities[selected, label], 1e-300)).mean())
    macro_recall = float(np.mean(list(recalls.values())))
    balanced_nll = float(np.mean(list(nlls.values())))
    return {"target_macro_recall": macro_recall, "target_balanced_nll": balanced_nll,
            "target_recall_by_label": recalls, "target_nll_by_label": nlls,
            "tail_macro_recall": macro_recall,
            "tail_balanced_nll": balanced_nll,
            "tail_recall_by_label": recalls, "tail_nll_by_label": nlls}


def _validation_report(model, loader, device, classes,
                       target_classes=TAIL_CLASSES, *, precision="float32",
                       channels_last=False) -> dict:
    labels, probability = collect_predictions(
        model, loader, device, precision=precision, channels_last=channels_last)
    return {**prediction_metrics(labels, probability, classes),
            **_tail_metrics(labels, probability, target_classes)}


def _candidate_complete(path: Path, identity: dict):
    manifest_path = path / "completion.json"
    if not manifest_path.exists():
        return None
    manifest = _read(manifest_path)
    if manifest.get("identity") != _json(identity):
        raise ValueError(f"Repair candidate identity changed: {path}")
    for name, digest in manifest.get("files", {}).items():
        if fingerprint(path / name) != digest:
            raise ValueError(f"Repair candidate artifact changed: {path / name}")
    return _read(path / "result.json")


def _train_candidate(directory: Path, identity: dict, teacher_config, teacher_checkpoint: Path,
                     anchor: VisionModel, dataset, labels, target_indices, preservation_indices,
                     channels, device, classes, validation_loader, localization_sha256,
                     budget: int, learning_rate: float, spec: NeuronSurgerySpec,
                     target_classes: tuple[int, ...]) -> dict:
    cached = _candidate_complete(directory, identity)
    if cached is not None:
        state = torch.load(directory / "teacher.pt", map_location="cpu", weights_only=True)
        validate_repair_provenance(state, base_checkpoint_sha256=fingerprint(teacher_checkpoint),
                                   localization_sha256=localization_sha256)
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    candidate = create_model(spec.teacher_model, teacher_config.data.num_classes)
    load_model_checkpoint(candidate, teacher_checkpoint,
                          metadata(spec.teacher_model, classes, teacher_config.data.image_size,
                                   teacher_config.data.source))
    history, verification = train_repair_candidate(
        candidate, anchor, dataset, labels, target_indices, preservation_indices, channels,
        device, epochs=spec.repair_epochs, samples_per_epoch=spec.repair_samples_per_epoch,
        batch_size=spec.repair_batch_size, learning_rate=learning_rate, momentum=0.9,
        temperature=4.0, kd_weight=spec.kd_weight, feature_weight=spec.feature_weight,
        preservation_ce_weight=spec.preservation_ce_weight,
        feature_stages=_configured_stages(spec), seed=REPAIR_SEED,
        precision=teacher_config.train.precision,
        channels_last=teacher_config.train.channels_last,
        fail_fast=(device.type != "cuda"
                   or teacher_config.train.cuda_mode == "deterministic"))
    validation = _validation_report(
        candidate, validation_loader, device, classes, target_classes,
        precision=teacher_config.train.precision,
        channels_last=teacher_config.train.channels_last)
    checkpoint = {**metadata(spec.teacher_model, classes, teacher_config.data.image_size,
                             teacher_config.data.source), "kind": "inference",
                  "epoch": spec.repair_epochs - 1, "student": candidate.cpu().state_dict(),
                  "repair": {"format_version": FORMAT_VERSION,
                             "base_checkpoint": str(teacher_checkpoint),
                             "base_checkpoint_sha256": fingerprint(teacher_checkpoint),
                             "localization_sha256": localization_sha256,
                             "channels": channels, "channel_budget": budget,
                             "learning_rate": learning_rate, "repair_seed": REPAIR_SEED,
                             "study_name": spec.name, "score_mode": spec.score_mode,
                             "causal_validation": spec.causal_validation,
                             "repair_subject": spec.repair_subject,
                             "kd_weight": spec.kd_weight,
                             "feature_weight": spec.feature_weight,
                             "preservation_ce_weight": spec.preservation_ce_weight,
                             "validation": validation, "verification": verification}}
    save_checkpoint(directory / "teacher.pt", checkpoint)
    result = {"budget": budget, "learning_rate": learning_rate, "channels": channels,
              "validation": validation, "verification": verification,
              "checkpoint": str(directory / "teacher.pt"),
              "checkpoint_sha256": fingerprint(directory / "teacher.pt")}
    write_json(directory / "history.json", history)
    write_json(directory / "result.json", result)
    write_json(directory / "completion.json", {"identity": _json(identity),
                                                "files": {name: fingerprint(directory / name)
                                                          for name in ("teacher.pt", "history.json", "result.json")}})
    return result


def _complete_student(config, protocol_sha256: str):
    path = Path(config.output_dir) / config.name
    manifest_path = path / "completion.json"
    names = ("config.json", "data.json", "history.json", "summary.json", "student.pt", "best.pt", "last.pt")
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"Completed student belongs to another protocol: {path}")
        for name, digest in manifest["files"].items():
            if fingerprint(path / name) != digest:
                raise ValueError(f"Completed student artifact changed: {path / name}")
        return _read(path / "summary.json")
    if (path / "summary.json").exists():
        if _read(path / "config.json") != _json(config.to_dict()):
            raise ValueError(f"Student configuration differs: {path}")
        summary = _read(path / "summary.json")
    else:
        last = path / "last.pt"
        summary = run_experiment(config, resume=str(last) if last.exists() else None)
    write_json(manifest_path, {"protocol_sha256": protocol_sha256,
                               "files": {name: fingerprint(path / name) for name in names}})
    return summary


def _save_predictions(path: Path, identity: dict, model, loader, classes):
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise ValueError(f"Incomplete prediction cache: {path}")
        recorded = _read(sidecar)
        if recorded.get("identity") != _json(identity) or recorded.get("sha256") != fingerprint(path):
            raise ValueError(f"Prediction cache changed: {path}")
        with np.load(path, allow_pickle=False) as cached:
            if cached["classes"].tolist() != classes:
                raise ValueError("Prediction class vocabulary changed")
            return cached["labels"], cached["probabilities"]
    labels, probabilities = collect_predictions(model, loader, torch.device("cpu"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, labels=labels, probabilities=probabilities,
                            classes=np.asarray(classes))
    os.replace(temporary, path)
    write_json(sidecar, {"identity": _json(identity), "sha256": fingerprint(path)})
    return labels, probabilities


def _quality_report(directory: Path, labels, probabilities, classes,
                    target_classes=TAIL_CLASSES) -> dict:
    quality = prediction_metrics(labels, probabilities, classes)
    quality["extended"] = extended_prediction_metrics(labels, probabilities, classes,
                                                       include_curve=False)
    quality.update(_tail_metrics(labels, probabilities, target_classes))
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", quality)
    np.savetxt(directory / "confusion.csv", np.asarray(quality["confusion_matrix"]), delimiter=",",
               fmt="%d", header=",".join(classes), comments="")
    return quality


def _model(config, checkpoint, classes):
    model = create_model(config.student.name, config.data.num_classes)
    state = load_model_checkpoint(model, checkpoint,
                                  metadata(config.student.name, classes, config.data.image_size,
                                           config.data.source))
    model.requires_grad_(False).eval()
    return model, state


def _mean_std(values):
    return {"values": values, "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def _render_report(comparison: dict) -> str:
    dataset = comparison.get("study", {}).get("dataset_source", "cifar10")
    title = f"# {dataset} neuron surgery and KD"
    if comparison["status"] in {"repair_only_complete", "student_repair_complete"}:
        delta = comparison["validation_delta"]
        return (f"# {dataset} neuron-surgery component ablation\n\n"
                f"Study: **{comparison['study']['name']}**.\n\n"
                f"Validation accuracy delta: **{delta['accuracy']:+.4f}**.  \n"
                f"Validation target-recall delta: **{delta['target_macro_recall']:+.4f}**.  \n"
                f"Validation target-NLL delta: **{delta['target_balanced_nll']:+.4f}**.\n")
    if comparison["status"] != "complete":
        return (title + "\n\n"
                f"Status: **{comparison['status']}**.\n\n"
                f"{comparison['reason']}\n")
    aggregate = comparison["aggregate"]
    lines = [title, "",
             f"Study: **{comparison['study']['name']}**.", "",
             ("AI-Lancet-inspired channel repair was selected on the development split "
              f"before the {comparison['evaluation_split']} split was loaded."), "",
             "| Seed | Baseline acc. | Repaired-teacher KD acc. | Acc. delta | Target recall delta |",
             "|---:|---:|---:|---:|---:|"]
    for row in comparison["students"]:
        lines.append(f"| {row['seed']} | {row['baseline']['accuracy']:.4f} | {row['candidate']['accuracy']:.4f} | "
                     f"{row['accuracy_delta']:+.4f} | {row['target_recall_delta']:+.4f} |")
    lines.extend(["", f"Mean target-recall delta: **{aggregate['target_recall_delta']['mean']:+.4f}**.",
                  f"Mean accuracy delta: **{aggregate['accuracy_delta']['mean']:+.4f}**.",
                  f"Success rule passed: **{comparison['success']['passed']}**.", "",
                  comparison["evaluation_interpretation"], ""])
    return "\n".join(lines)


def run_neuron_surgery_study(baseline="runs/cifar10-lt-multiseed",
                             output="runs/cifar10-lt-neuron-surgery", device="auto",
                             seeds=DEFAULT_SEEDS, dry_run=False, *,
                             spec: NeuronSurgerySpec | None = None):
    """Localize/repair one teacher, then run three matched repaired-teacher KD arms."""
    seeds = tuple(seeds)
    spec = NeuronSurgerySpec() if spec is None else spec
    spec.validate()
    baseline, teacher_config, teacher_checkpoint, students = _load_inputs(baseline, seeds, spec)
    _data_available(teacher_config)
    directory = Path(output).resolve()
    protocol = _protocol(baseline, directory, device, teacher_config, teacher_checkpoint,
                         students, spec)
    if dry_run:
        return {"dry_run": True, "output": str(directory), "protocol": protocol,
                "student_run_count": len(seeds) if spec.downstream_kd else 0,
                "repair_candidate_count": len(spec.channel_budgets) * len(spec.learning_rates)}
    resolved_device = resolve_device(device)
    deterministic = teacher_config.train.cuda_mode == "deterministic"
    seed_everything(REPAIR_SEED, deterministic=deterministic)
    configure_accelerator(resolved_device, teacher_config.train.cuda_mode)
    accelerator = runtime_metadata(
        resolved_device, teacher_config.train.precision,
        teacher_config.train.cuda_mode, teacher_config.train.channels_last)
    if directory.exists() and not (directory / "protocol.json").exists() and any(directory.iterdir()):
        raise FileExistsError(f"Nonempty study directory has no protocol: {directory}")
    if (directory / "protocol.json").exists() and _read(directory / "protocol.json") != protocol:
        raise ValueError("Neuron-surgery protocol or computational source changed; use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    _freeze(directory / "protocol.json", protocol)
    protocol_sha256 = fingerprint(directory / "protocol.json")
    _freeze(directory / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
                                               "torchvision": str(torchvision.__version__),
                                               "numpy": np.__version__, "platform": platform.platform(),
                                               "device": str(resolved_device),
                                               "accelerator": accelerator})
    final_manifest = directory / "report_manifest.json"
    if final_manifest.exists():
        manifest = _read(final_manifest)
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Completed report belongs to another protocol")
        for name, digest in manifest["files"].items():
            if fingerprint(directory / name) != digest:
                raise ValueError(f"Completed report artifact changed: {name}")
        return _read(directory / "comparison.json")

    torch.set_num_threads(teacher_config.train.threads)
    seed_everything(REPAIR_SEED, deterministic=deterministic)
    configure_accelerator(resolved_device, teacher_config.train.cuda_mode)
    diagnostic_data = build_data(teacher_config.data, replace(teacher_config.train, seed=REPAIR_SEED,
                                                               workers=0, device=device),
                                 include_test=False, diagnostic=True)
    if (spec.expected_train_counts is not None
            and tuple(diagnostic_data.provenance["train_per_class"]) != spec.expected_train_counts):
        raise ValueError("Frozen long-tail training counts changed")
    target_classes = resolve_target_classes(
        spec, diagnostic_data.provenance["train_per_class"])
    target_class_plan = {
        "policy": spec.target_class_policy,
        "classes": list(target_classes),
        "train_per_class": diagnostic_data.provenance["train_per_class"],
        "tie_break": "dataset label index",
    }
    _freeze(directory / "target_class_plan.json", target_class_plan)
    teacher = create_model(spec.teacher_model, teacher_config.data.num_classes)
    stages = _configured_stages(spec)
    if stages != localization_stages(teacher):
        raise ValueError(f"Configured repair stages {stages} do not match model map {localization_stages(teacher)}")
    teacher_state = load_model_checkpoint(
        teacher, teacher_checkpoint,
        metadata(spec.teacher_model, diagnostic_data.classes, teacher_config.data.image_size,
                 teacher_config.data.source))
    if teacher_state.get("epoch", -1) < 0:
        raise ValueError("Repair teacher checkpoint must come from a completed training epoch")
    prepare_model(teacher, resolved_device,
                  teacher_config.train.channels_last).requires_grad_(False).eval()
    data_identity = {"protocol_sha256": protocol_sha256, "teacher_sha256": fingerprint(teacher_checkpoint),
                     "data": diagnostic_data.provenance, "classes": diagnostic_data.classes,
                     "target_classes": list(target_classes)}

    def diagnostics_compute():
        records, embeddings = collect_teacher_diagnostics(teacher, diagnostic_data.train, resolved_device)
        return {"labels": np.asarray([row["true_label"] for row in records], dtype=np.int64),
                "predictions": np.asarray([row["predicted_label"] for row in records], dtype=np.int64),
                "confidence": np.asarray([row["confidence"] for row in records], dtype=np.float32),
                "nll": np.asarray([row["nll"] for row in records], dtype=np.float32),
                "embeddings": embeddings}

    diagnostics = _stage_npz(directory / "diagnostics.npz", data_identity, diagnostics_compute)
    records = _records_from_arrays(diagnostics)
    targets_identity = {**data_identity, "diagnostics_sha256": fingerprint(directory / "diagnostics.npz")}

    def targets_compute():
        targets = select_hard_tail_targets(records, target_classes,
                                           fraction=spec.target_fraction)
        counts = tuple(sum(row["true_label"] == label for row in targets)
                       for label in target_classes)
        if spec.expected_target_counts is not None and counts != spec.expected_target_counts:
            raise ValueError(f"Hard-target counts changed: {counts}")
        indices = [row["index"] for row in targets]
        preservation = select_preservation_indices(
            records, indices, seed=REPAIR_SEED,
            allow_sparse_fallback=spec.allow_sparse_class_fallback)
        preservation_pool = [row["index"] for row in records
                             if row["correct"] and row["index"] not in set(indices)]
        if not preservation_pool and spec.allow_sparse_class_fallback:
            preservation_pool = [row["index"] for row in records
                                 if row["index"] not in set(indices)]
        if not preservation_pool:
            raise ValueError("Repair requires at least one non-target preservation example")
        preservation_classes = sorted({records[index]["true_label"] for index in preservation})
        return {"targets": targets, "target_counts": list(counts),
                "target_classes": list(target_classes),
                "causal_preservation_indices": preservation,
                "causal_preservation_classes": preservation_classes,
                "causal_preservation_omitted_classes": sorted(
                    set(range(spec.dataset_num_classes)) - set(preservation_classes)),
                "training_preservation_indices": preservation_pool}

    target_plan = _stage_json(directory / "targets.json", targets_identity, targets_compute)
    companion_identity = {"protocol_sha256": protocol_sha256,
                          "targets_sha256": fingerprint(directory / "targets.json"),
                          "diagnostics_sha256": fingerprint(directory / "diagnostics.npz")}
    companions = _stage_json(
        directory / "companions.json", companion_identity,
        lambda: build_companion_records(records, diagnostics["embeddings"],
                                        target_plan["targets"], companions=spec.companion_count,
                                        allow_sparse_fallback=spec.allow_sparse_class_fallback))

    score_identity = {"protocol_sha256": protocol_sha256,
                      "companions_sha256": fingerprint(directory / "companions.json"),
                      "teacher_sha256": fingerprint(teacher_checkpoint)}
    scores = _stage_npz(
        directory / "differential_scores.npz", score_identity,
        lambda: differential_channel_scores(teacher, diagnostic_data.train.dataset, companions,
                                            resolved_device, batch_size=16, stages=stages,
                                            score_mode=spec.score_mode))
    consensus_identity = {"protocol_sha256": protocol_sha256,
                          "scores_sha256": fingerprint(directory / "differential_scores.npz")}
    consensus = _stage_json(
        directory / "consensus_scores.json", consensus_identity,
        lambda: consensus_channel_ranking(
            scores, target_classes=target_classes, repetitions=20, seed=REPAIR_SEED,
            top_fraction=0.25, stability_threshold=0.7, stages=stages))

    target_loader = DataLoader(Subset(diagnostic_data.train.dataset,
                                      [row["index"] for row in target_plan["targets"]]),
                               batch_size=128, shuffle=False, num_workers=0,
                               **loader_performance_kwargs(str(resolved_device), 0))
    preservation_loader = DataLoader(Subset(diagnostic_data.train.dataset,
                                            target_plan["causal_preservation_indices"]),
                                     batch_size=128, shuffle=False, num_workers=0,
                                     **loader_performance_kwargs(str(resolved_device), 0))
    causal_identity = {"protocol_sha256": protocol_sha256,
                       "consensus_sha256": fingerprint(directory / "consensus_scores.json"),
                       "targets_sha256": fingerprint(directory / "targets.json")}
    def causal_compute():
        if spec.causal_validation:
            return causal_channel_validation(
                teacher, target_loader, preservation_loader, consensus, resolved_device,
                target_classes=target_classes, accuracy_guardrail=0.005)
        ranking = [{"stage": row["stage"], "channel": row["channel"]}
                   for row in consensus["channels"] if row["eligible"]]
        return {"performed": False, "reason": "causal_validation_disabled",
                "channels": [], "ranking": ranking}

    causal = _stage_json(directory / "causal_ablation.json", causal_identity, causal_compute)
    localization_identity = {"protocol_sha256": protocol_sha256,
                             "causal_sha256": fingerprint(directory / "causal_ablation.json")}

    def localization_compute():
        ranking = causal["ranking"]
        budgets = [budget for budget in spec.channel_budgets if len(ranking) >= budget]
        return {"format_version": FORMAT_VERSION, "ranking": ranking,
                "available_channel_budgets": budgets,
                "candidate_channels": {str(budget): ranking[:budget] for budget in budgets}}

    localization = _stage_json(directory / "localization.json", localization_identity,
                               localization_compute)
    localization_sha256 = fingerprint(directory / "localization.json")
    if not localization["available_channel_budgets"]:
        comparison = {"status": "no_repair_selected",
                      "reason": "Too few channels passed the configured localization guardrails.",
                      "protocol_sha256": protocol_sha256,
                      "target_classes": list(target_classes)}
        write_json(directory / "comparison.json", comparison)
        _freeze_text(directory / "report.md", _render_report(comparison))
        _write_report_manifest(directory, protocol_sha256,
                               ["comparison.json", "report.md", "targets.json", "companions.json",
                                "target_class_plan.json",
                                "differential_scores.npz", "differential_scores.json",
                                "consensus_scores.json", "causal_ablation.json", "localization.json"])
        return comparison

    original_validation = _validation_report(
        teacher, diagnostic_data.val, resolved_device, diagnostic_data.classes,
        target_classes, precision=teacher_config.train.precision,
        channels_last=teacher_config.train.channels_last)
    _freeze(directory / "teacher_validation_baseline.json", original_validation)
    augmented_data = build_data(teacher_config.data,
                                replace(teacher_config.train, seed=REPAIR_SEED, workers=0, device=device),
                                include_test=False, diagnostic=False)
    if augmented_data.provenance != diagnostic_data.provenance:
        raise ValueError("Augmented and canonical repair datasets have different splits")
    labels = np.asarray([row["true_label"] for row in records], dtype=np.int64)
    candidates = []
    for budget in localization["available_channel_budgets"]:
        channels = localization["candidate_channels"][str(budget)]
        for learning_rate in spec.learning_rates:
            name = f"budget{budget}_lr{learning_rate:g}"
            identity = {"protocol_sha256": protocol_sha256,
                        "localization_sha256": localization_sha256,
                        "teacher_sha256": fingerprint(teacher_checkpoint),
                        "budget": budget, "learning_rate": learning_rate, "channels": channels}
            result = _train_candidate(
                directory / "repair_candidates" / name, identity, teacher_config,
                teacher_checkpoint, teacher, augmented_data.train.dataset, labels,
                [row["index"] for row in target_plan["targets"]],
                target_plan["training_preservation_indices"], channels, resolved_device,
                diagnostic_data.classes, diagnostic_data.val, localization_sha256,
                budget, learning_rate, spec, target_classes)
            candidates.append(result)
            write_json(directory / "progress.json", {"phase": "teacher_repair",
                                                       "completed": [row["checkpoint"] for row in candidates]})
    _freeze(directory / "candidate_results.json", candidates)
    eligible = [row for row in candidates
                if row["validation"]["accuracy"] >= original_validation["accuracy"] - 0.005
                and row["validation"]["target_macro_recall"] > original_validation["target_macro_recall"]]
    eligible.sort(key=lambda row: (-row["validation"]["target_macro_recall"],
                                   row["validation"]["target_balanced_nll"], row["budget"],
                                   row["learning_rate"]))
    selection = {"protocol_sha256": protocol_sha256, "localization_sha256": localization_sha256,
                 "baseline_validation": original_validation,
                 "eligible_count": len(eligible), "selected": eligible[0] if eligible else None,
                 "selection_frozen_before_evaluation": True}
    _freeze(directory / "teacher_selection.json", selection)
    if not eligible:
        comparison = {"status": "no_repair_selected",
                      "reason": "No repaired teacher strictly improved validation target recall within the 0.5-point accuracy guardrail.",
                      "protocol_sha256": protocol_sha256,
                      "target_classes": list(target_classes),
                      "teacher_selection": selection}
        write_json(directory / "comparison.json", comparison)
        _freeze_text(directory / "report.md", _render_report(comparison))
        candidate_files = []
        for row in candidates:
            relative = Path(row["checkpoint"]).parent.relative_to(directory)
            candidate_files.extend(str(relative / name) for name in
                                   ("teacher.pt", "history.json", "result.json", "completion.json"))
        _write_report_manifest(directory, protocol_sha256,
                               ["comparison.json", "report.md", "teacher_selection.json",
                                "candidate_results.json", "localization.json",
                                "target_class_plan.json", *candidate_files])
        return comparison

    selected = eligible[0]
    selected_state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    validate_repair_provenance(selected_state, base_checkpoint_sha256=fingerprint(teacher_checkpoint),
                               localization_sha256=localization_sha256)
    repaired_checkpoint = directory / "teacher_repaired.pt"
    if repaired_checkpoint.exists():
        export = _read(directory / "teacher_repaired.json")
        if export.get("sha256") != fingerprint(repaired_checkpoint):
            raise ValueError("Exported repaired teacher changed")
    else:
        save_checkpoint(repaired_checkpoint, selected_state)
        write_json(directory / "teacher_repaired.json",
                   {"source": selected["checkpoint"], "source_sha256": selected["checkpoint_sha256"],
                    "sha256": fingerprint(repaired_checkpoint),
                    "selection_sha256": fingerprint(directory / "teacher_selection.json")})

    if not spec.downstream_kd:
        direct_student = spec.repair_subject == "student"
        comparison = {
            "status": ("student_repair_complete" if direct_student else "repair_only_complete"),
            "protocol_sha256": protocol_sha256,
            "study": spec.to_dict(), "target_classes": list(target_classes),
            "teacher_selection": selection,
            "validation_delta": {
                "accuracy": (selected["validation"]["accuracy"]
                             - original_validation["accuracy"]),
                "tail_macro_recall": (selected["validation"]["tail_macro_recall"]
                                      - original_validation["tail_macro_recall"]),
                "tail_balanced_nll": (selected["validation"]["tail_balanced_nll"]
                                      - original_validation["tail_balanced_nll"]),
                "target_macro_recall": (selected["validation"]["target_macro_recall"]
                                        - original_validation["target_macro_recall"]),
                "target_balanced_nll": (selected["validation"]["target_balanced_nll"]
                                        - original_validation["target_balanced_nll"]),
            },
            "reason": ("Direct student repair selected on validation; final evaluation is delegated to the matched comparison."
                       if direct_student else
                       "Validation-only component ablation; downstream KD and evaluation were predisabled.")
        }
        write_json(directory / "comparison.json", comparison)
        _freeze_text(directory / "report.md", _render_report(comparison))
        candidate_files = []
        for row in candidates:
            relative = Path(row["checkpoint"]).parent.relative_to(directory)
            candidate_files.extend(str(relative / name) for name in
                                   ("teacher.pt", "history.json", "result.json", "completion.json"))
        _write_report_manifest(
            directory, protocol_sha256,
            ["comparison.json", "report.md", "teacher_selection.json", "teacher_repaired.pt",
             "teacher_repaired.json", "candidate_results.json", "localization.json",
             "target_class_plan.json", *candidate_files])
        return comparison

    repaired_configs = {}
    for seed, baseline_row in students.items():
        config = baseline_row["config"]
        repaired = replace(config, name=f"kd_repaired_{spec.name}_seed{seed}",
                           output_dir=str(directory),
                           teacher=replace(config.teacher, checkpoint=str(repaired_checkpoint)),
                           train=replace(
                               config.train, device=device,
                               workers=accelerator_workers(device, config.train.workers)),
                           surgery=SurgeryConfig())
        repaired.validate()
        repaired_configs[seed] = repaired
        summary = _complete_student(repaired, protocol_sha256)
        if summary["initial_student_sha256"] != baseline_row["summary"]["initial_student_sha256"]:
            raise ValueError(f"Seed {seed} student initialization differs from its baseline")
        if summary["data"] != baseline_row["summary"]["data"]:
            raise ValueError(f"Seed {seed} student data provenance differs from its baseline")
        write_json(directory / "progress.json", {"phase": "student_kd", "completed_seed": seed})

    # Evaluation construction begins only after teacher selection and every student run are fixed.
    evaluation_data = build_data(
        teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
        include_test=spec.evaluation_split == "test",
        include_confirmation=spec.evaluation_split == "confirmation", diagnostic=True)
    if (_training_provenance(evaluation_data.provenance)
            != _training_provenance(diagnostic_data.provenance)):
        raise ValueError("Final evaluation construction changed train/validation provenance")
    evaluation_loader = getattr(evaluation_data, spec.evaluation_split)
    if evaluation_loader is None:
        raise ValueError(f"Configured evaluation split {spec.evaluation_split} was not constructed")
    teacher.cpu()
    repaired_teacher, repaired_state = _model(
        teacher_config, repaired_checkpoint, evaluation_data.classes)
    validate_repair_provenance(repaired_state, base_checkpoint_sha256=fingerprint(teacher_checkpoint),
                               localization_sha256=localization_sha256)
    teacher_rows = []
    for condition, model, checkpoint in (("original", teacher, teacher_checkpoint),
                                         ("repaired", repaired_teacher, repaired_checkpoint)):
        identity = {"checkpoint_sha256": fingerprint(checkpoint), "condition": condition,
                    "evaluation_policy": protocol["evaluation_policy"],
                    "data": evaluation_data.provenance}
        y, p = _save_predictions(
            directory / "teacher_evaluation" / condition / "predictions.npz",
            identity, model, evaluation_loader, evaluation_data.classes)
        teacher_rows.append((condition, y, p,
                             _quality_report(directory / "teacher_evaluation" / condition, y, p,
                                             evaluation_data.classes, target_classes)))
    if not np.array_equal(teacher_rows[0][1], teacher_rows[1][1]):
        raise ValueError("Teacher evaluation labels differ")
    teacher_comparison = {"baseline": teacher_rows[0][3], "candidate": teacher_rows[1][3],
                          "accuracy": paired_comparison(teacher_rows[0][1], teacher_rows[0][2],
                                                        teacher_rows[1][2], seed=STATISTICS_SEED,
                                                        repetitions=BOOTSTRAP_REPETITIONS),
                          "calibration": paired_calibration_comparison(
                              teacher_rows[0][1], teacher_rows[0][2], teacher_rows[1][2],
                              seed=STATISTICS_SEED, repetitions=BOOTSTRAP_REPETITIONS)}
    write_json(directory / "teacher_comparison.json", teacher_comparison)

    student_rows = []
    for seed in seeds:
        baseline_config = students[seed]["config"]
        baseline_model, _ = _model(
            baseline_config, students[seed]["checkpoint"], evaluation_data.classes)
        candidate_checkpoint = directory / repaired_configs[seed].name / "student.pt"
        candidate_model, _ = _model(
            repaired_configs[seed], candidate_checkpoint, evaluation_data.classes)
        values = []
        for condition, model, checkpoint in (("baseline", baseline_model, students[seed]["checkpoint"]),
                                             ("candidate", candidate_model, candidate_checkpoint)):
            identity = {"checkpoint_sha256": fingerprint(checkpoint), "seed": seed,
                        "condition": condition, "evaluation_split": spec.evaluation_split,
                        "data": evaluation_data.provenance}
            y, p = _save_predictions(
                directory / "student_evaluation" / f"seed{seed}" / condition /
                "predictions.npz", identity, model, evaluation_loader, evaluation_data.classes)
            values.append((y, p, _quality_report(
                directory / "student_evaluation" / f"seed{seed}" / condition,
                y, p, evaluation_data.classes, target_classes)))
        if not np.array_equal(values[0][0], values[1][0]):
            raise ValueError(f"Seed {seed} paired evaluation labels differ")
        target_delta = (values[1][2]["target_macro_recall"]
                        - values[0][2]["target_macro_recall"])
        student_rows.append({"seed": seed, "baseline": values[0][2], "candidate": values[1][2],
                             "accuracy_delta": values[1][2]["accuracy"] - values[0][2]["accuracy"],
                             "target_recall_delta": target_delta,
                             "tail_recall_delta": target_delta,
                             "paired_accuracy": paired_comparison(values[0][0], values[0][1], values[1][1],
                                                                   seed=STATISTICS_SEED,
                                                                   repetitions=BOOTSTRAP_REPETITIONS),
                             "paired_calibration": paired_calibration_comparison(
                                 values[0][0], values[0][1], values[1][1], seed=STATISTICS_SEED,
                                 repetitions=BOOTSTRAP_REPETITIONS)})
    accuracy_deltas = [row["accuracy_delta"] for row in student_rows]
    tail_deltas = [row["tail_recall_delta"] for row in student_rows]
    target_delta_summary = _mean_std(tail_deltas)
    aggregate = {"accuracy_delta": _mean_std(accuracy_deltas),
                 "target_recall_delta": target_delta_summary,
                 "tail_recall_delta": target_delta_summary}
    success = {"mean_target_recall_at_least_1pp": aggregate["target_recall_delta"]["mean"] >= 0.01,
               "positive_target_seeds": sum(value > 0 for value in tail_deltas),
               "mean_tail_recall_at_least_1pp": aggregate["tail_recall_delta"]["mean"] >= 0.01,
               "positive_tail_seeds": sum(value > 0 for value in tail_deltas),
               "mean_accuracy_guardrail": aggregate["accuracy_delta"]["mean"] >= -0.005}
    success["passed"] = (success["mean_target_recall_at_least_1pp"]
                         and success["positive_target_seeds"] >= 2
                         and success["mean_accuracy_guardrail"])
    comparison = {"status": "complete", "protocol_sha256": protocol_sha256,
                  "study": spec.to_dict(), "evaluation_split": spec.evaluation_split,
                  "target_classes": list(target_classes),
                  "teacher_selection": selection, "teacher": teacher_comparison,
                  "students": student_rows, "aggregate": aggregate, "success": success,
                  "evaluation_interpretation": (
                      "Confirmatory sealed holdout excluded from all fitting and selection."
                      if spec.evaluation_split == "confirmation" else
                      "Exploratory because earlier workspace studies examined the official test set.")}
    write_json(directory / "comparison.json", comparison)
    _freeze_text(directory / "report.md", _render_report(comparison))
    manifest_files = ["comparison.json", "report.md", "teacher_selection.json",
                      "teacher_repaired.pt", "teacher_repaired.json", "teacher_comparison.json",
                      "candidate_results.json", "localization.json", "target_class_plan.json"]
    for seed in seeds:
        run = repaired_configs[seed].name
        manifest_files.extend([f"{run}/completion.json", f"{run}/student.pt"])
        for condition in ("baseline", "candidate"):
            base = f"student_evaluation/seed{seed}/{condition}"
            manifest_files.extend([f"{base}/predictions.npz", f"{base}/predictions.json",
                                   f"{base}/metrics.json", f"{base}/confusion.csv"])
    for condition in ("original", "repaired"):
        base = f"teacher_evaluation/{condition}"
        manifest_files.extend([f"{base}/predictions.npz", f"{base}/predictions.json",
                               f"{base}/metrics.json", f"{base}/confusion.csv"])
    for row in candidates:
        relative = Path(row["checkpoint"]).parent.relative_to(directory)
        manifest_files.extend(str(relative / name) for name in
                              ("teacher.pt", "history.json", "result.json", "completion.json"))
    _write_report_manifest(directory, protocol_sha256, manifest_files)
    return comparison
