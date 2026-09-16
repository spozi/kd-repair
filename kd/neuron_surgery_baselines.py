"""Resumable matched baselines for generalized neuron-surgery studies."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .checkpoints import fingerprint, write_json
from .config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                     ModelConfig, TrainConfig, from_dict)
from .engine import run_experiment
from .neuron_surgery_study import (NeuronSurgerySpec, _data_available,
                                   resolved_dataset_profile)
from .runtime import accelerator_workers


BASELINE_VERSION = 1


def baseline_configs(output: str | Path, root: str | Path, device: str,
                     spec: NeuronSurgerySpec) -> tuple[ExperimentConfig, list[ExperimentConfig]]:
    """Build the teacher and seed-matched classical-KD controls for a study spec."""
    spec.validate()
    output = str(Path(output).resolve())
    profile = resolved_dataset_profile(spec)
    data = DataConfig(
        source=spec.dataset_source, root=str(Path(root).resolve()),
        num_classes=spec.dataset_num_classes, image_size=spec.dataset_image_size,
        dataset_version=spec.dataset_version, dataset_profile=profile,
        horizontal_flip=spec.dataset_source in {"cifar10", "cifar100", "cinic10"},
        validation_fraction=0.1,
        confirmation_fraction=spec.confirmation_fraction, split_seed=spec.split_seed,
        imbalance_factor=spec.imbalance_factor if profile == "balanced" else 1.0)
    workers = accelerator_workers(device, 0)
    train = TrainConfig(
        epochs=spec.training_epochs, batch_size=128, learning_rate=0.05, momentum=0.9,
        weight_decay=0.0005, workers=workers, threads=4, device=device,
        seed=spec.teacher_seed)
    teacher = ExperimentConfig(
        name=spec.teacher_run, output_dir=output, data=data,
        student=ModelConfig(spec.teacher_model), teacher=ModelConfig(spec.teacher_model),
        distillation=DistillationConfig(method="supervised"), train=train,
        benchmark=BenchmarkConfig(warmup=20, iterations=100))
    teacher_checkpoint = str(Path(output) / spec.teacher_run / "student.pt")
    students = []
    for seed in spec.student_seeds:
        students.append(replace(
            teacher, name=spec.student_run_template.format(seed=seed),
            student=ModelConfig(spec.student_model),
            teacher=ModelConfig(spec.teacher_model, teacher_checkpoint),
            distillation=DistillationConfig(
                method="kd", temperature=4.0, weight=0.5, warmup_epochs=5,
                feature_weight=0.0),
            train=replace(train, seed=seed)))
    teacher.validate(require_teacher=False)
    for student in students:
        student.validate()
    return teacher, students


def _normalized_config(path: Path) -> dict:
    return from_dict(json.loads(path.read_text())).to_dict()


def _run_or_resume(config: ExperimentConfig) -> dict:
    directory = Path(config.output_dir) / config.name
    summary_path = directory / "summary.json"
    if summary_path.exists():
        if _normalized_config(directory / "config.json") != config.to_dict():
            raise ValueError(f"Completed baseline configuration changed: {directory}")
        summary = json.loads(summary_path.read_text())
        if fingerprint(summary["checkpoint"]) != fingerprint(directory / "student.pt"):
            raise ValueError(f"Completed baseline checkpoint reference changed: {directory}")
        return summary
    last = directory / "last.pt"
    return run_experiment(config, resume=str(last) if last.exists() else None)


def run_neuron_surgery_baselines(output: str, root: str, device: str,
                                 spec: NeuronSurgerySpec, *, dry_run: bool = False) -> dict:
    """Train or validate the matched baseline family required by one study."""
    teacher, students = baseline_configs(output, root, device, spec)
    _data_available(teacher)
    directory = Path(output).resolve()
    protocol = {
        "version": BASELINE_VERSION,
        "purpose": "Matched teacher and classical-KD controls for neuron surgery",
        "study": spec.to_dict(),
        "configs": [teacher.to_dict(), *(student.to_dict() for student in students)],
        "source_sha256": {
            name: fingerprint(Path(__file__).resolve().parent / name)
            for name in ("config.py", "data.py", "engine.py", "models.py",
                         "runtime.py", "neuron_surgery_baselines.py")
        },
    }
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    if dry_run:
        return {"dry_run": True, "output": str(directory), "protocol": protocol,
                "run_count": 1 + len(students)}
    protocol_path = directory / "protocol.json"
    if directory.exists() and any(directory.iterdir()) and not protocol_path.exists():
        raise FileExistsError(f"Nonempty baseline directory has no protocol: {directory}")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Baseline protocol changed; use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    manifest_path = directory / "baseline_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol_sha256") != fingerprint(protocol_path):
            raise ValueError("Completed baselines belong to another protocol")
        for name, digest in manifest.get("checkpoints", {}).items():
            checkpoint = directory / name / "student.pt"
            if fingerprint(checkpoint) != digest:
                raise ValueError(f"Completed baseline checkpoint changed: {checkpoint}")
        summaries = {config.name: json.loads(
            (directory / config.name / "summary.json").read_text())
                     for config in (teacher, *students)}
        return {"status": "complete", "output": str(directory), "manifest": manifest,
                "summaries": summaries}

    summaries = {}
    for config in (teacher, *students):
        summaries[config.name] = _run_or_resume(config)
        write_json(directory / "progress.json",
                   {"phase": "training", "completed": list(summaries)})
    manifest = {
        "protocol_sha256": fingerprint(protocol_path),
        "checkpoints": {name: fingerprint(row["checkpoint"])
                        for name, row in summaries.items()},
        "initial_student_sha256": {
            name: row["initial_student_sha256"] for name, row in summaries.items()
            if name != teacher.name},
    }
    write_json(manifest_path, manifest)
    write_json(directory / "progress.json", {"phase": "complete", "completed": list(summaries)})
    return {"status": "complete", "output": str(directory), "manifest": manifest,
            "summaries": summaries}
