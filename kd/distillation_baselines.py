"""Alternative distillation losses on the matched students of a completed repair study.

Each baseline student is the study's classical-KD control with only the distillation
loss changed: the same original teacher, data, splits, initial state, schedule and
checkpoint selection. Students are then scored on the study's own evaluation split with
its own target classes, so every arm is directly comparable with the study's students
distilled from the original and from the repaired teacher.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .checkpoints import fingerprint, write_json
from .config import from_dict
from .data import build_data
from .evaluation import paired_comparison
from .neuron_surgery_study import (BOOTSTRAP_REPETITIONS, STATISTICS_SEED, _complete_student,
                                   _data_available, _freeze, _json, _mean_std, _model,
                                   _quality_report, _read, _save_predictions)

BASELINE_VERSION = 1
# DKD and RLD are published with the response term at full strength beside full
# cross-entropy; LoCa is a plug-in on classical KD and keeps the control's own weight.
RESPONSE_WEIGHTS = {"dkd": 1.0, "rld": 1.0, "loca": None}
# The study's condition names, keyed to what each of its students was distilled from.
REFERENCES = {"baseline": "kd_original_teacher", "candidate": "kd_repaired_teacher"}


def _source(study: Path, data_root: str | None) -> dict:
    comparison = _read(study / "comparison.json")
    if comparison.get("status") != "complete":
        raise ValueError(f"Study has no repaired-teacher comparison to extend: {study}")
    spec = comparison["study"]
    baseline = study.parent.parent / "baselines" / "confirmatory"
    teacher = baseline / spec["teacher_run"] / "student.pt"
    if not teacher.is_file():
        raise FileNotFoundError(f"Missing original teacher: {teacher}")
    rows = {}
    for seed in spec["student_seeds"]:
        run = baseline / spec["student_run_template"].format(seed=seed)
        references = {condition: study / "student_evaluation" / f"seed{seed}" / condition
                      for condition in REFERENCES}
        for path in (run / "config.json", run / "summary.json", run / "student.pt",
                     *(directory / name for directory in references.values()
                       for name in ("predictions.npz", "predictions.json", "metrics.json"))):
            if not path.is_file():
                raise FileNotFoundError(f"Missing source artifact: {path}")
        config, summary = from_dict(_read(run / "config.json")), _read(run / "summary.json")
        if config.distillation.method != "kd" or config.train.seed != seed:
            raise ValueError(f"{run.name} is not the classical-KD control for seed {seed}")
        if summary["teacher"]["checkpoint_sha256"] != fingerprint(teacher):
            raise ValueError(f"{run.name} was not distilled from {teacher}")
        # Anchor the teacher on the study's own tree, just verified to be the checkpoint the
        # control used, so that a relocated archive still resolves.
        config = replace(config, teacher=replace(config.teacher, checkpoint=str(teacher)))
        if data_root is not None:
            config = replace(config, data=replace(config.data, root=str(Path(data_root).resolve())))
        rows[seed] = {"config": config, "summary": summary, "references": references}
    return {"comparison": comparison, "spec": spec, "baseline": baseline, "teacher": teacher,
            "rows": rows}


def run_distillation_baseline(study: str | Path, method: str, output: str | Path | None = None,
                              device: str | None = None, data_root: str | None = None,
                              dry_run: bool = False) -> dict:
    """Train and score one baseline loss on the matched students of a completed study."""
    if method not in RESPONSE_WEIGHTS:
        raise ValueError(f"Baseline method must be one of {sorted(RESPONSE_WEIGHTS)}")
    study = Path(study).resolve()
    source = _source(study, data_root)
    spec, rows = source["spec"], source["rows"]
    output = (Path(output) if output else
              study.parent.parent / "baselines" / "distillation" / method).resolve()
    configs = {}
    for seed, row in rows.items():
        control = row["config"]
        weight = RESPONSE_WEIGHTS[method]
        configs[seed] = replace(
            control, name=f"{method}_{spec['name']}_seed{seed}", output_dir=str(output),
            distillation=replace(control.distillation, method=method,
                                 weight=control.distillation.weight if weight is None else weight),
            train=control.train if device is None else replace(control.train, device=device))
    protocol = {
        "version": BASELINE_VERSION,
        "purpose": f"{method} distillation on a completed study's matched classical-KD students",
        "method": method,
        "source_study": str(study),
        "source_comparison_sha256": fingerprint(study / "comparison.json"),
        "teacher_checkpoint_sha256": fingerprint(source["teacher"]),
        "evaluation_split": spec["evaluation_split"],
        "target_classes": source["comparison"]["target_classes"],
        "configs": {str(seed): config.to_dict() for seed, config in configs.items()},
        "source_sha256": {name: fingerprint(Path(__file__).parent / name)
                          for name in ("distillation_baselines.py", "losses.py", "engine.py",
                                       "data.py", "models.py", "neuron_surgery_study.py")},
    }
    if dry_run:
        return {"dry_run": True, "method": method, "output": str(output),
                "runs": [config.name for config in configs.values()],
                "response_weight": next(iter(configs.values())).distillation.weight,
                "teacher_checkpoint_sha256": protocol["teacher_checkpoint_sha256"]}
    _data_available(next(iter(configs.values())))
    if output.exists() and any(output.iterdir()) and not (output / "protocol.json").exists():
        raise FileExistsError(f"Nonempty output has no protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _freeze(output / "protocol.json", protocol)
    protocol_sha256 = fingerprint(output / "protocol.json")
    manifest_path = output / "report_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"Completed {method} report belongs to another protocol: {output}")
        for name, digest in manifest["files"].items():
            if fingerprint(output / name) != digest:
                raise ValueError(f"Completed {method} artifact changed: {output / name}")
        return _read(output / "comparison.json")

    for seed, config in configs.items():
        summary = _complete_student(config, protocol_sha256)
        control = rows[seed]["summary"]
        for key in ("initial_student_sha256", "data"):
            if summary[key] != control[key]:
                raise ValueError(f"{config.name} and its classical-KD control differ in {key}")
        if summary["teacher"]["checkpoint_sha256"] != control["teacher"]["checkpoint_sha256"]:
            raise ValueError(f"{config.name} and its classical-KD control used different teachers")
        write_json(output / "progress.json", {"phase": "training", "completed": config.name})

    # As in the source study, the evaluation split is constructed only once every student
    # checkpoint is fixed, and it must be the split the study itself was scored on.
    teacher_config = from_dict(_read(source["baseline"] / spec["teacher_run"] / "config.json"))
    if data_root is not None:
        teacher_config = replace(teacher_config, data=replace(
            teacher_config.data, root=str(Path(data_root).resolve())))
    split = spec["evaluation_split"]
    evaluation_data = build_data(
        teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
        include_test=split == "test", include_confirmation=split == "confirmation", diagnostic=True)
    loader = getattr(evaluation_data, split)
    if loader is None:
        raise ValueError(f"Evaluation split {split} was not constructed")
    reference_data = _read(next(iter(rows.values()))["references"]["baseline"]
                           / "predictions.json")["identity"]["data"]
    if _json(evaluation_data.provenance) != reference_data:
        raise ValueError("Evaluation data differs from the split the study was scored on")

    targets = source["comparison"]["target_classes"]
    seed_rows = []
    for seed, config in configs.items():
        checkpoint = output / config.name / "student.pt"
        model, _ = _model(config, checkpoint, evaluation_data.classes)
        directory = output / "evaluation" / f"seed{seed}"
        identity = {"checkpoint_sha256": fingerprint(checkpoint), "seed": seed, "condition": method,
                    "evaluation_split": split, "data": evaluation_data.provenance}
        labels, probabilities = _save_predictions(directory / "predictions.npz", identity, model,
                                                  loader, evaluation_data.classes)
        metrics = _quality_report(directory, labels, probabilities, evaluation_data.classes, targets)
        versus = {}
        for condition, name in REFERENCES.items():
            reference = rows[seed]["references"][condition]
            with np.load(reference / "predictions.npz", allow_pickle=False) as cached:
                if not np.array_equal(cached["labels"], labels):
                    raise ValueError(f"{config.name} was scored on different examples than {name}")
                paired = paired_comparison(cached["labels"], cached["probabilities"], probabilities,
                                           seed=STATISTICS_SEED, repetitions=BOOTSTRAP_REPETITIONS)
            paired["target_recall_delta"] = (metrics["target_macro_recall"]
                                             - _read(reference / "metrics.json")["target_macro_recall"])
            versus[name] = paired
        seed_rows.append({"seed": seed, "metrics": metrics, "versus": versus})

    comparison = {
        "status": "complete", "protocol_sha256": protocol_sha256, "method": method,
        "study": spec["name"], "evaluation_split": split, "target_classes": targets,
        "response_weight": next(iter(configs.values())).distillation.weight,
        "sign": "Each delta is this method minus the reference student; positive favours this method.",
        "seeds": seed_rows,
        "aggregate": {
            "accuracy": _mean_std([row["metrics"]["accuracy"] for row in seed_rows]),
            "target_macro_recall": _mean_std([row["metrics"]["target_macro_recall"]
                                              for row in seed_rows]),
            "versus": {name: {key: _mean_std([row["versus"][name][key] for row in seed_rows])
                              for key in ("accuracy_delta", "target_recall_delta")}
                       for name in REFERENCES.values()},
        },
    }
    write_json(output / "comparison.json", comparison)
    files = ["comparison.json", "protocol.json"] + [
        f"evaluation/seed{seed}/{name}" for seed in configs
        for name in ("predictions.npz", "predictions.json", "metrics.json", "confusion.csv")]
    write_json(manifest_path, {"protocol_sha256": protocol_sha256,
                               "files": {name: fingerprint(output / name) for name in files}})
    return comparison


def run_matrix_distillation_baselines(matrix: str | Path, methods: tuple[str, ...] = tuple(RESPONSE_WEIGHTS),
                                      jobs: list[str] | None = None, device: str | None = None,
                                      data_root: str | None = None, dry_run: bool = False) -> dict:
    """Run each baseline on every matrix study that produced a repaired teacher."""
    matrix = Path(matrix).resolve()
    complete = {job["id"]: job for job in _read(matrix / "matrix_summary.json")["jobs"]
                if job["status"] == "complete"}
    selected = list(complete) if jobs is None else list(jobs)
    missing = sorted(set(selected) - set(complete))
    if missing:
        raise ValueError(f"Jobs without a repaired-teacher comparison: {missing}")
    report = {}
    for job_id in selected:
        study = matrix / complete[job_id]["dataset"] / complete[job_id]["profile"] / "studies" / "confirmatory"
        report[job_id] = {}
        for method in methods:
            result = run_distillation_baseline(study, method, device=device, data_root=data_root,
                                               dry_run=dry_run)
            report[job_id][method] = result if dry_run else result["aggregate"]
    return report
