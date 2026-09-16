"""Matched direct-student surgery, supervised, and KD comparison."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics

import numpy as np

from .checkpoints import fingerprint, write_json
from .config import DistillationConfig, ModelConfig, SurgeryConfig, from_dict
from .data import build_data
from .evaluation import paired_calibration_comparison, paired_comparison
from .runtime import accelerator_workers
from .neuron_surgery_study import (
    BOOTSTRAP_REPETITIONS,
    NeuronSurgerySpec,
    STATISTICS_SEED,
    _complete_student,
    _data_available,
    _freeze,
    _freeze_text,
    _model,
    _quality_report,
    _save_predictions,
    _training_provenance,
    run_neuron_surgery_study,
)


STUDY_VERSION = 1
DEFAULT_SEEDS = (142, 143, 144)
CONDITIONS = (
    "supervised",
    "student_surgery",
    "kd_original_teacher",
    "kd_repaired_teacher",
)


def _read(path: str | Path):
    return json.loads(Path(path).read_text())


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"values": [], "mean": None, "std": None, "count": 0}
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _source_inputs(source: Path, seeds: tuple[int, ...]) -> dict:
    campaign_progress = _read(source / "progress.json")
    if campaign_progress.get("phase") != "complete":
        raise ValueError("Source neuron-surgery campaign is not complete")
    study = source / "studies" / "confirmatory"
    baseline = source / "baselines" / "confirmatory"
    comparison = _read(study / "comparison.json")
    if comparison.get("status") != "complete" or not comparison.get("success", {}).get("passed"):
        raise ValueError("Source confirmatory study must be complete and successful")
    source_spec = NeuronSurgerySpec(**{
        **comparison["study"],
        "student_seeds": tuple(comparison["study"]["student_seeds"]),
        "target_classes": tuple(comparison["study"]["target_classes"]),
        "stages": (tuple(comparison["study"]["stages"])
                   if comparison["study"].get("stages") is not None else None),
        "channel_budgets": tuple(comparison["study"]["channel_budgets"]),
        "learning_rates": tuple(comparison["study"]["learning_rates"]),
        "expected_train_counts": (tuple(comparison["study"]["expected_train_counts"])
                                  if comparison["study"].get("expected_train_counts") is not None else None),
        "expected_target_counts": (tuple(comparison["study"]["expected_target_counts"])
                                   if comparison["study"].get("expected_target_counts") is not None else None),
    })
    if tuple(seeds) != source_spec.student_seeds:
        raise ValueError(f"Comparison requires source seeds {source_spec.student_seeds}")
    if (source_spec.student_model != "cifar_student" or source_spec.imbalance_factor != 100
            or source_spec.evaluation_split != "confirmation"):
        raise ValueError("Direct student surgery currently requires sealed factor-100 CifarCNN inputs")

    rows = {}
    for seed in seeds:
        run = source_spec.student_run_template.format(seed=seed)
        config_path = baseline / run / "config.json"
        checkpoint = baseline / run / "student.pt"
        summary_path = baseline / run / "summary.json"
        repaired_run = f"kd_repaired_{source_spec.name}_seed{seed}"
        repaired_checkpoint = study / repaired_run / "student.pt"
        for path in (config_path, checkpoint, summary_path, repaired_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"Missing source comparison artifact: {path}")
        config = from_dict(_read(config_path))
        summary = _read(summary_path)
        if config.distillation.method != "kd" or config.train.seed != seed:
            raise ValueError(f"Source seed {seed} is not a matched KD control")
        rows[seed] = {
            "config": config,
            "summary": summary,
            "kd_original": checkpoint,
            "kd_repaired": repaired_checkpoint,
            "source_predictions": {
                "kd_original_teacher": study / "student_evaluation" / f"seed{seed}" /
                                       "baseline" / "predictions.npz",
                "kd_repaired_teacher": study / "student_evaluation" / f"seed{seed}" /
                                       "candidate" / "predictions.npz",
            },
        }
        for path in rows[seed]["source_predictions"].values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing source predictions: {path}")
    return {"study": study, "baseline": baseline, "comparison": comparison,
            "spec": source_spec, "rows": rows}


def _direct_spec(source: NeuronSurgerySpec, seed: int, run_name: str) -> NeuronSurgerySpec:
    return replace(
        source,
        name=f"student-surgery-f100-confirmatory-seed{seed}",
        teacher_run=run_name,
        student_run_template=f"unused_student_seed{{seed}}",
        teacher_model=source.student_model,
        student_model=source.student_model,
        teacher_seed=seed,
        student_seeds=(seed,),
        repair_subject="student",
        kd_weight=0.0,
        feature_weight=0.0,
        preservation_ce_weight=1.0,
        downstream_kd=False,
    )


def _supervised_config(source_config, output: Path, seed: int, device: str):
    config = replace(
        source_config,
        name=f"supervised_f100_confirmatory_seed{seed}",
        output_dir=str(output),
        teacher=ModelConfig(source_config.student.name),
        distillation=DistillationConfig(method="supervised"),
        train=replace(
            source_config.train, seed=seed, device=device,
            workers=accelerator_workers(device, source_config.train.workers)),
        surgery=SurgeryConfig(),
    )
    config.validate(require_teacher=False)
    return config


def _load_source_predictions(path: Path, classes: list[str]):
    sidecar = _read(path.with_suffix(".json"))
    if sidecar.get("sha256") != fingerprint(path):
        raise ValueError(f"Source prediction cache changed: {path}")
    with np.load(path, allow_pickle=False) as values:
        if values["classes"].tolist() != classes:
            raise ValueError("Source prediction vocabulary changed")
        return values["labels"], values["probabilities"]


def _contrast(labels, baseline_probability, candidate_probability) -> dict:
    return {
        "accuracy": paired_comparison(
            labels, baseline_probability, candidate_probability,
            seed=STATISTICS_SEED, repetitions=BOOTSTRAP_REPETITIONS),
        "calibration": paired_calibration_comparison(
            labels, baseline_probability, candidate_probability,
            seed=STATISTICS_SEED, repetitions=BOOTSTRAP_REPETITIONS),
    }


def _render_report(comparison: dict) -> str:
    lines = ["# Direct student surgery versus KD", "",
             "All four arms use matched student seeds and the same CIFAR-10-LT split.", "",
             "| Seed | Supervised acc. | Student surgery acc. | Original-teacher KD acc. | Repaired-teacher KD acc. |",
             "|---:|---:|---:|---:|---:|"]
    for row in comparison["seeds"]:
        metrics = row["conditions"]
        surgery = metrics.get("student_surgery")
        surgery_accuracy = f"{surgery['accuracy']:.4f}" if surgery is not None else "n/a"
        lines.append(
            f"| {row['seed']} | {metrics['supervised']['accuracy']:.4f} | "
            f"{surgery_accuracy} | "
            f"{metrics['kd_original_teacher']['accuracy']:.4f} | "
            f"{metrics['kd_repaired_teacher']['accuracy']:.4f} |")
    lines.extend(["", "## Mean deltas versus supervised", "",
                  "| Arm | Accuracy delta | Tail-recall delta |",
                  "|---|---:|---:|"])
    for name in ("student_surgery", "kd_original_teacher", "kd_repaired_teacher"):
        row = comparison["aggregate"]["versus_supervised"][name]
        accuracy = row["accuracy_delta"]["mean"]
        tail = row["tail_recall_delta"]["mean"]
        accuracy_text = f"{accuracy:+.4f}" if accuracy is not None else "n/a"
        tail_text = f"{tail:+.4f}" if tail is not None else "n/a"
        lines.append(f"| {name} | {accuracy_text} | {tail_text} |")
    lines.extend(["", comparison["evaluation_interpretation"], ""])
    return "\n".join(lines)


def run_student_surgery_study(
        source_campaign="runs/cifar10-lt-neuron-campaign",
        output="runs/cifar10-lt-student-surgery",
        device="auto", seeds=DEFAULT_SEEDS, dry_run=False):
    """Compare supervised direct surgery against matched original and repaired-teacher KD."""
    source = Path(source_campaign).resolve()
    output = Path(output).resolve()
    seeds = tuple(seeds)
    inputs = _source_inputs(source, seeds)
    supervised_root = output / "supervised"
    surgery_root = output / "student_surgery"
    configs = {seed: _supervised_config(inputs["rows"][seed]["config"], supervised_root, seed, device)
               for seed in seeds}
    specs = {seed: _direct_spec(inputs["spec"], seed, configs[seed].name) for seed in seeds}
    source_checkpoints = {
        str(seed): {
            "kd_original": fingerprint(inputs["rows"][seed]["kd_original"]),
            "kd_repaired": fingerprint(inputs["rows"][seed]["kd_repaired"]),
            "initial_student_sha256": inputs["rows"][seed]["summary"]["initial_student_sha256"],
        } for seed in seeds
    }
    protocol = {
        "version": STUDY_VERSION,
        "purpose": "Four-arm direct student surgery, supervised, and teacher-KD comparison",
        "source_campaign": str(source),
        "source_comparison_sha256": fingerprint(inputs["study"] / "comparison.json"),
        "source_checkpoints": source_checkpoints,
        "seeds": list(seeds),
        "supervised_configs": {str(seed): configs[seed].to_dict() for seed in seeds},
        "student_surgery_specs": {str(seed): specs[seed].to_dict() for seed in seeds},
        "conditions": list(CONDITIONS),
        "direct_surgery_loss": "CE(target) + CE(preservation); no logit or feature KD",
        "selection": "Independent per-seed long-tailed validation selection with a -0.5pp accuracy guardrail",
        "evaluation": "Existing balanced confirmation split, loaded after all direct-repair selections are fixed",
        "source_sha256": {
            name: fingerprint(Path(__file__).parent / name)
            for name in ("student_surgery_study.py", "neuron_surgery.py",
                         "neuron_surgery_study.py", "engine.py", "data.py", "models.py",
                         "runtime.py")
        },
    }
    if dry_run:
        return {"dry_run": True, "output": str(output), "protocol": protocol,
                "supervised_run_count": len(seeds),
                "repair_candidate_count": len(seeds) * len(inputs["spec"].channel_budgets)
                                          * len(inputs["spec"].learning_rates)}
    _data_available(configs[seeds[0]])
    if output.exists() and any(output.iterdir()) and not (output / "protocol.json").exists():
        raise FileExistsError(f"Nonempty output has no protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _freeze(output / "protocol.json", protocol)
    protocol_sha256 = fingerprint(output / "protocol.json")
    manifest_path = output / "report_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Completed student-surgery report belongs to another protocol")
        for name, digest in manifest["files"].items():
            if fingerprint(output / name) != digest:
                raise ValueError(f"Completed student-surgery artifact changed: {name}")
        return _read(output / "comparison.json")

    summaries = {}
    surgeries = {}
    for seed in seeds:
        summaries[seed] = _complete_student(configs[seed], protocol_sha256)
        source_summary = inputs["rows"][seed]["summary"]
        if summaries[seed]["initial_student_sha256"] != source_summary["initial_student_sha256"]:
            raise ValueError(f"Seed {seed} supervised and KD initial states differ")
        if summaries[seed]["data"] != source_summary["data"]:
            raise ValueError(f"Seed {seed} supervised and KD data provenance differs")
        write_json(output / "progress.json", {"phase": "supervised", "completed_seed": seed})

    for seed in seeds:
        surgeries[seed] = run_neuron_surgery_study(
            supervised_root, surgery_root / f"seed{seed}", device, (seed,), spec=specs[seed])
        write_json(output / "progress.json", {"phase": "student_surgery",
                                                "completed_seed": seed,
                                                "status": surgeries[seed]["status"]})
    missing = [seed for seed in seeds
               if surgeries[seed].get("status") != "student_repair_complete"]
    evaluation_data = build_data(
        configs[seeds[0]].data, replace(configs[seeds[0]].train, workers=0, device="cpu"),
        include_test=False, include_confirmation=True, diagnostic=True)
    if evaluation_data.confirmation is None:
        raise ValueError("Confirmation evaluation split was not constructed")
    if _training_provenance(evaluation_data.provenance) != _training_provenance(
            inputs["rows"][seeds[0]]["summary"]["data"]):
        raise ValueError("Student comparison changed train/validation provenance")

    seed_rows = []
    for seed in seeds:
        checkpoints = {
            "supervised": supervised_root / configs[seed].name / "student.pt",
        }
        if seed not in missing:
            checkpoints["student_surgery"] = (surgery_root / f"seed{seed}"
                                               / "teacher_repaired.pt")
        probabilities = {}
        labels = None
        metrics = {}
        for condition, checkpoint in checkpoints.items():
            model, _ = _model(configs[seed], checkpoint, evaluation_data.classes)
            identity = {"protocol_sha256": protocol_sha256, "seed": seed,
                        "condition": condition, "checkpoint_sha256": fingerprint(checkpoint),
                        "data": evaluation_data.provenance}
            y, p = _save_predictions(
                output / "evaluation" / f"seed{seed}" / condition / "predictions.npz",
                identity, model, evaluation_data.confirmation, evaluation_data.classes)
            labels = y if labels is None else labels
            if not np.array_equal(labels, y):
                raise ValueError(f"Seed {seed} evaluation labels differ")
            probabilities[condition] = p
            metrics[condition] = _quality_report(
                output / "evaluation" / f"seed{seed}" / condition,
                y, p, evaluation_data.classes, specs[seed].target_classes)
        for condition, path in inputs["rows"][seed]["source_predictions"].items():
            y, p = _load_source_predictions(path, evaluation_data.classes)
            if not np.array_equal(labels, y):
                raise ValueError(f"Seed {seed} source KD evaluation labels differ")
            probabilities[condition] = p
            metrics[condition] = _quality_report(
                output / "evaluation" / f"seed{seed}" / condition,
                y, p, evaluation_data.classes, specs[seed].target_classes)
        contrasts = {}
        for condition in CONDITIONS[1:]:
            if condition not in probabilities:
                continue
            contrasts[f"{condition}_vs_supervised"] = _contrast(
                labels, probabilities["supervised"], probabilities[condition])
        contrasts["kd_repaired_vs_kd_original"] = _contrast(
            labels, probabilities["kd_original_teacher"], probabilities["kd_repaired_teacher"])
        seed_rows.append({"seed": seed, "conditions": metrics, "contrasts": contrasts,
                          "selected_surgery": (surgeries[seed]["teacher_selection"]["selected"]
                                               if seed not in missing else None),
                          "surgery_status": surgeries[seed]["status"]})
        write_json(output / "progress.json", {"phase": "evaluation", "completed_seed": seed})

    condition_aggregate = {}
    versus_supervised = {}
    for condition in CONDITIONS:
        available = [row for row in seed_rows if condition in row["conditions"]]
        condition_aggregate[condition] = {
            "accuracy": _mean_std([row["conditions"][condition]["accuracy"] for row in available]),
            "tail_macro_recall": _mean_std([
                row["conditions"][condition]["tail_macro_recall"] for row in available]),
        }
        if condition != "supervised":
            versus_supervised[condition] = {
                "accuracy_delta": _mean_std([
                    row["conditions"][condition]["accuracy"]
                    - row["conditions"]["supervised"]["accuracy"] for row in available]),
                "tail_recall_delta": _mean_std([
                    row["conditions"][condition]["tail_macro_recall"]
                    - row["conditions"]["supervised"]["tail_macro_recall"] for row in available]),
            }
    surgery_delta = versus_supervised["student_surgery"]
    success = {
        "all_seeds_selected": not missing,
        "mean_tail_recall_at_least_1pp": (
            surgery_delta["tail_recall_delta"]["mean"] is not None
            and surgery_delta["tail_recall_delta"]["mean"] >= 0.01),
        "positive_tail_seeds": sum(
            row["conditions"]["student_surgery"]["tail_macro_recall"]
            > row["conditions"]["supervised"]["tail_macro_recall"]
            for row in seed_rows if "student_surgery" in row["conditions"]),
        "mean_accuracy_guardrail": (
            surgery_delta["accuracy_delta"]["mean"] is not None
            and surgery_delta["accuracy_delta"]["mean"] >= -0.005),
    }
    success["passed"] = (success["all_seeds_selected"]
                         and success["mean_tail_recall_at_least_1pp"]
                         and success["positive_tail_seeds"] >= 2
                         and success["mean_accuracy_guardrail"])
    comparison = {
        "status": "complete" if not missing else "partial",
        "missing_student_surgery_seeds": missing,
        "protocol_sha256": protocol_sha256,
        "seeds": seed_rows,
        "aggregate": {"conditions": condition_aggregate,
                      "versus_supervised": versus_supervised},
        "direct_student_surgery_success": success,
        "evaluation_interpretation": (
            "Exploratory follow-up: selection excluded confirmation data, but this workspace had already "
            "examined the confirmation split in the preceding campaign."),
    }
    write_json(output / "comparison.json", comparison)
    _freeze_text(output / "report.md", _render_report(comparison))
    manifest_files = ["comparison.json", "report.md"]
    for seed in seeds:
        manifest_files.append(f"supervised/{configs[seed].name}/student.pt")
        selection_name = f"student_surgery/seed{seed}/teacher_selection.json"
        if (output / selection_name).is_file():
            manifest_files.append(selection_name)
        if seed not in missing:
            manifest_files.append(f"student_surgery/seed{seed}/teacher_repaired.pt")
    write_json(manifest_path, {"protocol_sha256": protocol_sha256,
                               "files": {name: fingerprint(output / name)
                                         for name in manifest_files}})
    write_json(output / "progress.json", {"phase": "complete", "completed": list(seeds)})
    return comparison
