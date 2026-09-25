"""Loss-balance ablation for DKD and RLD on Experiment 1 studies, on a single device.

Tests whether DKD's and RLD's published settings, rather than the methods themselves, explain
how they differ from classical KD: the weight of the teacher term relative to cross-entropy,
and DKD's x8 non-target term. Each variant retrains a study's classical-KD students from the
original teacher with only the loss settings changed, and a classical-KD rerun on the same
device is the matched reference.

Every student must reproduce its control's initial weights and data; each is scored on the
study's own test split with the code that scored the baselines. Outputs go to --output.

    python scripts/distillation_ablation.py --matrix runs/experiment1-multidataset \\
        --jobs bloodmnist-lt-if100 organamnist-balanced --device mps
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.checkpoints import fingerprint, write_json  # noqa: E402
from kd.config import from_dict  # noqa: E402
from kd.data import build_data  # noqa: E402
from kd.engine import run_experiment  # noqa: E402
from kd.neuron_surgery_study import _json, _model, _quality_report, _read, _save_predictions  # noqa: E402

# name: (method, teacher-term weight, beta). The reduced weights bring DKD's and RLD's
# teacher-to-cross-entropy balance (measured by baseline_comparison_stats.py) back to
# classical KD's.
VARIANTS = {
    "kd": ("kd", 0.5, 8.0),
    "dkd": ("dkd", 1.0, 8.0),
    "dkd_w0.15": ("dkd", 0.15, 8.0),
    "dkd_beta1": ("dkd", 1.0, 1.0),
    "rld": ("rld", 1.0, 8.0),
    "rld_w0.2": ("rld", 0.2, 8.0),
}


def _study(matrix: Path, job_id: str) -> tuple[dict, Path]:
    for job in _read(matrix / "matrix_summary.json")["jobs"]:
        if job["id"] == job_id and job["status"] == "complete":
            return job, matrix / job["dataset"] / job["profile"]
    raise ValueError(f"{job_id} is not a completed study in {matrix}")


def _train(control_dir: Path, profile: Path, spec: dict, variant: str, output: Path,
           device: str, data_root: str, workers: int | None) -> Path:
    method, weight, beta = VARIANTS[variant]
    control = from_dict(_read(control_dir / "config.json"))
    teacher = profile / "baselines" / "confirmatory" / spec["teacher_run"] / "student.pt"
    config = replace(
        control, name=f"{variant}_{control.name.removeprefix('kd_')}", output_dir=str(output),
        teacher=replace(control.teacher, checkpoint=str(teacher)),
        data=replace(control.data, root=data_root),
        train=replace(control.train, device=device,
                      **({} if workers is None else {"workers": workers})),
        distillation=replace(control.distillation, method=method, weight=weight, beta=beta))
    directory = output / config.name
    if not (directory / "summary.json").is_file():
        last = directory / "last.pt"
        print(f"[train] {config.name}", flush=True)
        run_experiment(config, resume=str(last) if last.is_file() else None)
    summary, reference = _read(directory / "summary.json"), _read(control_dir / "summary.json")
    for key in ("initial_student_sha256", "data"):
        if summary[key] != reference[key]:
            raise ValueError(f"{config.name} differs from its classical-KD control in {key}")
    if summary["teacher"]["checkpoint_sha256"] != fingerprint(teacher):
        raise ValueError(f"{config.name} was not distilled from the original teacher")
    return directory


def _score(directory: Path, evaluation, targets: list[int], seed: int) -> dict:
    config = from_dict(_read(directory / "config.json"))
    checkpoint = directory / "student.pt"
    model, _ = _model(config, checkpoint, evaluation.classes)
    identity = {"checkpoint_sha256": fingerprint(checkpoint), "seed": seed,
                "condition": directory.name, "evaluation_split": "test",
                "data": evaluation.provenance}
    labels, probabilities = _save_predictions(directory / "evaluation" / "predictions.npz", identity,
                                              model, evaluation.test, evaluation.classes)
    metrics = _quality_report(directory / "evaluation", labels, probabilities, evaluation.classes,
                              targets)
    history = _read(directory / "history.json")[5:]
    weights = (0.5, 0.5) if config.distillation.method in {"kd", "loca"} else (1.0, 1.0)
    weights = (weights[0], config.distillation.weight)
    recall = [row["recall"] for row in metrics["per_class"]]
    accuracy = [epoch["val"]["accuracy"] for epoch in history]
    late = accuracy[len(accuracy) // 2:]
    return {"accuracy": metrics["accuracy"], "target_recall": metrics["target_macro_recall"],
            "tail": statistics.mean(recall[i] for i in targets),
            "head": statistics.mean([recall[i] for i in range(len(recall)) if i not in targets] or [0.0]),
            "ratio": statistics.median(weights[1] * e["train"]["response"] / (weights[0] * e["train"]["ce"])
                                       for e in history if e["train"]["ce"] > 0),
            "swing": statistics.mean(abs(b - a) for a, b in zip(late, late[1:]))}


def run(matrix: Path, jobs: list[str], variants: list[str], seeds: list[int], device: str,
        data_root: str, output: Path, workers: int | None = None) -> dict:
    # Studies run in separate invocations share one report, so add to what is already there.
    report = _read(output / "report.json") if (output / "report.json").is_file() else {}
    for job_id in jobs:
        job, profile = _study(matrix, job_id)
        comparison = _read(profile / "studies" / "confirmatory" / "comparison.json")
        spec, targets = comparison["study"], comparison["target_classes"]
        # Balanced studies target every class, so there is no separate head.
        trained = {}
        for seed in seeds:
            control = profile / "baselines" / "confirmatory" / spec["student_run_template"].format(seed=seed)
            for variant in variants:
                trained[(variant, seed)] = _train(control, profile, spec, variant, output / job_id,
                                                  device, data_root, workers)
        # As in the study, the test split is built only once every checkpoint is fixed.
        teacher_config = from_dict(_read(profile / "baselines" / "confirmatory" / spec["teacher_run"]
                                         / "config.json"))
        teacher_config = replace(teacher_config, data=replace(teacher_config.data, root=data_root))
        evaluation = build_data(teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
                                include_test=True, diagnostic=True)
        reference = _read(profile / "studies" / "confirmatory" / "student_evaluation" / f"seed{seeds[0]}"
                          / "baseline" / "predictions.json")["identity"]["data"]
        if _json(evaluation.provenance) != reference:
            raise ValueError(f"{job_id}: test data differs from the split the study was scored on")
        report[job_id] = {variant: {seed: _score(trained[(variant, seed)], evaluation, targets, seed)
                                    for seed in seeds} for variant in variants}
        write_json(output / "report.json", report)
    return report


def render(report: dict) -> str:
    lines = ["# DKD and RLD loss-balance ablation", "",
             "Every student in a study shares its seed's initial weights, data, schedule and device; "
             "only the loss settings differ. Mean ± SD over seeds, in percent; ratio is the weighted "
             "teacher term over the weighted cross-entropy after warm-up.", ""]
    for job_id, variants in report.items():
        lines += [f"## {job_id}", "",
                  "| Variant | Accuracy | Target recall | Tail recall | Head recall | Teacher ÷ CE | Val swing (pp) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for variant, seeds in variants.items():
            rows = list(seeds.values())

            def cell(key, rows=rows):
                values = [100 * row[key] for row in rows]
                spread = statistics.stdev(values) if len(values) > 1 else 0.0
                return f"{statistics.mean(values):.2f} ± {spread:.2f}"

            lines.append(f"| {variant} | {cell('accuracy')} | {cell('target_recall')} | {cell('tail')} | "
                         f"{cell('head')} | {statistics.median(r['ratio'] for r in rows):.1f}× | "
                         f"{100 * statistics.median(r['swing'] for r in rows):.2f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--jobs", nargs="+", default=["bloodmnist-lt-if100", "organamnist-balanced"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs/distillation-ablation"))
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the controls' loader workers (0 avoids macOS's per-epoch "
                             "worker start-up); applies to every variant alike")
    args = parser.parse_args()
    output = args.output.resolve()
    report = run(args.matrix.resolve(), args.jobs, args.variants, args.seeds, args.device,
                 str(Path(args.data_root).resolve()), output, args.workers)
    text = render(report)
    (output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
