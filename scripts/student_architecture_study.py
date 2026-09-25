"""Does a repaired teacher's gain reach students of other architectures?

For each completed Experiment 1 study, each student architecture and each seed, two students
are distilled with the study's classical-KD recipe: one from the original teacher, one from
the repaired teacher the study selected. Within a pair everything else is identical, which
is checked: initial weights, data and schedule. Both are scored on the study's own test
split with the code that scored Experiment 1.

    python scripts/student_architecture_study.py --matrix runs/experiment1-multidataset \\
        --students resnet8 resnet20 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kd.checkpoints import fingerprint, write_json  # noqa: E402
from kd.config import from_dict  # noqa: E402
from kd.data import build_data  # noqa: E402
from kd.engine import run_experiment  # noqa: E402
from kd.neuron_surgery_study import _json, _read  # noqa: E402

from distillation_ablation import _score  # noqa: E402

TEACHERS = ("original", "repaired")


def _teachers(profile: Path, comparison: dict) -> dict[str, Path]:
    """The study's original teacher and its selected repair, found in this study tree."""
    spec, selected = comparison["study"], comparison["teacher_selection"]["selected"]
    original = profile / "baselines" / "confirmatory" / spec["teacher_run"] / "student.pt"
    repaired = (profile / "studies" / "confirmatory" / "repair_candidates"
                / f"budget{selected['budget']}_lr{selected['learning_rate']}" / "teacher.pt")
    if not repaired.is_file() or fingerprint(repaired) != selected["checkpoint_sha256"]:
        raise ValueError(f"Selected repaired teacher missing or changed: {repaired}")
    return {"original": original, "repaired": repaired}


def _train(control_dir: Path, teacher: Path, teacher_label: str, student: str, output: Path,
           device: str, data_root: str) -> Path:
    control = from_dict(_read(control_dir / "config.json"))
    config = replace(
        control, name=f"{student}_{teacher_label}_{control.name.removeprefix('kd_')}",
        output_dir=str(output), student=replace(control.student, name=student),
        teacher=replace(control.teacher, checkpoint=str(teacher)),
        data=replace(control.data, root=data_root), train=replace(control.train, device=device))
    directory = output / config.name
    if not (directory / "summary.json").is_file():
        last = directory / "last.pt"
        print(f"[train] {config.name}", flush=True)
        run_experiment(config, resume=str(last) if last.is_file() else None)
    summary = _read(directory / "summary.json")
    if summary["data"] != _read(control_dir / "summary.json")["data"]:
        raise ValueError(f"{config.name} differs from its classical-KD control in data")
    if summary["teacher"]["checkpoint_sha256"] != fingerprint(teacher):
        raise ValueError(f"{config.name} was not distilled from {teacher}")
    return directory


def run(matrix: Path, jobs: list[str] | None, students: list[str], seeds: list[int], device: str,
        data_root: str, output: Path) -> dict:
    report = _read(output / "report.json") if (output / "report.json").is_file() else {}
    for job in _read(matrix / "matrix_summary.json")["jobs"]:
        if job["status"] != "complete" or (jobs and job["id"] not in jobs):
            continue
        profile = matrix / job["dataset"] / job["profile"]
        comparison = _read(profile / "studies" / "confirmatory" / "comparison.json")
        spec, targets = comparison["study"], comparison["target_classes"]
        teachers = _teachers(profile, comparison)
        trained = {}
        for seed in seeds:
            control = profile / "baselines" / "confirmatory" / spec["student_run_template"].format(seed=seed)
            for student in students:
                for label in TEACHERS:
                    trained[(student, label, seed)] = _train(control, teachers[label], label, student,
                                                             output / job["id"], device, data_root)
                pair = [_read(trained[(student, label, seed)] / "summary.json")["initial_student_sha256"]
                        for label in TEACHERS]
                if pair[0] != pair[1]:
                    raise ValueError(f"{job['id']} {student} seed {seed}: the pair starts from different weights")
        teacher_config = from_dict(_read(profile / "baselines" / "confirmatory" / spec["teacher_run"]
                                         / "config.json"))
        teacher_config = replace(teacher_config, data=replace(teacher_config.data, root=data_root))
        evaluation = build_data(teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
                                include_test=True, diagnostic=True)
        reference = _read(profile / "studies" / "confirmatory" / "student_evaluation" / f"seed{seeds[0]}"
                          / "baseline" / "predictions.json")["identity"]["data"]
        if _json(evaluation.provenance) != reference:
            raise ValueError(f"{job['id']}: test data differs from the split the study was scored on")
        report[job["id"]] = {student: {label: {seed: _score(trained[(student, label, seed)], evaluation,
                                                            targets, seed) for seed in seeds}
                                       for label in TEACHERS} for student in students}
        write_json(output / "report.json", report)
    return report


def render(report: dict) -> str:
    lines = ["# Student architectures: repaired against original teacher", "",
             "Each cell is the repaired-teacher student minus the original-teacher student of the "
             "same architecture and seed, mean over seeds, in percentage points.", "",
             "| Study | Student | Δ accuracy | Δ target recall |", "|---|---|---:|---:|"]
    for job, students in report.items():
        for student, pair in students.items():
            delta = {key: statistics.mean(pair["repaired"][s][key] - pair["original"][s][key]
                                          for s in pair["original"]) for key in ("accuracy", "target_recall")}
            lines.append(f"| {job} | {student} | {100 * delta['accuracy']:+.2f} | "
                         f"{100 * delta['target_recall']:+.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--jobs", nargs="*", default=None, help="Study ids (default: every completed study)")
    parser.add_argument("--students", nargs="+", default=["resnet8", "resnet20"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs/student-architecture"))
    args = parser.parse_args()
    output = args.output.resolve()
    report = run(args.matrix.resolve(), args.jobs, args.students, args.seeds, args.device,
                 str(Path(args.data_root).resolve()), output)
    (output / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
