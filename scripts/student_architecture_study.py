"""Does a repaired teacher's gain reach students of other architectures?

For every Experiment 1 setting, each student architecture and each seed, a student is distilled
from the original teacher with the setting's classical-KD recipe. Where the setting selected a
repair, a second student is distilled from the repaired teacher, sharing everything else with
the first, which is checked: initial weights, data and schedule. Where no repair was selected
the repair effect is zero by definition and only the original-teacher students are trained;
they also serve as the pairing's controls for the alternative-loss comparison. Every student is
scored on the setting's own test split with the code that scored Experiment 1.

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
from pairing_cells import cells  # noqa: E402

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
        data_root: str, output: Path, dry_run: bool = False, skip_reported: Path | None = None) -> dict:
    report = _read(output / "report.json") if (output / "report.json").is_file() else {}
    reported = set()
    if skip_reported is not None:
        for path in skip_reported.glob("*/report.json"):
            reported |= set(_read(path))
    planned = {"settings": 0, "students": 0, "to_train": 0}
    for cell in cells("cnn-cnn", matrix, jobs=jobs):
        if cell.job in reported:
            continue
        teachers = {"original": cell.teacher, **({"repaired": cell.repaired} if cell.repaired else {})}
        planned["settings"] += 1
        for seed in seeds:
            control = cell.control(seed)
            for path in (control / "config.json", control / "summary.json", *teachers.values()):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing input for {cell.job}: {path}")
            for student in students:
                for label in teachers:
                    planned["students"] += 1
                    name = f"{student}_{label}_{cell.job}_seed{seed}"
                    planned["to_train"] += not (output / cell.job / name / "summary.json").is_file()
        if dry_run:
            continue
        trained = {}
        for seed in seeds:
            for student in students:
                for label, teacher in teachers.items():
                    trained[(student, label, seed)] = _train(cell.control(seed), teacher, label, student,
                                                             output / cell.job, device, data_root)
                if "repaired" in teachers:
                    pair = [_read(trained[(student, label, seed)] / "summary.json")["initial_student_sha256"]
                            for label in teachers]
                    if pair[0] != pair[1]:
                        raise ValueError(f"{cell.job} {student} seed {seed}: the pair starts from different weights")
        teacher_config = from_dict(_read(cell.root / "baselines" / "confirmatory" / cell.spec["teacher_run"]
                                         / "config.json"))
        teacher_config = replace(teacher_config, data=replace(teacher_config.data, root=data_root))
        split = cell.spec["evaluation_split"]
        evaluation = build_data(teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
                                include_test=split == "test", include_confirmation=split == "confirmation",
                                diagnostic=True)
        reference = cell.reference_data(seeds[0])
        # Settings without a repair never scored students, so there is no recorded split to match.
        if reference is not None and _json(evaluation.provenance) != reference:
            raise ValueError(f"{cell.job}: test data differs from the split the study was scored on")
        report[cell.job] = {student: {"repair": "selected" if cell.repaired else "none",
                                      **{label: {seed: _score(trained[(student, label, seed)], evaluation,
                                                              cell.target_classes, seed) for seed in seeds}
                                         for label in teachers}}
                            for student in students}
        write_json(output / "report.json", report)
    print(f"[plan] {planned['settings']} settings, {planned['students']} students, "
          f"{planned['to_train']} still to train", flush=True)
    return report


def render(report: dict) -> str:
    lines = ["# Student architectures: repaired against original teacher", "",
             "Each cell is the repaired-teacher student minus the original-teacher student of the "
             "same architecture and seed, mean over seeds, in percentage points.", "",
             "| Study | Student | Δ accuracy | Δ target recall |", "|---|---|---:|---:|"]
    for job, students in report.items():
        for student, pair in students.items():
            if "repaired" not in pair:
                lines.append(f"| {job} | {student} | 0 (no repair) | 0 (no repair) |")
                continue
            delta = {key: statistics.mean(pair["repaired"][s][key] - pair["original"][s][key]
                                          for s in pair["original"]) for key in ("accuracy", "target_recall")}
            lines.append(f"| {job} | {student} | {100 * delta['accuracy']:+.2f} | "
                         f"{100 * delta['target_recall']:+.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--jobs", nargs="*", default=None,
                        help="Setting ids (default: all 52, with or without a selected repair)")
    parser.add_argument("--students", nargs="+", default=["resnet8", "resnet20"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs/student-architecture"))
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and count students without training")
    parser.add_argument("--skip-reported", type=Path, default=None,
                        help="Skip settings already in any ROOT/*/report.json, e.g. earlier runs' parts")
    args = parser.parse_args()
    output = args.output.resolve()
    report = run(args.matrix.resolve(), args.jobs, args.students, args.seeds, args.device,
                 str(Path(args.data_root).resolve()), output, dry_run=args.dry_run,
                 skip_reported=args.skip_reported.resolve() if args.skip_reported else None)
    if args.dry_run:
        return
    (output / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
