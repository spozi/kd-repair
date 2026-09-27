"""Rerun Experiment 1 jobs with a different teacher architecture.

Each Experiment 1 job config is copied with only the teacher changed: its model, its run
names, and an explicit late-stage search space. Every other setting, including the
cifar_student student, splits, seeds, schedules, repair grid and selection gates, stays as
fixed for Experiment 1. The unchanged pipeline then trains the teacher and matched
classical-KD controls, localizes and repairs channels, and distils the selected repair.
Jobs run in order and resume.

    python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan \\
        --teacher cifar_resnet18 --jobs cifar100-lt-if10 cifar100-lt-if100 --device cuda:0

For the full 52-setting matrix, use --all-jobs. It skips settings whose comparison
already ended with either an accepted repair or a no-repair decision. Use --dry-run to
list the remaining settings without writing files or starting training.

The scaled-budget arm changes only the channel budgets, e.g. to {8, 16, 32} for teachers
with wider late stages. It reuses the fixed arm's teacher and classical-KD controls: those
are trained (or confirmed complete) under the fixed-arm config in --baseline-root, linked
into the scaled output, and only the localization, repair and distillation are rerun.

    python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan \\
        --teacher cifar_resnet18 --all-jobs --budgets 8 16 32 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# The late residual stages, as for torchvision ResNet teachers.
STAGES = {"cifar_resnet18": ["stage3", "stage4"], "cifar_resnet34": ["stage3", "stage4"],
          "cifar_resnet50": ["stage3", "stage4"]}
SHORT = {"cifar_resnet18": "r18", "cifar_resnet34": "r34", "cifar_resnet50": "r50"}
FINISHED = {"complete", "no_repair_selected"}


def config_for(plan: Path, job: str, teacher: str, output: Path, budgets: list[int] | None = None) -> Path:
    spec = json.loads((plan / "configs" / f"{job}.json").read_text())
    name = f"{job}-{SHORT[teacher]}"
    spec.update(name=name, teacher_model=teacher, teacher_run=f"teacher_{name}",
                student_run_template=f"kd_{name}_seed{{seed}}", stages=STAGES[teacher])
    if budgets is not None:
        spec["channel_budgets"] = budgets
    path = output / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2) + "\n"
    if path.is_file() and path.read_text() != text:
        raise ValueError(f"A different config already exists for {name}: {path}")
    path.write_text(text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="Experiment 1 matrix-plan directory")
    parser.add_argument("--teacher", required=True, choices=sorted(STAGES))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--jobs", nargs="+", help="Specific Experiment 1 job ids")
    selection.add_argument("--all-jobs", action="store_true", help="Use every config in the matrix plan")
    parser.add_argument("--dry-run", action="store_true", help="List pending jobs without writing or training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Scaled-budget arm: channel budgets replacing the Experiment 1 grid")
    parser.add_argument("--baseline-root", type=Path, default=Path("runs/teacher-architecture"),
                        help="Fixed-arm output whose teacher and controls the scaled arm reuses")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: runs/teacher-architecture, or runs/teacher-architecture-scaled with --budgets")
    args = parser.parse_args()
    if args.budgets is not None and (min(args.budgets) < 1 or sorted(set(args.budgets)) != args.budgets):
        parser.error("--budgets must be unique increasing positive integers")
    if args.output is None:
        args.output = Path("runs/teacher-architecture" + ("-scaled" if args.budgets else ""))
    scaled = args.budgets is not None
    if scaled and args.output.resolve() == args.baseline_root.resolve():
        parser.error("The scaled arm needs an output separate from --baseline-root")
    jobs = (sorted(path.stem for path in (args.plan / "configs").glob("*.json"))
            if args.all_jobs else args.jobs)
    if not jobs:
        parser.error(f"No job configs found in {args.plan / 'configs'}")
    failures = []
    pending = 0
    for job in jobs:
        source = args.plan / "configs" / f"{job}.json"
        if not source.is_file():
            parser.error(f"Missing job config: {source}")
        name = f"{job}-{SHORT[args.teacher]}"
        comparison = args.output / name / "studies" / "confirmatory" / "comparison.json"
        if comparison.is_file() and json.loads(comparison.read_text()).get("status") in FINISHED:
            print(f"[finished] {name}", flush=True)
            continue
        pending += 1
        base_root = args.baseline_root if scaled else args.output
        baselines = base_root / name / "baselines" / "confirmatory"
        if args.dry_run:
            shared = (" (fixed-arm baselines complete)" if (baselines / "baseline_manifest.json").is_file()
                      else " (fixed-arm baselines to train)") if scaled else ""
            print(f"[pending] {name}{shared}", flush=True)
            continue
        base_config = config_for(args.plan, job, args.teacher, base_root)
        config = config_for(args.plan, job, args.teacher, args.output, args.budgets) if scaled else base_config
        root = args.output / name
        study = root / "studies" / "confirmatory"
        print(f"[job] {name}", flush=True)
        steps = [[sys.executable, "-m", "kd", "neuron-surgery-baselines", "--study-config", str(base_config),
                  "--output", str(baselines), "--root", args.data_root, "--device", args.device],
                 [sys.executable, "-m", "kd", "neuron-surgery-study", "--study-config", str(config),
                  "--baseline", str(baselines), "--output", str(study), "--device", args.device]]
        if scaled:
            # Link the shared baselines where the student and alternative-loss scripts look for them.
            link = root / "baselines" / "confirmatory"
            link.parent.mkdir(parents=True, exist_ok=True)
            target = os.path.relpath(baselines.resolve(), link.parent.resolve())
            if not link.is_symlink():
                link.symlink_to(target, target_is_directory=True)
            elif os.readlink(link) != target:
                raise ValueError(f"{link} points elsewhere than the fixed-arm baselines")
        for step in steps:
            if subprocess.run(step).returncode:
                failures.append(name)
                print(f"[failed] {name}: {' '.join(step[3:5])}", flush=True)
                break
        else:
            print(f"[done] {name}", flush=True)
    if failures:
        raise SystemExit(f"Failed jobs: {failures}")
    print(f"[summary] {len(jobs)} planned, {pending} pending", flush=True)


if __name__ == "__main__":
    main()
