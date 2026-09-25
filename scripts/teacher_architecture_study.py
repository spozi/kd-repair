"""Rerun Experiment 1 jobs with a different teacher architecture.

Each Experiment 1 job config is copied with only the teacher changed: its model, its run
names, and an explicit late-stage search space. Every other setting, including the
cifar_student student, splits, seeds, schedules, repair grid and selection gates, stays as
fixed for Experiment 1. The unchanged pipeline then trains the teacher and matched
classical-KD controls, localizes and repairs channels, and distils the selected repair.
Jobs run in order and resume.

    python scripts/teacher_architecture_study.py --plan runs/experiment1-multidataset/matrix-plan \\
        --teacher cifar_resnet18 --jobs cifar100-lt-if10 cifar100-lt-if100 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The late residual stages, as for torchvision ResNet teachers.
STAGES = {"cifar_resnet18": ["stage3", "stage4"], "cifar_resnet34": ["stage3", "stage4"],
          "cifar_resnet50": ["stage3", "stage4"]}
SHORT = {"cifar_resnet18": "r18", "cifar_resnet34": "r34", "cifar_resnet50": "r50"}


def config_for(plan: Path, job: str, teacher: str, output: Path) -> Path:
    spec = json.loads((plan / "configs" / f"{job}.json").read_text())
    name = f"{job}-{SHORT[teacher]}"
    spec.update(name=name, teacher_model=teacher, teacher_run=f"teacher_{name}",
                student_run_template=f"kd_{name}_seed{{seed}}", stages=STAGES[teacher])
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
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs/teacher-architecture"))
    args = parser.parse_args()
    failures = []
    for job in args.jobs:
        config = config_for(args.plan, job, args.teacher, args.output)
        root = args.output / config.stem
        baselines, study = root / "baselines" / "confirmatory", root / "studies" / "confirmatory"
        print(f"[job] {config.stem}", flush=True)
        steps = [[sys.executable, "-m", "kd", "neuron-surgery-baselines", "--study-config", str(config),
                  "--output", str(baselines), "--root", args.data_root, "--device", args.device],
                 [sys.executable, "-m", "kd", "neuron-surgery-study", "--study-config", str(config),
                  "--baseline", str(baselines), "--output", str(study), "--device", args.device]]
        for step in steps:
            if subprocess.run(step).returncode:
                failures.append(config.stem)
                print(f"[failed] {config.stem}: {' '.join(step[3:5])}", flush=True)
                break
        else:
            print(f"[done] {config.stem}", flush=True)
    if failures:
        raise SystemExit(f"Failed jobs: {failures}")


if __name__ == "__main__":
    main()
