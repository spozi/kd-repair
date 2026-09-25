"""Train a supervised teacher of another architecture with an Experiment 1 teacher's recipe.

Takes an existing teacher's config.json and changes only the architecture, and optionally the
dataset source and profile, so a new teacher family can be compared with the original on the
same splits, schedule and seed. It reports validation accuracy only: architecture choices
must not see the sealed test split.

    python scripts/teacher_architecture_check.py \\
        --base runs/experiment1-multidataset/cifar10/lt-if10/baselines/confirmatory/teacher_cifar10-lt-if10/config.json \\
        --model cifar_resnet18 --profile balanced --epochs 81 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kd.config import from_dict  # noqa: E402
from kd.dataset_registry import dataset_recipe  # noqa: E402
from kd.engine import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="An Experiment 1 teacher config.json")
    parser.add_argument("--model", required=True, help="Teacher architecture, e.g. cifar_resnet18")
    parser.add_argument("--source", help="Dataset to use instead of the base config's")
    parser.add_argument("--profile", help="Dataset profile to use instead of the base config's")
    parser.add_argument("--epochs", type=int, help="Epochs instead of the base config's")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, default=Path("runs/architecture-check"))
    args = parser.parse_args()

    base = from_dict(json.loads(args.base.read_text()))
    data = replace(base.data, root=str(Path(args.data_root).resolve()))
    if args.source:
        _, recipe = dataset_recipe(args.source)
        data = replace(data, source=args.source, num_classes=recipe["classes"],
                       dataset_version=recipe["version"])
    if args.profile:
        data = replace(data, dataset_profile=args.profile)
    train = base.train
    if args.epochs:
        train = replace(train, epochs=args.epochs)
    if args.device:
        train = replace(train, device=args.device)
    name = f"{args.model}_{data.source}-{data.dataset_profile}"
    config = replace(base, name=name, output_dir=str(args.output.resolve()), data=data,
                     train=train, student=replace(base.student, name=args.model))
    directory = args.output / name
    if not (directory / "summary.json").is_file():
        last = directory / "last.pt"
        run_experiment(config, resume=str(last) if last.is_file() else None)
    summary = json.loads((directory / "summary.json").read_text())
    print(json.dumps({"name": name, "model": args.model, "source": data.source,
                      "profile": data.dataset_profile, "epochs": train.epochs,
                      "validation_accuracy": summary["validation"]["accuracy"],
                      "best_epoch": summary["best_epoch"],
                      "parameters": summary["inference"]["parameters"],
                      "mean_epoch_seconds": summary["training"]["mean_epoch_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
