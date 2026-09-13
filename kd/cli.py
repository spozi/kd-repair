"""Small CLI; experiments remain directly callable from Python."""

import argparse
import json
from pathlib import Path

from .config import from_dict, load_config
from .engine import evaluate_checkpoint, run_experiment
from .experiments import grid_configs, run_ablation, run_smoke


def read_config(path: str):
    if Path(path).suffix == ".json":
        return from_dict(json.loads(Path(path).read_text()))
    return load_config(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Reproducible image-classification knowledge distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train a supervised teacher/student or distill a student")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", help="Resume an epoch-boundary last.pt in the original run directory")
    sweep = subparsers.add_parser("sweep", help="Run baseline, KD/DKD grids, features, and stronger augmentation")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--temperatures", type=float, nargs="+", default=[2, 4, 8])
    sweep.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    sweep.add_argument("--dry-run", action="store_true", help="Print the initial grid without training")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate an already selected checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    smoke = subparsers.add_parser("smoke", help="Run offline synthetic teacher training and all ablations on CPU")
    smoke.add_argument("--output", default="runs/smoke")
    cifar = subparsers.add_parser("cifar10", help="Run the full held-out, repeated-seed CIFAR-10 study")
    cifar.add_argument("--output", default="runs/cifar10")
    cifar.add_argument("--root", default="data")
    cifar.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    cifar.add_argument("--teacher-epochs", type=int, default=30)
    cifar.add_argument("--student-epochs", type=int, default=20)
    cifar.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    calibration = subparsers.add_parser("calibration-study", help="Compare fixed KD against triggered MMCE across three seeds")
    calibration.add_argument("--baseline",default="runs/cifar10")
    calibration.add_argument("--output",default="runs/cifar10-calibration")
    calibration.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="mps")
    cpc = subparsers.add_parser("cpc-study", help="Compare matched KD and fixed-coefficient CPC with validation-fitted TS")
    cpc.add_argument("--baseline", default="runs/cifar10")
    cpc.add_argument("--output", default="runs/cifar10-cpc")
    cpc.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="mps")
    cpc.add_argument("--dry-run", action="store_true", help="Validate and print the protocol without training or writing artifacts")
    args = parser.parse_args(argv)
    try:
        if args.command == "cpc-study":
            from .cpc_study import run_cpc_study
            report = run_cpc_study(args.baseline,args.output,args.device,dry_run=args.dry_run)
        elif args.command == "calibration-study":
            from .calibration_study import run_calibration_study
            report = run_calibration_study(args.baseline,args.output,args.device)
        elif args.command == "cifar10":
            from .cifar_study import run_cifar_study
            report = run_cifar_study(args.output, args.root, args.device, args.teacher_epochs, args.student_epochs, tuple(args.seeds))
        elif args.command == "smoke":
            report = run_smoke(args.output)
        else:
            config = read_config(args.config)
            if args.command == "train":
                report = run_experiment(config, resume=args.resume)
            elif args.command == "evaluate":
                report = evaluate_checkpoint(config, args.checkpoint, args.split)
            elif args.dry_run:
                report = {"initial_grid": [c.to_dict() for c in grid_configs(config, args.temperatures, args.weights)],
                          "adaptive_steps": ["Best DKD + stage features", "Best preceding variant + strong augmentation"]}
            else:
                report = run_ablation(config, args.temperatures, args.weights)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"kd: {error}\n")
