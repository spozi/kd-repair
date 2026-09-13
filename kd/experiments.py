"""Controlled ablations: baseline, response sweeps, features, augmentation."""

import csv
from dataclasses import replace
from itertools import product
from pathlib import Path

from .checkpoints import write_json
from .config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                     ModelConfig, TrainConfig)
from .engine import run_experiment


def grid_configs(config: ExperimentConfig, temperatures=(2.0, 4.0, 8.0), weights=(0.25, 0.5, 0.75)) -> list[ExperimentConfig]:
    directory = str(Path(config.output_dir) / config.name)
    base = replace(config, output_dir=directory,
                   data=replace(config.data, augmentation="basic"),
                   distillation=replace(config.distillation, feature_weight=0.0))
    runs = [replace(base, name="supervised", distillation=replace(base.distillation, method="supervised"))]
    for method, temperature, weight in product(("kd", "dkd"), temperatures, weights):
        run = replace(base, name=f"{method}_t{temperature:g}_w{weight:g}",
                      distillation=replace(base.distillation, method=method, temperature=temperature, weight=weight))
        run.validate(require_teacher=False)
        runs.append(run)
    if len({run.name for run in runs}) != len(runs) or len(runs) == 1:
        raise ValueError("Sweep temperatures/weights must be nonempty and unique")
    return runs


def run_ablation(config: ExperimentConfig, temperatures=(2.0, 4.0, 8.0), weights=(0.25, 0.5, 0.75)) -> dict:
    runs = grid_configs(config, temperatures, weights)
    # Validate before creating any output, including when the base method was supervised.
    for run in runs:
        run.validate()
    directory = Path(config.output_dir) / config.name
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Sweep directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "plan.json", [run.to_dict() for run in runs])
    results = []

    def execute(run):
        result = run_experiment(run)
        results.append((run, result))
        write_json(directory / "progress.json", [r for _, r in results])

    for run in runs:
        execute(run)
    best_dkd, _ = max((pair for pair in results if pair[0].distillation.method == "dkd"),
                      key=lambda pair: pair[1]["validation"]["accuracy"])
    execute(replace(best_dkd, name="dkd_features",
                    distillation=replace(best_dkd.distillation,
                                         feature_weight=config.distillation.feature_weight or 1.0)))
    best_config, _ = max(results, key=lambda pair: pair[1]["validation"]["accuracy"])
    execute(replace(best_config, name="best_strong_augmentation",
                    data=replace(best_config.data, augmentation="strong")))
    baseline_seconds = results[0][1]["training"]["mean_epoch_seconds"]
    rows = []
    for run, result in results:
        row = {"name": run.name, "method": run.distillation.method,
               "temperature": run.distillation.temperature, "weight": run.distillation.weight,
               "feature_weight": run.distillation.feature_weight, "augmentation": run.data.augmentation,
               "accuracy": result["validation"]["accuracy"], "parameters": result["inference"]["parameters"],
               "estimated_flops": result["inference"]["estimated_flops"],
               "latency_median_ms": result["inference"]["latency_median_ms"],
               "mean_epoch_seconds": result["training"]["mean_epoch_seconds"],
               "training_overhead_ratio": result["training"]["mean_epoch_seconds"] / baseline_seconds,
               "extra_trainable_parameters": result["training"]["extra_trainable_parameters"],
               "within_budget": result["deployment_budget"]["passed"], "checkpoint": result["checkpoint"]}
        rows.append(row)
    eligible = [r for r in rows if r["within_budget"]]
    best = max(eligible, key=lambda r: r["accuracy"]) if eligible else None
    report = {"selection_split": "val", "seed": config.train.seed,
              "synthetic_smoke_only": config.data.source == "synthetic",
              "status": "selected" if best else "no_model_within_budget", "best": best, "runs": rows,
              "notes": "Training overhead is relative to the supervised student on this machine. Repeat seeds for statistical comparisons; evaluate the selected checkpoint on held-out test data separately."}
    write_json(directory / "comparison.json", report)
    with (directory / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if best:
        selected_config = next(run for run, _ in results if run.name == best["name"])
        write_json(directory / "selected_config.json", selected_config.to_dict())
    return report


def run_smoke(output: str) -> dict:
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Smoke directory is not empty: {directory}")
    teacher = ExperimentConfig(
        name="teacher", output_dir=str(directory), data=DataConfig(train_samples=64, val_samples=32, test_samples=32),
        student=ModelConfig("tiny_medium"), distillation=DistillationConfig(method="supervised"),
        train=TrainConfig(epochs=3, batch_size=16, learning_rate=0.05, device="cpu"),
        benchmark=BenchmarkConfig(warmup=1, iterations=3),
    )
    teacher_result = run_experiment(teacher)
    student = replace(teacher, name="ablations", student=ModelConfig("tiny_small"),
                      teacher=ModelConfig("tiny_medium", teacher_result["checkpoint"]),
                      train=replace(teacher.train, epochs=1),
                      distillation=DistillationConfig(method="dkd", warmup_epochs=0))
    report = run_ablation(student)
    write_json(directory / "smoke.json", report)
    return report

