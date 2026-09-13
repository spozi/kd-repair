"""Predeclared, resumable CIFAR-10 comparison with a final test phase."""

from dataclasses import replace
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .checkpoints import fingerprint, load_model_checkpoint, metadata, write_json
from .config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                     ModelConfig, TrainConfig, from_dict)
from .data import build_data
from .engine import resolve_device, run_experiment
from .evaluation import collect_predictions, paired_comparison, save_prediction_report
from .models import create_model


def study_configs(output: str, root: str, device: str, teacher_epochs: int,
                  student_epochs: int, seeds: tuple[int, ...]):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Study seeds must be nonempty and unique")
    data = DataConfig(source="cifar10", root=root, num_classes=10, image_size=32,
                      horizontal_flip=True, validation_fraction=0.1, split_seed=2026)
    teacher = ExperimentConfig(name="teacher", output_dir=output, data=data,
                               student=ModelConfig("cifar_teacher"),
                               distillation=DistillationConfig(method="supervised"),
                               train=TrainConfig(epochs=teacher_epochs, batch_size=128, learning_rate=0.05,
                                                 seed=41, device=device, threads=4),
                               benchmark=BenchmarkConfig(warmup=20, iterations=100))
    students = []
    for seed in seeds:
        for method in ("supervised", "kd", "dkd"):
            students.append(replace(teacher, name=f"{method}_seed{seed}", student=ModelConfig("cifar_student"),
                                    teacher=ModelConfig("cifar_teacher", str(Path(output) / "teacher" / "student.pt")),
                                    train=replace(teacher.train, epochs=student_epochs, seed=seed),
                                    distillation=DistillationConfig(method=method, temperature=4.0, weight=0.5,
                                                                    alpha=1.0, beta=8.0, warmup_epochs=5)))
    for config in [teacher, *students]:
        config.validate()
    return teacher, students


def run_cifar_study(output="runs/cifar10", root="data", device="auto", teacher_epochs=30,
                    student_epochs=20, seeds=(42, 43, 44)) -> dict:
    directory = Path(output)
    teacher, students = study_configs(output, root, device, teacher_epochs, student_epochs, tuple(seeds))
    configs = [teacher, *students]
    protocol = {"version": 1, "dataset": "Official CIFAR-10, torchvision checksum verification",
                "split": "45000 training / 5000 stratified validation / 10000 official test",
                "hyperparameters": "Fixed in advance: T=4, weight=0.5, DKD alpha=1 beta=8; no test tuning",
                "selection": "Best epoch on validation per run; method selected by mean validation accuracy across seeds",
                "teacher_policy": "One fixed teacher (seed 41), shared across all student seeds",
                "configs": [c.to_dict() for c in configs]}
    # JSON normalization makes tuple/list representations irrelevant on resume.
    protocol = json.loads(json.dumps(protocol))
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        stored_protocol = json.loads(protocol_path.read_text())
        normalized_protocol = dict(stored_protocol)
        normalized_protocol["configs"] = [from_dict(c).to_dict() for c in stored_protocol["configs"]]
        if json.loads(json.dumps(normalized_protocol)) != protocol:
            raise ValueError("Existing study protocol differs; choose a new output directory")
        # Preserve the archived protocol and its fingerprint for historical studies.
        protocol = stored_protocol
    elif any(directory.iterdir()):
        raise FileExistsError(f"Study directory already contains unrelated files: {directory}")
    else:
        write_json(protocol_path, protocol)
    summaries = {}
    for config in configs:
        run_directory = directory / config.name
        summary_path = run_directory / "summary.json"
        if summary_path.exists():
            saved_config = from_dict(json.loads((run_directory / "config.json").read_text()))
            if saved_config.to_dict() != config.to_dict():
                raise ValueError(f"Saved configuration changed for {config.name}")
            summaries[config.name] = json.loads(summary_path.read_text())
            print(f"Reusing completed {config.name}", flush=True)
        else:
            last = run_directory / "last.pt"
            summaries[config.name] = run_experiment(config, resume=str(last) if last.exists() else None)
        write_json(directory / "progress.json", {"phase": "training", "completed": list(summaries)})

    methods = ("supervised", "kd", "dkd")
    validation_means = {method: statistics.mean(summaries[f"{method}_seed{seed}"]["validation"]["accuracy"]
                                               for seed in seeds) for method in methods}
    selection = {"method": max(validation_means, key=validation_means.get),
                 "mean_validation_accuracy": validation_means,
                 "checkpoint_sha256": {name: fingerprint(result["checkpoint"]) for name, result in summaries.items()},
                 "selection_split": "val", "protocol_sha256": fingerprint(protocol_path)}
    selection_path = directory / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Selection changed after the test phase was unlocked; use a new study")
    write_json(selection_path, selection)
    # Only now is the official test split opened and predictions calculated.
    print("All training complete; selection frozen. Starting official test evaluation.", flush=True)
    data = build_data(teacher.data, teacher.train, include_test=True)
    device_object = resolve_device(device)
    results = {}
    predictions = {}
    for config in configs:
        name = config.name
        model = create_model(config.student.name, 10)
        load_model_checkpoint(model, summaries[name]["checkpoint"],
                              metadata(config.student.name, data.classes, 32, "cifar10"))
        model.to(device_object)
        labels, probabilities = collect_predictions(model, data.test, device_object)
        quality = save_prediction_report(directory / name / "test", labels, probabilities, data.classes)
        predictions[name] = (labels, probabilities)
        results[name] = {"test": quality, "validation": summaries[name]["validation"],
                         "inference": summaries[name]["inference"], "training": summaries[name]["training"],
                         "best_epoch": summaries[name]["best_epoch"],
                         "checkpoint": summaries[name]["checkpoint"]}
        if config.distillation.method != "supervised":
            baseline_labels, baseline = predictions[f"supervised_seed{config.train.seed}"]
            if not np.array_equal(labels, baseline_labels):
                raise ValueError("Paired evaluation requires identical test order")
            results[name]["vs_supervised"] = paired_comparison(labels, baseline, probabilities)
        print(f"TEST {name}: accuracy={quality['accuracy']:.4f} macro_f1={quality['macro_f1']:.4f}", flush=True)
        del model
        if device_object.type == "mps":
            torch.mps.empty_cache()
    aggregates = {}
    for method in methods:
        scores = [results[f"{method}_seed{seed}"]["test"]["accuracy"] for seed in seeds]
        aggregates[method] = {"test_accuracy_mean": statistics.mean(scores),
                              "test_accuracy_std": statistics.stdev(scores) if len(scores) > 1 else None,
                              "seeds": list(seeds), "test_accuracies": scores,
                              "mean_validation_accuracy": validation_means[method]}
        if method != "supervised":
            deltas = [results[f"{method}_seed{seed}"]["vs_supervised"]["accuracy_delta"] for seed in seeds]
            aggregates[method]["paired_seed_deltas"] = deltas
            aggregates[method]["mean_paired_delta"] = statistics.mean(deltas)
    report = {"protocol": protocol, "selection": selection, "aggregate": aggregates, "runs": results,
              "limitations": ["Compact six-convolution models and a fixed training budget; not a state-of-the-art CIFAR benchmark.",
                              "One teacher seed; student-seed variation is conditional on that teacher and this fixed data split.",
                              "Fixed hyperparameters, not an exhaustive validation sweep.",
                              "Wilson and paired bootstrap intervals measure test-example uncertainty, not retraining uncertainty.",
                              "Three seeds provide a limited estimate of training variation; no claim of general superiority."]}
    write_json(directory / "report.json", report)
    write_json(directory / "progress.json", {"phase": "complete", "completed": list(summaries)})
    return report
