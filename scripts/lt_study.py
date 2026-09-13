"""First diagnostic on long-tailed CIFAR-10: is overconfidence frequency-structured?

The question this answers is narrow and deliberately so. Under an exponential
imbalance the literature predicts (a) large miscalibration, (b) that it is
organised by class frequency — head classes overconfident — and (c) that a single
global temperature fitted on an equally-skewed validation split does not transfer
to the balanced test split. Sections 3-4 of docs/calibration-roadmap.md showed
none of those hold on balanced CIFAR-10, where every metric sat at its floor.

Training uses the long-tailed train split, temperature fitting uses the equally
long-tailed validation split, and evaluation uses the untouched balanced official
test split. Epochs are raised so total sample exposure roughly matches the
balanced 20-epoch studies (11,167 x 80 vs 45,000 x 20).

Exploratory: this reuses the official test set, which earlier studies in this
project have already examined. It is a diagnostic, not a confirmatory study, and
carries no protocol freeze.

    python scripts/lt_study.py
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.calibration_metrics import extended_prediction_metrics
from kd.checkpoints import load_model_checkpoint, metadata, write_json
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, TrainConfig)
from kd.data import build_data
from kd.engine import run_experiment
from kd.models import create_model
from kd.posthoc import LogitCalibrator, collect_logits, fit_calibrators

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "runs" / "cifar10-lt"
IMBALANCE = 100.0
EPOCHS = 80
TEMPS = (np.arange(1, 501) / 100).tolist()
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def base(name, model, method, epochs=EPOCHS, teacher=None):
    config = ExperimentConfig(
        name=name, output_dir=str(OUTPUT),
        data=DataConfig(source="cifar10", root=str(ROOT / "data"), num_classes=10, image_size=32,
                        horizontal_flip=True, validation_fraction=0.1, split_seed=2026,
                        imbalance_factor=IMBALANCE),
        student=ModelConfig(model),
        teacher=ModelConfig("cifar_teacher", teacher) if teacher else ModelConfig("cifar_teacher"),
        distillation=DistillationConfig(method=method, temperature=4.0, weight=0.5, warmup_epochs=5),
        train=TrainConfig(epochs=epochs, batch_size=128, learning_rate=0.05, momentum=0.9,
                          weight_decay=0.0005, workers=0, threads=4, device="auto", seed=42),
        benchmark=BenchmarkConfig(warmup=20, iterations=100))
    config.validate(require_teacher=method != "supervised")
    return config


def per_class_temperatures(logits, labels, grid=TEMPS):
    """One temperature per predicted class. Scaling a row by a scalar cannot
    change its argmax, so predictions are preserved exactly."""
    prediction = logits.argmax(1)
    temps = np.ones(10)
    counts = np.zeros(10, dtype=int)
    for k in range(10):
        m = prediction == k
        counts[k] = m.sum()
        if m.sum() == 0:
            continue
        best, best_t = np.inf, 1.0
        for t in grid:
            p = LogitCalibrator(temperature=float(t)).predict(logits[m])
            nll = -np.log(np.maximum(p[np.arange(m.sum()), labels[m]], 1e-300)).mean()
            if nll < best:
                best, best_t = nll, float(t)
        temps[k] = best_t
    return temps, counts


def evaluate(name, checkpoint, config, val_loader, test_loader):
    model = create_model(config.student.name, 10)
    load_model_checkpoint(model, checkpoint, metadata(config.student.name, CLASSES, 32, "cifar10"))
    model.requires_grad_(False).eval()
    device = torch.device("cpu")
    v_labels, v_logits = collect_logits(model, val_loader, device)
    t_labels, t_logits = collect_logits(model, test_loader, device)
    del model

    winners, _ = fit_calibrators(v_logits, v_labels, temperatures=TEMPS, gammas=[0.0])
    global_t = winners["temperature_nll"]["parameters"]["temperature"]
    class_t, fit_counts = per_class_temperatures(v_logits, v_labels)

    variants = {
        "uncalibrated": t_logits,
        f"global TS (T={global_t:.2f})": t_logits / global_t,
        "per-class TS": t_logits / class_t[t_logits.argmax(1)][:, None],
    }
    rows = {}
    for variant, scaled in variants.items():
        probability = LogitCalibrator().predict(scaled)
        assert np.array_equal(probability.argmax(1), t_logits.argmax(1))
        rows[variant] = extended_prediction_metrics(t_labels, probability, CLASSES, include_curve=False)
    return {"name": name, "rows": rows, "global_temperature": global_t,
            "per_class_temperature": class_t.tolist(), "validation_fit_counts": fit_counts.tolist(),
            "test_labels": t_labels, "test_logits": t_logits}


def frequency_gap(labels, probability, train_per_class):
    """Confidence minus accuracy per predicted class, ordered by train frequency."""
    prediction, confidence = probability.argmax(1), probability.max(1)
    correct = (prediction == labels).astype(float)
    out = []
    for k in range(10):
        m = prediction == k
        out.append({"class": CLASSES[k], "train_images": int(train_per_class[k]),
                    "predicted": int(m.sum()),
                    "confidence": float(confidence[m].mean()) if m.any() else float("nan"),
                    "accuracy": float(correct[m].mean()) if m.any() else float("nan")})
    return out


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    teacher_cfg = base("teacher_lt", "cifar_teacher", "supervised")
    data = build_data(teacher_cfg.data, teacher_cfg.train, include_test=True)
    train_per_class = data.provenance["train_per_class"]
    print(f"long-tailed train per class: {train_per_class} (total {sum(train_per_class)})")
    print(f"long-tailed   val per class: {data.provenance['val_per_class']}")
    print(f"balanced test: {len(data.test.dataset)} images\n")

    if not (OUTPUT / "teacher_lt" / "student.pt").exists():
        print("Training long-tailed teacher...")
        run_experiment(teacher_cfg)
    teacher_ckpt = str(OUTPUT / "teacher_lt" / "student.pt")

    arms = [("supervised_lt", base("supervised_lt", "cifar_student", "supervised")),
            ("kd_lt", base("kd_lt", "cifar_student", "kd", teacher=teacher_ckpt))]
    results = []
    for name, config in arms:
        if not (OUTPUT / name / "student.pt").exists():
            print(f"Training {name}...")
            run_experiment(config)
        results.append(evaluate(name, str(OUTPUT / name / "student.pt"), config, data.val, data.test))

    hdr = f"{'arm':<16}{'variant':<22}{'acc':>7}{'ECE':>9}{'cwECE':>9}{'AURC':>10}{'NLL':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        for variant, m in r["rows"].items():
            print(f"{r['name']:<16}{variant:<22}{100*m['accuracy']:>6.2f}%{100*m['ece_15_bins']:>8.3f}%"
                  f"{100*m['classwise_ece_15_bins']:>8.3f}%{m['aurc']:>10.5f}{m['nll']:>9.5f}")

    print("\nConfidence minus accuracy by predicted class, ordered head to tail")
    print("(uncalibrated, then after the global temperature):")
    for r in results:
        print(f"\n  {r['name']}  global T = {r['global_temperature']:.2f}")
        print(f"    {'class':<12}{'train n':>9}{'raw gap':>10}{'TS gap':>10}{'val fit n':>11}{'per-class T':>13}")
        raw = LogitCalibrator().predict(r["test_logits"])
        ts = LogitCalibrator().predict(r["test_logits"] / r["global_temperature"])
        a = frequency_gap(r["test_labels"], raw, train_per_class)
        b = frequency_gap(r["test_labels"], ts, train_per_class)
        for i in range(10):
            print(f"    {a[i]['class']:<12}{a[i]['train_images']:>9}"
                  f"{100*(a[i]['confidence']-a[i]['accuracy']):>+9.2f}%"
                  f"{100*(b[i]['confidence']-b[i]['accuracy']):>+9.2f}%"
                  f"{r['validation_fit_counts'][i]:>11}{r['per_class_temperature'][i]:>13.2f}")

    write_json(OUTPUT / "diagnostic.json",
               [{k: v for k, v in r.items() if k not in ("test_labels", "test_logits")} for r in results])
    print(f"\nSaved: {OUTPUT / 'diagnostic.json'}")
    print("Exploratory: reuses the official test set examined by earlier studies.")


if __name__ == "__main__":
    main()
