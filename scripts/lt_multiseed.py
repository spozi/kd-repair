"""Multi-seed, two-factor confirmation of the long-tailed calibration diagnostic.

Extends `scripts/lt_study.py` from one seed at imbalance factor 100 to three
student seeds at factors 100 and 10, so the frequency-ordered miscalibration and
the two post-hoc corrections can be reported with seed variation rather than as
single point estimates.

Design mirrors the balanced CIFAR-10 study: one teacher per imbalance factor
(seed 41), shared across student seeds, and epochs scaled so total sample
exposure is roughly constant against the balanced 45,000 x 20 baseline
(11,167 x 81 at factor 100; 18,391 x 49 at factor 10).

Calibrators are fitted on the equally long-tailed validation split and evaluated
on the untouched balanced official test split. Four post-hoc variants:

  uncalibrated              raw softmax
  TS (plain NLL)            one temperature, unweighted validation NLL
  TS (balanced NLL)         one temperature, samples weighted 1/n_class
  logit adjustment + TS     subtract tau*log(prior), then scale; balanced NLL

Only the last changes predictions; the accuracy column makes that visible.

Exploratory: reuses the official test set that earlier studies examined. Runs are
resumable — completed runs are reused rather than retrained.

    python scripts/lt_multiseed.py
"""

import json
from pathlib import Path
import statistics
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
from kd.posthoc import LogitCalibrator, collect_logits

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "runs" / "cifar10-lt-multiseed"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SEEDS = (42, 43, 44)
TEACHER_SEED = 41
# factor -> epochs matching the balanced 45,000 x 20 sample exposure
FACTORS = {100.0: 81, 10.0: 49}
METHODS = ("uncalibrated", "TS (plain NLL)", "TS (balanced NLL)", "logit adjustment + TS")
REPORTED = ("accuracy", "ece_15_bins", "classwise_ece_15_bins", "aurc", "nll")


def config_for(name, model, method, factor, epochs, seed, teacher=None):
    config = ExperimentConfig(
        name=name, output_dir=str(OUTPUT),
        data=DataConfig(source="cifar10", root=str(ROOT / "data"), num_classes=10, image_size=32,
                        horizontal_flip=True, validation_fraction=0.1, split_seed=2026,
                        imbalance_factor=factor),
        student=ModelConfig(model),
        teacher=ModelConfig("cifar_teacher", teacher) if teacher else ModelConfig("cifar_teacher"),
        distillation=DistillationConfig(method=method, temperature=4.0, weight=0.5, warmup_epochs=5),
        train=TrainConfig(epochs=epochs, batch_size=128, learning_rate=0.05, momentum=0.9,
                          weight_decay=0.0005, workers=0, threads=4, device="auto", seed=seed),
        benchmark=BenchmarkConfig(warmup=20, iterations=100))
    config.validate(require_teacher=method != "supervised")
    return config


def train_if_needed(config):
    path = OUTPUT / config.name
    if (path / "student.pt").exists():
        return str(path / "student.pt")
    print(f"Training {config.name} ({config.train.epochs} epochs)...", flush=True)
    run_experiment(config)
    return str(path / "student.pt")


def logits_for(config, checkpoint, val_loader, test_loader):
    model = create_model(config.student.name, 10)
    load_model_checkpoint(model, checkpoint, metadata(config.student.name, CLASSES, 32, "cifar10"))
    model.requires_grad_(False).eval()
    device = torch.device("cpu")
    out = (collect_logits(model, val_loader, device), collect_logits(model, test_loader, device))
    del model
    return out


def _fit(params, forward, Z, Y, weights):
    optimizer = torch.optim.LBFGS(params, max_iter=400, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        losses = torch.nn.functional.cross_entropy(forward(Z), Y, reduction="none")
        loss = (losses * weights).sum() / weights.sum() if weights is not None else losses.mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return [p.detach() for p in params]


def calibrated_variants(v_labels, v_logits, t_logits, log_prior):
    """Fit every variant on validation; return test probabilities for each."""
    Z = torch.tensor(v_logits, dtype=torch.float64)
    Y = torch.tensor(v_labels)
    Zt = torch.tensor(t_logits, dtype=torch.float64)
    counts = np.bincount(v_labels, minlength=10).astype(float)
    balanced = torch.tensor(1.0 / np.maximum(counts, 1))[Y]

    out, params = {"uncalibrated": Zt}, {}
    for label, weights in (("TS (plain NLL)", None), ("TS (balanced NLL)", balanced)):
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        (fitted,) = _fit([log_t], lambda z, p=log_t: z / torch.exp(p), Z, Y, weights)
        out[label] = Zt / torch.exp(fitted)
        params[label] = {"temperature": float(torch.exp(fitted))}

    tau = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    tau_f, log_t_f = _fit([tau, log_t],
                          lambda z, a=tau, b=log_t: (z - a * log_prior) / torch.exp(b),
                          Z, Y, balanced)
    out["logit adjustment + TS"] = (Zt - tau_f * log_prior) / torch.exp(log_t_f)
    params["logit adjustment + TS"] = {"tau": float(tau_f), "temperature": float(torch.exp(log_t_f))}
    return {k: LogitCalibrator().predict(v.numpy()) for k, v in out.items()}, params


def gap_spread(labels, probability):
    prediction, confidence = probability.argmax(1), probability.max(1)
    correct = (prediction == labels).astype(float)
    gaps = [confidence[prediction == k].mean() - correct[prediction == k].mean()
            for k in range(10) if (prediction == k).any()]
    return float(max(gaps) - min(gaps))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for factor, epochs in FACTORS.items():
        teacher_cfg = config_for(f"teacher_f{factor:g}", "cifar_teacher", "supervised",
                                 factor, epochs, TEACHER_SEED)
        data = build_data(teacher_cfg.data, teacher_cfg.train, include_test=True)
        counts = np.array(data.provenance["train_per_class"], dtype=float)
        log_prior = torch.tensor(np.log(counts / counts.sum()))
        print(f"\n=== imbalance factor {factor:g} | train {int(counts.sum())} images "
              f"({int(counts[0])}..{int(counts[-1])}) | {epochs} epochs ===", flush=True)
        teacher_ckpt = train_if_needed(teacher_cfg)

        for seed in SEEDS:
            for arm, method, teacher in (("supervised", "supervised", None), ("kd", "kd", teacher_ckpt)):
                name = f"{arm}_f{factor:g}_seed{seed}"
                config = config_for(name, "cifar_student", method, factor, epochs, seed, teacher)
                checkpoint = train_if_needed(config)
                (v_labels, v_logits), (t_labels, t_logits) = logits_for(config, checkpoint, data.val, data.test)
                variants, params = calibrated_variants(v_labels, v_logits, t_logits, log_prior)
                for label, probability in variants.items():
                    quality = extended_prediction_metrics(t_labels, probability, CLASSES, include_curve=False)
                    records.append({"factor": factor, "seed": seed, "arm": arm, "method": label,
                                    "parameters": params.get(label, {}),
                                    "gap_spread": gap_spread(t_labels, probability),
                                    **{k: quality[k] for k in REPORTED}})
                print(f"  {name}: acc={records[-1]['accuracy']:.4f} "
                      f"(logit-adjusted) ECE={records[-1]['ece_15_bins']:.4f}", flush=True)

    write_json(OUTPUT / "multiseed.json", records)
    header = (f"{'factor':>7}{'arm':>12}{'method':<24}{'acc':>16}{'ECE':>16}"
              f"{'cwECE':>16}{'AURC':>18}{'spread':>14}")
    print("\n" + header)
    print("-" * len(header))
    for factor in FACTORS:
        for arm in ("supervised", "kd"):
            for method in METHODS:
                group = [r for r in records if r["factor"] == factor and r["arm"] == arm
                         and r["method"] == method]
                if len(group) < 2:
                    continue

                def cell(key, scale=100.0, digits=2):
                    values = [r[key] * scale for r in group]
                    return f"{statistics.mean(values):.{digits}f}±{statistics.stdev(values):.{digits}f}"
                print(f"{factor:>7.0f}{arm:>12}  {method:<22}"
                      f"{cell('accuracy'):>16}{cell('ece_15_bins', digits=3):>16}"
                      f"{cell('classwise_ece_15_bins', digits=3):>16}"
                      f"{cell('aurc', scale=1, digits=5):>18}{cell('gap_spread', digits=1):>14}")
    print(f"\nSaved: {OUTPUT / 'multiseed.json'}")
    print("Values are mean±sd over seeds 42, 43, 44. Percentages except AURC.")
    print("Exploratory: reuses the official test set examined by earlier studies.")


if __name__ == "__main__":
    main()
