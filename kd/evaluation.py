"""Held-out classification reports and paired comparisons, independent of training."""

import math
from pathlib import Path

import numpy as np
import torch

from .checkpoints import write_json
from .runtime import autocast_context, move_images


def wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0 or not 0 <= correct <= count:
        raise ValueError("Accuracy interval requires 0 <= correct <= count and count > 0")
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def prediction_metrics(labels: np.ndarray, probabilities: np.ndarray, classes: list[str], *, extended=False) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(classes)) or not len(labels):
        raise ValueError("Predictions must match nonempty labels and the class vocabulary")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Predictions must be finite probability distributions")
    if np.any(labels < 0) or np.any(labels >= len(classes)):
        raise ValueError("Labels are outside the class vocabulary")
    predictions = probabilities.argmax(1)
    matched = predictions == labels
    count = len(labels)
    confusion = np.bincount(labels * len(classes) + predictions,
                            minlength=len(classes)**2).reshape(len(classes), len(classes))
    tp = confusion.diagonal()
    support = confusion.sum(1)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(len(classes)), where=confusion.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=precision + recall > 0)
    confidence = probabilities.max(1)
    bin_indices = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(abs((confidence[bin_indices == i] - matched[bin_indices == i]).sum()) for i in range(15)) / count
    one_hot = np.eye(len(classes))[labels]
    result = {"samples": count, "accuracy": float(matched.mean()),
            "accuracy_wilson_95": wilson_interval(int(matched.sum()), count),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "nll": float(-np.log(np.maximum(probabilities[np.arange(count), labels], 1e-300)).mean()),
            "brier_score": float(np.square(probabilities - one_hot).sum(1).mean()),
            "ece_15_bins": float(ece), "mean_confidence": float(confidence.mean()),
            "confusion_matrix": confusion.tolist(), "confusion_axes": "rows=true, columns=predicted",
            "per_class": [{"class": name, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i, name in enumerate(classes)],
            "uncertainty_note": "Wilson interval describes test-sample uncertainty for this fitted model, not variation from retraining."}
    if extended:
        from .calibration_metrics import extended_prediction_metrics
        result["extended"] = extended_prediction_metrics(labels, probabilities, classes)
    return result


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device, *, precision="float32",
                        channels_last=False) -> tuple[np.ndarray, np.ndarray]:
    previous_mode = model.training
    model.eval()
    labels, probabilities = [], []
    try:
        for images, targets in loader:
            images = move_images(images, device, channels_last)
            with autocast_context(device, precision):
                logits = model(images).logits.float()
            probabilities.append(logits.softmax(1))
            labels.append(targets.numpy())
    finally:
        model.train(previous_mode)
    return np.concatenate(labels), torch.cat(probabilities).cpu().numpy()


def save_prediction_report(directory: Path, labels, probabilities, classes: list[str], *, extended=False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    report = prediction_metrics(labels, probabilities, classes, extended=extended)
    np.savez_compressed(directory / "predictions.npz", labels=labels, probabilities=probabilities,
                        classes=np.asarray(classes))
    write_json(directory / "metrics.json", report)
    np.savetxt(directory / "confusion.csv", np.asarray(report["confusion_matrix"]), delimiter=",", fmt="%d",
               header=",".join(classes), comments="")
    return report


def paired_comparison(labels, baseline_probabilities, candidate_probabilities, *, seed=2026, repetitions=2000) -> dict:
    """Paired test-example bootstrap and exact two-sided McNemar test."""
    labels = np.asarray(labels)
    baseline = np.asarray(baseline_probabilities).argmax(1) == labels
    candidate = np.asarray(candidate_probabilities).argmax(1) == labels
    differences = candidate.astype(float) - baseline.astype(float)
    generator = np.random.default_rng(seed)
    # The three possible paired differences form a multinomial distribution;
    # resampling counts is exactly equivalent to resampling the test examples.
    counts = np.array([(differences == value).sum() for value in (-1, 0, 1)])
    samples = generator.multinomial(len(labels), counts / len(labels), size=repetitions)
    deltas = (samples[:, 2] - samples[:, 0]) / len(labels)
    lost, gained = int(counts[0]), int(counts[2])
    discordant = lost + gained
    if not discordant:
        p_value = 1.0
    else:
        k = np.arange(min(lost, gained) + 1)
        log_terms = np.array([math.lgamma(discordant + 1) - math.lgamma(int(i) + 1)
                              - math.lgamma(discordant - int(i) + 1) - discordant * math.log(2) for i in k])
        maximum = log_terms.max()
        p_value = min(1.0, 2 * math.exp(float(maximum)) * float(np.exp(log_terms - maximum).sum()))
    return {"accuracy_delta": float(differences.mean()),
            "paired_bootstrap_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "baseline_only_correct": lost, "candidate_only_correct": gained,
            "mcnemar_exact_two_sided_p": p_value, "bootstrap_repetitions": repetitions,
            "note": "Conditional on this paired fitted model pair; exploratory unadjusted p-value, not a claim across training seeds."}


def paired_calibration_comparison(labels, baseline_probabilities, candidate_probabilities,
                                  *, seed=2026, repetitions=2000) -> dict:
    """Paired bootstrap with ECE bins recomputed for each resampled test set."""
    labels = np.asarray(labels, dtype=np.int64)
    count = len(labels)
    if count == 0 or repetitions < 1:
        raise ValueError("Paired calibration requires nonempty data and positive repetitions")

    def statistics_for(probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        confidence = probabilities.max(1)
        correct = probabilities.argmax(1) == labels
        bins = np.minimum((confidence * 15).astype(int), 14)
        residual = confidence - correct
        nll = -np.log(np.maximum(probabilities[np.arange(count), labels], 1e-300))
        brier = np.square(probabilities - np.eye(probabilities.shape[1])[labels]).sum(1)
        ece = np.abs(np.bincount(bins, weights=residual, minlength=15)).sum()/count
        return bins, residual, nll, brier, ece

    b = statistics_for(baseline_probabilities)
    c = statistics_for(candidate_probabilities)
    rng = np.random.default_rng(seed)
    bootstraps = {"ece": [], "nll": [], "brier": []}
    for start in range(0, repetitions, 64):
        size = min(64, repetitions-start)
        indices = rng.integers(count, size=(size, count))
        offsets = np.arange(size)[:, None]*15

        def ece_resampled(stats):
            sums = np.bincount((stats[0][indices]+offsets).ravel(), weights=stats[1][indices].ravel(),
                               minlength=size*15).reshape(size,15)
            return np.abs(sums).sum(1)/count

        bootstraps["ece"].extend((ece_resampled(c)-ece_resampled(b)).tolist())
        bootstraps["nll"].extend((c[2]-b[2])[indices].mean(1).tolist())
        bootstraps["brier"].extend((c[3]-b[3])[indices].mean(1).tolist())
    result = {"ece_delta": float(c[4]-b[4]), "nll_delta": float((c[2]-b[2]).mean()),
              "brier_delta": float((c[3]-b[3]).mean()), "bootstrap_repetitions": repetitions,
              "note": "Candidate minus control; negative is better. Intervals condition on fitted models and resample paired test examples."}
    for name, values in bootstraps.items():
        result[f"{name}_paired_bootstrap_95"] = np.quantile(values,[.025,.975]).tolist()
    return result
