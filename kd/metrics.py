"""Quality, confidence, inference cost, and explicit deployment budgets."""

import math
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from .config import BenchmarkConfig
from .models import VisionModel


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate(model: VisionModel, loader, device: torch.device, *, confidence_observer=None, extended=False) -> dict:
    """Evaluate once; optionally emit detached CPU top-confidence/correctness batches."""
    previous_mode = model.training
    model.eval()
    count = correct = 0
    sums = dict(nll=0.0, confidence=0.0, entropy=0.0, correct_confidence=0.0, incorrect_confidence=0.0)
    bins = torch.zeros(15, 3, dtype=torch.float64)
    diagnostic_labels, diagnostic_probabilities = [], []
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite logits during evaluation")
            log_prob = F.log_softmax(logits, dim=1)
            prob = log_prob.exp()
            if extended:
                diagnostic_labels.append(labels.detach().cpu())
                diagnostic_probabilities.append(prob.detach().cpu())
            confidence, predictions = prob.max(dim=1)
            matched = predictions.eq(labels)
            count += labels.numel()
            correct += matched.sum().item()
            sums["nll"] += F.cross_entropy(logits, labels, reduction="sum").item()
            sums["confidence"] += confidence.sum().item()
            sums["entropy"] += (-(prob * log_prob).sum(dim=1)).sum().item()
            sums["correct_confidence"] += confidence[matched].sum().item()
            sums["incorrect_confidence"] += confidence[~matched].sum().item()
            cpu_confidence = confidence.cpu().double()
            cpu_matched = matched.cpu().double()
            if confidence_observer is not None:
                confidence_observer(cpu_confidence, cpu_matched)
            bin_indices = (cpu_confidence * 15).long().clamp(max=14)
            bins[:, 0] += torch.bincount(bin_indices, minlength=15)
            bins[:, 1] += torch.bincount(bin_indices, weights=cpu_confidence, minlength=15)
            bins[:, 2] += torch.bincount(bin_indices, weights=cpu_matched, minlength=15)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Cannot evaluate an empty dataset")
    nonempty = bins[:, 0] > 0
    ece = (bins[nonempty, 1] - bins[nonempty, 2]).abs().sum().item() / count
    result = {"samples": count, "accuracy": correct / count, "nll": sums["nll"] / count,
            "mean_confidence": sums["confidence"] / count, "entropy": sums["entropy"] / count,
            "ece_15_bins": ece,
            "correct_confidence": sums["correct_confidence"] / correct if correct else None,
            "incorrect_confidence": sums["incorrect_confidence"] / (count - correct) if count > correct else None}
    if extended:
        from .calibration_metrics import extended_prediction_metrics
        result["extended"] = extended_prediction_metrics(torch.cat(diagnostic_labels).numpy(),
                                                          torch.cat(diagnostic_probabilities).numpy())
    return result


@torch.inference_mode()
def benchmark(model: VisionModel, image_size: int, device: torch.device, config: BenchmarkConfig) -> dict:
    """Measure model-only batch-one latency; count Conv2d/Linear MACs explicitly."""
    previous_mode = model.training
    model.eval()
    images = torch.zeros(1, 3, image_size, image_size, device=device)
    macs = 0

    def count_ops(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    hooks = [m.register_forward_hook(count_ops) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        model(images)
    finally:
        for hook in hooks:
            hook.remove()
    try:
        for _ in range(config.warmup):
            model(images)
        synchronize(device)
        timings = []
        for _ in range(config.iterations):
            start = time.perf_counter()
            model(images)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    finally:
        model.train(previous_mode)
    ordered = sorted(timings)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "conv_linear_macs": macs, "estimated_flops": 2 * macs,
            "flops_convention": "2 FLOPs per Conv2d/Linear MAC; excludes normalization, activation, pooling, elementwise ops",
            "latency_median_ms": statistics.median(timings),
            "latency_p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "batch_size": 1, "image_size": image_size, "iterations": config.iterations,
            "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
            "latency_scope": "model forward only; excludes loading, preprocessing and host-to-device transfer"}


def check_budget(cost: dict, config: BenchmarkConfig) -> dict:
    violations = []
    for metric, maximum in (("parameters", config.max_parameters), ("estimated_flops", config.max_flops),
                            ("latency_median_ms", config.max_latency_ms)):
        if maximum is not None and cost[metric] > maximum:
            violations.append({"metric": metric, "actual": cost[metric], "limit": maximum})
    return {"passed": not violations, "violations": violations}
