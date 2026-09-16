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
from .runtime import autocast_context, move_images, move_labels, resolve_precision


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate(model: VisionModel, loader, device: torch.device, *, confidence_observer=None,
             extended=False, precision="float32", channels_last=False) -> dict:
    """Evaluate once; optionally emit detached CPU top-confidence/correctness batches."""
    previous_mode = model.training
    model.eval()
    count = 0
    correct = torch.zeros((), dtype=torch.long, device=device)
    sums = {name: torch.zeros((), device=device) for name in
            ("nll", "confidence", "entropy", "correct_confidence", "incorrect_confidence")}
    bin_device = torch.device("cpu") if device.type == "mps" else device
    bins = torch.zeros(15, 3, dtype=torch.float64, device=bin_device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    diagnostic_labels, diagnostic_probabilities = [], []
    observer_confidence, observer_correct = [], []
    try:
        for images, labels in loader:
            images = move_images(images, device, channels_last)
            labels = move_labels(labels, device)
            with autocast_context(device, precision):
                logits = model(images).logits.float()
            finite &= torch.isfinite(logits).all()
            log_prob = F.log_softmax(logits, dim=1)
            prob = log_prob.exp()
            if extended:
                diagnostic_labels.append(labels.detach())
                diagnostic_probabilities.append(prob.detach())
            confidence, predictions = prob.max(dim=1)
            matched = predictions.eq(labels)
            count += labels.numel()
            correct += matched.sum()
            sums["nll"] += F.cross_entropy(logits, labels, reduction="sum")
            sums["confidence"] += confidence.sum()
            sums["entropy"] += (-(prob * log_prob).sum(dim=1)).sum()
            sums["correct_confidence"] += confidence[matched].sum()
            sums["incorrect_confidence"] += confidence[~matched].sum()
            if confidence_observer is not None:
                observer_confidence.append(confidence.detach())
                observer_correct.append(matched.detach())
            bin_confidence = confidence.cpu().double() if device.type == "mps" else confidence.double()
            bin_matched = matched.cpu().double() if device.type == "mps" else matched.double()
            bin_indices = (bin_confidence * 15).long().clamp(min=0, max=14)
            bins[:, 0] += torch.bincount(bin_indices, minlength=15)
            bins[:, 1] += torch.bincount(bin_indices, weights=bin_confidence, minlength=15)
            bins[:, 2] += torch.bincount(bin_indices, weights=bin_matched, minlength=15)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Cannot evaluate an empty dataset")
    if not bool(finite.cpu()):
        raise FloatingPointError("Non-finite logits during evaluation")
    if confidence_observer is not None:
        confidence_observer(torch.cat(observer_confidence).cpu().double(),
                            torch.cat(observer_correct).cpu().double())
    bins = bins.cpu()
    correct = int(correct.cpu())
    sums = {name: value.item() for name, value in sums.items()}
    nonempty = bins[:, 0] > 0
    ece = (bins[nonempty, 1] - bins[nonempty, 2]).abs().sum().item() / count
    result = {"samples": count, "accuracy": correct / count, "nll": sums["nll"] / count,
            "mean_confidence": sums["confidence"] / count, "entropy": sums["entropy"] / count,
            "ece_15_bins": ece,
            "correct_confidence": sums["correct_confidence"] / correct if correct else None,
            "incorrect_confidence": sums["incorrect_confidence"] / (count - correct) if count > correct else None}
    if extended:
        from .calibration_metrics import extended_prediction_metrics
        result["extended"] = extended_prediction_metrics(
            torch.cat(diagnostic_labels).cpu().numpy(),
            torch.cat(diagnostic_probabilities).cpu().numpy())
    return result


@torch.inference_mode()
def benchmark(model: VisionModel, image_size: int, device: torch.device, config: BenchmarkConfig,
              *, precision="float32", channels_last=False) -> dict:
    """Measure model-only batch-one latency; count Conv2d/Linear MACs explicitly."""
    previous_mode = model.training
    model.eval()
    images = torch.zeros(1, 3, image_size, image_size, device=device)
    if device.type == "cuda" and channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    macs = 0

    def count_ops(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    hooks = [m.register_forward_hook(count_ops) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with autocast_context(device, precision):
            model(images)
    finally:
        for hook in hooks:
            hook.remove()
    try:
        for _ in range(config.warmup):
            with autocast_context(device, precision):
                model(images)
        synchronize(device)
        timings = []
        for _ in range(config.iterations):
            start = time.perf_counter()
            with autocast_context(device, precision):
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
            "precision": resolve_precision(precision, device),
            "channels_last": bool(device.type == "cuda" and channels_last),
            "latency_scope": "model forward only; excludes loading, preprocessing and host-to-device transfer"}


def check_budget(cost: dict, config: BenchmarkConfig) -> dict:
    violations = []
    for metric, maximum in (("parameters", config.max_parameters), ("estimated_flops", config.max_flops),
                            ("latency_median_ms", config.max_latency_ms)):
        if maximum is not None and cost[metric] > maximum:
            violations.append({"metric": metric, "actual": cost[metric], "limit": maximum})
    return {"passed": not violations, "violations": violations}
