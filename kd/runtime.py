"""Shared accelerator policy for training, evaluation, and repair studies."""

from __future__ import annotations

from contextlib import nullcontext
import re
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PRECISIONS = ("auto", "float32", "float16", "bfloat16")
CUDA_MODES = ("fast", "deterministic")
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?\Z")


def valid_device_spec(value: str) -> bool:
    return value in {"auto", "cpu", "mps"} or _CUDA_DEVICE.fullmatch(value) is not None


def resolve_device(requested: str) -> torch.device:
    """Resolve an accelerator and validate explicit CUDA device indices."""
    if not valid_device_spec(requested):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device.index} is unavailable; found {torch.cuda.device_count()} device(s)")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def cuda_requested(requested: str) -> bool:
    """Return whether a loader should prepare batches for CUDA transfer."""
    return requested.startswith("cuda") or requested == "auto" and torch.cuda.is_available()


def accelerator_workers(requested: str, configured: int) -> int:
    """Retain an explicit worker count or provide a conservative accelerator default."""
    return configured or (4 if requested == "auto" or requested.startswith("cuda") else 0)


def configure_accelerator(device: torch.device, cuda_mode: str = "fast") -> None:
    """Apply one explicit CUDA kernel policy before models are constructed."""
    if cuda_mode not in CUDA_MODES:
        raise ValueError(f"cuda_mode must be one of {CUDA_MODES}")
    if device.type != "cuda":
        return
    fast = cuda_mode == "fast"
    torch.backends.cudnn.benchmark = fast
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.allow_tf32 = fast
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.set_float32_matmul_precision("high" if fast else "highest")


def cuda_bf16_supported(device: torch.device) -> bool:
    """Check native BF16 tensor-core support on the selected CUDA device."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if requested == "auto":
        if device.type != "cuda":
            return "float32"
        return "bfloat16" if cuda_bf16_supported(device) else "float16"
    if requested != "float32" and device.type != "cuda":
        raise ValueError(f"{requested} mixed precision is supported only on CUDA")
    if requested == "bfloat16" and not cuda_bf16_supported(device):
        raise ValueError("This CUDA device does not support bfloat16")
    return requested


def autocast_context(device: torch.device, precision: str):
    precision = resolve_precision(precision, device)
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    resolved = resolve_precision(precision, device)
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and resolved == "float16")


def prepare_model(model: nn.Module, device: torch.device, channels_last: bool = True) -> nn.Module:
    model.to(device)
    if device.type == "cuda" and channels_last:
        model.to(memory_format=torch.channels_last)
    return model


def move_images(images: Tensor, device: torch.device, channels_last: bool = True) -> Tensor:
    kwargs = {"device": device, "non_blocking": device.type == "cuda"}
    if device.type == "cuda" and channels_last and images.ndim == 4:
        kwargs["memory_format"] = torch.channels_last
    return images.to(**kwargs)


def move_labels(labels: Tensor, device: torch.device) -> Tensor:
    return labels.to(device=device, non_blocking=device.type == "cuda")


def loader_performance_kwargs(requested_device: str, workers: int, *,
                              persistent_workers: bool = False,
                              prefetch_factor: int = 2) -> dict:
    """DataLoader options that are valid for both zero and multi-worker loaders."""
    result = {
        "pin_memory": cuda_requested(requested_device),
        "persistent_workers": bool(persistent_workers and workers > 0),
    }
    if workers > 0:
        result["prefetch_factor"] = prefetch_factor
    return result


def runtime_metadata(device: torch.device, precision: str, cuda_mode: str,
                     channels_last: bool) -> dict:
    resolved = resolve_precision(precision, device)
    result = {
        "device": str(device),
        "precision": resolved,
        "requested_precision": precision,
        "channels_last": bool(device.type == "cuda" and channels_last),
        "cuda_mode": cuda_mode if device.type == "cuda" else None,
        "non_blocking_transfers": device.type == "cuda",
        "fused_sgd": device.type == "cuda" and cuda_mode == "fast",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu_name=properties.name,
            compute_capability=list(torch.cuda.get_device_capability(device)),
            total_memory_bytes=properties.total_memory,
            tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        )
    return result


def cuda_preflight(requested_device: str = "cuda", precision: str = "auto", *,
                   batch_size: int = 128, iterations: int = 10) -> dict:
    """Exercise the CUDA data-transfer, autocast, backward, and optimizer path."""
    if batch_size < 1 or iterations < 1:
        raise ValueError("batch_size and iterations must be positive")
    device = resolve_device(requested_device)
    if device.type != "cuda":
        raise ValueError("CUDA preflight requires a cuda or cuda:N device")
    configure_accelerator(device, "fast")
    resolved = resolve_precision(precision, device)
    torch.cuda.reset_peak_memory_stats(device)
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    )
    prepare_model(model, device, channels_last=True).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, fused=True)
    scaler = make_grad_scaler(device, resolved)
    host_images = torch.randn(batch_size, 3, 32, 32, pin_memory=True)
    host_labels = torch.randint(10, (batch_size,), pin_memory=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_value = None
    output_dtype = None
    for _ in range(iterations):
        images = move_images(host_images, device, channels_last=True)
        labels = move_labels(host_labels, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, resolved):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CUDA preflight loss")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_value = loss.detach()
        output_dtype = str(logits.dtype)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "status": "ok",
        "runtime": runtime_metadata(device, resolved, "fast", True),
        "batch_size": batch_size,
        "iterations": iterations,
        "samples_per_second": batch_size * iterations / seconds,
        "final_loss": float(loss_value.cpu()),
        "output_dtype": output_dtype,
        "grad_scaler_enabled": scaler.is_enabled(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
