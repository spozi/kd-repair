"""A trainable calibration loss and an epoch-boundary feedback controller.

Unweighted MMCE, Eq. 7/8 of Kumar et al., ICML 2018:
https://proceedings.mlr.press/v80/kumar18a.html
This does not implement the paper's reweighted correct/incorrect variant.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .config import CalibrationConfig


class MMCELoss(nn.Module):
    """Smoothed norm of the biased empirical kernel calibration embedding.

    Correctness is discrete; gradients flow through confidence and the kernel.
    Includes diagonal pairs, costs O(batch_size**2), and uses raw T=1 outputs.
    sqrt(x + epsilon) - sqrt(epsilon) avoids an infinite derivative at zero.
    """

    def __init__(self, bandwidth: float = 0.4, epsilon: float = 1e-12):
        super().__init__()
        if not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("MMCE bandwidth and epsilon must be finite and positive")
        self.bandwidth, self.epsilon = bandwidth, epsilon

    def forward(self, logits, labels):
        if logits.ndim != 2 or labels.shape != logits.shape[:1] or labels.numel() == 0:
            raise ValueError("MMCE requires nonempty [batch, classes] logits and matching labels")
        # Retain float64 for numerical gradient checks; promote low precision.
        values = logits if logits.dtype == torch.float64 else logits.float()
        confidence, predicted = values.softmax(1).max(1)
        residual = confidence - predicted.eq(labels).to(values.dtype)
        kernel = torch.exp(-(confidence[:, None] - confidence[None, :]).abs() / self.bandwidth)
        squared = (residual[:, None] * residual[None, :] * kernel).mean().clamp_min(0)
        return (squared + self.epsilon).sqrt() - math.sqrt(self.epsilon)


@dataclass
class ControllerState:
    consecutive: int = 0
    activated_after_epoch: int | None = None
    last_observed_epoch: int = 0


class CalibrationController:
    """One-way activation with hysteresis; decisions affect the next epoch only."""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.state = ControllerState()

    def weight_for_epoch(self, epoch: int) -> float:
        """epoch is one-based; a decision at k cannot affect training epoch k."""
        activation = self.state.activated_after_epoch
        if not self.config.enabled or activation is None or epoch <= activation:
            return 0.0
        return self.config.max_weight * min((epoch - activation) / self.config.ramp_epochs, 1.0)

    def observe(self, epoch: int, metrics: dict) -> dict:
        if epoch <= self.state.last_observed_epoch:
            raise ValueError("Controller observations must advance by epoch")
        ece = metrics["ece_15_bins"]
        gap = metrics["mean_confidence"] - metrics["accuracy"]
        if not math.isfinite(ece) or not math.isfinite(gap):
            raise ValueError("Controller requires finite validation metrics")
        eligible = self.config.enabled and epoch >= self.config.monitor_start_epoch
        breached = eligible and ece > self.config.ece_threshold and gap > self.config.overconfidence_threshold
        self.state.last_observed_epoch = epoch
        if self.state.activated_after_epoch is None:
            self.state.consecutive = self.state.consecutive + 1 if breached else 0
            if self.state.consecutive >= self.config.patience:
                self.state.activated_after_epoch = epoch
        return {"observed_epoch": epoch, "ece": ece, "confidence_minus_accuracy": gap,
                "threshold_breached": breached, **self.state_dict(),
                "next_epoch_weight": self.weight_for_epoch(epoch + 1)}

    def state_dict(self) -> dict:
        return asdict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = ControllerState(**state)
