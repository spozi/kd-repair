"""Validation feedback and loss strategies for AdaFocal / AdaDualFocal.

AdaFocal: https://arxiv.org/html/2211.11838 (Algorithm 1).
Dual focal: https://proceedings.mlr.press/v202/tao23a/tao23a.pdf (Eq. 3, §5.2).
The positive branch uses dual focal for AdaDualFocal; the negative branch
retains AdaFocal's inverse focal. KD composition is a separate experiment.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import SupervisedLossConfig


class AdaptiveFocalController(nn.Module):
    """Epoch-boundary feedback; buffers travel with the objective checkpoint.

    Validation bins use top-class confidence; training assignment uses p(true).
    Quantile boundaries approximate equal mass. Ties stay together in the lower
    interval, and empty bins retain gamma. No gradients through feedback.
    """

    def __init__(self, config: SupervisedLossConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("boundaries", torch.arange(1, config.bins).float() / config.bins)
        self.register_buffer("gamma", torch.ones(config.bins))
        self.register_buffer("last_epoch", torch.tensor(0, dtype=torch.long))

    def gamma_for(self, true_probability):
        indices = torch.bucketize(true_probability.detach().contiguous(), self.boundaries, right=False)
        return self.gamma[indices].to(true_probability.dtype)

    def snapshot(self) -> dict:
        return {"boundaries": self.boundaries.detach().cpu().tolist(),
                "gamma": self.gamma.detach().cpu().tolist(), "last_epoch": self.last_epoch.item()}

    @torch.no_grad()
    def observe(self, epoch: int, confidence: torch.Tensor, correct: torch.Tensor) -> dict:
        if epoch != self.last_epoch.item() + 1:
            raise ValueError("Adaptive focal observations must cover consecutive one-based epochs")
        confidence = confidence.detach().to(device="cpu", dtype=torch.float64)
        correct = correct.detach().to(device="cpu", dtype=torch.float64)
        if confidence.ndim != 1 or not confidence.numel() or correct.shape != confidence.shape:
            raise ValueError("Feedback requires matching nonempty confidence and correctness vectors")
        if not torch.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
            raise ValueError("Validation confidence must be finite and in [0, 1]")
        if not ((correct == 0) | (correct == 1)).all():
            raise ValueError("Validation correctness must be binary")
        before = self.snapshot()
        c = self.config
        quantiles = torch.arange(1, c.bins, dtype=torch.float64) / c.bins
        boundaries = torch.quantile(confidence, quantiles).to(self.boundaries.dtype)
        indices = torch.bucketize(confidence, boundaries.double(), right=False)
        counts = torch.bincount(indices, minlength=c.bins)
        mean_confidence = torch.bincount(indices, weights=confidence, minlength=c.bins) / counts.clamp_min(1)
        accuracy = torch.bincount(indices, weights=correct, minlength=c.bins) / counts.clamp_min(1)
        gap = mean_confidence - accuracy
        gamma = self.gamma.detach().cpu().double()
        focal = gamma >= 0
        # Log-space clipping avoids overflow for large update rates.
        exponent = gamma.abs().clamp_min(torch.finfo(torch.float64).tiny).log()
        exponent += torch.where(focal, c.update_rate * gap, -c.update_rate * gap)
        cap = torch.where(focal, math.log(c.gamma_max), math.log(-c.gamma_min))
        magnitude = torch.minimum(exponent, cap).exp()
        updated = torch.where(focal, magnitude, -magnitude)
        switched = magnitude < c.switch_threshold
        updated = torch.where(switched, torch.where(focal, -c.switch_threshold, c.switch_threshold), updated)
        updated = torch.where(counts > 0, updated, gamma)
        self.boundaries.copy_(boundaries.to(self.boundaries.device))
        self.gamma.copy_(updated.to(self.gamma))
        self.last_epoch.fill_(epoch)
        return {"observed_epoch": epoch, "used": before, "next": self.snapshot(),
                "counts": counts.tolist(), "confidence": mean_confidence.tolist(),
                "accuracy": accuracy.tolist(), "signed_error": gap.tolist(),
                "ece_quantile_bins": (gap.abs() * counts).sum().item() / confidence.numel()}


class AdaptiveFocalLoss(nn.Module):
    """Mean loss at T=1, with differentiable probability weighting.

    Dual probability is the largest STRICTLY BELOW p(true), or zero if none;
    it is not necessarily the runner-up predicted class on incorrect samples.
    """

    def __init__(self, config: SupervisedLossConfig):
        super().__init__()
        if config.method not in {"adafocal", "adadualfocal"}:
            raise ValueError("AdaptiveFocalLoss requires an adaptive method")
        self.controller = AdaptiveFocalController(config)
        self.dual = config.method == "adadualfocal"

    def forward(self, logits, target):
        if logits.ndim != 2 or logits.shape[1] < 2 or not logits.shape[0]:
            raise ValueError("Adaptive focal requires nonempty [batch, classes] logits")
        if target.shape != logits.shape[:1] or target.dtype != torch.long:
            raise ValueError("Targets must be a matching int64 vector")
        values = logits if logits.dtype == torch.float64 else logits.float()
        log_probabilities = F.log_softmax(values, dim=1)
        log_true = log_probabilities.gather(1, target[:, None]).squeeze(1)
        p_true = log_true.exp()
        gamma = self.controller.gamma_for(p_true)
        base = 1 - p_true
        if self.dual:
            probabilities = log_probabilities.exp()
            p_dual = probabilities.masked_fill(probabilities >= p_true[:, None], 0).max(1).values
            base = base + p_dual
        base = torch.where(gamma >= 0, base, 1 + p_true)
        # Fractional powers at saturated p=1 otherwise have infinite derivatives.
        weight = (gamma.abs() * base.clamp_min(torch.finfo(values.dtype).tiny).log()).exp()
        return -(weight * log_true).mean()
