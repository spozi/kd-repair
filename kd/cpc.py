"""Normalized pairwise calibration constraints for raw student logits.

Binary discrimination and exclusion follow equations 15–16 of Cheng and
Vasconcelos, CVPR 2022, expressed with numerically stable softplus terms.
https://jiacheng-cheng.github.io/assets/papers/cvpr22.pdf
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseCalibrationLoss(nn.Module):
    """Average target discrimination and non-target pair exclusion.

Each example averages C-1 discrimination terms and (C-1)(C-2)/2
unordered exclusion pairs. Exclusion averages both pair directions and has
minimum log(2). Its empty binary-class case is a differentiable zero.

The pair-index buffer is deterministic, non-persistent derived state: it
moves with the module and is rebuilt on a class-count/device change. A fresh
module can therefore restore an objective checkpoint without buffer-shape
or class-count assumptions. This loss has no trainable parameters.
"""

    def __init__(self, discrimination_weight: float = 0.1,
                 exclusion_weight: float = 0.1, warmup_epochs: int = 0):
        super().__init__()
        if any(not math.isfinite(weight) or weight < 0
               for weight in (discrimination_weight, exclusion_weight)):
            raise ValueError("CPC weights must be finite and nonnegative")
        if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("CPC warmup_epochs must be a nonnegative integer")
        self.discrimination_weight = float(discrimination_weight)
        self.exclusion_weight = float(exclusion_weight)
        self.warmup_epochs = warmup_epochs
        self.register_buffer("_pair_indices", torch.empty((2, 0), dtype=torch.long),
                             persistent=False)
        self._cached_classes = 0

    def ramp(self, epoch: int) -> float:
        """One-based linear ramp matching the project's KD/DKD warmup convention.

        Zero epochs (the default) means full weight from the first training
        epoch, exactly reproducing the original always-on CPC behavior.
        """
        if not self.warmup_epochs:
            return 1.0
        return min((epoch + 1) / self.warmup_epochs, 1.0)

    def components(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """Return unweighted scalar discrimination/exclusion batch means."""
        if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2
                or not logits.is_floating_point()):
            raise ValueError("CPC requires nonempty floating [batch, classes>=2] logits")
        if labels.shape != logits.shape[:1] or labels.dtype != torch.long:
            raise ValueError("CPC requires matching one-dimensional int64 labels")
        if labels.device != logits.device:
            raise ValueError("CPC logits and labels must be on the same device")
        classes = logits.shape[1]
        if bool(((labels < 0) | (labels >= classes)).any()):
            raise ValueError("CPC labels must be valid class indices")
        # Preserve float64 for gradient checks and promote low-precision logits.
        values = logits if logits.dtype == torch.float64 else logits.float()
        target = values.gather(1, labels[:, None])
        target_mask = F.one_hot(labels, num_classes=classes).bool()
        discrimination = (F.softplus(values - target)
                          .masked_fill(target_mask, 0).sum(1) / (classes - 1)).mean()
        if classes == 2:
            exclusion = (values[:, :1] * 0).sum()
        else:
            if self._cached_classes != classes or self._pair_indices.device != values.device:
                # Build on CPU as triu_indices need not be implemented on every
                # accelerator. This happens once per class count/device.
                self._pair_indices = torch.triu_indices(classes, classes, offset=1).to(values.device)
                self._cached_classes = classes
            left, right = self._pair_indices.unbind(0)
            difference = values[:, left] - values[:, right]
            pair_terms = 0.5 * (F.softplus(difference) + F.softplus(-difference))
            target_pair = (left[None, :] == labels[:, None]) | (right[None, :] == labels[:, None])
            pair_count = (classes - 1) * (classes - 2) // 2
            exclusion = (pair_terms.masked_fill(target_pair, 0).sum(1) / pair_count).mean()
        return {"discrimination": discrimination, "exclusion": exclusion}

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        components = self.components(logits, labels)
        return (self.discrimination_weight * components["discrimination"]
                + self.exclusion_weight * components["exclusion"])
