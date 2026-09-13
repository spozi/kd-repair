"""Independent loss strategies and a composable supervised/distillation objective.

DKD follows Zhao et al., CVPR 2022 (https://arxiv.org/abs/2203.08679).
The target/rest distributions are computed in log space to avoid log(0).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DistillationConfig
from .calibration import MMCELoss
from .cpc import PairwiseCalibrationLoss
from .models import ModelOutput


def _check_logits(student: Tensor, teacher: Tensor, target: Tensor) -> None:
    if student.ndim != 2 or student.shape != teacher.shape or student.shape[1] < 2:
        raise ValueError("KD requires matching [batch, classes] logits with at least two classes")
    if target.shape != student.shape[:1] or target.dtype != torch.long or target.numel() == 0:
        raise ValueError("Targets must be a nonempty int64 vector matching the batch")


class LogitsKD(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, student: Tensor, teacher: Tensor, target: Tensor) -> Tensor:
        _check_logits(student, teacher, target)
        t = self.temperature
        return F.kl_div(F.log_softmax(student.float() / t, dim=1),
                        F.softmax(teacher.detach().float() / t, dim=1),
                        reduction="batchmean") * t**2


class DecoupledKD(nn.Module):
    def __init__(self, temperature: float, alpha: float = 1.0, beta: float = 8.0):
        super().__init__()
        self.temperature, self.alpha, self.beta = temperature, alpha, beta

    def forward(self, student: Tensor, teacher: Tensor, target: Tensor) -> Tensor:
        _check_logits(student, teacher, target)
        s = student.float() / self.temperature
        t = teacher.detach().float() / self.temperature
        target_indices = target[:, None]
        mask = torch.zeros_like(s, dtype=torch.bool).scatter_(1, target_indices, True)

        def split(logits: Tensor) -> tuple[Tensor, Tensor]:
            # Remove target entries entirely for NCKD: no finite mask constant
            # and no 0 * -inf term in KL, even for extremely confident teachers.
            others = logits[~mask].reshape(logits.shape[0], logits.shape[1] - 1)
            binary_logits = torch.cat((logits.gather(1, target_indices),
                                       others.logsumexp(dim=1, keepdim=True)), dim=1)
            return F.log_softmax(binary_logits, dim=1), F.log_softmax(others, dim=1)

        s_binary, s_others = split(s)
        t_binary, t_others = split(t)
        target_loss = F.kl_div(s_binary, t_binary.exp(), reduction="batchmean")
        other_loss = F.kl_div(s_others, t_others.exp(), reduction="batchmean")
        return (self.alpha * target_loss + self.beta * other_loss) * self.temperature**2


class StageFeatureLoss(nn.Module):
    """Explicit semantic stage mapping with trainable channel/spatial alignment.

    This is projected stage matching, not a reproduction of ReviewKD. Projections
    exist before optimizer creation and are used only during training.
    """

    def __init__(self, student_channels: dict[str, int], teacher_channels: dict[str, int],
                 pairs: tuple[tuple[str, str], ...]):
        super().__init__()
        if not pairs:
            raise ValueError("At least one feature stage pair is required")
        self.pairs = pairs
        self.projections = nn.ModuleList()
        for student_stage, teacher_stage in pairs:
            if student_stage not in student_channels or teacher_stage not in teacher_channels:
                raise ValueError(f"Unknown feature pair: {student_stage} -> {teacher_stage}")
            self.projections.append(nn.Conv2d(student_channels[student_stage],
                                              teacher_channels[teacher_stage], 1, bias=False))

    def forward(self, student: dict[str, Tensor], teacher: dict[str, Tensor]) -> Tensor:
        losses = []
        for (s_name, t_name), projection in zip(self.pairs, self.projections):
            s, t = projection(student[s_name]), teacher[t_name].detach()
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            # Normalize each spatial channel vector so width and magnitude do
            # not dominate which semantic stages are learned.
            losses.append((F.normalize(s.float(), dim=1) - F.normalize(t.float(), dim=1))
                          .square().sum(dim=1).mean())
        return torch.stack(losses).mean()


class DistillationObjective(nn.Module):
    def __init__(self, config: DistillationConfig, feature_loss: StageFeatureLoss | None = None,
                 calibration_loss: MMCELoss | None = None, supervised_loss: nn.Module | None = None,
                 *, cpc_loss: PairwiseCalibrationLoss | None = None):
        super().__init__()
        self.config = config
        self.feature_loss = feature_loss
        self.calibration_loss = calibration_loss
        self.calibration_weight = 0.0
        self.supervised_loss = supervised_loss
        self.cpc_loss = cpc_loss
        if cpc_loss is not None and (config.method != "kd" or calibration_loss is not None
                                     or supervised_loss is not None or feature_loss is not None):
            raise ValueError("CPC requires standard KD without additional loss interventions")
        self.response_loss = {"supervised": lambda: None,
                              "kd": lambda: LogitsKD(config.temperature),
                              "dkd": lambda: DecoupledKD(config.temperature, config.alpha, config.beta)}[config.method]()

    def forward(self, student: ModelOutput, teacher: ModelOutput | None,
                target: Tensor, epoch: int) -> dict[str, Tensor]:
        ce = F.cross_entropy(student.logits, target)
        supervised = ce if self.supervised_loss is None else self.supervised_loss(student.logits, target)
        response = ce.new_zeros(())
        features = ce.new_zeros(())
        calibration = ce.new_zeros(())
        if self.calibration_loss is not None and self.calibration_weight > 0:
            calibration = self.calibration_loss(student.logits, target)
        if self.response_loss is None:
            return {"total": supervised + self.calibration_weight * calibration, "ce": ce,
                    "supervised": supervised, "response": response,
                    "features": features, "calibration": calibration}
        if teacher is None:
            raise ValueError("Distillation requires teacher output")
        response = self.response_loss(student.logits, teacher.logits, target)
        if self.feature_loss is not None:
            features = self.feature_loss(student.features, teacher.features)
        d = self.config
        warmup = min((epoch + 1) / d.warmup_epochs, 1.0) if d.warmup_epochs else 1.0
        # Classical KD uses a convex CE/KD mixture; DKD keeps full-strength CE.
        ce_weight = 1.0 - d.weight if d.method == "kd" else 1.0
        total = ce_weight * supervised + warmup * (d.weight * response + d.feature_weight * features)
        total = total + self.calibration_weight * calibration
        result = {"total": total, "ce": ce, "supervised": supervised, "response": response,
                  "features": features, "calibration": calibration}
        if self.cpc_loss is not None:
            components = self.cpc_loss.components(student.logits, target)
            weighted = (self.cpc_loss.discrimination_weight * components["discrimination"]
                        + self.cpc_loss.exclusion_weight * components["exclusion"])
            result.update(cpc_discrimination=components["discrimination"],
                          cpc_exclusion=components["exclusion"], cpc_weighted=weighted)
            result["total"] = total + self.cpc_loss.ramp(epoch) * weighted
        return result
