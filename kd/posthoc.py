"""Post-hoc maps on frozen logits; no model training or new dependencies.

Focal temperature scaling: Komisarenko & Kull, ECAI 2024, Proposition 4.
https://arxiv.org/html/2408.11598#S4.SS2
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


def _log_probabilities(logits, temperature):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite, nonempty [samples, classes] logits")
    z = (z - z.max(1, keepdims=True)) / temperature
    normalizer = np.logaddexp.reduce(z, axis=1, keepdims=True)
    log_q = z - normalizer
    # Complement log-probability without subtracting a rounded probability of 1.
    prefix = np.logaddexp.accumulate(z, axis=1)
    suffix = np.logaddexp.accumulate(z[:, ::-1], axis=1)[:, ::-1]
    left = np.concatenate((np.full((len(z), 1), -np.inf), prefix[:, :-1]), axis=1)
    right = np.concatenate((suffix[:, 1:], np.full((len(z), 1), -np.inf)), axis=1)
    return log_q, np.logaddexp(left, right) - normalizer


def _focal_map(log_q, log_complement, gamma):
    if gamma == 0:
        return log_q
    # h(q)=q/[(1-q)^gamma * (1-gamma*q*log(q)/(1-q))], then normalize h.
    # -q*log(q)/(1-q) tends to 1 as q tends to 1.
    numerator = -np.exp(log_q) * log_q
    denominator = -np.expm1(log_q)
    ratio = np.divide(numerator, denominator, out=np.ones_like(log_q), where=denominator > 0)
    log_h = log_q - gamma * log_complement - np.log1p(gamma * ratio)
    return log_h - np.logaddexp.reduce(log_h, axis=1, keepdims=True)


@dataclass(frozen=True)
class LogitCalibrator:
    temperature: float = 1.0
    gamma: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if not math.isfinite(self.gamma) or self.gamma < -.5 or self.gamma > 5:
            raise ValueError("Focal gamma must be in the supported [-0.5, 5] range")

    def log_probabilities(self, logits):
        return _focal_map(*_log_probabilities(logits, self.temperature), self.gamma)

    def predict(self, logits):
        return np.exp(self.log_probabilities(logits))

    def to_dict(self):
        return asdict(self)


def fit_calibrators(logits, labels, *, temperatures=None, gammas=None):
    """Fixed finite grid; select NLL and 15-bin ECE variants on fitting data only.

    gamma=0 nests ordinary temperature scaling within the focal family.
    No metric-specific refitting on test data. Ties preserve the first candidate.
    """
    temperatures = np.arange(1, 501) / 100 if temperatures is None else np.asarray(temperatures)
    gammas = (0., -.5, -.25, .05, .25, .37, .5, .75, 1., 5.) if gammas is None else tuple(gammas)
    if not len(temperatures) or 0 not in gammas:
        raise ValueError("Grid must contain temperatures and gamma=0")
    for t in temperatures:
        for g in gammas:
            LogitCalibrator(float(t), float(g))
    z = np.asarray(logits, dtype=np.float64)
    _log_probabilities(z, 1.)
    y = np.asarray(labels)
    if y.shape != (len(z),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError("Labels must be integer class indices matching the logits")
    correct = z.argmax(1) == y
    row = np.arange(len(z))
    winners = {}
    scores = []
    for temperature in temperatures:
        log_q, log_complement = _log_probabilities(z, float(temperature))
        for gamma in gammas:
            lp = _focal_map(log_q, log_complement, gamma)
            confidence = np.exp(lp.max(1))
            bins = np.minimum((confidence * 15).astype(int), 14)
            metrics = {"nll":float(-lp[row, y].mean()),
                       "ece":float(np.abs(np.bincount(bins, weights=confidence-correct, minlength=15)).sum()/len(z))}
            record = {"temperature":float(temperature), "gamma":float(gamma), **metrics}
            scores.append(record)
            for family in (("temperature", "focal_temperature") if gamma == 0 else ("focal_temperature",)):
                for criterion in ("nll", "ece"):
                    key = family + "_" + criterion
                    if key not in winners or metrics[criterion] < winners[key]["fit_score"]:
                        winners[key] = {"parameters":{"temperature":float(temperature), "gamma":float(gamma)},
                                        "fit_criterion":criterion, "fit_score":metrics[criterion], "fit_metrics":metrics}
    return winners, scores


@torch.inference_mode()
def collect_logits(model, loader, device):
    previous = model.training
    model.eval()
    logits, labels = [], []
    try:
        for images, targets in loader:
            logits.append(model(images.to(device)).logits.detach().cpu().numpy())
            labels.append(targets.numpy())
    finally:
        model.train(previous)
    if not logits:
        raise ValueError("Cannot collect logits from an empty dataset")
    return np.concatenate(labels), np.concatenate(logits)
