"""Oracle upper bounds for post-hoc calibrator families.

A conditional calibration null (`calibration_metrics.conditional_calibration_null`)
answers "how small could this metric get if the model were already perfect?". This
module answers the complementary question: "how small could this metric get if we
were allowed to cheat?"

An oracle bound refits a calibrator family **on the split it is evaluated on**. The
resulting number is unachievable by any honest procedure, so it upper-bounds what
every member of that family could ever deliver in this setting. Comparing it against
the conditional null decides whether a family is worth pursuing at all: when the
oracle already falls inside the null band, no method in the family can produce a
difference the metric can resolve, and the experiment should not be run.

Oracle values are planning diagnostics. They are never results, and every record
returned here carries an `interpretation` field saying so.

Fitting minimises negative log-likelihood by default. `criterion="ece"` fits the
reported calibration error directly, which gives the tighter bound for ECE claims;
the NLL bound is retained because it is the standard fitting objective and because a
family whose NLL oracle already sits below the null is exhausted a fortiori.
"""

from __future__ import annotations

import numpy as np
import torch

from .calibration_metrics import extended_prediction_metrics

FAMILIES = ("temperature", "per_class_temperature", "vector", "matrix", "dirichlet")
PREDICTION_PRESERVING = ("temperature", "per_class_temperature")


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _ece(probabilities, labels, bins):
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(np.float64)
    index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    return float(np.abs(np.bincount(index, weights=confidence - correct,
                                    minlength=bins)).sum() / len(labels))


def _features(logits, family):
    if family == "dirichlet":
        return np.log(np.maximum(_softmax(logits), 1e-12))
    return logits


def _apply(features, parameters, family, predictions=None):
    x = torch.as_tensor(features, dtype=torch.float64)
    if family == "temperature":
        return x / torch.exp(parameters[0])
    if family == "per_class_temperature":
        # A row scaled by a scalar keeps its argmax, so predictions are preserved.
        return x / torch.exp(parameters[0])[predictions][:, None]
    if family == "vector":
        weight, bias = parameters
        return x * torch.exp(weight) + bias
    weight, bias = parameters
    return x @ weight.T + bias


def _initial(family, classes):
    zero = lambda n: torch.zeros(n, dtype=torch.float64, requires_grad=True)
    if family == "temperature":
        return [zero(1)]
    if family == "per_class_temperature":
        return [zero(classes)]
    if family == "vector":
        return [zero(classes), zero(classes)]
    return [torch.eye(classes, dtype=torch.float64).requires_grad_(True), zero(classes)]


def fit_family(logits, labels, family, *, criterion="nll", bins=15,
               weights=None, penalty=0.0, steps=400, ece_grid=None):
    """Fit one calibrator family. Returns the fitted torch parameters.

    `weights` reweights the fitting objective per example, which matters whenever
    the fitting split is class-imbalanced: an unweighted objective on a skewed
    split is dominated by frequent classes and yields a badly biased fit.

    `penalty` applies ODIR-style regularisation (Kull et al.) to the off-diagonal
    weights and the bias, which the richer families need on small fitting splits.
    Leave it at zero for oracle bounds: an unregularised fit is the most flexible
    member of the family and therefore the correct upper bound.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown calibrator family {family!r}; choose from {FAMILIES}")
    if criterion not in ("nll", "ece"):
        raise ValueError("criterion must be 'nll' or 'ece'")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or labels.shape != (len(logits),) or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [samples, classes >= 2] logits and matching labels")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    classes = logits.shape[1]
    features = _features(logits, family)
    predictions = logits.argmax(1)

    if criterion == "ece":
        # ECE is piecewise constant in the parameters, so gradient descent cannot
        # be used. Only the one- and K-parameter scaling families admit a direct
        # grid search; richer families keep the NLL bound, which is looser.
        grid = np.arange(1, 501) / 100 if ece_grid is None else np.asarray(ece_grid)
        if family == "temperature":
            best = min(grid, key=lambda t: _ece(_softmax(logits / t), labels, bins))
            return [torch.tensor([np.log(float(best))], dtype=torch.float64)]
        if family == "per_class_temperature":
            temperatures = np.ones(classes)
            for k in range(classes):
                mask = predictions == k
                if not mask.any():
                    continue
                temperatures[k] = min(grid, key=lambda t: _ece(_softmax(logits[mask] / t),
                                                               labels[mask], bins))
            return [torch.tensor(np.log(temperatures), dtype=torch.float64)]
        raise ValueError(f"criterion='ece' is only defined for {PREDICTION_PRESERVING}")

    parameters = _initial(family, classes)
    target = torch.as_tensor(labels, dtype=torch.long)
    sample_weight = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(parameters, max_iter=steps, line_search_fn="strong_wolfe")

    off_diagonal = ~torch.eye(classes, dtype=torch.bool)

    def regulariser():
        if not penalty:
            return 0.0
        if family in ("vector", "matrix", "dirichlet"):
            weight, bias = parameters
            spread = (weight[off_diagonal] ** 2).mean() if weight.ndim == 2 else 0.0
            return penalty * (spread + (bias ** 2).mean())
        return penalty * (parameters[0] ** 2).mean()

    def closure():
        optimizer.zero_grad()
        scaled = _apply(features, parameters, family, predictions)
        losses = torch.nn.functional.cross_entropy(scaled, target, reduction="none")
        loss = losses.mean() if sample_weight is None else \
            (losses * sample_weight).sum() / sample_weight.sum()
        loss = loss + regulariser()
        loss.backward()
        return loss
    optimizer.step(closure)
    return [p.detach() for p in parameters]


def apply_family(logits, parameters, family):
    """Map logits through a fitted family, returning probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    scaled = _apply(_features(logits, family), parameters, family, logits.argmax(1))
    return torch.softmax(scaled, 1).numpy()


def oracle_bound(labels, logits, family, classes=None, *, criterion="nll", bins=15):
    """Upper-bound a family by fitting it on the split it is scored on.

    NOT A RESULT. The returned metrics are unachievable; they exist to be compared
    against a conditional null before committing compute to a new method.
    """
    labels = np.asarray(labels)
    parameters = fit_family(logits, labels, family, criterion=criterion, bins=bins)
    probabilities = apply_family(logits, parameters, family)
    quality = extended_prediction_metrics(labels, probabilities, classes,
                                          bins=bins, include_curve=False)
    return {"family": family, "criterion": criterion,
            "predictions_preserved": bool(np.array_equal(probabilities.argmax(1),
                                                         np.asarray(logits).argmax(1))),
            "metrics": quality,
            "interpretation": "Oracle fitted on the evaluation split. An unachievable "
                              "upper bound for this family, never a reported result. "
                              "Compare against a conditional calibration null: an oracle "
                              "inside the null band means the family cannot produce a "
                              "resolvable difference in this setting."}


def oracle_table(labels, logits, classes=None, *, families=FAMILIES, criterion="nll", bins=15):
    """Oracle bounds for several families, ordered as given."""
    return [oracle_bound(labels, logits, family, classes, criterion=criterion, bins=bins)
            for family in families]
