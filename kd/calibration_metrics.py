"""Calibration and selective-risk diagnostics for fixed probability matrices.

All metrics are fractions, including AURC. Confidence ties stay together in
equal-mass bins and use the expected error under uniform ordering for ranking.
Simulated calibration references condition on the supplied probabilities; they
are neither irreducible noise floors nor confidence intervals for true ECE.
"""

from __future__ import annotations

import numpy as np


def _validate(labels, probabilities, classes=None):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if (probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] < 2
            or labels.shape != (len(probabilities),)
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Expected nonempty N-by-C probabilities and N integer labels, C >= 2")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-7, rtol=1e-5)):
        raise ValueError("Each row must be a finite probability distribution")
    if (labels < 0).any() or (labels >= probabilities.shape[1]).any():
        raise ValueError("Labels are outside the class vocabulary")
    if classes is not None and len(classes) != probabilities.shape[1]:
        raise ValueError("Class names must match the probability columns")
    return labels.astype(np.int64, copy=False), probabilities


def _mass_ids(group_counts, bins):
    """Choose distinct tie boundaries nearest k*N/bins; ties choose lower cut.

    Counts refer to confidence groups in descending order. Empty bootstrap
    groups are removed before choosing cuts. This exactly matches recomputing
    the same equal-mass convention on the explicitly resampled observations.
    """
    active = np.flatnonzero(group_counts)
    if len(active) <= 1:
        return np.zeros(len(group_counts), dtype=np.int64), 1
    cumulative = np.cumsum(group_counts[active])
    possible = cumulative[:-1]
    targets = np.arange(1, bins) * cumulative[-1] / bins
    upper = np.minimum(np.searchsorted(possible, targets), len(possible) - 1)
    lower = np.maximum(upper - 1, 0)
    cuts = np.where(np.abs(possible[lower] - targets) <= np.abs(possible[upper] - targets),
                    lower, upper)
    cuts = np.unique(active[cuts] + 1)
    return np.searchsorted(cuts, np.arange(len(group_counts)), side="right"), len(cuts) + 1


class _Prepared:
    """Probability-only indexes shared by bootstrap and categorical simulation."""

    def __init__(self, probabilities, bins):
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ValueError("bins must be a positive integer")
        self.p = probabilities
        self.n, self.c = probabilities.shape
        self.bins = bins
        self.prediction = probabilities.argmax(1)
        self.confidence = probabilities.max(1)
        self.width_ids = np.minimum((self.confidence * bins).astype(np.int64), bins - 1)
        negative_confidence, self.rank_ids = np.unique(-self.confidence, return_inverse=True)
        self.rank_confidence = -negative_confidence
        self.groups = len(negative_confidence)
        self.harmonic = np.concatenate(([0.0], np.cumsum(1 / np.arange(1, self.n + 1))))
        self.class_ids = (np.minimum((probabilities * bins).astype(np.int64), bins - 1)
                          + np.arange(self.c)[None, :] * bins).ravel()
        self.predicted_ids = self.width_ids + self.prediction * bins


class _Observed:
    def __init__(self, prepared, labels):
        self.prepared, self.labels = prepared, labels
        self.correct = (prepared.prediction == labels).astype(np.float64)
        self.residual = prepared.confidence - self.correct
        self.class_residual = prepared.p.copy()
        self.class_residual[np.arange(prepared.n), labels] -= 1
        self.nll = -np.log(np.maximum(prepared.p[np.arange(prepared.n), labels], 1e-300))
        self.brier = np.square(self.class_residual).sum(1)

    def parts(self, weights=None):
        p = self.prepared
        if weights is None:
            weights = np.ones(p.n, dtype=np.int64)
        count = int(weights.sum())
        width_count = np.bincount(p.width_ids, weights=weights, minlength=p.bins)
        width_gap = np.bincount(p.width_ids, weights=weights * self.residual, minlength=p.bins)
        rank_count = np.bincount(p.rank_ids, weights=weights, minlength=p.groups).astype(np.int64)
        rank_error = np.bincount(p.rank_ids, weights=weights * (1 - self.correct), minlength=p.groups)
        mass_ids, mass_count = _mass_ids(rank_count, p.bins)
        mass_gap = np.bincount(mass_ids[p.rank_ids], weights=weights * self.residual,
                               minlength=mass_count)
        class_gap = np.bincount(p.class_ids, weights=(weights[:, None] * self.class_residual).ravel(),
                                minlength=p.c * p.bins).reshape(p.c, p.bins)
        class_ece = np.abs(class_gap).sum(1) / count
        predicted_count = np.bincount(p.prediction, weights=weights, minlength=p.c)
        predicted_correct = np.bincount(p.prediction, weights=weights * self.correct, minlength=p.c)
        predicted_confidence = np.bincount(p.prediction, weights=weights * p.confidence, minlength=p.c)
        predicted_gap = np.bincount(p.predicted_ids, weights=weights * self.residual,
                                    minlength=p.c * p.bins).reshape(p.c, p.bins)
        values = {
            "accuracy": float(np.dot(weights, self.correct) / count),
            "nll": float(np.dot(weights, self.nll) / count),
            "brier_score": float(np.dot(weights, self.brier) / count),
            f"ece_{p.bins}_bins": float(np.abs(width_gap).sum() / count),
            f"ece_equal_mass_{p.bins}_bins": float(np.abs(mass_gap).sum() / count),
            f"ece_l2_{p.bins}_bins": float(np.sqrt(np.divide(width_gap**2, width_count,
                out=np.zeros(p.bins), where=width_count > 0).sum() / count)),
            f"classwise_ece_{p.bins}_bins": float(class_ece.mean()),
        }
        return (values, rank_count, rank_error, mass_ids, mass_count, class_ece,
                predicted_count, predicted_correct, predicted_confidence, predicted_gap)

    def statistics(self, weights=None):
        parts = self.parts(weights)
        values, rank_count, rank_error = parts[:3]
        values.update(_ranking(rank_count, rank_error, self.prepared.harmonic))
        return values


def _ranking(counts, errors, harmonic, include_curve=False):
    present = counts > 0
    counts, errors = counts[present], errors[present]
    ends = np.cumsum(counts)
    starts = ends - counts
    prior_errors = np.cumsum(errors) - errors
    rates = errors / counts
    offsets = prior_errors - starts * rates
    total = int(ends[-1])
    aurc = np.sum(counts * rates + offsets * (harmonic[ends] - harmonic[starts])) / total
    values = {"aurc": float(aurc)}
    for coverage in (80, 90):
        retained = max(1, int(np.ceil(total * coverage / 100)))
        group = np.searchsorted(ends, retained, side="left")
        expected_errors = prior_errors[group] + (retained - starts[group]) * rates[group]
        values[f"selective_accuracy_{coverage}"] = float(1 - expected_errors / retained)
    if include_curve:
        retained = np.arange(1, total + 1)
        risk = np.repeat(rates, counts) + np.repeat(offsets, counts) / retained
        values["risk_coverage_curve"] = {
            "coverage": (retained / total).tolist(), "risk": risk.tolist(),
            "tie_convention": "Expected risk under uniform random ordering within each exact-confidence tie",
        }
    return values


def _bin_report(confidence, correct, ids, number):
    rows = []
    for group in range(number):
        mask = ids == group
        count = int(mask.sum())
        rows.append({"bin": group, "count": count,
                     "minimum_confidence": float(confidence[mask].min()) if count else None,
                     "maximum_confidence": float(confidence[mask].max()) if count else None,
                     "mean_confidence": float(confidence[mask].mean()) if count else None,
                     "accuracy": float(correct[mask].mean()) if count else None,
                     "signed_gap": float((confidence[mask] - correct[mask]).mean()) if count else None})
    return rows


def extended_prediction_metrics(labels, probabilities, classes=None, *, bins=15, include_curve=True):
    """Return opt-in metrics and diagnostics; merge with existing report keys.

    Class-wise ECE is the equal-weight mean of marginal one-vs-rest class ECEs.
    Predicted-class diagnostics condition on the model's argmax and are distinct.
    Empty predicted classes have ``None`` values. Equal-mass groups use the
    nearest available tie boundary to each target quantile, choosing the lower
    count on equal distances; duplicate boundaries are removed. Selective
    accuracy retains ceil(coverage*N) examples, averaging boundary ties.
    """
    labels, probabilities = _validate(labels, probabilities, classes)
    prepared = _Prepared(probabilities, bins)
    observed = _Observed(prepared, labels)
    (values, rank_count, rank_error, mass_ids, mass_count, class_ece,
     predicted_count, predicted_correct, predicted_confidence, predicted_gap) = observed.parts()
    values.update(_ranking(rank_count, rank_error, prepared.harmonic, include_curve))
    names = list(classes) if classes is not None else [str(i) for i in range(prepared.c)]
    values["classwise_calibration"] = [
        {"class": name, "target_support": int((labels == i).sum()), "samples": prepared.n,
         "mean_probability": float(probabilities[:, i].mean()),
         "target_frequency": float((labels == i).mean()), "ece": float(class_ece[i])}
        for i, name in enumerate(names)]
    values["predicted_class_calibration"] = []
    for i, name in enumerate(names):
        count = int(predicted_count[i])
        values["predicted_class_calibration"].append({
            "class": name, "support": count,
            "mean_confidence": float(predicted_confidence[i] / count) if count else None,
            "accuracy": float(predicted_correct[i] / count) if count else None,
            "signed_gap": float((predicted_confidence[i] - predicted_correct[i]) / count) if count else None,
            "ece": float(np.abs(predicted_gap[i]).sum() / count) if count else None,
        })
    values["equal_width_bins"] = _bin_report(prepared.confidence, observed.correct, prepared.width_ids, bins)
    values["equal_mass_bins"] = _bin_report(prepared.confidence, observed.correct, mass_ids[prepared.rank_ids], mass_count)
    values["equal_mass_effective_bins"] = mass_count
    values["metric_conventions"] = {
        "units": "Fractions; smaller ECE, NLL, Brier and AURC are better",
        "top_label_ece": f"{bins} equal-width bins; confidence=1 belongs to the final bin",
        "l2_ece": "Square root of sample-weighted squared bin confidence-accuracy gaps",
        "classwise_ece": "Unweighted mean of marginal one-vs-rest ECE across all probability columns",
        "equal_mass_ece": "Nearest tie-preserving cumulative-count cuts; lower cut wins ties; duplicate cuts removed",
        "ranking": "Expected risk under uniform ordering within exact confidence ties",
        "selective_accuracy": "Retain ceil(coverage*N) predictions; expected correct count at a tied boundary",
        "empty_groups": "Zero-support predicted classes and empty bins have null diagnostic values",
    }
    return values


def _reference(observed, samples):
    samples = np.asarray(samples, dtype=np.float64)
    if observed is None:
        return {"observed": None, "null_mean": None, "null_central_95": None, "observed_percentile": None}
    # Midrank empirical percentile handles discrete references without implying
    # a continuous p-value (which this descriptive diagnostic is not).
    percentile = 100 * (np.count_nonzero(samples < observed) + .5 * np.count_nonzero(samples == observed)) / len(samples)
    return {"observed": float(observed), "null_mean": float(samples.mean()),
            "null_central_95": np.quantile(samples, [.025, .975]).tolist(),
            "observed_percentile": float(percentile)}


def conditional_calibration_null(labels, probabilities, classes=None, *, seed=2026, repetitions=2000, bins=15):
    """Simulate coherent categorical labels conditional on fixed probabilities.

    Each replicate draws one label per example, reused for every class and group.
    Only calibration diagnostics are simulated; ranking is not a calibration null.
    """
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    labels, probabilities = _validate(labels, probabilities, classes)
    prepared = _Prepared(probabilities, bins)
    observed = extended_prediction_metrics(labels, probabilities, classes, bins=bins, include_curve=False)
    keys = [f"ece_{bins}_bins", f"ece_equal_mass_{bins}_bins", f"ece_l2_{bins}_bins", f"classwise_ece_{bins}_bins"]
    draws = {key: np.empty(repetitions) for key in keys}
    class_draws = np.empty((repetitions, prepared.c))
    class_support_draws = np.empty((repetitions, prepared.c), dtype=np.int64)
    predicted_draws = {key: np.zeros((repetitions, prepared.c)) for key in ("accuracy", "signed_gap", "ece")}
    rng = np.random.default_rng(seed)
    cumulative = probabilities.cumsum(1)
    cumulative[:, -1] = 1
    for repetition in range(repetitions):
        sampled_labels = (rng.random((prepared.n, 1)) >= cumulative).sum(1)
        class_support_draws[repetition] = np.bincount(sampled_labels, minlength=prepared.c)
        parts = _Observed(prepared, sampled_labels).parts()
        for key in keys:
            draws[key][repetition] = parts[0][key]
        class_draws[repetition] = parts[5]
        count, correct, confidence, gap = parts[6:]
        predicted_draws["accuracy"][repetition] = np.divide(correct, count, out=np.zeros(prepared.c), where=count > 0)
        predicted_draws["signed_gap"][repetition] = np.divide(confidence - correct, count, out=np.zeros(prepared.c), where=count > 0)
        predicted_draws["ece"][repetition] = np.divide(np.abs(gap).sum(1), count, out=np.zeros(prepared.c), where=count > 0)
    return {
        "repetitions": repetitions, "seed": seed,
        "metrics": {key: _reference(observed[key], draws[key]) for key in keys},
        "classwise_calibration": [
            {"class": row["class"], "samples": prepared.n,
             "mean_probability": row["mean_probability"],
             "target_support": _reference(row["target_support"], class_support_draws[:, i]),
             "target_frequency": _reference(row["target_frequency"], class_support_draws[:, i] / prepared.n),
             "ece": _reference(row["ece"], class_draws[:, i])}
            for i, row in enumerate(observed["classwise_calibration"])],
        "predicted_class_calibration": [
            {"class": row["class"], "support": row["support"],
             "mean_confidence": row["mean_confidence"],
             **{key: _reference(row[key], predicted_draws[key][:, i]) for key in predicted_draws}}
            for i, row in enumerate(observed["predicted_class_calibration"])],
        "note": "Conditional perfect-calibration simulation with one categorical label per example. Central ranges are simulation references, not hard noise floors, true-ECE confidence intervals, or quantities to subtract from observed ECE. Percentiles are descriptive midranks, not adjusted hypothesis tests.",
    }


def paired_seed_comparison(labels, baseline_probabilities, candidate_probabilities, *, seed=2026,
                           repetitions=2000, bins=15, indices=None):
    """Paired test bootstrap of the mean difference across matched model seeds.

    Supply sequences of N-by-C matrices in identical image and seed order. Each
    replicate resamples the SAME N image indices across every seed pair. Optional
    ``indices`` is an iterable of one-dimensional N-index vectors, permitting
    coordination with another contrast. It must yield exactly ``repetitions``
    resamples. Confidence grouping, bin populations, quantile boundaries, ranks
    and retained counts are recalculated for each replicate via sample weights;
    this is exactly equivalent to explicitly duplicated resampled rows.
    """
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    baselines, candidates = list(baseline_probabilities), list(candidate_probabilities)
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("Matched nonempty baseline and candidate seed sequences are required")
    prepared_pairs, observed = [], []
    for baseline, candidate in zip(baselines, candidates):
        baseline_labels, baseline = _validate(labels, baseline)
        candidate_labels, candidate = _validate(labels, candidate)
        if baseline.shape != candidate.shape:
            raise ValueError("Paired probability matrices must have the same shape")
        b = _Observed(_Prepared(baseline, bins), baseline_labels)
        c = _Observed(_Prepared(candidate, bins), candidate_labels)
        baseline_values, candidate_values = b.statistics(), c.statistics()
        keys = list(baseline_values)
        observed.append([candidate_values[key] - baseline_values[key] for key in keys])
        prepared_pairs.append((b, c))
    observed = np.asarray(observed)
    samples = np.empty((repetitions, len(prepared_pairs), len(keys)))
    rng = np.random.default_rng(seed)
    index_iterator = iter(indices) if indices is not None else None
    count = len(labels)
    for repetition in range(repetitions):
        if index_iterator is None:
            selected = rng.integers(count, size=count)
        else:
            try:
                selected = np.asarray(next(index_iterator))
            except StopIteration as error:
                raise ValueError("indices yielded fewer resamples than repetitions") from error
            if (selected.shape != (count,) or not np.issubdtype(selected.dtype, np.integer)
                    or (selected < 0).any() or (selected >= count).any()):
                raise ValueError("Each paired resample must contain N valid integer image indices")
        weights = np.bincount(selected, minlength=count)
        for pair, (b, c) in enumerate(prepared_pairs):
            baseline_values, candidate_values = b.statistics(weights), c.statistics(weights)
            samples[repetition, pair] = [candidate_values[key] - baseline_values[key] for key in keys]
    if index_iterator is not None and next(index_iterator, None) is not None:
        raise ValueError("indices yielded more resamples than repetitions")

    def summarize(deltas, bootstrap):
        return {key: {"delta": float(deltas[i]), "paired_bootstrap_95": np.quantile(bootstrap[:, i], [.025, .975]).tolist()}
                for i, key in enumerate(keys)}

    return {
        "metrics": summarize(observed.mean(0), samples.mean(1)),
        "per_seed": [{"pair_index": i, "metrics": summarize(observed[i], samples[:, i])} for i in range(len(prepared_pairs))],
        "seed_delta_standard_deviation": {
            key: float(observed[:, i].std(ddof=1)) if len(prepared_pairs) > 1 else None
            for i, key in enumerate(keys)},
        "bootstrap_repetitions": repetitions, "seed": seed, "paired_seeds": len(prepared_pairs),
        "external_indices": indices is not None,
        "note": "Candidate minus control. The same test-image indices are used across every seed pair; rankings, tie expectations and equal-mass boundaries are recomputed. Intervals condition on these fitted models and measure test-example uncertainty, not uncertainty over retraining. Negative favors the candidate for losses, calibration errors and AURC; positive favors it for accuracy.",
    }


def paired_extended_comparison(labels, baseline_probabilities, candidate_probabilities, *, seed=2026,
                               repetitions=2000, bins=15, indices=None):
    """Single fitted model-pair wrapper around ``paired_seed_comparison``."""
    return paired_seed_comparison(labels, [baseline_probabilities], [candidate_probabilities],
                                  seed=seed, repetitions=repetitions, bins=bins, indices=indices)

