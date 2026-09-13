"""Paper §2-§5: calibrator families, conditional nulls, and oracle bounds.

Reproduces the balanced-CIFAR-10 tables showing that top-label and class-wise ECE
both sit at their estimators' floors, and that the best oracle for the whole
logit-affine family falls *below* the top-label floor.

Honest variants are fitted on the validation split and scored on test. Oracle rows
are fitted on test on purpose, are unachievable, and are never results.

Reads only saved logit archives; trains nothing and modifies nothing.

    python scripts/calibrator_families.py [--run runs/cifar10-posthoc/seed42]
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.calibration_metrics import conditional_calibration_null, extended_prediction_metrics
from kd.floors import FAMILIES, apply_family, fit_family, oracle_bound
from kd.posthoc import LogitCalibrator, fit_calibrators

TEMPERATURES = (np.arange(1, 501) / 100).tolist()
REPORTED = (("accuracy", 100, 2), ("ece_15_bins", 100, 3), ("ece_equal_mass_15_bins", 100, 3),
            ("ece_l2_15_bins", 100, 3), ("classwise_ece_15_bins", 100, 3),
            ("nll", 1, 5), ("brier_score", 1, 5))


def load(run: Path):
    with np.load(run / "validation_logits.npz", allow_pickle=True) as v:
        validation = (v["labels"].astype(int), v["logits"].astype(np.float64),
                      [str(c) for c in v["classes"]])
    with np.load(run / "test_logits.npz", allow_pickle=True) as t:
        test = (t["labels"].astype(int), t["logits"].astype(np.float64))
    return validation, test


def per_class_temperatures(logits, labels, criterion="nll"):
    """One temperature per predicted class; argmax is preserved by construction."""
    prediction = logits.argmax(1)
    temperatures = np.ones(logits.shape[1])
    for k in range(logits.shape[1]):
        mask = prediction == k
        if not mask.any():
            continue
        best, score = 1.0, np.inf
        for t in TEMPERATURES:
            probability = LogitCalibrator(temperature=float(t)).predict(logits[mask])
            if criterion == "nll":
                value = -np.log(np.maximum(
                    probability[np.arange(mask.sum()), labels[mask]], 1e-300)).mean()
            else:
                confidence = probability.max(1)
                correct = (probability.argmax(1) == labels[mask]).astype(float)
                index = np.minimum((confidence * 15).astype(int), 14)
                value = np.abs(np.bincount(index, weights=confidence - correct,
                                           minlength=15)).sum() / mask.sum()
            if value < score:
                best, score = float(t), value
        temperatures[k] = best
    return temperatures


PENALTIES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def cross_validated_penalty(logits, labels, family, classes, folds=5, seed=2026):
    """Choose ODIR strength by k-fold NLL inside the fitting split only."""
    order = np.random.default_rng(seed).permutation(len(labels))
    parts = np.array_split(order, folds)
    best, chosen = np.inf, PENALTIES[0]
    for penalty in PENALTIES:
        total = 0.0
        for i in range(folds):
            held = parts[i]
            rest = np.concatenate([parts[j] for j in range(folds) if j != i])
            fitted = fit_family(logits[rest], labels[rest], family, penalty=penalty)
            probability = apply_family(logits[held], fitted, family)
            total += extended_prediction_metrics(labels[held], probability, classes,
                                                 include_curve=False)["nll"] * len(held)
        if total < best:
            best, chosen = total, penalty
    return chosen


def row(name, labels, probabilities, classes):
    quality = extended_prediction_metrics(labels, probabilities, classes, include_curve=False)
    cells = "".join(f"{scale * quality[key]:>{9 if digits == 3 else 10}.{digits}f}"
                    + ("%" if scale == 100 else "")
                    for key, scale, digits in REPORTED)
    return name, quality, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/cifar10-posthoc/seed42"))
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    (v_labels, v_logits, classes), (t_labels, t_logits) = load(args.run)
    print(f"validation {len(v_labels)} images | test {len(t_labels)} images | "
          f"{len(classes)} classes\n")

    header = (f"{'variant':<30}{'acc':>10}{'ECE-w15':>10}{'ECE-m15':>10}"
              f"{'ECE-L2':>10}{'cwECE':>10}{'NLL':>10}{'Brier':>10}")

    # ---- honest variants: fitted on validation, scored on test ----
    variants = {"raw": t_logits}
    winners, _ = fit_calibrators(v_logits, v_labels, temperatures=TEMPERATURES, gammas=[0.0])
    for criterion in ("nll", "ece"):
        temperature = winners[f"temperature_{criterion}"]["parameters"]["temperature"]
        variants[f"global TS ({criterion} fit, T={temperature:.2f})"] = t_logits / temperature
        temperatures = per_class_temperatures(v_logits, v_labels, criterion)
        variants[f"per-class TS ({criterion} fit)"] = \
            t_logits / temperatures[t_logits.argmax(1)][:, None]

    print("FITTED ON VALIDATION, SCORED ON TEST")
    print(header)
    print("-" * len(header))
    probabilities = {}
    for name, scaled in variants.items():
        probabilities[name] = LogitCalibrator().predict(scaled)
        _, _, cells = row(name, t_labels, probabilities[name], classes)
        print(f"{name:<30}{cells}")
    for family in ("vector", "matrix", "dirichlet"):
        penalty = cross_validated_penalty(v_logits, v_labels, family, classes)
        parameters = fit_family(v_logits, v_labels, family, penalty=penalty)
        probability = apply_family(t_logits, parameters, family)
        probabilities[family] = probability
        name, _, cells = row(f"{family} scaling (lam={penalty:g})", t_labels, probability, classes)
        print(f"{name:<30}{cells}")

    # ---- conditional null for the reference variant ----
    reference = next(k for k in variants if k.startswith("global TS (nll"))
    print(f"\nCONDITIONAL NULL for '{reference}' ({args.repetitions} label simulations)")
    null = conditional_calibration_null(t_labels, probabilities[reference], classes,
                                        repetitions=args.repetitions)
    print(f"{'metric':<30}{'observed':>12}{'null mean':>12}{'null 95%':>26}{'pctile':>9}")
    print("-" * 89)
    for metric, record in null["metrics"].items():
        low, high = record["null_central_95"]
        print(f"{metric:<30}{100 * record['observed']:>11.3f}%{100 * record['null_mean']:>11.3f}%"
              f"{f'[{100 * low:.3f}%, {100 * high:.3f}%]':>26}"
              f"{record['observed_percentile']:>9.1f}")

    # ---- oracle bounds: fitted on test, NOT results ----
    print("\nORACLE BOUNDS — fitted on the test split. Unachievable; never results.")
    print(header)
    print("-" * len(header))
    oracles = {}
    for family in FAMILIES:
        criterion = "ece" if family in ("temperature", "per_class_temperature") else "nll"
        record = oracle_bound(t_labels, t_logits, family, classes, criterion=criterion)
        oracles[family] = record
        cells = "".join(f"{scale * record['metrics'][key]:>{9 if d == 3 else 10}.{d}f}"
                        + ("%" if scale == 100 else "") for key, scale, d in REPORTED)
        print(f"{f'{family} ({criterion} fit)':<30}{cells}")

    floor = null["metrics"]["ece_15_bins"]
    best = min(oracles.values(), key=lambda r: r["metrics"]["ece_15_bins"])
    print(f"\nbest oracle top-label ECE : {100 * best['metrics']['ece_15_bins']:.3f}%"
          f"  ({best['family']})")
    print(f"conditional null          : {100 * floor['null_mean']:.3f}% "
          f"[{100 * floor['null_central_95'][0]:.3f}%, {100 * floor['null_central_95'][1]:.3f}%]")
    exhausted = best["metrics"]["ece_15_bins"] <= floor["null_central_95"][1]
    print(f"family exhausted          : {exhausted}"
          + ("  (no honest method in this family can produce a resolvable difference)"
             if exhausted else ""))


if __name__ == "__main__":
    main()
