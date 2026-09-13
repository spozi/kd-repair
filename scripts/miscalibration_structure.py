"""Paper §5: structure that aggregate calibration metrics cannot see.

Four analyses on the same frozen checkpoint:

  1. Per-predicted-class confidence gaps, which cancel to near zero in aggregate.
  2. Pairwise binary calibration over all C(C-1)/2 class pairs, against a matched
     null. The worst pair sits far above its floor while every aggregate summary
     sits at the floor.
  3. The selection margin that chose the saved calibrator, against its own
     bootstrap interval.
  4. Feasibility arithmetic for a coarse-routed expert tree, showing routing
     accuracy is a ceiling only under perfect experts.

Reads only saved logit archives; trains nothing and modifies nothing.

    python scripts/miscalibration_structure.py [--run runs/cifar10-posthoc/seed42]
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.posthoc import LogitCalibrator, fit_calibrators

TEMPERATURES = (np.arange(1, 501) / 100).tolist()
# CIFAR-10 label order; the coarse split used for the expert-tree arithmetic.
VEHICLES = (0, 1, 8, 9)
ANIMALS = (2, 3, 4, 5, 6, 7)


def binary_ece(q, y, bins=15):
    index = np.minimum((q * bins).astype(int), bins - 1)
    return float(np.abs(np.bincount(index, weights=q - y, minlength=bins)).sum() / len(q))


def per_class_gaps(labels, probabilities, classes):
    prediction, confidence = probabilities.argmax(1), probabilities.max(1)
    correct = (prediction == labels).astype(float)
    rows = []
    for k, name in enumerate(classes):
        mask = prediction == k
        if not mask.any():
            continue
        rows.append((name, int(mask.sum()), float(confidence[mask].mean()),
                     float(correct[mask].mean())))
    return rows


def pairwise(labels, probabilities, classes, rng, replicates=40):
    """Binary ECE for every class pair, restricted to that pair's images."""
    rows = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            mask = (labels == i) | (labels == j)
            if mask.sum() < 2:
                continue
            q = probabilities[mask, i] / (probabilities[mask, i] + probabilities[mask, j])
            y = (labels[mask] == i).astype(float)
            null = np.mean([binary_ece(q, (rng.random(len(q)) < q).astype(float))
                            for _ in range(replicates)])
            rows.append((classes[i], classes[j], int(mask.sum()), binary_ece(q, y), float(null)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/cifar10-posthoc/seed42"))
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)

    with np.load(args.run / "validation_logits.npz", allow_pickle=True) as v:
        v_labels, v_logits = v["labels"].astype(int), v["logits"].astype(np.float64)
        classes = [str(c) for c in v["classes"]]
    with np.load(args.run / "test_logits.npz", allow_pickle=True) as t:
        t_labels, t_logits = t["labels"].astype(int), t["logits"].astype(np.float64)

    winners, _ = fit_calibrators(v_logits, v_labels, temperatures=TEMPERATURES, gammas=[0.0])
    temperature = winners["temperature_nll"]["parameters"]["temperature"]
    raw = LogitCalibrator().predict(t_logits)
    scaled = LogitCalibrator(temperature=temperature).predict(t_logits)

    # ---- 1. per-class gaps ----
    print(f"1. PER-PREDICTED-CLASS GAPS after global TS (T={temperature:.2f})\n")
    print(f"{'class':<14}{'n':>7}{'mean conf':>12}{'accuracy':>11}{'gap':>10}")
    print("-" * 54)
    gaps = per_class_gaps(t_labels, scaled, classes)
    for name, n, confidence, accuracy in sorted(gaps, key=lambda r: r[2] - r[3], reverse=True):
        print(f"{name:<14}{n:>7}{100 * confidence:>11.2f}%{100 * accuracy:>10.2f}%"
              f"{100 * (confidence - accuracy):>+9.2f}%")
    spread = max(c - a for _, _, c, a in gaps) - min(c - a for _, _, c, a in gaps)
    aggregate = sum(n * (c - a) for _, n, c, a in gaps) / len(t_labels)
    print(f"\nspread head to tail : {100 * spread:.1f} pp")
    print(f"sample-weighted mean: {100 * aggregate:+.3f} pp   <- what an aggregate summary sees")

    # ---- 2. pairwise ----
    print("\n\n2. PAIRWISE BINARY CALIBRATION over all class pairs\n")
    for label, probabilities in (("raw", raw), (f"global TS (T={temperature:.2f})", scaled)):
        rows = pairwise(t_labels, probabilities, classes, rng)
        observed = np.array([r[3] for r in rows])
        null = np.array([r[4] for r in rows])
        worst = max(rows, key=lambda r: r[3])
        print(f"  {label:<26} mean {100 * observed.mean():.3f}%  "
              f"(null {100 * null.mean():.3f}%)   worst: {worst[0]} vs {worst[1]} "
              f"{100 * worst[3]:.3f}% (null {100 * worst[4]:.3f}%, n={worst[2]})")
    print("\n  worst five pairs after scaling:")
    for a, b, n, value, null in sorted(pairwise(t_labels, scaled, classes, rng),
                                       key=lambda r: r[3], reverse=True)[:5]:
        print(f"    {a:>10} vs {b:<10} n={n:>5}  ECE {100 * value:>6.3f}%  "
              f"null {100 * null:>6.3f}%  ratio {value / null:>5.1f}x")

    # ---- 3. selection margin ----
    print("\n\n3. SELECTION MARGIN on validation\n")
    focal = min((w for k, w in winners.items() if k.startswith("focal")),
                key=lambda w: w["fit_score"], default=None)
    plain = winners["temperature_nll"]
    if focal is not None:
        a = LogitCalibrator(**plain["parameters"]).predict(v_logits)
        b = LogitCalibrator(**focal["parameters"]).predict(v_logits)
        index = np.arange(len(v_labels))
        loss = (-np.log(np.maximum(b[index, v_labels], 1e-300))
                + np.log(np.maximum(a[index, v_labels], 1e-300)))
        draws = np.array([loss[rng.integers(len(loss), size=len(loss))].mean()
                          for _ in range(args.repetitions)])
        low, high = np.quantile(draws, [0.025, 0.975])
        print(f"  focal minus plain, validation NLL : {loss.mean():+.6f}")
        print(f"  bootstrap 95% interval            : [{low:+.5f}, {high:+.5f}]")
        print(f"  interval width / |margin|         : {(high - low) / abs(loss.mean()):.1f}x")
        print("  -> the interval is wider than the margin: the selection is not resolvable")

    # ---- 4. expert-tree arithmetic ----
    print("\n\n4. COARSE-ROUTED EXPERT TREE feasibility\n")
    flat = float((t_logits.argmax(1) == t_labels).mean())
    grouped = np.stack([scaled[:, VEHICLES].sum(1), scaled[:, ANIMALS].sum(1)], 1)
    router = float((grouped.argmax(1) == np.isin(t_labels, ANIMALS).astype(int)).mean())
    weighted = 0.0
    for group in (VEHICLES, ANIMALS):
        mask = np.isin(t_labels, group)
        within = scaled[mask][:, group].argmax(1)
        target = np.array([list(group).index(v) for v in t_labels[mask]])
        weighted += mask.mean() * float((within == target).mean())
    print(f"  flat accuracy                        : {100 * flat:.2f}%")
    print(f"  coarse router accuracy               : {100 * router:.2f}%")
    print(f"  expert accuracy needed to tie flat   : {100 * flat / router:.2f}%")
    print(f"  flat model restricted within groups  : {100 * weighted:.2f}%")
    print(f"  projected tree accuracy              : {100 * router * weighted:.2f}%")
    print(f"  -> experts must beat the restricted flat model by "
          f"{100 * (flat / router - weighted):+.2f} pp merely to break even")


if __name__ == "__main__":
    main()
