"""Paper §3-§5: does "oracle below the null" generalise beyond one checkpoint?

The central empirical claim rests on a single seed-42 KD student: the best oracle
for the logit-affine family falls below the conditional null, so no honest method
in that family can produce a resolvable top-label ECE difference. That is one
checkpoint. This script repeats the test on every trained checkpoint in `runs/`,
spanning three settings, several training objectives, three seeds and two
imbalance factors.

For each run: fit a temperature on its own validation split, score on test,
simulate the conditional null, and compute oracle bounds for every family.

The verdict needs **two** comparisons, not one. Asking only whether the oracle
reaches the floor conflates two opposite situations: a checkpoint that is already
calibrated (nothing to fix) and a badly miscalibrated one whose error the family
can remove entirely. The decision rule is therefore:

    observed <= null_high                  -> "no signal"  : do not run the experiment
    observed >  null_high, oracle <= high   -> "family suffices" : fit the family properly
    observed >  null_high, oracle >  high   -> "family insufficient" : a new method is justified

Only the third verdict warrants inventing a method.

Logits are collected once on CPU and cached next to each checkpoint as
`floor_cache_{split}.npz`. Caches written by a protocol-frozen study
(`validation_logits.npz` / `test_logits.npz`) are reused read-only and never
overwritten.

    python scripts/floor_survey.py [--limit N] [--repetitions 2000]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.calibration_metrics import conditional_calibration_null
from kd.checkpoints import load_model_checkpoint, metadata
from kd.config import from_dict
from kd.data import build_data
from kd.floors import FAMILIES, oracle_bound
from kd.models import create_model
from kd.posthoc import LogitCalibrator, collect_logits, fit_calibrators

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = (("balanced", ROOT / "runs/cifar10"),
            ("cpc", ROOT / "runs/cifar10-cpc"),
            ("long-tailed", ROOT / "runs/cifar10-lt-multiseed"))
TEMPERATURES = (np.arange(1, 501) / 100).tolist()


def verdict(observed, oracle, null_high):
    """Two-axis decision: is there signal, and can this family capture it?"""
    if observed <= null_high:
        return "no signal"
    return "family suffices" if oracle <= null_high else "family insufficient"


def discover():
    for setting, directory in SETTINGS:
        if not directory.is_dir():
            continue
        for run in sorted(p for p in directory.iterdir() if (p / "student.pt").exists()):
            if (run / "config.json").exists():
                yield setting, run


def logits_for(run: Path, split: str, loader, config, classes):
    """Reuse a study's signed cache if present; otherwise cache our own."""
    signed = run / f"{'validation' if split == 'val' else 'test'}_logits.npz"
    if signed.exists():
        with np.load(signed, allow_pickle=True) as data:
            return data["labels"].astype(int), data["logits"].astype(np.float64)
    cache = run / f"floor_cache_{split}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            return data["labels"].astype(int), data["logits"].astype(np.float64)
    model = create_model(config.student.name, config.data.num_classes)
    load_model_checkpoint(model, str(run / "student.pt"),
                          metadata(config.student.name, classes,
                                   config.data.image_size, config.data.source))
    model.requires_grad_(False).eval()
    labels, values = collect_logits(model, loader, torch.device("cpu"))
    del model
    np.savez_compressed(cache, labels=labels, logits=values, classes=np.asarray(classes))
    return labels.astype(int), values.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()

    runs = list(discover())[: args.limit]
    print(f"{len(runs)} checkpoints found\n")
    rows, bundles = [], {}
    for index, (setting, run) in enumerate(runs, start=1):
        config = from_dict(json.loads((run / "config.json").read_text()))
        key = json.dumps(config.data.to_dict() if hasattr(config.data, "to_dict")
                         else config.data.__dict__, sort_keys=True, default=str)
        if key not in bundles:
            bundles[key] = build_data(config.data, config.train, include_test=True)
        data = bundles[key]
        print(f"[{index}/{len(runs)}] {setting}/{run.name}", flush=True)

        v_labels, v_logits = logits_for(run, "val", data.val, config, data.classes)
        t_labels, t_logits = logits_for(run, "test", data.test, config, data.classes)
        winners, _ = fit_calibrators(v_logits, v_labels, temperatures=TEMPERATURES, gammas=[0.0])
        temperature = winners["temperature_nll"]["parameters"]["temperature"]
        probabilities = LogitCalibrator(temperature=temperature).predict(t_logits)

        null = conditional_calibration_null(t_labels, probabilities, data.classes,
                                            repetitions=args.repetitions)["metrics"]["ece_15_bins"]
        best = min(
            (oracle_bound(t_labels, t_logits, family, data.classes,
                          criterion="ece" if family in ("temperature", "per_class_temperature")
                          else "nll")
             for family in FAMILIES),
            key=lambda record: record["metrics"]["ece_15_bins"])
        rows.append({
            "setting": setting, "run": run.name, "temperature": temperature,
            "observed": null["observed"], "null_mean": null["null_mean"],
            "null_high": null["null_central_95"][1], "percentile": null["observed_percentile"],
            "oracle": best["metrics"]["ece_15_bins"], "oracle_family": best["family"],
            "verdict": verdict(null["observed"], best["metrics"]["ece_15_bins"],
                               null["null_central_95"][1])})

    header = (f"{'setting':<12}{'run':<26}{'T':>6}{'ECE':>9}{'null hi':>9}"
              f"{'pct':>7}{'oracle':>9}{'family':>24}{'  verdict'}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['setting']:<12}{r['run']:<26}{r['temperature']:>6.2f}"
              f"{100 * r['observed']:>8.3f}%{100 * r['null_high']:>8.3f}%"
              f"{r['percentile']:>7.1f}{100 * r['oracle']:>8.3f}%"
              f"{r['oracle_family']:>24}   {r['verdict']}")

    print()
    for setting, _ in SETTINGS:
        group = [r for r in rows if r["setting"] == setting]
        if not group:
            continue
        counts = {v: sum(r["verdict"] == v for r in group)
                  for v in ("no signal", "family suffices", "family insufficient")}
        print(f"  {setting:<12} n={len(group):<3} " +
              "  ".join(f"{v}: {n}" for v, n in counts.items() if n) +
              f"   median observed {100 * np.median([r['observed'] for r in group]):.3f}%"
              f"   median oracle {100 * np.median([r['oracle'] for r in group]):.3f}%")
    output = ROOT / "runs" / "floor_survey.json"
    output.write_text(json.dumps(rows, indent=2, default=float) + "\n")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
