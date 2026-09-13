"""Exploratory single-seed CPC weight/warmup sweep, validation only.

This is deliberately outside the frozen `kd.cpc_study` protocol: it varies the
hyperparameters that study held fixed (discrimination/exclusion weight,
warmup), and it never touches the official test split, which has already been
examined by the KD, calibration, adaptive-focal, post-hoc and CPC studies. Only
after a candidate looks promising here should a matched, multi-seed,
test-evaluated study be built (mirroring `kd.cpc_study`'s protocol-freeze
discipline) to confirm it.

Two rows are free: the existing `kd_control_seed42` and `kd_cpc_seed42` runs
from `runs/cifar10-cpc/` already have frozen validation logits and are reused
without retraining. New variants train fresh 20-epoch students with the exact
seed-42 CIFAR-10 KD recipe `kd.cpc_study.intervention_configs` enforces,
varying only `cpc.discrimination_weight`, `cpc.exclusion_weight` and
`cpc.warmup_epochs`.

Run from the project root in the kd environment:

    python scripts/cpc_sweep.py
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.calibration_metrics import extended_prediction_metrics
from kd.config import (BenchmarkConfig, CPCConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig, from_dict)
from kd.data import build_data
from kd.engine import run_experiment
from kd.models import create_model
from kd.checkpoints import load_model_checkpoint, metadata
from kd.posthoc import LogitCalibrator, collect_logits, fit_calibrators


ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "runs" / "cifar10-cpc"
OUTPUT = ROOT / "runs" / "cifar10-cpc-sweep"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CAT, DOG = CLASSES.index("cat"), CLASSES.index("dog")
TEMPERATURES = (np.arange(1, 501) / 100).tolist()

# (name, discrimination_weight, exclusion_weight, warmup_epochs)
# Rationale: cpc_discrimination tracked ~0.06 by epoch 20 in the original run
# and largely duplicates work CE/response already do (see history.json); the
# term that stalled was cpc_exclusion (~1.95, barely moving from its epoch-5
# value of ~2.06) while the KD response term was still ~0.5 of the total loss
# at epoch 20. These variants (a) drop discrimination toward zero to stop
# competing with CE for no benefit, (b) raise exclusion weight since it is the
# only term doing calibration work the rest of the objective does not, and
# (c) ramp CPC in instead of running it full-strength against a still-huge
# early-epoch response loss (7.99 at epoch 1 in the original run).
VARIANTS = [
    ("excl_only_w0.3_warmup0", 0.00, 0.30, 0),
    ("excl_only_w0.3_warmup8", 0.00, 0.30, 8),
    ("low_disc_w0.05_excl_w0.15_warmup8", 0.05, 0.15, 8),
    # Corrected variant. The first three variants above were designed on the
    # assumption that exclusion governs confusable-pair calibration. It cannot:
    # `PairwiseCalibrationLoss.components` masks out every pair touching the
    # true label, so on a cat image the exclusion term never sees the cat-dog
    # relationship at all. Discrimination is the term that penalizes
    # softplus(logit_dog - logit_cat) directly. This raises it instead, keeps
    # exclusion at its original modest weight, and retains the warmup that
    # measurably helped accuracy and AURC.
    ("disc_w0.2_excl_w0.1_warmup8", 0.20, 0.10, 8),
]


def base_config(name: str, discrimination: float, exclusion: float, warmup_epochs: int) -> ExperimentConfig:
    teacher_checkpoint = str(ROOT / "runs" / "cifar10" / "teacher" / "student.pt")
    config = ExperimentConfig(
        name=name, output_dir=str(OUTPUT),
        data=DataConfig(source="cifar10", root=str(ROOT / "data"), num_classes=10, image_size=32,
                        horizontal_flip=True, validation_fraction=0.1, split_seed=2026),
        student=ModelConfig("cifar_student"), teacher=ModelConfig("cifar_teacher", teacher_checkpoint),
        distillation=DistillationConfig(method="kd", temperature=4.0, weight=0.5, warmup_epochs=5),
        train=TrainConfig(epochs=20, batch_size=128, learning_rate=0.05, momentum=0.9,
                          weight_decay=0.0005, workers=0, threads=4, device="auto", seed=42),
        benchmark=BenchmarkConfig(warmup=20, iterations=100),
        cpc=CPCConfig(enabled=True, discrimination_weight=discrimination,
                     exclusion_weight=exclusion, warmup_epochs=warmup_epochs))
    config.validate()
    return config


def evaluate_condition(name: str, checkpoint: str, config: ExperimentConfig, validation) -> dict:
    model = create_model(config.student.name, config.data.num_classes)
    load_model_checkpoint(model, checkpoint,
                          metadata(config.student.name, CLASSES, config.data.image_size, config.data.source))
    model.requires_grad_(False).eval()
    labels, logits = collect_logits(model, validation, torch.device("cpu"))
    winners, _ = fit_calibrators(logits, labels, temperatures=TEMPERATURES, gammas=[0.0])
    temperature = winners["temperature_nll"]["parameters"]["temperature"]
    probability = LogitCalibrator(temperature=temperature).predict(logits)
    metrics = extended_prediction_metrics(labels, probability, CLASSES, include_curve=False)
    metrics.update(temperature=temperature)
    m = (labels == CAT) | (labels == DOG)
    q = probability[m, CAT] / (probability[m, CAT] + probability[m, DOG])
    yb = (labels[m] == CAT).astype(float)
    metrics.update(catdog_pairwise_ece=pairwise_ece(q, yb), catdog_n=int(m.sum()))
    return {"name": name, "checkpoint": checkpoint, "_q": q, "_y": yb, **metrics}


def pairwise_ece(q: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    ids = np.minimum((q * bins).astype(int), bins - 1)
    return float(np.abs(np.bincount(ids, weights=q - y, minlength=bins)).sum() / len(q))


def paired_catdog_interval(control: dict, candidate: dict, *, seed: int = 2026,
                           repetitions: int = 2000) -> dict:
    """Paired bootstrap of the cat/dog pairwise ECE difference.

    Every condition scores the identical validation images in the same order,
    so resampling one index vector and rescoring both sides keeps the pairing.
    """
    if not np.array_equal(control["_y"], candidate["_y"]):
        raise ValueError("Paired comparison requires identical cat/dog label vectors")
    rng = np.random.default_rng(seed)
    n = len(control["_y"])
    deltas = []
    for _ in range(repetitions):
        index = rng.integers(n, size=n)
        deltas.append(pairwise_ece(candidate["_q"][index], candidate["_y"][index])
                      - pairwise_ece(control["_q"][index], control["_y"][index]))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"delta": candidate["catdog_pairwise_ece"] - control["catdog_pairwise_ece"],
            "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # Free rows: reuse the frozen study's checkpoints and validation split.
    control_cfg = from_dict(json.loads((BASELINE / "kd_control_seed42" / "config.json").read_text()))
    cpc_cfg = from_dict(json.loads((BASELINE / "kd_cpc_seed42" / "config.json").read_text()))
    validation = build_data(control_cfg.data, control_cfg.train, include_test=False).val
    rows.append(evaluate_condition("control (reused)", str(BASELINE / "kd_control_seed42" / "student.pt"),
                                   control_cfg, validation))
    rows.append(evaluate_condition("cpc_0.1_0.1_warmup0 (reused, original study)",
                                   str(BASELINE / "kd_cpc_seed42" / "student.pt"), cpc_cfg, validation))

    # New rows: train fresh, then evaluate the same way.
    for name, disc, excl, warmup in VARIANTS:
        config = base_config(name, disc, excl, warmup)
        run_dir = OUTPUT / name
        if (run_dir / "student.pt").exists():
            print(f"Reusing already-trained {name}")
        else:
            print(f"Training {name}: discrimination={disc} exclusion={excl} warmup_epochs={warmup}")
            run_experiment(config)
        rows.append(evaluate_condition(name, str(run_dir / "student.pt"), config, validation))

    print()
    hdr = f"{'condition':<42}{'acc':>7}{'AURC':>10}{'cwECE':>8}{'cat/dog ECE':>13}{'T':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<42}{100*r['accuracy']:>6.2f}%{r['aurc']:>10.5f}"
              f"{100*r['classwise_ece_15_bins']:>7.3f}%{100*r['catdog_pairwise_ece']:>12.3f}%{r['temperature']:>7.2f}")

    control = rows[0]
    print(f"\ncat/dog pairwise ECE vs control, paired bootstrap over the same "
          f"{control['catdog_n']} validation images (negative favours the variant):")
    for r in rows[1:]:
        interval = paired_catdog_interval(control, r)
        r["catdog_vs_control"] = interval
        print(f"  {r['name']:<44}{100*interval['delta']:>+8.3f} pp  "
              f"[{100*interval['ci_low']:>+7.3f}, {100*interval['ci_high']:>+7.3f}]")

    saved = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (OUTPUT / "sweep_summary.json").write_text(json.dumps(saved, indent=2, default=float))
    print(f"\nSaved: {OUTPUT / 'sweep_summary.json'}")
    print("\nValidation only. No test-set access. Not a substitute for a matched,")
    print("multi-seed, protocol-frozen confirmation of any winning candidate.")


if __name__ == "__main__":
    main()
