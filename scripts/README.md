# Analysis scripts

Every number in [`../docs/calibration-roadmap.md`](../docs/calibration-roadmap.md)
and [`../docs/paper-outline.md`](../docs/paper-outline.md) is produced here. The
two diagnostics the paper contributes are tested library code, not scripts:

| Diagnostic | Location | Tests |
|---|---|---|
| Conditional calibration null | `kd.calibration_metrics.conditional_calibration_null` | `tests/test_cpc_study.py` |
| Family oracle bound | `kd.floors.oracle_bound`, `oracle_table` | `tests/test_floors.py` |

Run everything from the project root in the `kd` environment:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd
```

## Section map

| Paper § | Script | Trains? | Runtime | Reads |
|---|---|---|---|---|
| §2, §4, §5 | `calibrator_families.py` | no | ~2 min | `runs/cifar10-posthoc/seed42/*_logits.npz` |
| §5 | `miscalibration_structure.py` | no | ~2 min | same |
| §6 | `cpc_sweep.py` | 3 runs | ~15 min | `runs/cifar10-cpc/`, writes `runs/cifar10-cpc-sweep/` |
| §7 | `lt_study.py` | 3 runs | ~15 min | writes `runs/cifar10-lt/` |
| §7 | `lt_multiseed.py` | 14 runs | ~60 min | writes `runs/cifar10-lt-multiseed/` |

The two no-training scripts reproduce the bulk of the paper's tables in about
four minutes from saved logits, and modify nothing.

### `calibrator_families.py` — §2, §4, §5

Calibrator comparison (raw, global TS by NLL and by ECE, per-class TS, vector,
matrix, Dirichlet), the conditional null for all four ECE variants, and oracle
bounds for all five families.

Honest fits use validation and are scored on test; ODIR strength is chosen by
5-fold cross-validation *inside* validation. Oracle rows are fitted on test
deliberately and are never results.

Produces the headline: best oracle 0.691% top-label ECE against a 0.770%
[0.450%, 1.130%] null — the logit-affine family is exhausted.

```bash
python scripts/calibrator_families.py [--run runs/cifar10-posthoc/seed42]
```

### `miscalibration_structure.py` — §5

Four analyses of structure that aggregate metrics cannot see:

1. Per-predicted-class confidence gaps (+2.85% dog to −2.99% horse) which cancel
   to a sample-weighted mean of +0.052 pp.
2. Pairwise binary calibration over all 45 class pairs, each against **its own**
   null.
3. The selection margin that chose the saved calibrator (+0.000083) against its
   bootstrap interval (9.8× wider).
4. Coarse-routed expert-tree feasibility arithmetic.

```bash
python scripts/miscalibration_structure.py [--run runs/cifar10-posthoc/seed42]
```

> **Per-pair floors are not interchangeable.** Across the 45 pairs the null
> ranges 0.252%–2.256%, because it tracks each pair's confidence distribution;
> pairs with mid-range confidences have substantially higher floors. Comparing
> one pair against the pooled mean null overstated cat/dog as "5× its floor" in
> an earlier draft; against its own null it is 1.9×. Always compare a stratum
> against its own null.

### `cpc_sweep.py` — §6

Validation-only sweep over CPC weights and warmup. Reuses the frozen
`runs/cifar10-cpc/` baselines and trains three fresh students. No test access.

### `lt_study.py`, `lt_multiseed.py` — §7

Long-tailed CIFAR-10. `lt_study.py` is the single-seed diagnostic at imbalance
factor 100; `lt_multiseed.py` is the three-seed, two-factor confirmation. Both
train on the long-tailed split, fit calibrators on the equally long-tailed
validation split, and evaluate on the untouched balanced test split. Both are
resumable — completed runs are reused.

## Conventions

- Scripts read saved artifacts and never mutate `runs/` directories that belong
  to a completed study.
- Anything fitted on the split it is scored on is labelled an oracle and is not a
  result.
- Analyses that reuse the official CIFAR-10 test split are exploratory
  follow-ups; earlier studies in this project already examined it.
- `sys.path` is adjusted at the top of each script so they run from the project
  root without installing the package.
