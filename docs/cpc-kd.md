# Pairwise calibration during student KD

CPC is an optional student training intervention with two fixed coefficients. It reuses the existing frozen teacher, student architecture, optimizer and checkpoint-selection policy. The separate loss module is injected into `DistillationObjective`; it has no trainable parameters and adds no model inference operations.

**The first six-run study is complete.** At coefficients 0.1/0.1, CPC did not improve the primary ranking endpoint beyond KD+TS. The small top-label ECE reduction after TS was inconclusive under the paired test-image bootstrap. See the [full results](../runs/cifar10-cpc/report.md) for per-seed metrics, uncertainty, calibration-null references and training cost.

| Condition | Mean test accuracy ↑ | Mean test AURC ↓ | Mean test ECE ↓ |
|---|---:|---:|---:|
| KD+TS | 85.77% | 0.02965 | 1.154% |
| KD+CPC+TS | 85.69% | 0.03009 | 0.968% |

These results describe three student seeds with one teacher and this fixed coefficient choice. They do not establish that CPC is ineffective in other settings.

The [implementation plan](cpc-implementation-plan.md) records the study design. The first setting uses coefficient 0.1 for each component, a project starting point rather than a claim of optimal weights. It is a CPC–KD adaptation, not a reproduction of the paper's architecture or training schedule.

## Loss and integration

The [CPC paper](https://jiacheng-cheng.github.io/assets/papers/cvpr22.pdf), equations 15–16, supplies binary discrimination and binary exclusion constraints. This implementation averages them as:

```text
BD = mean_j!=y softplus(z_j - z_y)
BE = mean_i<j,i!=y,j!=y [softplus(z_i-z_j) + softplus(z_j-z_i)] / 2
```

Average both components across the batch. CIFAR-10 has nine target comparisons and 36 non-target pairs per example. Exclusion is minimized at log(2); it is not expected to converge to zero. For two classes the empty exclusion term is a differentiable zero.

The study uses `0.5*CE + 0.5*warmup*16*KL(teacher/4 || student/4) + 0.1*BD + 0.1*BE`. The response warmup lasts five epochs. CPC acts on raw student logits from epoch 1 and has no trigger or ramp. Teacher outputs remain detached. The existing MMCE controller continues to control MMCE alone.

```toml
[cpc]
enabled = true
discrimination_weight = 0.1
exclusion_weight = 0.1
```

Old configurations omit this section and retain CPC-disabled behavior. The first implementation rejects combining CPC with MMCE, adaptive focal supervision, feature losses or nonstandard KD. Component weights must be finite and nonnegative; the study rejects an intervention with both weights zero.

Training logs `cpc_discrimination`, `cpc_exclusion`, and `cpc_weighted`. Resume validates the coefficients and restores existing optimizer/scheduler/RNG state. Pair indices are deterministic non-persistent buffers, rebuilt on demand. Exports contain only the student's original weights and metadata.

## Run and evaluate

From the project root in the existing `kd` Conda environment:

```bash
python -m unittest discover -s tests -v
python -m kd cpc-study --baseline runs/cifar10 --output runs/cifar10-cpc --device mps --dry-run
python -m kd cpc-study --baseline runs/cifar10 --output runs/cifar10-cpc --device mps
python -m kd.cpc_reporting --study runs/cifar10-cpc
```

The study trains fresh control/CPC pairs for seeds 42, 43 and 44. It keeps 45,000 training and 5,000 validation images, the existing split seed 2026, 20 epochs, matched SGD settings and augmentation. Checkpoints are selected by validation accuracy. It then fits an independent NLL temperature to each selected model's validation logits, using the fixed grid 0.01–5.00 in steps of 0.01. No temperature or checkpoint is selected on test data.

All checkpoint and calibration choices are persisted before loading the 10,000-image test split. Six CPU forward evaluations produce twelve conditions:

| Training | Raw probabilities | Validation-calibrated probabilities |
|---|---|---|
| Standard KD | KD | KD+TS |
| KD with CPC | KD+CPC | KD+CPC+TS |

The primary comparison is **KD+CPC+TS versus KD+TS** on mean AURC, with a mean accuracy-loss guardrail of 0.5 percentage points. AURC averages error rates as progressively less-confident predictions are retained; lower is better. A numerical improvement alone does not establish a conclusive result. The report also shows accuracy, NLL, Brier, class-wise calibration, three top-label ECE estimators and selective accuracy at 80%/90% coverage.

## Statistical interpretation

ECE uses 15 equal-width bins for continuity. Equal-mass binning keeps confidence ties together; L2 ECE uses the square root of weighted squared bin gaps. Class-wise ECE averages marginal one-versus-rest ECE over classes, and is distinct from the separate predicted-class gap table.

AURC and selective accuracy use the expected ordering within confidence ties. Temperature scaling keeps the predicted class but can change confidence rankings across examples, so ranking is recomputed after scaling.

Conditional calibration-null references draw coherent categorical labels from each model's fixed probabilities. They describe finite-sample behavior under this conditional null, not a hard measurement floor or uncertainty over model fitting. Paired test bootstrap intervals similarly condition on fitted models. Shared resample indices across seeds respect the common test images; seed variation is shown separately. This repeatedly examined CIFAR-10 test set makes the experiment exploratory.

The old logit archives can be diagnosed without inference or fitting:

```bash
python -m kd.calibration_diagnostics \
  --predictions runs/cifar10-posthoc/seed42/test_logits.npz \
  --temperature 1.46 \
  --output runs/cifar10-posthoc/seed42/extended_diagnostics.json
```

This command refuses to overwrite an existing diagnostic file. The supplied temperature is fixed, not estimated using the archive's labels.
