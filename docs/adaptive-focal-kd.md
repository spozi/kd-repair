# AdaFocal and AdaDualFocal during student KD

These options replace the student's supervised cross-entropy term with an adaptive focal loss. The teacher stays frozen in evaluation mode, and its logits still supply the original distillation term. They are training methods, not post-hoc calibration. This integration has been checked with numerical and synthetic end-to-end tests; no CIFAR-10 accuracy or ECE improvement is claimed for this combination.

## Run configurations

The two ready-to-run configurations reuse the existing CIFAR-10 teacher, 45,000/5,000 training/validation split, 20-epoch student schedule, and seed 42. They use separate new run names and do not enable the previous MMCE intervention.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd
python -m kd train --config configs/cifar10_kd_adafocal.toml
python -m kd train --config configs/cifar10_kd_adadualfocal.toml
```

The commands above start training when executed. To initialize from an existing compatible student, set `student.checkpoint` and choose a new run name. This starts a fresh optimizer and adaptive controller. To resume an interrupted adaptive run, keep its configuration and directory and add `--resume runs/cifar10-adaptive-focal/RUN_NAME/last.pt`.

## Loss composition

For classical KD the objective is

```text
(1 - weight) * adaptive_supervised_loss(student_logits, labels)
    + warmup * weight * T² * KL(teacher/T || student/T)
    + optional feature and MMCE terms, if explicitly configured
```

The supervised probabilities are evaluated at T=1. Existing KD temperature, weighting, and response warmup semantics are preserved. For DKD, the adaptive supervised loss has coefficient 1, following the existing full-strength supervised convention. A classical KD weight of 1 is rejected with an adaptive supervised loss because it would remove that loss entirely.

Let `p` be the student's true-class probability and `g` the signed gamma assigned to it:

| Method | g >= 0 | g < 0 |
|---|---|---|
| AdaFocal | `-(1-p)^g log(p)` | `-(1+p)^abs(g) log(p)` |
| AdaDualFocal | `-(1-p+q)^g log(p)` | `-(1+p)^abs(g) log(p)` |

Here `q` is the largest class probability strictly below `p`, or zero when none exists, following the authors' DFL implementation. Incorrect predictions and ties matter: `q` is not generally the second-largest model output. Gradients flow through `p` and the selected `q`; discrete bin/competitor selection is not differentiated.

The AdaDualFocal composition uses dual focal in the positive branch and retains AdaFocal's standard inverse-focal branch. The DFL paper describes combining its loss with AdaFocal's schedule, but the public code provides the standalone DFL loss; this branch convention is explicit here, and the added KD term is an experimental extension rather than a reproduction of the papers' benchmarks.

## Validation feedback

`AdaptiveFocalController` starts with uniform bin boundaries and gamma=1. After each student validation pass it computes quantile boundaries from top-prediction confidence and records each bin's mean confidence `C`, accuracy `A`, and signed error `e=C-A`. The next epoch's training examples are assigned by their true-class probability using those boundaries.

Positive gamma updates as `g * exp(update_rate * e)`; negative gamma updates as `g * exp(-update_rate * e)`. Bounds limit the magnitude. If the resulting magnitude falls below `switch_threshold`, the controller switches branches and initializes the magnitude at that threshold. Defaults are 15 bins, rate 1, bounds [-2, 20], and threshold 0.2. Updates are computed in log space to avoid exponential overflow.

Equal-confidence ties stay together in the lower interval. Therefore bins may be empty or only approximately equal in size; empty bins retain their gamma. This avoids assigning contradictory gamma values to identical probabilities. Bins are ordered by confidence; gamma is retained by bin position when boundaries move.

Feedback comes from the existing unaugmented student validation pass, with no additional teacher pass and no test access. Evaluation's optional observer collects detached CPU confidence/correctness vectors. Validation does not contribute gradients, but it is a tuning set and must not be interpreted as an independent final evaluation.

## Design, checkpoints, and verification

- `SupervisedLossConfig` validates the strategy and controller settings; `cross_entropy` is the backward-compatible default.
- `AdaptiveFocalLoss` is a loss strategy injected into `DistillationObjective`. It composes the controller rather than putting adaptive rules into the trainer's batch loop.
- The controller's boundaries, gamma vector, and last observed epoch are PyTorch buffers in the objective state dictionary. They move with the device and resume alongside optimizer, scheduler, and RNG state.
- `history.json` records `adaptive_focal.used` for the just-completed epoch, the validation bin statistics, and `adaptive_focal.next` for the following epoch. `train.ce` remains ordinary cross-entropy for comparison; `train.supervised` is the actual supervised objective.
- `summary.json` identifies the supervised strategy and explicitly labels the final controller state. Best-model selection still uses validation accuracy. The exported `student.pt` contains only student weights and no teacher or controller.

Verification includes independent probability calculations, double-precision gradient checks, saturation stability, both switch directions, bounded updates, bin ties, teacher weights/BatchNorm isolation, and exact CPU resume through real student validation for both methods. Synthetic integration checks do not establish calibration benefit on CIFAR-10.

References: [AdaFocal, Algorithm 1](https://arxiv.org/html/2211.11838#S5), [Dual Focal Loss, Eq. 3 and §5.2](https://proceedings.mlr.press/v202/tao23a/tao23a.pdf), and [the authors' DFL implementation](https://github.com/Linwei94/ICML2023-DualFocalLoss/blob/main/dual_focal_loss.py).
