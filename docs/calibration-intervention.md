# Calibration-loss intervention

This follow-up tests whether a training-time MMCE penalty improves the previously observed calibration of CIFAR-10 KD students. It compares six fresh runs: fixed KD versus triggered MMCE for seeds 42, 43, and 44, using the existing trained teacher. Every run preserves the original 45,000/5,000 split, 20-epoch training budget, initialization/data seed, model, augmentation, and optimizer.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd
python -m kd calibration-study --baseline runs/cifar10 --output runs/cifar10-calibration --device mps
MPLCONFIGDIR=/private/tmp/kd-matplotlib python -m kd.calibration_reporting --study runs/cifar10-calibration
```

The existing CIFAR study's configuration files and teacher artifact must be present. No teacher retraining is required. The new study writes to its own directory and can resume completed epochs/runs when the protocol and source hashes match.

## Intervention rule

The detector examines the unaugmented validation set after each epoch, starting at epoch 5. It requires three consecutive observations with both:

- ECE above 0.03 (3%).
- Mean predicted confidence minus accuracy above 0.01 (one percentage point).

The next training epoch receives a calibration coefficient of 1/3, followed by 2/3 and then 1. The coefficient stays at 1 afterwards. One-way activation avoids repeated on/off oscillation; this is a fixed experimental rule, not a learned optimal controller. It detects sustained overconfidence and does not establish its cause.

The objective is:

```text
existing classical KD objective + gamma × unweighted MMCE norm
```

For each training example, let `r` be the student's maximum softmax probability at ordinary temperature 1, and `c` indicate whether its top prediction is correct. The squared penalty is the batch mean over all pairs of `(r_i-c_i)(r_j-c_j) exp(-|r_i-r_j|/0.4)`. We use its square root with a small smoothing constant for numerical stability. Gradients flow through probabilities and the kernel, not through discrete correctness. There is no gradient from validation labels into model weights; validation drives the controller's scalar schedule only.

This is the unweighted empirical MMCE formulation, including diagonal pairs, rather than the paper's reweighted correct/incorrect variant. It has quadratic batch-size cost; our batch size is 128. The teacher and the original KD temperature/weight remain fixed. The penalty can affect accuracy, so its calibration benefit must be evaluated alongside classification quality.

## Predeclared evaluation

The same accuracy-based checkpoint selection is used for control and intervention: highest validation accuracy, earliest epoch on ties. No ECE-based checkpoint cherry-picking or post-training temperature scaling is used. The final test phase begins only after every run and checkpoint selection is complete.

The primary criterion is lower mean test ECE with at most a 0.5-percentage-point loss of mean test accuracy. NLL and Brier score are reported separately, including paired bootstrap confidence intervals. All outcomes are retained; an unsuccessful intervention is not automatically retuned using test results.

This uses the previously examined CIFAR-10 official test set, so it is an exploratory follow-up rather than a new independent confirmation. The controller uses validation data only. Three student seeds and one fixed teacher provide limited evidence about transfer to other models, datasets, or teacher initializations.

## Design and artifacts

- `MMCELoss` is a loss strategy composed into `DistillationObjective`.
- `CalibrationController` owns the trigger counter and activation epoch. It consumes scalar validation metrics and produces a weight for a subsequent epoch.
- Controller state, optimizer, scheduler, and random-generator states are checkpointed together. Resume restores the same intervention schedule.
- Every epoch records the observed ECE, confidence gap, threshold decision, activation state, actual coefficient, and calibration-loss value in `history.json`.
- `protocol.json` freezes settings, source hashes, and teacher identity before training. `selection.json` freezes checkpoint hashes before final test evaluation.
- `report.json`, the generated `report.md`, `results.csv`, figures, and per-run `test/predictions.npz` make the results reviewable.

The calibration penalty and controller have no inference parameters and add no inference operations. Student-only artifacts retain the original architecture.

For custom experiments with the ordinary `train` command, add a `[calibration]` section to the configuration. `enabled` defaults to false; available settings are `max_weight`, `kernel_bandwidth`, `monitor_start_epoch`, `ece_threshold`, `overconfidence_threshold`, `patience`, and `ramp_epochs`.

Reference: Kumar, Sarawagi, and Jain, [Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings, ICML 2018](https://proceedings.mlr.press/v80/kumar18a.html).
