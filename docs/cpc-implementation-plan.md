# Implementation plan: CPC during student knowledge distillation

Status: implemented and the first six-run study completed, 6 September 2026. This document preserves the design for option E from the [calibration roadmap](calibration-roadmap.md). See the [completed report](../runs/cifar10-cpc/report.md) and [implementation guide](cpc-kd.md).

**Objective:** determine whether CPC improves confidence-based error ranking and probability quality beyond standard KD plus validation-fitted temperature scaling. Implement the measurement prerequisites first, then a fixed-coefficient CPC loss, then a matched three-seed study.

## 1. Decisions for the first experiment

| Decision | Planned setting |
|---|---|
| Training arms | Fresh standard KD and fresh KD+CPC for each seed: six student runs |
| Seeds | 42, 43, 44 |
| Teacher | Existing `runs/cifar10/teacher/student.pt`, frozen; record its SHA256 |
| Student | Existing `cifar_student`, initialized afresh with matched initialization within each seed |
| Data | Existing 45,000 training / 5,000 stratified validation / 10,000 test split; split seed 2026 |
| Training recipe | 20 epochs, batch 128, SGD learning rate 0.05, momentum 0.9, weight decay 0.0005, cosine schedule, current basic augmentation |
| KD settings | Training temperature 4, KD weight 0.5, five-epoch response-loss warmup |
| CPC settings | Fixed binary-discrimination weight 0.1 and binary-exclusion weight 0.1, active from epoch 1 |
| Checkpoint selection | Highest validation accuracy, earliest epoch on ties |
| Post-hoc calibration | Ordinary TS fitted separately for each selected checkpoint by validation NLL |
| Primary comparison | KD+CPC+TS minus KD+TS, mean test AURC across the three paired seeds |
| Accuracy guardrail | Mean accuracy decrease no greater than 0.5 percentage points |
| Execution device | MPS training after a device check; CPU final inference for all arms |

The two CPC weights are explicit project starting values, not claimed optimal or taken from a matching paper experiment. This first experiment has no coefficient search. Numerical smoke checks may identify implementation errors, but must not become undocumented performance tuning. Any later coefficient search needs a separate validation-only protocol and a recorded search budget.

Keep MMCE, AdaFocal, AdaDualFocal, feature losses, architecture changes and adaptive CPC activation out of these two arms. This isolates the effect of adding CPC to standard KD. Historical checkpoints remain references; fresh controls provide the primary comparison under the same implementation and execution conditions.

## 2. Loss specification

Use the binary discrimination and binary exclusion losses in equations 15–16 of [Cheng and Vasconcelos, CVPR 2022](https://jiacheng-cheng.github.io/assets/papers/cvpr22.pdf). For logits `z`, true label `y`, and `C` classes, an equivalent implementation using `softplus` is:

```text
L_BD = mean over j != y of softplus(z_j - z_y)

L_BE = mean over unordered pairs i < j, i != y, j != y of
       0.5 * (softplus(z_i - z_j) + softplus(z_j - z_i))
```

For CIFAR-10, each example contributes nine target-versus-alternative comparisons and 36 non-target pairs. Preserve the normalization above; do not sum all pairs without adjusting the coefficients. The exclusion term has minimum log(2), rather than zero. Average each component over the batch. For binary classification, define the empty exclusion term as a differentiable zero.

The project objective is:

```text
L = 0.5 * CE(student, label)
    + 0.5 * r(epoch) * 4^2 * KL(teacher/4 || student/4)
    + 0.1 * L_BD(student raw logits, label)
    + 0.1 * L_BE(student raw logits, label)

r(epoch) = min(epoch / 5, 1), with one-based epochs
```

This is a CPC–KD adaptation. Existing KD code already includes the temperature-squared factor in its response loss; do not apply it twice. CPC uses ordinary raw student logits and receives neither the KD temperature nor an extra cross-entropy term. Its coefficients remain fixed while the existing KD response warmup operates.

Binary exclusion can conflict with the teacher's distinctions among non-target classes. Record component losses and outcomes; an improvement is a hypothesis, not a consequence of the paper's theorem.

## 3. Design and file changes

Use the project's existing strategy, composition and dependency-injection patterns. Keep the training loop architecture-independent and put experiment policy in the study runner.

| File | Planned change | Boundary to preserve |
|---|---|---|
| `kd/cpc.py` — new | `PairwiseCalibrationLoss` with a component calculation and a weighted scalar forward; stable softplus; cached non-trainable pair indices | No data loading, controller, optimizer or teacher access |
| `kd/config.py` | Frozen `CPCConfig(enabled=False, discrimination_weight=0.1, exclusion_weight=0.1)`; parsing and finite, nonnegative coefficient validation | Old JSON/TOML files default to CPC disabled; disabled means exact existing behavior |
| `kd/losses.py` | Add an optional keyword-only `cpc_loss` dependency to `DistillationObjective`; compute components once and add their weighted sum | Preserve existing positional constructor arguments and CE/KD/MMCE semantics |
| `kd/engine.py` | Construct CPC when enabled; aggregate its logged components; include effective settings in summaries and resume checks | Teacher remains frozen; model-only export and validation-accuracy selection stay intact |
| `kd/calibration_metrics.py` — new | Pure NumPy calibration, ranking, bin summaries and conditional-null diagnostics | No model forward or fitting; deterministic explicit RNG inputs |
| `kd/evaluation.py` | Add optional extended reports and paired ranking/calibration comparisons using the new helpers | Retain existing metric keys and the current ECE definition |
| `kd/metrics.py` | Add an opt-in full-prediction diagnostic path for validation using the same helpers | Keep current default return values and adaptive-focal confidence observer behavior |
| `kd/posthoc.py` | Reuse `LogitCalibrator` and `fit_calibrators(..., gammas=[0.0])`; consume only the NLL-fitted TS result | Do not change the completed post-hoc study's choices or artifacts |
| `kd/cpc_study.py` — new | Protocol, matched run construction, resume, checkpoint and calibrator selection, test gate, result aggregation | Reuse `run_experiment`; no second training loop |
| `kd/cpc_reporting.py` — new | Render Markdown, CSV, risk–coverage and calibration plots from completed artifacts | Reporting must not train, fit temperatures, or rerun inference |
| `kd/cli.py` | Add `cpc-study` with a read-only `--dry-run` | CLI delegates to the study runner |
| `configs/cifar10_kd_cpc.toml` — new | Complete seed-42 standalone recipe with fixed CPC coefficients | Study derives matched seed configurations from the baseline configs |
| `tests/test_cpc.py`, `tests/test_calibration_metrics.py`, `tests/test_cpc_study.py` — new | Mathematical, integration, resume and evaluation-gating tests | Use existing `unittest` and offline fixtures |
| `docs/cpc-kd.md`, `docs/design.md`, `README.md` | Document mathematics, settings, commands, result interpretation and module boundaries | Distinguish completed measurements from planned experiments |

### Separate CPC from the MMCE controller

Currently `run_experiment` sets `objective.calibration_weight` from `CalibrationController` every epoch, and `calibration_loss` is used for MMCE. Simply injecting CPC there would retain MMCE's trigger or leave CPC inactive. The planned separate `cpc_loss` dependency avoids this coupling without changing MMCE's established configuration and resume contract.

Log `cpc_discrimination`, `cpc_exclusion` and `cpc_weighted` only when CPC is enabled. Extend the trainer's fixed metric aggregation for these named terms; do not introduce an unconstrained plugin registry. The objective should expose a component result from one calculation, so logging never computes the pair losses twice or retains graph tensors between batches.

The initial configuration validator should reject enabled CPC combined with MMCE or adaptive focal, and restrict this study to standard KD with cross-entropy supervision. Both zero coefficients are useful for exact-baseline regression tests; the study itself must reject an enabled all-zero intervention. Pair-index buffers must support device transfer and exact state restoration. No new model inference parameters are added.

## 4. Measurement contract: prerequisites A and B

| Output | Exact convention |
|---|---|
| Top-label ECE | Existing 15 equal-width bins and weighted absolute gap, preserving `ece_15_bins` |
| Equal-mass ECE | Target 15 quantile groups; keep identical confidences together, allow fewer groups and record effective counts/boundaries |
| L2 ECE | Square root of the bin-count-weighted mean squared confidence–accuracy gap; equal-width bins |
| Class-wise ECE | For each class, bin its probability against its one-hot target over all examples; average the ten class ECE values equally |
| Predicted-class diagnostics | Support, confidence, accuracy, signed gap and within-group top-label ECE for each predicted class; distinguish this from marginal class-wise ECE |
| AURC | Sort by descending maximum probability; mean cumulative error rate over retained counts 1 through N; smaller is better |
| Selective accuracy | Accuracy among the highest-confidence 80% and 90% of predictions |
| Existing probability metrics | Accuracy, NLL, multiclass Brier and macro F1 |

Resolve confidence ties using expected risk under uniform random ordering within each tied group, computed analytically. At a coverage boundary, use the expected correct count for the required fraction of the tied group. This avoids arbitrary input-order effects and does not use labels to choose a tie order. Document these conventions so ranking values can be reproduced.

Keep extended diagnostics opt-in. Selected-checkpoint reports should include all of them; expensive simulations and resampling run only after fitting and selection, not every epoch. Where epoch histories or controller logs show the old ECE alone, label those as monitoring values rather than inferential comparisons.

### Conditional calibration-null references

For each fixed probability matrix, draw 2,000 synthetic label vectors from its categorical distributions, with an explicit RNG seed. Recompute the same calibration metrics and group summaries, reporting their null means, central 95% simulation ranges and the observed percentile. Use one categorical label per example so class-wise draws are coherent.

These are conditional perfect-calibration references, not hard noise floors, confidence intervals on true calibration error, or estimates to subtract from observed ECE. Include references next to calibration metrics in the detailed report. Zero-support predicted classes should produce a documented unavailable value, not a misleading zero. Add a reproducible diagnostic command for the existing logit archives; do not silently treat the roadmap's unavailable throwaway analyses as verified inputs.

## 5. Experiment phases and leakage controls

1. **Dry run:** validate configs, dataset availability, teacher identity, seeds, coefficients, metric definitions, output paths and expected run count. Print the planned six runs without writing a protocol, initializing models, downloading data or training.
2. **Protocol:** before full training, atomically persist effective configs, source hashes, environment, dataset split provenance, initial model-state hashes, pair ordering, coefficient values, metric conventions and RNG seeds. Freeze the primary endpoint and comparison.
3. **Training:** interleave control and CPC within each seed. Save epoch histories, optimizer/scheduler/RNG/objective states and best/last checkpoints through the existing engine. Verify equal initial student tensors and batch/augmentation streams within each seed. Record measured training time.
4. **Checkpoint selection:** after all six runs complete, persist their validation-selected epochs and checkpoint hashes. No test dataset construction is allowed yet.
5. **Calibration:** collect each selected model's validation logits and fit ordinary TS by NLL on the existing T grid, 0.01–5.00 in steps of 0.01. Save all six temperatures and fit provenance before allowing test access. An optimum at a grid boundary is reported, without automatic grid expansion.
6. **Test:** collect test logits once per model on CPU; calculate raw and TS probabilities from those same logits. Confirm TS preserves all class predictions within each model; do not assert equality between control and CPC models. Verify class order, test labels, sample counts and checkpoint hashes.
7. **Statistics and reporting:** write the four evaluation conditions per seed, paired deltas, seed summaries, curves, null references and the conclusion. Reports read saved results and may be regenerated without rerunning training or fitting.

| Evaluation condition | Checkpoint | Probability transformation |
|---|---|---|
| KD | Fresh control for that seed | Ordinary softmax |
| KD+TS | Same control checkpoint | Its own validation-NLL temperature |
| KD+CPC | Fresh CPC checkpoint for that seed | Ordinary softmax |
| KD+CPC+TS | Same CPC checkpoint | Its own validation-NLL temperature |

Do not reuse seed 42's temperature for other checkpoints. Temperature scaling preserves the chosen class but can change rankings across examples in multiclass models, so recompute AURC after scaling.

### Resume and provenance

Use a new root such as `runs/cifar10-cpc/`. Resume completed phases only when their hashes match. An interrupted training run resumes its own `last.pt`; a completed run is verified and skipped. Validation caches must carry checkpoint and preprocessing identity. Test loading remains blocked unless every checkpoint and calibration selection exists and matches its artifact.

Never overwrite historical studies or update their recorded source hashes to match new code. Preserve historical manifests even after implementation changes. Final-run manifests should cover imported computational dependencies as well as the study runner. Keep reporting code separate so editorial changes need not invalidate computation provenance.

Suggested artifact layout:

```text
runs/cifar10-cpc/
  protocol.json
  environment.json
  progress.json
  checkpoint_selection.json
  calibration_selection.json
  kd_control_seed42/             # plus seeds 43 and 44
  kd_cpc_seed42/                 # plus seeds 43 and 44
    config.json, history.json, summary.json, best.pt, last.pt, student.pt
    validation_logits.npz
    calibrator.json
    test_logits.npz
    evaluation/raw/
    evaluation/temperature_nll/
  report.json
  report.md
  results.csv
  figures/
```

## 6. Decision rule and uncertainty

Predeclare the primary difference as mean AURC(KD+CPC+TS) minus mean AURC(KD+TS); negative favors CPC. A preliminary numerical success requires a negative difference and mean accuracy delta at least -0.005. Report whether uncertainty supports that direction separately; passing a numerical rule alone is not a conclusive finding.

Use 2,000 paired test-example bootstrap resamples with rankings, coverage and calibration bins recomputed in each replicate. For the across-seed mean difference, draw the same test-image indices for all seed pairs in each replicate because all seeds share the test set. Also report each seed's paired result and mean ± sample standard deviation across seeds. Do not pool the three predictions per image into 30,000 independent observations or treat a test-example interval as uncertainty over retraining.

Display NLL, Brier and class-wise calibration outcomes even if AURC fails to improve. A calibration improvement without a ranking improvement is a valid secondary result. Compare raw KD+CPC against raw KD as a secondary contrast; no multiple-comparison significance claims from whichever metric happens to improve. The existing test set has been examined repeatedly, so all findings remain exploratory.

## 7. Verification and acceptance criteria

| Layer | Required checks |
|---|---|
| CPC mathematics | Independent small-loop reference for both terms; nine/36 mask counts on ten classes; C=2 exclusion behavior; no diagonal pairs; pair normalization; equal-logit exclusion equals log(2) |
| Numerical behavior | Float64 gradient checks, finite loss/gradients on large positive and negative logits, batch-duplication invariance, class-permutation and common-logit-shift invariance |
| Objective | Zero CPC gives exactly existing KD; nonzero terms add once; gradients reach student only; no duplicate CE or T-squared factor |
| Teacher and export | Teacher gradients absent, weights and BatchNorm buffers unchanged; exported student contains no CPC dependencies or new model parameters |
| Configuration/resume | Old configs default off; unsupported combinations fail clearly; changed coefficients reject resume; interrupted and uninterrupted tiny CPU runs match exactly |
| Metrics | Hand-computed ECE/class-wise examples; equal-mass ties and empty groups; perfectly ordered and reversed rankings; tied-confidence ordering invariance; all-correct/all-wrong cases |
| Null/resampling | Deterministic RNG, coherent categorical labels, known synthetic calibrated example, correct same-image pairing across seeds |
| Study | Offline tiny integration fixture proves no test construction before all selections; completed runs skipped; tampered manifests rejected; calibrator/checkpoint pairing verified |
| Device | Small CPC forward/backward and training smoke on MPS if available; no expectation of bitwise CPU/MPS agreement |
| Regression | Existing `unittest` suite passes; old evaluation keys, MMCE activation and adaptive-focal resume behavior are preserved |
| Reporting | Tables agree with JSON; explicit units and fitting criteria; result headline corresponds to its named artifact; report regeneration needs no inference |

Completion means code, tests, documented commands and a dry-run protocol are ready. Completion of the actual experiment is separate: all six runs, six validation-fitted temperatures, twelve evaluation rows, paired analyses and a readable report must exist before reporting a CPC outcome.

## 8. Delivery order and estimated execution cost

| Stage | Deliverable | Dependency |
|---|---|---|
| 1 | Extended metrics, conditional-null diagnostics and existing-logit diagnostic command | None |
| 2 | CPC loss and independent mathematical tests | Paper equations verified |
| 3 | Configuration, objective/trainer integration, logging and resume tests | Stage 2 |
| 4 | Matched study runner, CLI dry run, artifact gates and TS evaluation | Stages 1 and 3 |
| 5 | Report renderer, documentation and full regression/device smoke checks | Stage 4 |
| 6 | Freeze protocol, execute six runs and publish local results | All implementation checks pass; subsequent execution task |

The previous standard-KD run took about 4.5 minutes for 20 training epochs on MPS. Six matched runs therefore imply roughly 27 minutes of training before CPC overhead, validation, calibration and reporting. Budget approximately 35–60 minutes for experiment execution, subject to an actual device measurement; this excludes implementation time and any future tuning. The roadmap's approximately 14 minutes covers only the three CPC runs, not fresh controls.

Planned commands, to become available after implementation:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd
python -m unittest discover -s tests -v
python -m kd cpc-study --baseline runs/cifar10 --output runs/cifar10-cpc --device mps --dry-run
python -m kd cpc-study --baseline runs/cifar10 --output runs/cifar10-cpc --device mps
python -m kd.cpc_reporting --study runs/cifar10-cpc
```

Corruption evaluation, a no-calibration-set experiment, adaptive CPC scheduling, and coefficient/component ablations remain separate follow-ups. A failed first setting does not establish that CPC cannot help; a successful one still needs additional seeds or fresh data for stronger confirmation.
