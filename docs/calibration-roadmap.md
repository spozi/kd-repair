# Calibration roadmap: what is finished, and what to measure next

This document records what has been tried to reduce this CIFAR-10 KD student's
overconfidence, what worked, and what did not. Sections 1–7 analyse existing
artifacts on balanced CIFAR-10. Section 8 records a completed three-seed study
and a follow-up sweep. Section 9 assesses an architecture that was considered
and, on the evidence, not built. Section 11 changes the dataset setting, which
is where the effects finally became measurable.

**Summary.**

1. **Post-hoc calibration is finished on balanced CIFAR-10.** Top-label and
   class-wise ECE sit at their estimators' noise floors (§3, §4), and this holds
   for 8 of 10 checkpoints across supervised, KD, DKD and the teacher (§12).
2. **The residual error is pairwise.** cat/dog pairwise ECE is ~4.1% against
   its own ~2.18% null — 1.9x its floor and above every simulated draw, where the
   aggregate summaries sit *at* their floors (§7). A single global temperature
   barely moves it.
3. **Training-time pairwise constraints did not fix it.** CPC missed its
   predeclared rule across three seeds, and five weight/warmup configurations all
   came out worse than the plain control on the best-powered pairwise endpoint
   (§8).
4. **A coarse-routed expert tree is not recommended.** It projects to 85.65% vs
   86.00% flat before any calibration benefit, and the closest published
   evaluations find top-down classifiers dominated by flat softmax (§9).
5. **The binding constraint is statistical power, not method choice.** Single-pair
   endpoints need roughly 9× more data than CIFAR-10 contains (§8).
6. **Changing the setting worked.** On long-tailed CIFAR-10 the miscalibration is
   28× its noise floor, ordered perfectly by class frequency (Spearman +1.0000),
   and global temperature scaling leaves 13.5% ECE on the table. A class-balanced
   fitting objective and logit adjustment recover it: +12.3 pp accuracy, 81% less
   class-wise ECE, AURC halved, on two post-hoc parameters (§11).
7. **Across all 30 trained checkpoints, none justified a new method** (§12). Every
   one was either already at the floor or fixable by the existing logit-affine
   family; none fell in the "family insufficient" quadrant that would warrant
   inventing one.

Unless a row is explicitly labelled an oracle, every number is fitted on
validation and evaluated on the split named in its section.

## 1. Reading the existing result correctly

[The post-hoc report](../runs/cifar10-posthoc/seed42/report.md) is sound: T = 1.46
removes about 78% of the calibration error with accuracy and every predicted
class unchanged. Two points need adding.

**The saved selection was decided by noise.** FTS was chosen over ordinary TS on
a validation NLL margin of **+0.000083**, whose bootstrap 95% interval is
**[-0.00033, +0.00048]** — five times wider than the margin, straddling zero.
The report already declines to claim a test difference; the stronger statement is
that the *selection rule itself* was a coin flip. An NLL argmin over a 5,000-point
grid fitted to 5,000 validation images is not a decision procedure when the
candidates are this close. A selection rule that requires a minimum margin, or
that prefers the simpler family on ties, would be more honest.

**The headline and the saved artifact disagree.** The summary quotes 0.994%
(TS, ECE fit) while `selected_calibrator.json` holds FTS at 1.143% test ECE. Both
facts are disclosed further down, but a reader who loads the saved artifact gets
a different number from the one in the first sentence.

## 2. Step 0 — the metrics that were missing

`ece_15_bins` in [`metrics.py`](../kd/metrics.py) and
[`evaluation.py`](../kd/evaluation.py) is top-label ECE only: 15 equal-width
bins, L1 norm, binned on `probabilities.max(1)`. Adding equal-mass binning, an L2
variant, and class-wise ECE changes what the results say.

Fitted on validation, evaluated on the 10,000-image test set:

| Variant | Acc | ECE-w15 | ECE-m15 | ECE-L2 | classwise-ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw KD | 86.00% | 4.990% | 4.874% | 6.373% | 1.111% | 0.45559 | 0.21067 |
| Global TS, NLL fit (T=1.46) | 86.00% | 1.101% | 0.889% | 1.935% | 0.498% | 0.41455 | 0.20405 |
| Global TS, ECE fit (T=1.49) | 86.00% | **0.994%** | 0.895% | 1.897% | 0.520% | 0.41445 | 0.20406 |
| Per-class TS, NLL fit | 86.00% | 1.048% | 1.019% | 1.806% | 0.480% | 0.41421 | 0.20387 |
| Per-class TS, ECE fit | 86.00% | 1.170% | 1.098% | 1.803% | 0.520% | 0.41666 | 0.20398 |
| Vector scaling (λ=0.1) | 86.19% | 1.417% | 1.111% | 2.399% | **0.432%** | 0.41316 | 0.20328 |
| Matrix scaling (λ=10) | 86.13% | 1.176% | 0.924% | 1.921% | 0.453% | **0.41266** | 0.20309 |
| Dirichlet ODIR (λ=10) | 86.08% | 1.140% | 0.907% | **1.813%** | 0.474% | 0.41377 | **0.20307** |

Top-label histogram binning is reported separately because it emits a calibrated
score rather than a distribution — the calibrated confidence can fall below
another class's raw probability, so no probability vector should be fabricated
from it:

| Variant | Acc | ECE-w15 | ECE-m15 | ECE-L2 |
|---|---:|---:|---:|---:|
| M2B + HB, 50 points/bin | 86.00% | 1.859% | 1.856% | 3.061% |
| M2B + HB, 100 points/bin | 86.00% | 1.733% | 1.999% | 2.259% |
| TS (1.46) then M2B + HB | 86.00% | 1.920% | 1.509% | 3.079% |

Ranking is essentially untouched by temperature scaling, which matters later:

| | AURC ↓ | Selective accuracy @80% | @90% |
|---|---:|---:|---:|
| Raw | 2.979% | 93.84% | 90.31% |
| TS (T=1.46) | 2.964% | 93.77% | 90.47% |

Spearman rank correlation between raw and scaled confidence is **0.9985**.

## 3. The measurement floor

Expected calibration error is a biased estimator: a *perfectly* calibrated model
still registers a nonzero value from finite-sample binning. Simulating correctness
draws from the calibrated confidences themselves gives the value an ideal
calibrator would score on this test set.

| Metric | Observed after global TS | Null (perfectly calibrated) | Percentile |
|---|---:|---:|---:|
| Top-label ECE, 15 equal-width bins | 1.101% | 0.772% [0.454%, 1.155%] | 95.6th |
| Class-wise ECE | 0.498% | 0.416% [0.364%, 0.473%] | 99.8th |

Roughly 0.77 of the 1.10 percentage points is binning noise. Class-wise ECE is
the more informative metric: it is unambiguously above its null where top-label
ECE is borderline.

## 4. The oracle bound

Each calibrator family was refitted **on the test set** — deliberately cheating —
to bound what any map in that family could ever achieve. These are diagnostics
for planning, never results, and must not be quoted as performance.

| Oracle (fitted on test) | Acc | ECE-w15 | classwise-ECE | NLL |
|---|---:|---:|---:|---:|
| Oracle global TS | 86.00% | 0.994% | 0.520% | 0.41445 |
| Oracle per-class TS | 86.00% | 1.000% | 0.453% | 0.41352 |
| Oracle vector scaling | 86.04% | 0.909% | 0.399% | 0.41095 |
| Oracle matrix scaling | 86.30% | **0.694%** | 0.375% | 0.40266 |
| Oracle Dirichlet | 86.40% | 1.086% | **0.319%** | 0.40312 |
| *Noise floor* | — | *0.772%* | *0.416%* | — |

Two conclusions follow.

- **Top-label ECE is exhausted.** The best oracle, 0.694%, falls *below* the
  0.772% noise floor. With perfect test knowledge, no logit-affine map yields a
  top-label ECE distinguishable from a perfectly calibrated model.
- **Class-wise ECE is effectively closed too.** Honest vector scaling reaches
  0.432%, inside the null band [0.364%, 0.473%]. Twenty parameters fitted on
  5,000 images already bring class-wise calibration to statistical
  indistinguishability from perfect.

## 5. Where the residual structure is

Global temperature scaling drives the *aggregate* confidence gap to near zero
while leaving substantial class-conditional error that cancels out. After
T = 1.46, grouped by predicted class:

| Predicted class | n | Mean confidence | Accuracy | Gap | Within-class ECE |
|---|---:|---:|---:|---:|---:|
| dog | 1053 | 81.20% | 78.35% | **+2.85%** | 3.228% |
| airplane | 1058 | 85.43% | 83.65% | +1.78% | 2.461% |
| ship | 1011 | 91.72% | 90.70% | +1.02% | 1.936% |
| frog | 1003 | 88.98% | 88.24% | +0.75% | 2.053% |
| truck | 1002 | 91.78% | 91.62% | +0.17% | 1.970% |
| automobile | 1003 | 94.17% | 94.32% | −0.15% | 1.882% |
| bird | 984 | 80.70% | 81.20% | −0.50% | 3.061% |
| deer | 1017 | 84.25% | 84.76% | −0.51% | 2.162% |
| cat | 904 | 73.59% | 76.11% | −2.52% | 3.192% |
| horse | 965 | 87.68% | 90.67% | **−2.99%** | 3.057% |

Per-class marginal calibration error concentrates on the same classes: cat
0.913% and dog 0.935%, against 0.278%–0.541% for the rest.

Per-class temperatures fitted by validation NLL track these gaps sensibly — dog
1.65 (most overconfident, softened most), automobile 1.29 (already calibrated) —
yet the paired test improvement is inconclusive:

> per-class TS − global TS, top-label ECE: **−0.053 pp, 95% CI [−0.394, +0.322]**

The ECE-fitted variant is worse still (1.170%) with erratic temperatures
(automobile 1.00, bird 1.57): 15 bins fitted to ~500 samples per class overfits.
**The binding constraint is calibration-set size, not calibrator flexibility.**

The confusion structure and the miscalibration structure are the same structure.
Symmetric confusion counts on test (1,400 total errors):

| Pair | Count |
|---|---:|
| cat ↔ dog | 209 |
| airplane ↔ ship | 85 |
| bird ↔ cat | 78 |
| cat ↔ frog | 77 |
| bird ↔ deer | 77 |
| airplane ↔ bird | 74 |
| bird ↔ frog | 69 |
| automobile ↔ truck | 67 |

The top three pairs account for 26.6% of all errors. cat↔dog alone is 15%.

## 6. Why the next steps need different endpoints

Temperature scaling is a near-monotone transform of confidence, so it cannot
change which predictions the model ranks as most trustworthy — AURC moves from
2.979% to 2.964%, and selective accuracy is flat. Whatever post-hoc scaling does
to ECE, it leaves the decision-relevant ordering alone.

This is the argument for training-time methods, and also the trap to avoid:
scoring them on in-distribution top-label ECE guarantees another inconclusive
result, for the same reason the
[MMCE intervention](calibration-intervention.md) returned −0.13 pp with every
paired interval spanning zero. The measurement cannot resolve anything below
about 0.8%.

Recommended endpoints instead:

1. **Ranking quality** — AURC, risk–coverage curves, selective accuracy at fixed
   coverage. Post-hoc scaling is flat here by construction; a method that changes
   the learned representation can move it.
2. **Calibration under distribution shift** — CIFAR-10-C-style corruptions
   (gaussian noise, blur, contrast, pixelate, JPEG) generated locally, requiring
   no download and preserving the project's offline discipline. A temperature
   fitted on clean validation degrades under shift.
3. **The no-calibration-set setting.** Post-hoc TS *costs* a 5,000-image
   calibration split. A training-time method does not, so it can use all 50,000
   images. Comparing `45k train + TS` against `50k train + train-time calibration`
   is a fair, equal-data head-to-head with real deployment meaning, and gives the
   training-time arm nine times more data for its calibration signal than TS
   ever had.

## 7. Pairwise structure: the one signal well above its floor

Sections 3–5 leave a puzzle. Top-label and class-wise ECE are both at their
floors, yet the per-class gaps in §5 are large. The resolution is that the
residual error is *pairwise*, and neither aggregate metric can see it.

Decomposing the 10-way problem into its 45 binary sub-problems — for each pair
`(i, j)`, restricting to images whose true label is `i` or `j` and scoring
`p_i / (p_i + p_j)` — gives, on the test split:

| | Mean pairwise binary ECE | Null (mean over pairs) | Worst pair |
|---|---:|---:|---:|
| Raw KD | 1.352% | 0.686% | **cat vs dog 5.377%** |
| Global TS (T = 1.46) | 1.014% | 0.865% | **cat vs dog 4.098%** |

**cat/dog sits at 1.9x its own noise floor — 4.098% observed against a 2.184%
null [1.415%, 3.073%], above all 2,000 simulated draws — and a single global
temperature barely touches it** (5.38% → 4.10%). One scalar cannot fix 45
sub-problems.

A caution that applies to this table and cost us a correction: the 0.865% column
is the null *averaged over all 45 pairs*, and must not be used as the floor for
any individual pair. Per-pair floors range from 0.252% to 2.256% — a 3x spread —
because the floor depends on that pair's confidence distribution, and harder
pairs with mid-range confidences have substantially higher floors. An earlier
version of this document compared cat/dog's 4.098% against the pooled 0.865%
and reported "five times its floor". The correct comparison is against cat/dog's
own null, giving 1.9x. The effect is real; the magnitude was overstated.

The worst offenders after scaling are all animal pairs: cat/dog 4.098%,
cat/frog 2.088%, dog/horse 1.844%, cat/deer 1.840%, bird/frog 1.631%.

This is the only quantity in the investigation that is unambiguously above its
own measurement floor, and it is what motivated the training-time work in §8 and
the architectural work in §9. Note that it is above its floor but not by the
margin first claimed, which weakens — without eliminating — the case for having
pursued §8 at all.

Two supporting diagnostics on the same checkpoint:

- **Routing is feasible at the coarse level.** A vehicle/animal split read off
  the existing model's own summed probabilities is 97.52% accurate. Routing into
  a dedicated cat/dog group is only 85.65% accurate, because separating cat and
  dog from the other animals *is* the hard problem.
- **The discriminative information is already present.** Binary cat-vs-dog AUC
  from the shared representation is **0.9373**. A dedicated head has little
  discrimination left to add; what is broken is the calibration of that decision
  (4.098% ECE), not the decision itself.

Confusion structure on test (1,400 total errors): cat↔dog 209, airplane↔ship 85,
bird↔cat 78, cat↔frog 77, bird↔deer 77. The top three pairs are 26.6% of all
errors, cat↔dog alone 15%. The confusion structure and the miscalibration
structure are the same structure.

## 8. What CPC actually did — a closed negative result

CPC was implemented ([`kd/cpc.py`](../kd/cpc.py)), run as a matched, protocol-frozen,
three-seed study ([`kd/cpc_study.py`](../kd/cpc_study.py),
[`runs/cifar10-cpc/report.md`](../runs/cifar10-cpc/report.md)), then swept over
weights and warmup ([`scripts/cpc_sweep.py`](../scripts/cpc_sweep.py)). It does not
work here.

**The three-seed study missed its predeclared rule.** The rule was "mean AURC
decreases and mean accuracy drops by at most 0.5 pp". AURC went the wrong way
(+0.000443). Everything else was noise-dominated: accuracy −0.073 pp
[−0.390, +0.250], AURC [−0.00041, +0.00130]. Class-wise ECE was unmoved:
0.507% (KD+TS) vs 0.508% (KD+CPC+TS), paired change +0.00075 pp.

**A five-configuration sweep found no setting that helps.** Single seed 42,
validation only, temperature fitted per condition:

| Condition | Acc | AURC | classwise ECE | cat/dog ECE |
|---|---:|---:|---:|---:|
| Control (no CPC) | 85.94% | **0.02649** | 0.635% | **4.100%** |
| CPC 0.1/0.1, warmup 0 (the study's setting) | 85.68% | 0.02745 | 0.649% | 4.000% |
| exclusion-only 0.3, warmup 0 | 85.74% | 0.02869 | **0.623%** | 5.213% |
| exclusion-only 0.3, warmup 8 | **86.32%** | 0.02768 | 0.642% | 4.728% |
| disc 0.05 / excl 0.15, warmup 8 | 85.74% | 0.02754 | 0.634% | 4.897% |
| disc 0.2 / excl 0.1, warmup 8 | 85.82% | 0.02858 | 0.670% | 4.533% |

On the best-powered pairwise endpoint — the mean over all 45 pairs, which uses
every validation image rather than the 1,000 cat/dog ones — **all five
configurations are worse than the plain control**:

| Condition | Mean pairwise ECE (45 pairs) | Δ vs control, 95% CI |
|---|---:|---|
| Control | **1.281%** | — |
| exclusion-only 0.3, warmup 8 | 1.295% | +0.014 pp [−0.111, +0.153] |
| CPC 0.1/0.1, warmup 0 | 1.299% | +0.018 pp [−0.076, +0.182] |
| exclusion-only 0.3, warmup 0 | 1.338% | +0.057 pp [−0.052, +0.202] |
| disc 0.2 / excl 0.1, warmup 8 | 1.339% | +0.058 pp [−0.061, +0.198] |
| disc 0.05 / excl 0.15, warmup 8 | 1.389% | +0.108 pp [−0.028, +0.223] |

No interval excludes zero, but the direction is unanimous across a 6× range of
discrimination weight, a 3× range of exclusion weight, and with/without warmup.
AURC agrees: 5/5 worse than control, matching the three-seed test result.

**Two things worth carrying forward.**

*Warmup works.* Holding weights fixed at exclusion 0.3, ramping over 8 epochs
instead of full strength from epoch 1 gave 86.32% vs 85.74% accuracy and 0.02768
vs 0.02869 AURC. `CPCConfig.warmup_epochs` is implemented, tested and defaults to
0, so the frozen `runs/cifar10-cpc` study reproduces bit-for-bit.

*Single-pair endpoints are not viable on CIFAR-10.* The cat/dog subset is 1,000
validation and 2,000 test images. Paired bootstrap CI half-widths on its ECE are
±1.3 to ±2.5 pp, **wider than the entire spread between the six conditions**
(1.2 pp). Resolving a 0.5 pp change would need roughly 9× more data than the
dataset contains. Only the all-45-pairs aggregate has usable power here.

A caution recorded for future work: point estimates in the sweep initially
appeared to show exclusion-heavy settings degrading cat/dog calibration, and a
mechanistic story was constructed around the pair mask in
`PairwiseCalibrationLoss.components` (which does exclude every pair touching the
true label, so exclusion structurally cannot see cat-vs-dog on cat images). The
code fact is true; the empirical claim was not supported once intervals were
computed. Compute the intervals before proposing the mechanism.

## 9. The coarse-routed 2-expert tree: assessment before building

The remaining untested member of the decomposition family is a hard-routed tree:
a vehicle/animal router, then a 4-class vehicle expert and a 6-class animal
expert. §7 shows the router would be 97.52% accurate. **The evidence does not
support building it, and a no-training feasibility calculation explains why.**

### The arithmetic

Routing errors are unrecoverable, so tree accuracy ≈ routing accuracy × expert
accuracy within the correct group. Using the existing model restricted to each
group as a conservative stand-in for a dedicated expert:

| Quantity | Value |
|---|---:|
| Flat 10-way accuracy | 86.00% |
| Coarse router accuracy | 97.52% |
| Oracle ceiling (perfect experts) | 97.52% |
| Expert accuracy needed merely to tie flat | **88.19%** |
| Flat-restricted vehicle expert (4 classes, n=4000) | 93.50% |
| Flat-restricted animal expert (6 classes, n=6000) | 84.05% |
| Weighted flat-restricted expert accuracy | 87.83% |
| **Projected tree accuracy** | **85.65%** vs 86.00% flat |

The tree starts **0.35 pp behind**, and dedicated experts must beat the flat
model's own within-group discrimination by +0.36 pp just to break even. That is
not impossible — a 6-class expert has more capacity per class — but it is a far
narrower margin than the "97.5% routing ceiling vs 86% flat" framing suggests.
Routing accuracy is a ceiling only under perfect experts.

An earlier note in this project described that ceiling as though it were
achievable headroom. It is not, and the corrected arithmetic above supersedes it.

### What the literature says

The evidence is mostly discouraging, and it targets this exact architecture.

- Valmadre's multi-operating-point evaluation finds that **top-down classifiers
  are dominated by a naive flat softmax classifier across the entire operating
  range** [1]. This is the closest published analogue to the proposed design and
  it is a direct negative.
- CALM-CXR builds a two-stage hierarchical protocol with **stage-specific
  temperature scaling and coherent probability composition** — essentially the
  calibration-aware version of this plan — and reports that against a matched
  flat baseline, "paired comparisons did not establish consistent overall
  superiority of the hierarchical organization" [2].
- Inter-level error propagation is described as "a crucial problem" in top-down
  hierarchical classification [3], and multiple papers exist solely to mitigate
  it [4, 5, 6].
- Naive application of existing calibration methods to cascade systems
  "sometimes performs worse"; making cascades work needed a purpose-built
  objective [7]. Errors compound across stages, and independently calibrating
  each stage gives no joint guarantee [8].

Supporting evidence exists but is narrower. Expert calibration is sufficient to
calibrate the whole model under a broad class of distribution shifts **in
hard-routed models**, though insufficient in soft-routed ones [9] — which
constrains the design but does not predict a gain. MoEs can yield more reliable
uncertainty than ensembles under OOD data, with more experts helping [10], and
MoE theory shows a router can learn cluster-center features that split a problem
into sub-problems individual experts solve [11]. None of this is in-distribution
calibration on a 10-class benchmark.

### Recommendation

**Do not build the full tree as a calibration intervention.** The arithmetic says
it starts behind on accuracy; the closest published evaluations say top-down
loses to flat softmax; and §8 has already shown that this project's measurement
resolution cannot detect the effect sizes in play.

If it is built anyway, do it as a staged experiment with a cheap kill-gate:

| Stage | Work | Cost | Kill condition |
|---|---|---|---|
| 1 | Train the two experts only; reuse the flat model as router | ~10 min | Weighted expert accuracy < 88.19% → stop, the tree cannot tie flat |
| 2 | Compose, recalibrate **after** aggregation, evaluate on validation | none | Mean 45-pair ECE not below control's 1.281% → stop |
| 3 | Only then, 3 seeds and a protocol-frozen test evaluation | ~1 hr | — |

Stage 1 answers the question for ten minutes of compute and requires no new
study machinery. Route hard, never soft [9]; recalibrate after aggregation, not
before; and expect the augmentation interaction that already applies to this
pipeline's random crop and horizontal flip.

## 10. Where this leaves the programme

Three families have now been tested against this checkpoint:

| Family | Outcome |
|---|---|
| Post-hoc scaling (TS, per-class TS, M2B+HB, vector/matrix/Dirichlet) | Exhausted — both aggregate metrics at their floors, oracle-confirmed |
| Training-time pairwise constraints (CPC, 5 configurations) | No detectable benefit; consistent small AURC harm |
| Coarse-routed expert tree | Not built; arithmetic and literature both negative |

The original finding stands unmoved: **cat/dog pairwise ECE of ~4.1% against a
its own ~2.18% null is real, though smaller than first reported, and nothing
tried has shifted it.**

Two defensible positions from here:

1. **Conclude the thread.** "Temperature scaling is sufficient on this model, and
   no intervention tested beats it" is a legitimate result, unusually well
   bounded by the oracle and noise-floor analysis in §3–§4.
2. **Change the setting, not the method.** The binding constraint is statistical
   power, not hyperparameters. Confusable-pair calibration needs a dataset with
   more images per pair, or a task where the pair structure carries more of the
   error mass than 15%.

Option 2 was taken. **§11 records the result: it worked.** The balanced-CIFAR-10
conclusions in §1–§9 stand unchanged, but they are conclusions about a setting
whose effects were too small to measure, not about calibration generally.

## 11. Changing the setting: long-tailed CIFAR-10

Section 10 concluded that the binding constraint was statistical power, not
method choice, and that the productive move was to change the setting. That was
done. Under an exponential class imbalance every blocker above disappears.

`data.imbalance_factor` ([`config.py`](../kd/config.py)) applies an exponential
profile via `long_tailed_subset` ([`data.py`](../kd/data.py)) to the training and
validation splits. The official test split is never resampled, so it stays
balanced at 1,000 images per class and every metric keeps the power it had.
A factor of exactly 1.0 returns the indices unchanged and adds no provenance
keys, so the completed balanced studies still verify against their saved
`data.json` byte-for-byte.

At factor 100 the training split is 11,167 images (4,500 head to 45 tail) and
validation is 1,242 (500 head to **5 tail**). Epochs are raised to 81 so total
sample exposure matches the balanced 45,000 x 20 studies.

### The effect is 28x its floor, where it was 6x before

| | Balanced (§2–§4) | Long-tailed, factor 100 |
|---|---:|---:|
| Uncalibrated ECE | 4.99% | **22.62%** |
| Noise floor | 0.77% | ~0.8% |
| ECE after global TS | 1.10% (78% cut) | **13.53%** (40% cut) |
| Per-class confidence-gap spread | ±3 pp | **69.9 pp** |

### The miscalibration is exactly frequency-ordered

Spearman rank correlation between class train count and confidence gap is
**+1.0000**, Pearson r against log count **+0.99**, on both the supervised and
KD arms. Head classes are overconfident, tail classes underconfident, monotonically:

| Class | Train n | Raw gap | After global TS |
|---|---:|---:|---:|
| airplane | 4500 | +43.36% | +35.87% |
| automobile | 2698 | +32.33% | +27.71% |
| bird | 1617 | +30.79% | +20.93% |
| cat | 969 | +27.07% | +15.17% |
| deer | 581 | +15.51% | +4.67% |
| dog | 348 | +5.67% | −4.00% |
| frog | 209 | −7.61% | −17.93% |
| horse | 125 | −9.86% | −20.85% |
| ship | 75 | −17.19% | −27.23% |
| truck | 45 | −22.87% | −34.04% |

A single temperature shifts every class the same direction, so it buys reduced
head overconfidence at the cost of worse tail underconfidence (−22.87% → −34.04%).
This is the §5 pathology at twenty times the magnitude, and here it is
structural rather than sub-noise.

### Two post-hoc corrections, single seed 42

| Arm | Method | Acc | ECE | classwise ECE | AURC | Gap spread |
|---|---|---:|---:|---:|---:|---:|
| supervised | TS, plain NLL (T=1.61) | 62.28% | 13.530% | 4.934% | 0.19864 | 69.9 pp |
| supervised | TS, **balanced** NLL (T=2.67) | 62.28% | 3.119% | 4.935% | 0.20013 | 70.2 pp |
| supervised | **logit adjustment + TS** (τ=1.58, T=1.79) | **74.62%** | 3.299% | **0.924%** | **0.09123** | **8.1 pp** |
| KD | TS, plain NLL (T=1.85) | 64.06% | 13.465% | 4.715% | 0.18181 | 65.0 pp |
| KD | TS, **balanced** NLL (T=2.79) | 64.06% | 3.800% | 4.563% | 0.18283 | 65.7 pp |
| KD | **logit adjustment + TS** (τ=1.69, T=1.87) | **75.94%** | **0.967%** | **0.936%** | **0.08296** | **12.2 pp** |

Three findings.

**Fitting a temperature by unweighted NLL on a skewed calibration set is close to
a latent bug.** Weighting validation samples by `1 / n_class` so each class counts
equally, with no other change, moves the fitted temperature from 1.61 to 2.67 and
cuts ECE from 13.5% to 3.1%. Same one-parameter method, same model.

**Logit adjustment is the effective correction.** Subtracting `tau * log(prior)`
before scaling gives **+12.3 pp accuracy**, cuts class-wise ECE by 81%, halves
AURC, and collapses the 70 pp gap spread to 8 pp — on two parameters, post-hoc.
Unlike temperature scaling it changes predictions, so it sits outside the
argmax-preserving family that `posthoc_study` asserts; that assertion needs the
same relaxation §4's affine family needs.

**AURC finally moves.** §6 established that post-hoc scaling cannot change
ranking (Spearman 0.9985 between raw and scaled confidence). Logit adjustment
halves AURC because it is not a monotone transform of confidence — it reorders
across classes. The barrier in §6 is a property of *global* scaling, not of
post-hoc methods generally.

### What this settles about the original hypothesis

Per-class treatment does beat a single global parameter — decisively, once the
setting contains per-class structure worth correcting. But naive per-class
fitting fails here: with 2–7 validation images for tail classes, independent
per-class temperature scaling produced degenerate fits (ship T=0.01 at the grid
boundary, horse T=0.42) and worsened top-label ECE relative to global TS. What
works is a *structured, low-parameter* correction that borrows strength across
classes — two parameters, not ten.

That is the decomposition hypothesis in its defensible form: decompose the
*correction* along a known structural axis, do not fit each class independently.

### Three-seed, two-factor confirmation

[`scripts/lt_multiseed.py`](../scripts/lt_multiseed.py) repeats the above over
seeds 42/43/44 at imbalance factors 100 and 10, with one teacher per factor
(seed 41) and epochs scaled per factor to hold sample exposure roughly constant.
Mean ± sd over seeds; percentages except AURC.

| Factor | Arm | Method | Acc | ECE | classwise ECE | AURC | Gap spread |
|---:|---|---|---:|---:|---:|---:|---:|
| 100 | supervised | uncalibrated | 63.01±1.23 | 21.529±0.298 | 5.541±0.146 | 0.19197±0.01045 | 62.8±1.4 |
| 100 | supervised | TS (plain NLL) | 63.01±1.23 | 12.666±1.048 | 4.854±0.297 | 0.19142±0.01030 | 66.0±1.1 |
| 100 | supervised | TS (balanced NLL) | 63.01±1.23 | 3.629±0.603 | 4.682±0.182 | 0.19211±0.00975 | 66.6±1.3 |
| 100 | supervised | **logit adjustment + TS** | **74.01±1.01** | **2.764±0.767** | **0.984±0.079** | **0.09349±0.00615** | **14.5±1.0** |
| 100 | kd | uncalibrated | 64.74±0.28 | 23.848±0.238 | 5.663±0.061 | 0.17344±0.00352 | 58.7±1.9 |
| 100 | kd | TS (plain NLL) | 64.74±0.28 | 13.587±0.313 | 4.778±0.110 | 0.17278±0.00371 | 62.7±2.3 |
| 100 | kd | TS (balanced NLL) | 64.74±0.28 | 3.927±0.290 | 4.529±0.089 | 0.17353±0.00401 | 65.0±1.0 |
| 100 | kd | **logit adjustment + TS** | **76.03±0.33** | **1.566±0.504** | **1.083±0.103** | **0.08146±0.00137** | **14.8±1.3** |
| 10 | supervised | uncalibrated | 79.18±0.61 | 7.368±0.634 | 2.103±0.139 | 0.06220±0.00555 | 25.7±0.2 |
| 10 | supervised | TS (plain NLL) | 79.18±0.61 | 2.797±0.781 | 1.815±0.201 | 0.06200±0.00560 | 25.8±0.7 |
| 10 | supervised | TS (balanced NLL) | 79.18±0.61 | 1.384±0.672 | 1.809±0.182 | 0.06200±0.00559 | 25.7±0.9 |
| 10 | supervised | **logit adjustment + TS** | **81.60±0.78** | **1.044±0.472** | **0.637±0.153** | **0.04849±0.00383** | **8.4±2.5** |
| 10 | kd | uncalibrated | 80.18±0.48 | 9.992±0.449 | 2.435±0.105 | 0.05727±0.00235 | 25.8±2.4 |
| 10 | kd | TS (plain NLL) | 80.18±0.48 | 2.635±0.361 | 1.876±0.070 | 0.05677±0.00234 | 26.7±2.0 |
| 10 | kd | TS (balanced NLL) | 80.18±0.48 | 1.137±0.262 | 1.881±0.051 | 0.05677±0.00233 | 26.8±1.8 |
| 10 | kd | **logit adjustment + TS** | **82.88±0.24** | **0.867±0.273** | **0.580±0.052** | **0.04183±0.00095** | **7.1±2.1** |

Four things this settles.

**Every effect is many standard deviations wide.** At factor 100 the KD arm's
plain-NLL and balanced-NLL temperatures differ by 9.7 pp of ECE with a seed
spread of ~0.3 pp. Contrast §8, where every CPC comparison straddled zero. This
is the first setting in the project where the interventions are separable from
noise at three seeds.

**The plain-NLL failure is structural, not a small-sample artifact.** Factor 10's
validation tail holds 50 images rather than 5, yet plain-NLL temperature scaling
still leaves roughly twice the ECE of the balanced-NLL fit (2.80 vs 1.38
supervised; 2.64 vs 1.14 KD). The defect is the skewed fitting objective itself.

**Severity scales monotonically with imbalance.** Uncalibrated ECE runs 21.5% at
factor 100 against 7.4% at factor 10, gap spread 62.8 pp against 25.7 pp. The
setting behaves as a controlled dial, which is what makes it useful.

**KD buys accuracy and costs calibration.** KD is more accurate than supervised at
both factors (+1.7 pp at 100, +1.0 pp at 10) but *more* miscalibrated before
correction (23.8% vs 21.5%; 10.0% vs 7.4%). After logit adjustment it is both the
most accurate and the best calibrated arm — 0.867% ECE at factor 10, at the
measurement floor.

Residual headroom is factor-dependent. At factor 10 logit adjustment reaches the
noise floor and the problem is essentially solved post-hoc; at factor 100 it
leaves 1.6–2.8% ECE against a ~0.8% floor, so training-time methods still have
something to attack there.

### Status and limitations

- Three student seeds, two imbalance factors, one teacher per factor. The
  single-seed numbers above replicate within about one standard deviation.
- Exploratory. Reuses the official test set earlier studies examined; no protocol
  freeze. A confirmatory study would need the `cpc_study.py` treatment.
- Accuracy is far below the balanced studies (62–64% vs 86%) because the training
  split is 11,167 images. Comparisons are within this setting only.
- Validation is as skewed as training, which is the realistic setting and the
  reason plain-NLL fitting fails. Checkpoint selection on validation accuracy is
  therefore head-biased.
- `tau` is fitted rather than set to the theoretical 1.0; fitted values were
  1.58–1.69.

## 12. The floor survey: does any of this generalise?

Sections 3-4 rest on one checkpoint. `scripts/floor_survey.py` repeats the test
on every trained checkpoint in `runs/` — 30 in total, spanning three settings,
four training objectives (supervised, KD, DKD, teacher), three seeds and two
imbalance factors. Results in `runs/floor_survey.json`.

### The decision rule needed correcting first

The original test asked only whether the family oracle reached the floor, and
called a family "exhausted" when it did. **That is wrong.** It marks as exhausted
both a checkpoint that is already calibrated and a long-tailed checkpoint at
13.9% ECE whose error the family could remove entirely — opposite situations
demanding opposite actions. The first run of the survey returned "30/30
exhausted", which is how the error surfaced.

The rule needs both axes:

| observed vs null | oracle vs null | Verdict | Action |
|---|---|---|---|
| at floor | at floor | **no signal** | do not run the experiment |
| above floor | at floor | **family suffices** | fit the existing family properly |
| above floor | above floor | **family insufficient** | a new method is justified |

### Results

| Setting | n | No signal | Family suffices | Family insufficient | Median observed ECE |
|---|---:|---:|---:|---:|---:|
| Balanced CIFAR-10 | 10 | **8** | 2 | 0 | 1.034% |
| CPC study | 6 | **5** | 1 | 0 | 1.087% |
| Long-tailed | 14 | 0 | **14** | 0 | 7.662% |

**The §3-§4 conclusion generalises.** Eight of ten balanced checkpoints have
observed ECE inside the null band after temperature scaling, across supervised,
KD, DKD and the teacher. The two exceptions clear the band marginally
(`dkd_seed44` 1.236% vs 1.188%; `kd_seed43` 1.288% vs 1.134%).

**Every long-tailed checkpoint is fixable by the affine family.** Observed
2.2-13.9% against null upper bounds of 1.2-1.5%, oracles at 0.58-1.02%. The gap
between honest performance (13.5% under plain-NLL fitting) and the oracle is
therefore a **fitting problem, not a family problem** — precisely what §11 found
when a class-balanced objective and logit adjustment recovered most of it. The
diagnostic and the experiment agree.

**No checkpoint landed in "family insufficient."** Across every setting tested,
nothing justified inventing a method. That is the retrospective explanation of
§8: CPC was a new method built for a regime that never called for one, and the
two-axis rule would have said so before any training run.

## 13. Protocol requirements

Any further work must keep the discipline the existing studies established:

- Predeclare the protocol and freeze source hashes before training, as
  [`cifar_study.py`](../kd/cifar_study.py),
  [`calibration_study.py`](../kd/calibration_study.py) and
  [`cpc_study.py`](../kd/cpc_study.py) already do.
- Freeze checkpoint selection before the test split is opened. Selection stays on
  validation accuracy; do not select checkpoints on ECE.
- Declare the primary endpoint in advance, and check its power first. Given §3
  and §8, neither top-label ECE nor any single-pair metric is admissible as a
  primary endpoint on CIFAR-10.
- Report the null/noise reference next to every calibration number, and the
  interval next to every difference, so sub-noise effects are not read as
  findings.
- Vector, matrix and Dirichlet calibration change `argmax`, so the
  "every prediction preserved" assertion in `posthoc_study` must be relaxed for
  that family rather than silently removed.

## 14. Limitations

- Sections 2–5 and 7 are one checkpoint (seed 42) on one fixed split. §8's
  three-seed study is the only multi-seed evidence here; the sweep is single-seed.
- The official test set has been examined repeatedly across the KD, calibration,
  adaptive-focal, post-hoc and CPC studies. All of this is exploratory follow-up,
  not independent confirmation. The §8 sweep and §9 arithmetic are deliberately
  validation-only for that reason.
- Noise-floor and conditional-null simulations condition on observed confidences
  and assume independent correctness draws. They bound binning noise, not model
  uncertainty, and are not confidence intervals on true ECE.
- Oracle rows are fitted on test and bound a family's best case. They are not
  achievable and must never be quoted as results.
- The §9 projection uses the flat model restricted to each group as a stand-in
  for a dedicated expert. A trained expert may do better; the projection is a
  reference point, not a prediction.
- Bootstrap intervals resample test or validation images. They do not capture
  training-seed or calibration-fit variability.
- **Everything here is measured with ECE, which is not a proper scoring rule.**
  The "no signal" verdicts are statements about what ECE can resolve, not about
  whether better calibration is achievable. Measured on the seed-42 checkpoint,
  Brier ranks matrix and Dirichlet scaling *better* than temperature scaling
  (−0.00096 [−0.00191, −0.00000] and −0.00098 [−0.00194, −0.00001]) where
  top-label ECE ranks them *worse* (+0.076 and +0.039 pp): a proper score
  resolves a difference ECE calls noise. Two things limit how far that goes —
  those variants change argmax and gain 0.08-0.19 pp accuracy, and Brier
  conflates calibration with discrimination. On the argmax-preserving comparison
  (per-class TS vs global TS, accuracy pinned at 86.00%) Brier also spans zero
  (−0.000179 [−0.000436, +0.000078]), and the §8 CPC null is likewise robust to
  the metric (pooled ΔBrier +0.00064 [−0.00226, +0.00354]). Reporting a proper
  score as primary, with a calibration-refinement decomposition to isolate the
  calibration term, would put these conclusions on firmer ground.

## 15. References

- Cheng and Vasconcelos, Calibrating Deep Neural Networks by Pairwise Constraints, CVPR 2022. DOI: `10.1109/cvpr52688.2022.01334`
- Gupta and Ramdas, [Top-label calibration and multiclass-to-binary reductions](https://arxiv.org/abs/2107.08353), 2021.
- Kull et al., [Beyond temperature scaling: Dirichlet calibration](https://arxiv.org/abs/1910.12656), NeurIPS 2019.
- Frenkel and Goldberger, Network Calibration by Class-based Temperature Scaling, EUSIPCO 2021. DOI: `10.23919/eusipco54536.2021.9616219`
- Patel et al., [Multi-Class Uncertainty Calibration via Mutual Information Maximization-based Binning](https://arxiv.org/abs/2006.13092), 2020.
- Nixon et al., [Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685), 2019.
- Padhy et al., [Revisiting One-vs-All Classifiers for Predictive Uncertainty and OOD Detection](https://arxiv.org/abs/2007.05134), 2020.
- Rahaman and Thiery, [Uncertainty Quantification and Deep Ensembles](https://arxiv.org/abs/2007.08792), 2020.
- Wen et al., [Combining Ensembles and Data Augmentation can Harm your Calibration](https://arxiv.org/abs/2010.09875), 2020.
- Zhang et al., A Survey on Learning to Reject, Proceedings of the IEEE 2023. DOI: `10.1109/jproc.2023.3238024`

Hierarchical and mixture-of-experts evidence cited in §9:

1. Valmadre, [Hierarchical classification at multiple operating points](https://arxiv.org/abs/2210.10929), 2022.
2. Samarla et al., CALM-CXR: a calibration-aware lung-masked workflow for hierarchical chest X-ray classification, *MethodsX* 2026. DOI: `10.1016/j.mex.2026.104081`
3. Qiu et al., Hierarchical classification based on coarse- to fine-grained knowledge transfer, *Int. J. Approx. Reason.* 2022. DOI: `10.1016/j.ijar.2022.07.002`
4. Guo et al., Hierarchical classification with multi-path selection based on granular computing, *Artificial Intelligence Review* 2020. DOI: `10.1007/s10462-020-09899-2`
5. Wang et al., Hierarchical classification with exponential weighting of multi-granularity paths, *Inf. Sci.* 2024. DOI: `10.1016/j.ins.2024.120715`
6. Wang et al., On the Adversarial Robustness of Hierarchical Classification, IEEE SMC 2024. DOI: `10.1109/smc54092.2024.10831681`
7. Enomoto et al., [Learning to Cascade: Confidence Calibration for Improving the Accuracy and Computational Cost of Cascade Inference Systems](https://arxiv.org/abs/2104.09286), AAAI 2021. DOI: `10.1609/aaai.v35i8.16900`
8. Kotte, [PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees](https://arxiv.org/abs/2605.18812), 2026.
9. Wong et al., [Toward Calibrated Mixture-of-Experts Under Distribution Shift](https://arxiv.org/abs/2606.20544), 2026.
10. Pavlitska et al., Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation, ICCVW 2025. DOI: `10.1109/iccvw69036.2025.00038`
11. Chen et al., [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813), 2022.

## 16. Reproducing this analysis

The §2–§5 and §7 numbers read only saved logit archives and change nothing. The
§8 sweep trains three students (~5 min each on MPS) and touches no test data.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd

# CPC weight/warmup sweep (validation only; reuses the frozen study's baselines)
python scripts/cpc_sweep.py

# The completed three-seed CPC study, re-rendered without training or inference
python -m kd.cpc_reporting --study runs/cifar10-cpc

# Two-axis floor survey over every trained checkpoint (logits cached after the
# first run, so re-runs take minutes rather than an hour)
python scripts/floor_survey.py
```

The post-hoc analysis scripts behind §2–§5 and §7 are not yet in the repository;
those numbers were produced by equivalent throwaway scripts. Landing them
alongside the step-0 metrics would make this document fully reproducible.
