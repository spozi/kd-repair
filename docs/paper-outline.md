# Paper outline: calibration evaluation below the noise floor

Working title options:

1. **Calibration evaluation below the noise floor: how much of a reported ECE improvement is measurable?**
2. Before you run the experiment: measurement floors and oracle bounds for calibration research
3. Expected calibration error has a floor, and we routinely report improvements beneath it

**Thesis.** Expected calibration error is a biased estimator with a non-trivial
floor that depends on sample size, binning scheme and the model's confidence
distribution. A substantial fraction of reported calibration improvements are
smaller than the floor of the estimator used to measure them. Two cheap
diagnostics — a conditional null and a family oracle bound — establish, *before*
an experiment is run, whether the effect being sought is measurable at all.

**Two target versions.** The core (§1–§7, §9–§11) is a self-contained analysis
paper suitable for TMLR, where novelty is not an acceptance criterion and the
existing artifacts largely suffice. Adding §8 (the literature survey) upgrades it
to the pitfalls-and-best-practices genre published by *Patterns*, *Data Mining
and Knowledge Discovery* and *Environmental Science & Technology*. §8 is the
load-bearing new work; everything else is mostly writing up what exists.

Throughout, **[HAVE]** marks material already produced in this repository and
**[NEED]** marks work still to do.

---

## 1. Introduction

- Calibration is now a standard reported quantity, and ECE its standard summary.
- Papers routinely report ECE improvements of a few tenths of a percentage point
  and treat them as findings.
- ECE is a *biased* estimator: a perfectly calibrated model scores strictly
  greater than zero. The bias depends on N, bin count, binning scheme and the
  confidence distribution — none of which are held constant across papers.
- Contribution:
  1. A two-axis decision rule built from two diagnostics — a conditional null
     (is there signal?) and a family oracle bound (can the family capture it?) —
     that says, before any method is compared, whether an experiment is worth
     running at all.
  2. A 30-checkpoint survey applying the rule across three settings, four
     training objectives, three seeds and two imbalance factors, in which **no
     checkpoint warranted a new calibration method**.
  3. Three case studies: a setting with no signal, an experiment run in it
     anyway, and a setting with large resolvable signal (positive control).
  4. **[NEED]** A literature survey estimating how often published ECE deltas
     fall below their own floors.
  5. A checklist for reporting calibration results.
- Explicit non-claim: we do not propose a calibration method, and we do not claim
  ECE should be abandoned. We claim it should be reported against its floor.

## 2. Background and setup

- ECE definitions in use: equal-width vs equal-mass binning, L1 vs L2 norm,
  top-label vs class-wise vs marginal. Nixon et al. established that metric
  choice reorders methods; we build on that.
- Estimator bias: why binning a finite sample yields a positive expected value
  under perfect calibration.
- Relationship to existing formal work: T-Cal casts miscalibration detection as
  hypothesis testing; Posocco et al. compare ECE estimators. **Position this
  paper as the practical, prospective counterpart** — not a new test, but a
  design-time discipline built from simulation any practitioner can run.

## 3. Diagnostic 1 — the conditional null **[HAVE]**

- Procedure: take the model's own predicted confidences, simulate correctness
  draws from them, recompute the metric. This is the value an *ideal* calibrator
  registers on this test set at this N and binning.
- Report observed value, null mean, null central 95%, and the observed
  percentile within the null.
- **State the assumption plainly**: correctness is drawn independently
  conditional on the predicted confidence. This bounds *binning and sampling
  noise*, not model or epistemic uncertainty, and it is not a confidence interval
  on true ECE. Reviewers will press on this; address it in the text, not a
  footnote.
- Reference implementation exists (`kd/calibration_metrics.py`,
  `conditional_calibration_null`).

## 4. Diagnostic 2 — the family oracle bound, and the decision rule **[HAVE]**

- Procedure: refit each calibrator family *on the split it is scored on*,
  deliberately cheating, to upper-bound what any member of that family could
  achieve.
- Purpose: separates "our method is not good enough" from "no method in this
  family can help here."
- Must be labelled unambiguously as a planning diagnostic, never a result.
- Worked example, balanced CIFAR-10 KD student (seed 42):

  | Oracle (fitted on test) | Top-label ECE | class-wise ECE |
  |---|---:|---:|
  | Global temperature scaling | 0.994% | 0.520% |
  | Per-class TS | 0.991% | 0.407% |
  | Vector scaling | 0.909% | 0.399% |
  | Matrix scaling | **0.691%** | 0.374% |
  | Dirichlet (ODIR) | 1.088% | **0.320%** |
  | *Conditional null* | *0.770% [0.450%, 1.130%]* | *0.416%* |

### 4.1 The oracle alone is not a decision rule

An early version of this analysis asked only whether the oracle reached the
floor, and concluded a family was "exhausted" when it did. **That test is wrong,
and the survey below shows why**: it marks as exhausted both a checkpoint that is
already calibrated and a checkpoint at 13.9% ECE whose error the family could
remove entirely. Those call for opposite actions.

The decision needs both axes — is there signal, and can the family capture it:

| observed vs null | oracle vs null | Verdict | Action |
|---|---|---|---|
| at floor | at floor | **no signal** | do not run the experiment |
| above floor | at floor | **family suffices** | fit the existing family properly |
| above floor | above floor | **family insufficient** | a new method is justified |

Only the third verdict warrants inventing a method. We report this correction
explicitly rather than silently, because the single-axis version is the intuitive
one and readers will reach for it.

### 4.2 Survey across 30 checkpoints **[HAVE]**

`scripts/floor_survey.py` applies the rule to every trained checkpoint in the
repository: three settings, four training objectives (supervised, KD, DKD, and a
teacher), three seeds, and two imbalance factors.

| Setting | n | No signal | Family suffices | Family insufficient | Median observed ECE |
|---|---:|---:|---:|---:|---:|
| Balanced CIFAR-10 | 10 | **8** | 2 | 0 | 1.034% |
| CPC study | 6 | **5** | 1 | 0 | 1.087% |
| Long-tailed | 14 | 0 | **14** | 0 | 7.662% |

Three findings, each usable as a section headline.

1. **The single-checkpoint result generalises.** Eight of ten balanced
   checkpoints have observed ECE *inside* the null band after temperature
   scaling. The two exceptions clear it marginally (1.236% vs 1.188%; 1.288% vs
   1.134%). This is not an artifact of one seed or one training objective.
2. **Every long-tailed checkpoint is fixable by the affine family.** Observed
   2.2-13.9% against null upper bounds of 1.2-1.5%, with oracles at 0.58-1.02%.
   The miscalibration is entirely within the family's reach, so the gap between
   honest performance (13.5% under plain-NLL fitting) and the oracle is a
   **fitting problem, not a family problem** — which is exactly what §7 finds
   when a class-balanced objective and logit adjustment recover most of it.
3. **No checkpoint landed in "family insufficient."** Across every setting
   tested, nothing justified a new method. That is the retrospective explanation
   of §6: CPC was a new method built for a regime that never called for one.

The third finding is the paper's strongest single sentence: *across 30
checkpoints spanning three settings, four training objectives, three seeds and
two imbalance factors, not one warranted inventing a calibration method.*

## 5. Case study 1 — a setting with no signal **[HAVE]**

Balanced CIFAR-10, a compact KD student, 10,000-image test split.

- Raw 4.990% → 1.101% after temperature scaling: a 78% reduction that looks
  decisive and is largely unmeasurable. Observed sits at the 95.6th percentile of
  its own null.
- Class-wise ECE 0.498% vs null 0.416% [0.364, 0.473] — the 99.8th percentile.
  **Different metrics have different amounts of headroom in the same setting**;
  the aggregate top-label number is the least informative.
- Two failure modes this produces in practice, both observed here:
  - **Selection by noise.** Focal TS was selected over ordinary TS on a
    validation-NLL margin of **+0.000083**, whose bootstrap 95% interval is
    **[−0.00033, +0.00048]** — five times wider than the margin. An argmin over a
    5,000-point grid is not a decision procedure at these margins.
  - **Headline/artifact divergence.** The reported summary quotes the best
    variant (0.994%) while the saved artifact holds a different one (1.143%).
- What aggregate metrics hide: per-predicted-class confidence gaps of +2.85%
  (dog) to −2.99% (horse) cancel to near zero; pairwise cat-vs-dog ECE is 4.098%
  against its own 2.184% null [1.415%, 3.073%] — 1.9x its floor and above all
  2,000 simulated draws. **Structure can be clearly above its floor while every
  aggregate summary sits at the floor.**
- **A worked instance of the paper's own thesis, worth including as such.** Per-pair
  floors range 0.252%-2.256% across the 45 pairs, because the floor tracks each
  pair's confidence distribution. An earlier draft of this analysis compared
  cat/dog against the *pooled* 0.865% mean null and reported "five times its
  floor"; against its own null it is 1.9x. Pooled floors are not substitutes for
  per-stratum floors, and the paper should say so with this example.

## 6. Case study 2 — running the experiment anyway **[HAVE]**

A training-time pairwise-calibration intervention (CPC), evaluated in a setting
§4.2 classifies as "no signal" for five of its six checkpoints.

- Three-seed protocol-frozen study with a predeclared success rule: missed it.
  Accuracy −0.073 pp [−0.390, +0.250], AURC +0.00044 [−0.00041, +0.00130],
  class-wise ECE change +0.00075 pp. Every interval spans zero.
- Five-configuration weight/schedule sweep spanning a 6× range of one loss term
  and 3× of another: all five worse than control on the best-powered pairwise
  endpoint, none significantly.
- **A power analysis that should have preceded the experiment**: the single-pair
  endpoint has bootstrap CI half-widths of ±1.3 to ±2.5 pp, *wider than the entire
  1.2 pp spread between the six conditions*. Resolving a 0.5 pp change needs
  roughly 9× more data than the dataset contains.
- Include as a cautionary sub-case: an intermediate analysis proposed a
  mechanistic explanation from point estimates, which the intervals then failed to
  support. Compute intervals before proposing mechanisms.
- Also note a metric that post-hoc scaling *structurally* cannot move: AURC goes
  2.979% → 2.964% under TS, Spearman 0.9985 between raw and scaled confidence.
  Choosing an endpoint the intervention cannot affect is a separate, common error.

## 7. Case study 3 — a resolvable setting (positive control) **[HAVE]**

Essential to the paper's credibility: the diagnostics must *pass* things, not
only fail them.

- Long-tailed CIFAR-10 (exponential profile, factors 100 and 10), balanced test
  split, three seeds.
- Uncalibrated ECE 22.6% against a ~0.8% floor — **28× the floor**, versus 6× in
  §5. Effects are many standard deviations wide where §6's were not.
- Miscalibration is perfectly frequency-ordered: Spearman **+1.0000** between
  class train count and confidence gap, head to tail spanning ~66 pp.
- The same diagnostics that killed §5 confirm real effects here, and the
  multi-seed table separates every method cleanly.
- **A practical pitfall worth its own subsection**: fitting a temperature by
  unweighted NLL on a skewed calibration set gives the wrong temperature
  (1.61 vs 2.67) and leaves 13.5% ECE where 3.1% was available — same method,
  same model, only the fitting objective's weighting changed. Confirmed at both
  imbalance factors, so it is not a small-sample artifact.
- Note honestly that the *methods* here (logit adjustment, class-balanced
  objectives) reproduce established long-tailed-calibration results; they appear
  as a validated positive control, not as a contribution.

## 8. Survey: how often are reported deltas below their floor? **[NEED]**

The load-bearing new work, and what distinguishes the *Patterns* version from
the TMLR version.

- **Sampling frame.** Papers reporting ECE on standard image benchmarks
  (CIFAR-10/100, ImageNet) with post-hoc or training-time calibration, drawn from
  a defined venue and date range. Target 50–100 papers.
- **Extraction.** Test-set size, bin count, binning scheme, norm, reported
  baseline and method ECE, the delta, and whether logits/checkpoints are released.
- **Open design problem, flag it early.** The exact null depends on the
  confidence distribution, which papers do not report. Proposed resolution, in
  descending order of rigour:
  1. For papers releasing logits or checkpoints, compute the null exactly.
  2. Otherwise, compute a **reference floor band** by simulating across a family
     of plausible confidence profiles matched to the paper's reported accuracy and
     mean confidence, and report the delta against that band.
  3. Report the fraction of papers that publish too little to determine their own
     floor. **This is itself a finding**, and mirrors the reporting-completeness
     results in the leakage and reproducibility literature.
- **Pre-register the analysis.** The paper argues against post-hoc selection; it
  must not select its own threshold after seeing results. State in advance what
  fraction would count as supporting versus refuting the thesis.
- Deliverable: a table of papers, their floors, their deltas, and the fraction
  falling below — plus released extraction data and code.

## 9. Guidelines and checklist **[HAVE, as raw material]**

Every paper in this genre ends here. Draft from `docs/calibration-roadmap.md` §12.

1. Report the conditional null beside every calibration number, and an interval
   beside every difference.
2. Choose the primary endpoint before running, and check its power first.
3. Prefer metrics with headroom in *your* setting; report at least one
   class-conditional metric alongside any aggregate.
4. Do not select among near-tied candidates by argmin over a fine grid; require a
   minimum margin, or prefer the simpler family on ties.
5. Apply the two-axis rule (§4.1) before investing compute in a new method:
   compute the observed value against its null, *and* the family oracle against
   the same null. Only "family insufficient" justifies a new method.
6. Verify that the endpoint is one the intervention can move at all.
7. Report bin count, binning scheme, norm and test-set size — the minimum needed
   for a reader to reconstruct your floor.
8. State whether the reported number matches the released artifact.

## 10. Limitations

To write plainly, not defensively:

- The conditional null assumes independent correctness draws given confidence. It
  bounds binning and sampling noise only.
- Oracle bounds are family-specific; a richer family could exceed them.
- Case studies use one architecture family and one dataset family at small scale.
  Floors are computed per setting, so the *method* transfers, but the specific
  numbers do not.
- The official test split was reused across several studies in this project;
  every case study is exploratory follow-up, and this must be stated in the paper
  rather than buried.
- The §7 methods reproduce known long-tailed results and are not claimed as novel.
- **[NEED]** Literature-survey coverage will be a sample, not a census, with its
  selection criteria stated.
- **ECE is not a proper scoring rule, and this paper does not argue it should be
  preferred to one.** Ferrer et al. argue calibration metrics should play no role
  in assessing posterior quality; Gruber and Buettner derive proper calibration
  errors with better estimation properties. Our position is narrower: ECE is what
  the field reports, so its floor is worth knowing.
  A concrete instance of the tension, measured on our own seed-42 checkpoint:
  Brier ranks matrix and Dirichlet scaling **better** than temperature scaling
  (ΔBrier −0.00096 [−0.00191, −0.00000] and −0.00098 [−0.00194, −0.00001]),
  while top-label ECE ranks them **worse** (+0.076 and +0.039 pp). Brier resolves
  a difference ECE calls noise. Two caveats keep this from overturning the paper:
  those variants change argmax and gain 0.08-0.19 pp accuracy, and Brier conflates
  calibration with discrimination; on the argmax-preserving comparison
  (per-class TS vs global TS, accuracy pinned at 86.00%) Brier also spans zero
  (−0.000179 [−0.000436, +0.000078]), agreeing with ECE. The honest statement is
  that our "no signal" verdicts are statements about ECE's resolution, not about
  whether better calibration is achievable under a proper score. A reviewer will
  raise this; the paper should raise it first.

## 11. Related work

- Calibration metrics and their pathologies: Nixon et al.; Posocco et al.;
  Vaicenavicius et al.; Lane's metrics review.
- Formal testing: T-Cal (minimax optimal calibration test) — the theoretical
  counterpart; position this work as the design-time, simulation-based practice.
- Calibration surveys: Filho et al.; Dong et al. (class imbalance).
- Benchmarks: CalArena.
- Methodological-pitfalls genre: Lones; Hewamalage et al.; Zhu et al.; Kapoor et
  al. (leakage, REFORMS); Raschka; Ferrer et al.
- Long-tailed calibration, for §7 attribution: Menon et al. (logit adjustment);
  Chen et al.; Hoque et al.; Guo et al.; Obadinma et al.; Zhong et al.

---

## Work plan

| Stage | Work | Effort | Blocking? |
|---|---|---|---|
| A | Land the analysis scripts in `scripts/`, so every number is reproducible | ~1 week | yes |
| B | Extend case studies to CIFAR-100 and a second architecture | ~2 weeks | for the journal version |
| C | Write §1–§7, §9–§11 | ~3 weeks | — |
| D | Survey (§8): frame, extract, compute floors, analyse | ~6 weeks | only for the journal version |
| E | Release code, extraction data, and a floor-calculator utility | ~1 week | recommended |

**TMLR route**: A + C, plus a trimmed B. Roughly 4–6 weeks.
**Patterns / DMKD route**: all stages. Roughly 3 months.

Recommendation: write the TMLR version first — it is mostly writing up existing
artifacts and is publishable without the survey. If it is received well, the
survey extension converts it into the journal tutorial.

## Assets inventory

| Section | Source |
|---|---|
| §3 null, §4 oracle | `kd/calibration_metrics.py`; roadmap §3–§4 |
| §5 exhausted setting | `runs/cifar10-posthoc/seed42/`; roadmap §1–§5, §7 |
| §6 CPC | `runs/cifar10-cpc/`, `scripts/cpc_sweep.py`; roadmap §8 |
| §7 long-tailed | `runs/cifar10-lt/`, `runs/cifar10-lt-multiseed/`; roadmap §11 |
| §9 checklist | roadmap §12 |
| §10 limitations | roadmap §13 |
| §11 related work | roadmap §14 |

The `kd` package's protocol freezing, source hashing and resumable studies are a
secondary asset: they make the artifact release unusually strong for this genre,
and are worth a short paragraph in §10 or an appendix.
