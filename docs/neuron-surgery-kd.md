# Channel-level teacher repair before knowledge distillation

This study adapts AI-Lancet's error-neuron localization to the resources already
available in this project. It is an inspired adaptation, not a reproduction:
CIFAR-10 has no trigger masks or annotations identifying a removable error
region, so each hard image is compared with five nearby, correctly classified
images from the same class.

The pipeline uses knowledge distillation twice:

1. The original teacher preserves logits and stage features while selected
   channel-connected weights are repaired.
2. The selected repaired teacher supervises three fresh compact students with
   the project's existing classical KD objective.

No samples are deleted. The earlier factor-100 deletion arm lost 2.17 percentage
points of balanced test accuracy, so those hard images now provide repair signal.

## Preflight

The predeclared study expects these completed artifacts:

- `runs/cifar10-lt-multiseed/teacher_f100/student.pt`
- `runs/cifar10-lt-multiseed/kd_f100_seed42/student.pt`
- `runs/cifar10-lt-multiseed/kd_f100_seed43/student.pt`
- `runs/cifar10-lt-multiseed/kd_f100_seed44/student.pt`

Validate their configurations and hashes without creating data, models, or output:

```bash
python -m kd neuron-surgery-study \
  --baseline runs/cifar10-lt-multiseed \
  --output runs/cifar10-lt-neuron-surgery \
  --device auto \
  --seeds 42 43 44 \
  --dry-run
```

Remove `--dry-run` to execute the study. A nonempty output directory is accepted
only when it contains the identical frozen protocol. Completed stages are reused
after their identities and content hashes are checked.

## Fixed protocol

- Target classes are labels 5-9: dog, frog, horse, ship, and truck.
- Every teacher mistake is retained, then high-loss correct examples fill a 20%
  quota per class. The current split yields 161 targets.
- Five correctly classified same-class companions are selected by cosine distance
  between normalized pooled `stage3` features.
- `stage2` and `stage3` channels receive differential activation-gradient scores.
- Twenty within-class bootstraps form an equal-class consensus. A channel must
  appear in the top quartile in at least 70% of replicates.
- One-channel zero ablations must improve target margin without reducing balanced
  preservation accuracy by more than 0.5 percentage points.
- Cumulative channel budgets 2, 4, and 8 are repaired at learning rates 0.001 and
  0.005 for 20 epochs.
- Only producing convolution rows, corresponding BatchNorm affine entries, and
  downstream consumer weights can change. BatchNorm buffers and every unselected
  parameter coordinate are checked for exact equality.

Teacher candidates are selected on long-tailed validation tail recall, subject to
an overall accuracy noninferiority margin of 0.5 percentage points. If no candidate
strictly improves tail recall, the study records `no_repair_selected` and stops
before constructing the official test dataset or training students.

## Outputs

The study directory records `protocol.json`, canonical diagnostics, targets,
companions, differential scores, consensus rankings, causal ablations, all repair
candidates, and `teacher_selection.json`. A successful gate also writes
`teacher_repaired.pt`, three seed-matched student runs, test predictions,
`teacher_comparison.json`, `comparison.json`, and `report.md`.

The repaired teacher uses the ordinary `cifar_teacher` checkpoint contract. Its
extra `repair` metadata binds it to the original teacher and localization hashes;
the existing student trainer therefore consumes it without a special model path.

The downstream success rule requires at least a one-percentage-point mean tail
recall gain, positive gains in at least two seeds, and no more than a 0.5-point
mean overall accuracy loss. Test evidence remains exploratory because previous
workspace studies already examined the official CIFAR-10 test set.

## Completed factor-100 study

The run in `runs/cifar10-lt-neuron-surgery` retained eight causal channels, all
from `stage3`: 81, 21, 95, 79, 4, 102, 104, and 13. The validation-selected
candidate used all eight channels with learning rate 0.001. Relative to the
original teacher, its validation tail recall rose from 46.46% to 48.51%, while
overall accuracy fell from 87.76% to 87.36%, remaining inside the fixed margin.
All 9,552 permitted coordinates changed, every other parameter coordinate stayed
byte-identical, and all BatchNorm buffers remained fixed.

On the balanced test set, teacher accuracy rose from 66.30% to 67.91% and tail
recall rose from 48.24% to 52.10%. Across student seeds 42-44, mean accuracy rose
by 0.87 percentage points and mean tail recall rose by 1.76 points. Tail recall
improved in seeds 43 and 44 and declined by 0.12 points in seed 42, satisfying
the predeclared two-of-three rule. The complete per-seed metrics, calibration
bootstraps, and class recalls are in the [generated report](../runs/cifar10-lt-neuron-surgery/report.md)
and [comparison data](../runs/cifar10-lt-neuron-surgery/comparison.json).

## Generalized protocol

Version 4 accepts a frozen JSON `NeuronSurgerySpec` while retaining the original
factor-100 defaults. The supported repair maps are:

- `CifarCNN`: `stage2` and `stage3`, with the original convolution, BatchNorm,
  next-stage, and four-cell classifier mapping.
- torchvision ResNet-18/34/50: `stage3` and `stage4`, using the final basic or
  bottleneck convolution row, matching BatchNorm affine entries, the next
  residual stage's first convolution and projection consumers, or the final
  classifier column.

The optional `confirmation_fraction` reserves a deterministic balanced subset
of the CIFAR-10 training archive. It is excluded from fitting, epoch selection,
repair localization, and teacher selection; its loader is not constructed until
the repaired teacher and all student checkpoints are frozen.

Component ablations can set `kd_weight = 0`, `score_mode = "gradient_only"`, or
`causal_validation = false`. Setting `downstream_kd = false` makes these
validation-only teacher ablations, avoiding redundant 81-epoch student runs.
See [the campaign protocol](neuron-surgery-campaign.md) for the fixed matrix.

### Multi-dataset matrix

The matrix protocol covers CIFAR-10, CIFAR-100, SVHN, CINIC-10, and GTSRB at
balanced, IF10, IF50, and IF100 profiles. Catalog profile identity is part of
the frozen baseline and study protocols. Balanced studies target every class;
long-tailed studies deterministically target the lowest-frequency half of the
classes, with class-index tie breaking. The resolved classes and training counts
are written to `target_class_plan.json` before localization.

Some IF100 classes can contain fewer than five usable training examples. Matrix
studies therefore declare `allow_sparse_class_fallback = true`: companions first
use correct non-target examples, then deterministic same-class examples with
replacement, and use gradient-only attribution only for singleton classes.
Unavailable classes are omitted from the causal preservation sample, and that
coverage is recorded in `targets.json`. The fixed CIFAR-10 protocol keeps this
fallback disabled.

Generate the immutable 20-job plan and run it across the available GPUs with:

```bash
python -m kd neuron-surgery-matrix-plan \
  --output runs/experiment1-multidataset/matrix-plan
scripts/run_gpu_experiment1_4gpu.sh --gpu-ids 0,1,2,3
```

The final `matrix_summary.json` and `matrix_summary.md` retain one row per
dataset/profile. Predictions are never pooled across datasets.
