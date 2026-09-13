# Sample/slice surgery before knowledge distillation

This intervention uses a frozen compatible checkpoint to score canonical images from one dataset split. It reports individual failures, true-class slices, and directed confusion slices such as `stop -> yield`. A training-split plan can then remove the selected dataset-local positions before the existing KD objective sees them.

## Plan generation

```bash
python -m kd surgery-plan \
  --config configs/teacher.toml \
  --checkpoint runs/teacher/student.pt \
  --output runs/surgery/teacher-plan.json \
  --split train \
  --fraction 0.05 \
  --score high_confidence_error \
  --strategy hybrid \
  --min-slice-size 2
```

Diagnostics disable training augmentation and shuffling, preserving the same underlying split positions. The plan contains:

- checkpoint identity and SHA-256;
- a dataset contract and position hash;
- per-sample confidence, true-class probability, NLL, prediction margin, and selection reason;
- true-class accuracy, NLL, confidence, and overconfidence gap;
- directed confusion-slice support and severity;
- the requested budget and actual selected count.

`high_confidence_error` selects only wrong predictions and ranks them by confidence. It may legitimately produce an empty plan. `high_loss` ranks every sample by NLL and is more aggressive because valid, informative hard examples can rank highly.

The strategies differ only in how the bounded sample list is assembled:

- `sample` ranks globally.
- `slice` round-robins through the worst eligible true-class or confusion slices.
- `hybrid` takes half its budget from bad slices and fills the rest from the global ranking.

Use `--max-samples` to impose an absolute cap in addition to `--fraction`.

## Apply during KD

Add this section to a KD configuration:

```toml
[surgery]
action = "drop"
plan = "runs/surgery/teacher-plan.json"
max_drop_fraction = 0.10
```

Only plans generated on `train` can modify training. Before constructing the optimizer, the runner verifies the dataset source, class order, image size, split recipe, local position signature, and sample bounds. It refuses an empty plan, duplicate indices, a changed plan during resume, and any removal above `max_drop_fraction`.

The original/final train sizes, plan path and hash, selection method, checkpoint hash, and removed fraction are written to `data.json`, `summary.json`, and training checkpoints.

## Experimental protocol

Run at least these matched arms with the same split, seed, teacher, student initialization, augmentation, and schedule:

1. KD without surgery.
2. KD with the planned surgery.
3. Supervised training with the same surgery, to separate a data effect from a distillation effect.

Select the intervention using training diagnostics and validation metrics only. Keep the test split sealed until the plan, drop bound, and model choice are fixed. Report overall and class-wise accuracy, NLL, calibration, and selective-risk metrics: removing hard or minority examples can improve aggregate accuracy while making important slices worse.

## Completed seed-42 CIFAR-10 pilot

The first pilot used the frozen CIFAR-10 teacher, `high_confidence_error`, `hybrid`, and a 5% budget. The teacher made 2,044 canonical training errors, so the plan removed 4.54% of training rather than filling the budget with correct hard examples. Cat and dog were the weakest teacher slices; the largest directed confusions were cat to dog (253 images) and dog to cat (215 images).

| Test metric | Existing KD | Surgery + KD | Candidate minus control |
|---|---:|---:|---:|
| Accuracy | 86.00% | 85.14% | -0.86 pp |
| NLL | 0.4556 | 0.4739 | +0.0183 |
| ECE, 15 bins | 0.0499 | 0.0583 | +0.0084 |
| Macro F1 | 0.8595 | 0.8507 | -0.0088 |
| AURC | 0.0298 | 0.0317 | +0.0019 |

The paired accuracy bootstrap interval was [-1.40, -0.29] percentage points; 481 test examples were correct only for baseline KD and 395 only for surgery + KD. Cat recall fell by 3.6 percentage points. These intervals condition on this fitted model pair, and this is only one training seed, so they are not an across-seed efficacy claim.

This arm is rejected: deleting every teacher-misclassified training image removed useful hard examples and worsened the target slice. The full plan, predictions, class metrics, and paired comparisons remain in [`runs/cifar10-surgery`](../runs/cifar10-surgery/evaluation/comparison.json) as negative evidence. The next defensible intervention is bounded downweighting or human label review of high-confidence disagreements, evaluated across matched seeds, rather than broader deletion.

## Completed seed-42 CIFAR-10 long-tail pilot

The long-tail pilot used the factor-100 exponential training profile and its existing frozen teacher/KD baseline. The teacher made 72 mistakes among 11,167 training examples (0.64%), well below the 5% cap, so the plan removed 72 examples and retained 11,095. The largest eligible slices were dog to cat (7 examples) and airplane to bird (4).

| Balanced test metric | Existing long-tail KD | Surgery + long-tail KD | Candidate minus control |
|---|---:|---:|---:|
| Accuracy / balanced accuracy | 64.06% | 61.89% | -2.17 pp |
| NLL | 1.5791 | 1.7334 | +0.1542 |
| ECE, 15 bins | 0.2372 | 0.2616 | +0.0244 |
| Macro F1 | 0.6236 | 0.5985 | -0.0251 |
| AURC | 0.1824 | 0.1976 | +0.0152 |

The paired accuracy bootstrap interval was [-2.82, -1.49] percentage points; 725 test examples were correct only for baseline KD and 508 only for surgery + KD. Recall changed by +0.9 pp for airplane, +1.0 for automobile, and +1.0 for bird, but fell by 6.3 pp for cat, 5.9 for deer, 6.7 for frog, 4.9 for ship, and 3.3 for truck. This is one seed and one teacher, so the paired interval is conditional on these fitted models and does not measure variation across retraining.

This second deletion arm is also rejected. The long-tail plan and matching configuration are in [`configs/cifar10_lt_kd_surgery.toml`](../configs/cifar10_lt_kd_surgery.toml); paired metrics and per-class reports are in [`runs/cifar10-lt-surgery/evaluation/comparison.json`](../runs/cifar10-lt-surgery/evaluation/comparison.json). The observed test harm is consistent with the balanced pilot: teacher-misclassified points still carried useful training signal, including for classes already underrepresented in the long-tail training set.
