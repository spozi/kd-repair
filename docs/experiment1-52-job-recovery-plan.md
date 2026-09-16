# Experiment 1: 52-Job Recovery Plan

## Objective

Finish all 52 dataset/profile jobs under the current long-tail sampling protocol while
preserving every existing result. The recovery is successful when the queue exits zero,
all 52 study comparison files exist, and the generated matrix summary reports 52 complete
jobs with no missing entries.

## Current State

The run completed 52 queue entries with 24 successes and 28 failures. There are 41
existing `comparison.json` files because 17 results from the earlier protocol are still
on disk alongside the 24 current-protocol results.

| Failure group | Jobs | Cause | Recovery |
|---|---:|---|---|
| CINIC-10, SVHN, CIFAR-100, CIFAR-10, GTSRB | 20 | Their output directories contain baseline protocols created before the one-example-per-class sampling floor was added. | Archive these five output trees and regenerate all four profiles for each dataset. |
| EuroSAT | 4 | The registry extracted images at `eurosat/EuroSAT_RGB`, while torchvision expects `eurosat/2750`. | Materialize the expected directory alias, validate loading, and resume. |
| Caltech101 | 4 | The registry extracted `101_ObjectCategories` at the version root, while torchvision expects it below `caltech101/`. | Materialize the expected directory alias, validate loading, and resume. |

The 24 current-protocol successes must remain in place: PathMNIST, FashionMNIST,
OrganAMNIST, BloodMNIST, DermaMNIST, and STL-10, each with all four profiles.

## Phase 1: Preserve Evidence

1. Record the current Git commit, Python package versions, CUDA/GPU information, final
   supervisor log, and counts of baseline/study manifests.
2. Copy the final supervisor log and the full matrix plan protocol into a timestamped
   recovery-audit directory under `runs/`.
3. Do not delete or edit checkpoints, manifests, summaries, or comparison files.

Expected checkpoint before proceeding:

```text
24 current-protocol jobs complete
28 jobs requiring recovery
52 jobs present in matrix-plan/protocol.json
```

## Phase 2: Make Registry Fetches Loadable

Implement a small post-extraction materialization step in `kd/dataset_registry.py` and
call it on every fetch, including when the archive is already present:

- EuroSAT: create `eurosat/2750` as a relative symlink to `EuroSAT_RGB`.
- Caltech101: create `caltech101/101_ObjectCategories` as a relative symlink to
  `../101_ObjectCategories`.
- Refuse to replace an existing non-symlink path or a symlink with an unexpected target.
- Keep this normalization idempotent so repeated fetches are harmless.

This belongs in the registry preparation layer rather than `kd/data.py`. The baseline
protocol hashes `data.py` but does not hash `dataset_registry.py`, so this repairs the
prepared data without invalidating the 24 successful current-protocol jobs.

Add focused tests to `tests/test_dataset_registry.py` that prepare representative archive
layouts and assert that torchvision can discover the resulting roots. Cover repeated
materialization and conflicting paths. Then run:

```bash
.venv/bin/python -m unittest tests.test_dataset_registry
.venv/bin/python -m unittest discover -s tests
```

Re-fetch the two local datasets to apply normalization and then verify all profiles:

```bash
.venv/bin/python -m kd dataset fetch eurosat --profile balanced --root data
.venv/bin/python -m kd dataset fetch caltech101 --profile balanced --root data

for dataset in eurosat caltech101; do
  for profile in balanced lt-if10 lt-if50 lt-if100; do
    .venv/bin/python -m kd dataset verify "$dataset" --profile "$profile" --root data
  done
done
```

Add a loader smoke check after archive verification. It must instantiate each dataset,
read at least one sample, and confirm the expected class count: 10 for EuroSAT and 101
for Caltech101. Archive verification alone is insufficient because it allowed both layout
errors through the original preflight.

## Phase 3: Validate Protocol Compatibility

Before moving any output, generate dry-run baseline protocols for one successful current
job and compare them with the stored protocol. Use PathMNIST balanced as the sentinel.
The stored and generated JSON must be identical.

Also dry-run one profile for each repaired dataset and all three formerly failing
long-tail cases:

- EuroSAT balanced
- Caltech101 balanced
- CIFAR-100 IF100
- GTSRB IF50
- GTSRB IF100

Do not begin GPU training until all five dry runs construct their datasets successfully
and retain at least one training and validation example per class.

## Phase 4: Archive Incompatible Outputs

Move the five old-protocol dataset trees into a timestamped archive on the same
filesystem. Moving them is reversible and leaves the 24 valid trees untouched.

```bash
RECOVERY_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LEGACY_ROOT="runs/experiment1-legacy-${RECOVERY_TAG}"
mkdir -p "$LEGACY_ROOT"

for dataset in cinic10 svhn cifar100 cifar10 gtsrb; do
  test -d "runs/experiment1-multidataset/$dataset"
  mv "runs/experiment1-multidataset/$dataset" "$LEGACY_ROOT/$dataset"
done
```

After the move, confirm that the six successful current-protocol dataset trees are still
present and that the legacy archive contains exactly the five moved trees. Keep
`runs/experiment1-multidataset/matrix-plan` in place; it already describes the complete
52-job matrix and must not be regenerated with a subset.

## Phase 5: Preflight Without Training

Run the complete matrix in dry-run mode on the same output root. This validates the
matrix plan, all prepared datasets, both GPUs, and every baseline protocol without
starting training:

```bash
scripts/run_gpu_experiment1_4gpu.sh \
  --python .venv/bin/python \
  --gpu-ids 0,1 \
  --datasets cinic10,pathmnist,svhn,fashionmnist,organamnist,cifar100,cifar10,gtsrb,eurosat,bloodmnist,caltech101,dermamnist,stl10 \
  --profiles balanced,lt-if10,lt-if50,lt-if100 \
  --output-root runs/experiment1-multidataset \
  --skip-bootstrap \
  --skip-fetch \
  --skip-tests \
  --dry-run-only
```

Required result: `All 52 baseline protocols validated without training.` Any protocol
or dataset error at this stage blocks the GPU rerun.

## Phase 6: Resume the Full Matrix

Run the same complete matrix without `--dry-run-only`:

```bash
scripts/run_gpu_experiment1_4gpu.sh \
  --python .venv/bin/python \
  --gpu-ids 0,1 \
  --datasets cinic10,pathmnist,svhn,fashionmnist,organamnist,cifar100,cifar10,gtsrb,eurosat,bloodmnist,caltech101,dermamnist,stl10 \
  --profiles balanced,lt-if10,lt-if50,lt-if100 \
  --output-root runs/experiment1-multidataset \
  --skip-bootstrap \
  --skip-fetch \
  --skip-tests
```

The 24 valid jobs should resume and finish quickly. The queue should train only the 20
archived old-protocol jobs plus the eight repaired EuroSAT/Caltech101 jobs. Monitor the
new supervisor log for `FAILED`, protocol mismatch, missing dataset, CUDA, and checkpoint
integrity errors.

## Phase 7: Acceptance Checks

After a zero-exit queue, run the matrix summary and verify its invariants:

```bash
.venv/bin/python -m kd neuron-surgery-matrix-summary \
  --plan runs/experiment1-multidataset/matrix-plan \
  --output-root runs/experiment1-multidataset

find runs/experiment1-multidataset -name comparison.json | wc -l
```

Acceptance criteria:

- The rerun reports `0 of 52 jobs failed` or the equivalent all-success message.
- Exactly 52 `comparison.json` files exist below the active output root.
- `matrix_summary.json` reports `job_count: 52`, `complete_count: 52`, and
  `all_jobs_finished: true`.
- Every baseline manifest validates its checkpoint fingerprints.
- No active result references the timestamped legacy archive.
- The full unit-test suite passes at the recovery commit.

Keep the legacy archive until the final summary and all checkpoint hashes have been
reviewed. It can then be retained as provenance or removed in a separate, explicit
cleanup operation.
