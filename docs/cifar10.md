# CIFAR-10 experiment protocol

This study compares a supervised student, classical logits KD, and DKD on real CIFAR-10 images. It is a controlled comparison of compact CNNs under a fixed training budget, not a reproduction of the original DKD paper's benchmark architectures or an exhaustive hyperparameter search.

## Run

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kd

# Download once; torchvision verifies the official archive and batch checksums.
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True)"

# Train one teacher and nine students, then evaluate the official test split.
python -m kd cifar10 --output runs/cifar10 --root data --device mps

# Export a readable report, CSV table, and standalone figures.
MPLCONFIGDIR=/private/tmp/kd-matplotlib python -m kd.reporting --study runs/cifar10
```

Use `--device cpu`, `cuda`, or `auto` on another machine. On this Mac, the restricted tool sandbox hid MPS availability; the GPU is available when the process runs outside that sandbox. The existing Conda environment already contains all training and plotting dependencies.

The study command automatically resumes completed runs or the last completed epoch of an interrupted run. It refuses to reuse an output directory with a different protocol. Training lengths can be changed using `--teacher-epochs` and `--student-epochs`, and seeds using `--seeds`; use a new output directory for a different protocol. A run interrupted before its first checkpoint has no recoverable training state.

## Data isolation

- Use all 50,000 official training examples, partitioned into **45,000 train / 5,000 validation**.
- Split independently within every class: 4,500 training and 500 validation images per class. The fixed split seed is **2026**, independent of all model initialization seeds.
- Reserve all **10,000 official test examples** for the final phase. Training does not instantiate the official test dataset.
- Training augmentation is a 32×32 random crop after four-pixel padding and a random horizontal flip. Teacher and student receive exactly the same tensor.
- Evaluation preserves all original 32×32 pixels. Both networks use fixed channel mean/std of 0.5, mapping inputs to [-1, 1]; no statistics are estimated from validation or test images.
- Every run saves split sizes, class counts, and SHA-256 hashes of its training/validation indices in `data.json`.

## Models and schedule

Both adapters use three semantic stages, each containing two 3×3 convolution/BatchNorm/ReLU blocks and a 2×2 max pool. A 2×2 adaptive average pool and linear classifier produce the logits. The teacher widths are 32/64/128 and the student widths are 16/32/64. This retains spatial detail appropriate for tiny images without the aggressive ImageNet ResNet stem.

The predeclared default protocol is:

| Setting | Teacher | Student |
|---|---|---|
| Initialization | From scratch, seed 41 | From scratch, seeds 42, 43, 44 |
| Epochs | 30 | 20 for each method/seed |
| Batch size | 128 | 128 |
| Optimizer | SGD, momentum 0.9 | Same |
| Initial learning rate | 0.05 | 0.05 |
| Schedule | Cosine decay over the run | Same |
| Weight decay | 0.0005 | 0.0005 |
| Teacher | None | One fixed teacher shared by all KD/DKD runs |

Classical KD uses `(1-weight)*CE + warmup*weight*T²*KL`, while DKD keeps full CE and adds the decoupled response terms. For both, `T=4`, `weight=0.5`, and warmup lasts five epochs. DKD uses `alpha=1`, `beta=8`. These settings are fixed before training; they are not selected using the test set. Feature transfer and stronger augmentation are excluded from this particular study to keep the method comparison focused.

## Selection and evaluation

The best epoch of each run is selected by validation accuracy. After every run finishes, the method is selected by mean validation accuracy across the three student seeds. `selection.json` records that decision, the protocol hash, and the SHA-256 of every selected checkpoint **before** the official test phase starts. Test results never change this decision.

The final phase evaluates every predeclared model on the same test examples and saves:

- Top-1 accuracy and a Wilson 95% interval for each fitted model.
- Mean accuracy and sample standard deviation across student seeds.
- Confusion matrix, per-class precision/recall/F1, macro F1, and balanced accuracy.
- NLL, multiclass Brier score, and expected calibration error using 15 equal-width confidence bins.
- Per-seed paired accuracy differences against supervised training, a 2,000-resample paired bootstrap interval, and an exact two-sided McNemar test.
- Student parameter count, estimated Conv2d/Linear FLOPs, warmed-up synchronized batch-one latency, and training time/overhead.

Wilson and paired bootstrap intervals quantify finite test-example uncertainty **conditional on fitted models**. They do not capture training randomness. Three student seeds give a limited estimate of that randomness, conditional on one teacher and one train/validation split. McNemar p-values are exploratory and unadjusted for multiple comparisons. Do not interpret them as evidence that one method generally dominates all alternatives.

## Artifacts

`runs/cifar10/` contains `protocol.json`, `progress.json`, `selection.json`, and the completed `report.json`. The executed study also records hardware/software, dataset checksum, and source hashes in `environment.json`. Every model directory contains effective config, split provenance, learning history, resumable checkpoints, an inference artifact, and `test/metrics.json`, `test/predictions.npz`, and `test/confusion.csv`. The report renderer adds `report.md`, `results.csv`, and two PNG figures. NumPy prediction archives contain labels, all class probabilities, and the ordered class names so comparisons can be independently reproduced.

For individual CIFAR runs with the ordinary `train` command, set `data.source="cifar10"`, `num_classes=10`, `image_size=32`, and the desired `root`. `download=true` enables first-time downloading. `train_samples`, `val_samples`, and `test_samples` apply only to synthetic data; CIFAR is never silently subsampled by those fields.

Dataset source: [CIFAR-10 project](https://cave.cs.toronto.edu/kriz/cifar.html), Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. Data loading follows the [torchvision CIFAR10 API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
