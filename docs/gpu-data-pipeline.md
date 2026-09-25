# Proposal: GPU-resident data and augmentation

Status: proposed, not implemented. Written on 2026-09-25 while the architecture studies ran on a
2 × RTX 5060 Ti Vast.ai instance.

## Problem

Runs with the compact 32×32 models are limited by the CPU, not the GPU. On the instance above:

| Measurement | Value |
|---|---|
| CPU quota of the container (`/sys/fs/cgroup/cpu.max`) | 768000/100000, i.e. 7.68 cores, although `nproc` reports 32 |
| Time the container had been throttled | 12.5 hours of cumulative CPU time |
| Runnable processes at a time (8 trainers plus loader workers) | about 31 |
| GPU busy time (`nvidia-smi dmon`) | 14–29%, memory bandwidth 4–10% |
| Per-image CPU cost of the current pipeline (CIFAR-10, one core) | about 105 µs |
| Throughput reached in practice | about 2,200 images/s per run, about 18,000 images/s in total |

Every epoch of every student, each image passes through Python on the CPU: a PIL image is built
from an array (CINIC-10, Caltech-101 and EuroSAT also re-decode a PNG or JPEG file), padded and
randomly cropped, flipped, converted and normalized, then batched in a loader worker and handed
to the training process. Loader workers are started anew every epoch (`persistent_workers` is
off), and validation runs every epoch. A model with 75,000 parameters needs far less GPU time per
batch than the CPU needs to prepare it, so the GPU sits idle.

Consequences:

- NVIDIA MPS does not help: it lets kernels from several processes share a busy GPU, but here
  the GPU is waiting for data.
- A faster GPU such as an RTX 5090 does not help either, except for the ResNet teachers.
- Throughput scales with *allocated* CPU cores. When renting, check the effective core count of an
  offer, not the host's.

## Proposed change

1. Decode each dataset split once into a uint8 tensor of shape `[N, 3, 32, 32]` held on the GPU.
   The fixed resize to 32×32 (Caltech-101, EuroSAT, STL-10, GTSRB) is deterministic, so applying it
   once gives the same pixels as applying it every epoch. The largest training set, CINIC-10,
   takes about 280 MB.
2. Per batch, on the GPU: zero-pad by 4, crop at random offsets, flip at random, convert to float
   and normalize, with batched tensor operations and a seeded `torch.Generator`.
3. Evaluation keeps its deterministic preprocessing (convert and normalize, no augmentation).
4. Keep the current CPU pipeline as the default, and select the new one by a config setting, so
   existing configs and frozen protocols keep their meaning.

Expected effect: the CPU then runs only the training loop, the GPU becomes the limit, and the
architecture studies should take roughly 2–4 hours instead of about 14 on the instance above.
This is an estimate until measured.

Even on a laptop CPU, the batched operations cost about 21 µs per image against 105 µs for the
current pipeline; on the GPU the cost is close to zero.

## Tests to add

- For a fixed crop offset and flip, the GPU output equals torchvision's `RandomCrop(32, padding=4)`
  followed by `RandomHorizontalFlip`, `ToTensor` and `Normalize`, pixel for pixel.
- Over many draws, crop offsets are uniform over the 9 × 9 positions and flips occur half the time.
- Evaluation tensors equal the current pipeline's exactly.
- The same seed gives the same augmentation sequence, and a resumed run continues it.

## Effect on comparability

The augmentation keeps the same operations and distribution, but its random draws differ from the
CPU pipeline's. Any comparison must therefore run both of its arms on the same pipeline:

- Student-architecture study: each original/repaired pair shares the pipeline, so it is fine.
- Teacher-architecture study: every model is new, so it is fine.
- Matched-balance DKD/RLD: needs a classical-KD reference retrained on the new pipeline rather
  than the Experiment 1 students.

`kd/data.py` is fingerprinted by the frozen study protocols, so re-running a study completed before
the change needs the code it was run with. Recorded results are unaffected.

## Cheaper alternative

Keep the CPU pipeline but set `persistent_workers = true` and fewer loader workers per run. This
avoids restarting workers every epoch and reduces contention, for an expected gain of at most
1.3–2×, because the per-image Python work remains.

## Related bug to fix at the same time

The first run to use a dataset profile saves its split indices through a temporary file with a
fixed name (`splits/<profile>.npz.tmp`) and then renames it. Two processes that reach the same new
profile at the same moment collide, and one fails with `FileNotFoundError` on the rename. Use a
unique temporary name, for example with `tempfile.NamedTemporaryFile(dir=...)`, as the registry
settings writer in `kd/dataset_registry.py` does. `write_json` and `save_checkpoint` in
`kd/checkpoints.py` use the same fixed-name pattern and are safe only because each run writes its
own directory. Until then, create all split files in one process before starting parallel runs.
