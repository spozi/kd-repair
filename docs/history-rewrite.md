# Commit history rewrite, 2 October 2026

Commit messages on `main` were edited to remove co-author trailers. No file changed:
every rewritten commit has the same tree as the commit it replaces, so the code at
each point in history is byte-identical. Only the commit hashes differ.

Provenance records written before the rewrite cite the old hashes. To resolve one,
find it in the table below; the tree hash is unchanged and identifies the same code
under either commit. The twelve commits that predate the first trailer, and the
`source-v0.x` tags that point into them, kept their hashes and are not listed.

Cited in saved run artifacts and export scripts:

| Old | New | Role |
|---|---|---|
| `89992ca` | `059d629` | Experiment 1 source revision |
| `7482848` | `d6251aa` | Code deployed for the full matrix |
| `d544c0f` | `c30ba78` | Replacement deployed 2026-09-28 |

All rewritten commits, oldest first:

| Old | New | Tree | Date | Subject |
|---|---|---|---|---|
| `c391a8d` | `5983611` | `9e0a1d4` | 2026-09-16 | Stream setup progress and provision the environment automatically |
| `f873a98` | `3fb5c33` | `69a5e05` | 2026-09-16 | Use the public HTTPS dataset registry and never prompt for credentials |
| `855645e` | `d7a69b9` | `ccf2044` | 2026-09-16 | Install a PyTorch build that matches the GPU architecture |
| `1c47f1a` | `1312042` | `b9c592b` | 2026-09-16 | Run the job queue on any number of GPUs |
| `5089dcd` | `c862872` | `c2a8e40` | 2026-09-16 | Scale neuron surgery matrix to all datasets |
| `17af1ad` | `43d7c92` | `3f80446` | 2026-09-17 | Repair dataset layouts for matrix recovery |
| `89992ca` | `059d629` | `8c6340a` | 2026-09-17 | Normalize Caltech101 images as RGB |
| `caf91fb` | `565c78d` | `772caf2` | 2026-09-25 | Add DKD, RLD and LoCa distillation baselines |
| `a22f627` | `236d81b` | `f82328c` | 2026-09-25 | Tabulate Experiment 1 students against the distillation baselines |
| `ba0214e` | `b848514` | `6a89a3e` | 2026-09-25 | Pin the GPU server environment to torch 2.14.0 with CUDA 13.2 |
| `fff1041` | `57224fe` | `1caacd0` | 2026-09-25 | Run several distillation baselines on each GPU |
| `9299898` | `b51582b` | `1097626` | 2026-09-25 | Prepare datasets and relocate the data root for baseline runs |
| `f42fba7` | `3dc61bf` | `ce36557` | 2026-09-25 | Show baseline progress with a bar and ETA |
| `a0f3e49` | `87c7c72` | `41e2ceb` | 2026-09-25 | Add a step-by-step guide for running the baselines on a rented GPU |
| `5d6c19e` | `cf61fae` | `8fc53b5` | 2026-09-25 | Anchor the distillation-baseline test's data root in its temp directory |
| `efea171` | `309261f` | `7994e7c` | 2026-09-25 | Fetch each dataset archive from its fastest source |
| `35f363e` | `3562a8c` | `e7a0083` | 2026-09-25 | Fetch baseline datasets in parallel and accept mirrors in the launcher |
| `5d50d23` | `e17c412` | `84c1aa9` | 2026-09-25 | Keep experiment results out of the public repository |
| `913bc30` | `3b55c87` | `037adf7` | 2026-09-25 | Rank dataset sources by their streaming rate, not their start-up time |
| `60328ec` | `cf5096d` | `2bfa4ac` | 2026-09-25 | Compute the Experiment 1 matrix statistics reported in the paper |
| `cdea3f6` | `d8f07bb` | `75f9ec4` | 2026-09-25 | Diagnose the distillation baselines and ablate their loss balance |
| `013f7e7` | `0f3df09` | `32c0c51` | 2026-09-25 | Add CIFAR-stem ResNets as teacher and student architectures |
| `c394bfe` | `7583926` | `071966a` | 2026-09-25 | Add CIFAR ResNet students and the architecture studies |
| `570ad63` | `6a6a199` | `ba94988` | 2026-09-25 | Propose a GPU-resident data and augmentation pipeline |
| `c0a9fba` | `5949a6c` | `05aaecf` | 2026-09-25 | Compute statistics for the architecture studies |
| `b63a1b3` | `4edc387` | `30f1ea2` | 2026-09-27 | Extend the architecture and loss studies to the full setting matrix |
| `c189a63` | `65f3957` | `431a4ad` | 2026-09-27 | Audit full-matrix coverage and reuse finished CNN loss runs |
| `7482848` | `d6251aa` | `17ea2a1` | 2026-09-27 | Merge alternative-loss reports under a lock so cells can run in parallel |
| `d544c0f` | `c30ba78` | `b6c8460` | 2026-09-28 | Confirm reused baselines on the device their protocol records |
