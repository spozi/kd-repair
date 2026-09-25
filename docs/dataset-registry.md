# Private dataset registry

The source repository is public and contains only recipes, checksums, split rules, and code. Raw
archives are cached in a companion Gitea repository through Git LFS. Training remains usable
without Gitea by falling back to the canonical upstream source.

## Access boundary

The mirror at `syafiq/kd-repair` is public: read access needs no credentials, and anonymous HTTPS
clones and Git LFS transfers both work. Never put a Gitea token, private SSH key, or authenticated
URL in this repository, an experiment config, or a run artifact.

Read access uses the HTTPS URL, which is the default for every fetch:

```text
https://gitea.izzus.dev/syafiq/kd-repair.git
```

Publishing new archives still requires an authenticated SSH remote. Trusted automation should use a
publication token held by the Gitea secret store. Public GitHub workflows must not receive it.

## Bootstrap and publish to the registry

Publication is the one operation that still needs credentials. Start from an empty directory, using
an SSH remote that is allowed to push:

```bash
python -m kd dataset registry-init /path/to/kd-repair-data
cd /path/to/kd-repair-data
git init -b main
git lfs install
git remote add origin ssh://git@gitea.izzus.dev:2222/syafiq/kd-repair.git
```

Place an upstream archive at
`datasets/NAME/VERSION/archives/UPSTREAM_FILENAME`. Review its terms for private archival, then
write and verify the binding manifest:

```bash
python -m kd dataset mirror-manifest NAME --registry-root . --terms-reviewed
python -m kd dataset registry-verify --registry-root .
git lfs fsck
```

Commit the LFS pointers, manifests, and workflow, push `main`, and create the immutable annotated
tag `catalog-v1.0.0`. Do not move or replace a published catalog tag.

## Configure a machine once

After cloning the public source on a new machine, persist the private registry location in the
user configuration directory. The file is written atomically with mode `0600` and contains no
credential:

```bash
python -m kd dataset registry-config \
  --url https://gitea.izzus.dev/syafiq/kd-repair.git \
  --ref catalog-v1.2.0
python -m kd dataset registry-status
```

Future fetches automatically use this setting:

```bash
python -m kd dataset list
python -m kd dataset fetch cifar100 --version 1.0 --profile lt-if100 --root data
python -m kd dataset verify cifar100 --version 1.0 --profile lt-if100 --root data
```

The default configuration path is `$XDG_CONFIG_HOME/kd-repair/datasets.json`, or
`~/.config/kd-repair/datasets.json` when `XDG_CONFIG_HOME` is unset. Environment variables
`KD_DATASET_REGISTRY` and `KD_DATASET_REGISTRY_REF` remain higher-priority temporary overrides.
Use `python -m kd dataset registry-config --remove` to remove the persisted setting.

For an HTTPS registry the fetcher reads the tag's manifest from `/raw/tag/<ref>/manifests/...` and
downloads archives from Gitea's `/media/tag/<ref>/...` URLs, so no Git checkout or Git LFS is
needed. SSH and local registries still clone with LFS smudging disabled and pull only the paths
named by the manifest. If the registry is unavailable, the fetcher records the fallback reason
without exposing credentials. Official test data are never long-tail resampled.

## Source selection and mirrors

Every archive is pinned by the catalog's MD5 and, when the registry lists it, a SHA-256, so any
host that serves the same bytes is interchangeable. For each archive the fetcher:

1. Lists its sources: the registry, each configured mirror, then the canonical upstream URL.
2. Downloads the first 4 MiB from every source at once (up to 8 s) and ranks them by speed.
   Archives under 32 MiB skip this test and use the order above.
3. Downloads from the fastest source into `FILE.part`. If a source fails, stalls for 60 s, or runs
   below a quarter of an untried source's measured speed for 30 s, it moves to the next source and
   continues from the same byte.
4. Keeps the file only if its checksums match; otherwise it discards it and tries the next source.
   A rerun after an interruption resumes the `.part` file.

Each fetch prints the measured speeds and the source used, and the dataset's `.kd-dataset.json`
records the winning source (`private_registry`, `mirror`, or `upstream`) for each archive. The
source never enters a run's data identity: runs record the catalog hash and each archive's
SHA-256, which are identical whichever host delivered the bytes. For the same reason, mirrors are
configured per machine and never added to the catalog.

A mirror is any static HTTP(S) or `file://` location serving the registry tree,
`BASE/datasets/NAME/VERSION/archives/FILE`. Add mirrors once per machine, or per command:

```bash
python -m kd dataset registry-config \
  --url https://gitea.izzus.dev/syafiq/kd-repair.git --ref catalog-v1.2.0 \
  --mirror https://huggingface.co/datasets/USER/kd-repair/resolve/main
export KD_DATASET_MIRRORS="https://mirror-a.example/kd https://mirror-b.example/kd"
```

Re-running `registry-config` without `--mirror` keeps the configured mirrors; `--clear-mirrors`
removes them. To create a mirror, check out the registry with its LFS objects and upload the
`datasets/` folder to any static host close to your GPU servers (a Hugging Face dataset repository,
an S3 or R2 bucket, or `python -m http.server` on a machine in the same data center). A second
public copy is subject to the same redistribution terms as the Gitea registry, so confirm each
dataset's license allows it:

```bash
git clone https://gitea.izzus.dev/syafiq/kd-repair.git kd-repair-data && cd kd-repair-data
git checkout catalog-v1.2.0 && git lfs pull
hf upload USER/kd-repair datasets datasets --repo-type dataset
```

## Catalog protocol

The current catalog includes CIFAR-10, CIFAR-100, SVHN, CINIC-10, GTSRB, Fashion-MNIST,
PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST, Caltech-101, EuroSAT, and STL-10. Profiles are
`balanced`, `lt-if10`, `lt-if50`, and `lt-if100`, all using split seed 2026. Official validation
and test splits are preserved. Caltech-101 and EuroSAT use pinned stratified protocol splits because
their archives do not provide official test partitions.
The registry validator rejects malformed manifests, altered objects, unsafe paths, unreviewed terms,
and total mirrored storage above 100 GiB.
