# Private dataset registry

The source repository is public and contains only recipes, checksums, split rules, and code. Raw
archives may be cached in a separate private Gitea repository through Git LFS. Training remains
usable without Gitea by falling back to the canonical upstream source.

## Security boundary

Make `syafiq/kd-repair` private before its first push. Confirm that an unauthenticated API request
and clone cannot discover the repository. Never put a Gitea token, private SSH key, or authenticated
URL in this repository, an experiment config, or a run artifact.

Local access uses the SSH URL:

```text
ssh://git@gitea.izzus.dev:2222/syafiq/kd-repair.git
```

Trusted automation should use separate read-only and publication tokens held by the Gitea secret
store. Public GitHub workflows must not receive either token.

## Bootstrap the private repository

Start from an empty directory after the remote has been made private:

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

## Fetch for an experiment

Configure the mirror only in the local shell:

```bash
export KD_DATASET_REGISTRY=ssh://git@gitea.izzus.dev:2222/syafiq/kd-repair.git
export KD_DATASET_REGISTRY_REF=catalog-v1.0.0
python -m kd dataset list
python -m kd dataset fetch cifar100 --version 1.0 --profile lt-if100 --root data
python -m kd dataset verify cifar100 --version 1.0 --profile lt-if100 --root data
```

The fetcher clones with LFS smudging disabled and pulls only paths named by the selected mirror
manifest. If the private mirror is unavailable, it downloads from upstream and records the fallback
reason without exposing credentials. Official test data are never long-tail resampled.

## Catalog protocol

Version 1 includes CIFAR-10, CIFAR-100, SVHN train/test, CINIC-10, and GTSRB. Profiles are
`balanced`, `lt-if10`, `lt-if50`, and `lt-if100`, all using split seed 2026. CIFAR, SVHN, and GTSRB
use a stratified 10 percent validation holdout; CINIC-10 retains its official validation split.
The registry validator rejects malformed manifests, altered objects, unsafe paths, unreviewed terms,
and total mirrored storage above 100 GiB.
