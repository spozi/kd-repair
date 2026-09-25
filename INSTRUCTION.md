# Running the distillation baselines on a rented GPU

These steps train the DKD, RLD and LoCa baseline students for the 23 Experiment 1 studies that
produced a repaired teacher, then write the results table to `docs/distillation-baselines-results.md`.
That file is generated locally and kept out of git, because this repository is public. The work is
69 runs of three students each. Every student keeps the training settings of the classical-KD
student it is compared with (batch size 128, same schedule and precision), so the comparison
changes only the distillation loss.

Replace `PORT` and `USER@HOST` with the SSH details your provider gives you.

## 1. Rent a machine

| Resource | Requirement | Why |
|---|---|---|
| GPU | RTX 5060 Ti 16 GB or newer | Experiment 1 ran on an RTX 5060 Ti |
| NVIDIA driver | `nvidia-smi` shows "CUDA Version" 13.0 or higher (R580+) | The pinned PyTorch is built for CUDA 13.2 |
| CPU | 16 or more vCPUs | Each run uses about five CPU processes |
| Disk | 20 GB or more free | 1.9 GB of extracted results, plus the datasets |

No system CUDA toolkit is needed; PyTorch brings its own.

## 2. Copy the Experiment 1 results to the server

On your Mac:

```bash
cd ~/Developments/kd-autonomous-car
rsync -avP -e "ssh -p PORT" \
  compiled-results/experiment1-results-all-52-20260917T103004Z.tar.zst USER@HOST:~/
```

The archive holds every teacher, KD student and cached prediction the baselines are compared with.

## 3. Set up the server

```bash
ssh -p PORT USER@HOST

# Only if `conda` is not installed:
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b && ~/miniforge3/bin/conda init bash && exec bash

git clone https://github.com/spozi/kd-repair.git kd-autonomous-car
cd kd-autonomous-car
conda env create -f environment.yml
conda activate kd
tar --zstd -xf ~/experiment1-results-all-52-20260917T103004Z.tar.zst
ls runs/experiment1-multidataset/matrix_summary.json   # must exist
```

`environment.yml` installs Python 3.12, PyTorch 2.14.0 with CUDA 13.2, NumPy 2.5.2, git-lfs and
zstd, the same stack Experiment 1 used.

## 4. Check the GPU

```bash
python -m kd cuda-check --device cuda:0 --precision auto
python -m unittest discover -s tests
```

`cuda-check` should report `bfloat16` and your GPU's name, and all tests should pass. If either
fails, stop here and keep the output.

## 5. Run the baselines

Run inside tmux so a dropped SSH connection does not kill the job:

```bash
tmux new -s baselines        # if tmux is missing: apt-get install -y tmux
scripts/run_gpu_distillation_baselines.sh --dry-run-only
scripts/run_gpu_distillation_baselines.sh --skip-fetch --runs-per-gpu 4
```

- The dry run downloads and verifies the 11 datasets into `data/`, three at a time, and checks that
  all 69 runs are set up correctly, without training. Each archive comes from whichever of Gitea
  and the dataset's original host is fastest from the server; the log shows the measured speeds.
  If a download slows down, it switches to the other source at the same byte.
- The second command trains. It keeps up to four runs on each GPU, and starts another only while
  the GPU has at least 3 GB free.
- Detach with `Ctrl-b` then `d`. Reattach with `tmux attach -t baselines`.

## 6. Watch it

Every minute the launcher prints a status report like this (also saved in
`logs/distillation-baselines/supervisor-*.log`):

```text
[03:36:31Z] status: 21/69 runs ended (0 failed), 4 running, elapsed 00:41:10
    [###########-------------------]  37.5%  students 70/207  runs scored 21/69  ETA 01:08:34
      cifar10-lt-if100       rld  [########----]  student 3/3  epoch 12/49
      eurosat-lt-if100       rld  [############]  scoring on the test split
    GPU 0: 4 runs, 6120/16311 MiB used
```

- The top bar is the share of all training done, weighted by training images, so a large balanced
  study counts for more than a small long-tailed one. The ETA uses the pace since this launcher
  started, so it settles after the first few minutes.
- Each run in progress gets its own bar: which of its three students is training, at which epoch,
  or that all three are done and it is being scored.
- The counts come from files the runs write after every epoch, so they stay correct after a restart.
- The dataset step before training shows its own download progress (bytes, rate and ETA).

To watch progress without attaching to tmux, open a second SSH session:

```bash
cd kd-autonomous-car && conda activate kd
python scripts/baseline_progress.py --watch 30     # redraws every 30 s; Ctrl-c to leave
nvidia-smi dmon                                    # GPU utilization; `top` for the CPU
```

| What you see | What to do |
|---|---|
| GPU `sm` column low, CPU near 100% | `Ctrl-c`, restart with `--runs-per-gpu 2` |
| GPU and CPU both have room | `Ctrl-c`, restart with `--runs-per-gpu 6` |
| A run reports `FAILED` | Let the others finish, read its log in `logs/distillation-baselines/`, then rerun the same command |
| Server rebooted or the job was stopped | Rerun the same command |
| Every dataset source is slow from this server | Add a closer mirror with `--mirror URL` (see [`docs/dataset-registry.md`](docs/dataset-registry.md#source-selection-and-mirrors)); partial downloads resume |

Stopping and rerunning is always safe: finished students are kept and interrupted epochs resume.

## 7. Bring the results home before releasing the server

The results table is regenerated only once every run has succeeded.

On the server:

```bash
tar --zstd -cf baselines-results.tar.zst docs/distillation-baselines-results.md \
  logs/distillation-baselines runs/experiment1-multidataset/*/*/baselines/distillation
```

On your Mac:

```bash
cd ~/Developments/kd-autonomous-car
rsync -avP -e "ssh -p PORT" USER@HOST:~/kd-autonomous-car/baselines-results.tar.zst .
# Remove the interrupted DKD run trained on this Mac so it cannot mix with the server's.
rm -rf runs/experiment1-multidataset/cifar10/lt-if10/baselines/distillation
tar --zstd -xf baselines-results.tar.zst
```

Check that `docs/distillation-baselines-results.md` has no `pending` cells before releasing the
server.

## More detail

[`docs/cuda.md`](docs/cuda.md) covers the environment files, the launcher's options, and why the
batch size is not raised to fill the GPU.
