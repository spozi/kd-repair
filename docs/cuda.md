# CUDA execution

The project uses one explicit CUDA device per process. This is intentional for the compact 32x32
models: independent dataset, profile, and seed jobs usually utilize a multi-GPU server better than
splitting one small model with data parallelism.

## Environment

The server environment is pinned in two files at the repository root:

| File | Contents |
|---|---|
| `requirements-cuda.txt` | `torch==2.14.0+cu132`, `torchvision==0.29.0+cu132`, NumPy, SciPy, Pillow, Matplotlib |
| `environment.yml` | Conda layer (Python 3.12, pip, git-lfs) that installs the file above and this project |

PyTorch 2.14.0 is the latest release and CUDA 13.2 the newest CUDA it has stable wheels for.
Experiment 1 was produced with exactly this build, so new students remain matched to its controls.
The wheels bundle the CUDA runtime and cuDNN; the host needs only an NVIDIA driver that supports
CUDA 13 (R580 or newer). No system CUDA toolkit is required.

```bash
git clone <repository> kd-autonomous-car && cd kd-autonomous-car
conda env create -f environment.yml    # later updates: conda env update -f environment.yml --prune
conda activate kd
```

Without Conda, use any Python 3.12:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cuda.txt && pip install -e '.[reports]'
```

The multi-GPU launcher provisions the same pins itself (see below).

## Preflight

With the environment active, fetch the selected datasets and run:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("built CUDA", torch.version.cuda)
print("available", torch.cuda.is_available())
print("devices", torch.cuda.device_count())
PY
python -m kd cuda-check --device cuda:0 --precision auto
python -m unittest discover -s tests
```

The preflight performs pinned-memory transfer, channels-last convolution, autocast, backward, and an
optimizer update. It reports the selected precision, GPU model, compute capability, cuDNN version,
throughput, and peak allocated memory.

## Runtime policy

The `train` section accepts these accelerator controls:

| Setting | CUDA default | Meaning |
|---|---|---|
| `device` | `auto` | `cuda`, `cuda:N`, `mps`, or `cpu` |
| `precision` | `auto` | BF16 when supported, otherwise scaled FP16; `float32` disables AMP |
| `cuda_mode` | `fast` | TF32 and cuDNN autotuning; use `deterministic` for the audit mode |
| `channels_last` | `true` | NHWC-compatible memory layout for convolution kernels |
| `workers` | `0` | Set 4-8 on a server, then measure GPU utilization |
| `persistent_workers` | `false` | Faster epoch transitions when true, but exact loader-RNG resume is not retained |
| `prefetch_factor` | `2` | Batches queued per worker |

Loader memory is pinned automatically when `device` is `cuda:N`, `cuda`, or an `auto` value that
resolves to CUDA. Transfers use `non_blocking=True`, and fast mode uses fused CUDA SGD. FP16 scaler
state is included in epoch checkpoints and restored on resume.

Neuron localization and causal channel ranking remain FP32 because small numerical changes can reorder
channels. Repair candidate optimization and downstream student KD use the configured mixed precision.
Final cached neuron-surgery comparisons remain CPU evaluations so paired artifacts are independent of
the GPU model.

## Running studies

For the fresh, sealed CIFAR-10-LT factor-100 Experiment 1 workflow, use the server launcher:

```bash
scripts/run_gpu_experiment1.sh --device cuda:0
```

The script configures and verifies the private dataset cache, runs the CUDA preflight and tests,
generates matched baselines, then runs neuron repair and downstream KD. It logs to `logs/` and can be
rerun with the same arguments to resume validated artifacts. Run
`scripts/run_gpu_experiment1.sh --help` for dry-run, stage-only, path, and registry overrides. Install
a CUDA-enabled PyTorch build in the active environment before invoking it.

For a multi-GPU server, run the complete 13-dataset matrix:

```bash
scripts/run_gpu_experiment1_4gpu.sh --gpu-ids 0,1,2,3
```

The default matrix contains CIFAR-10, CIFAR-100, SVHN, CINIC-10, GTSRB, Fashion-MNIST,
PathMNIST, BloodMNIST, DermaMNIST, OrganAMNIST, Caltech-101, EuroSAT, and STL-10 under balanced,
IF10, IF50, and IF100 profiles (52 studies). Shared dataset preparation, CUDA checks, and tests run once. A dynamic queue
keeps one isolated worker per GPU active, starts the next study when a GPU becomes free, and writes a
dataset/profile report under `runs/experiment1-all-datasets/`. Use `--datasets` or `--profiles` for a
smaller matrix. Every job can resume independently.

The launcher provisions Python itself: an activated virtualenv is reused as-is, otherwise it creates
the `kd` Conda environment (or a local `.venv` when Conda is absent) and installs
`requirements-cuda.txt` plus this project. Installs are skipped when every pinned version is already
installed and the wheel has kernels for each GPU; `--force-bootstrap` reinstalls them. A GPU the pinned
build cannot run stops the launcher with its compute capability; `--pytorch-index` then swaps in the
same PyTorch release from another CUDA wheel index. Long operations are no longer silent — dataset downloads and Git LFS transfers report
transferred bytes, rate, and ETA, pip shows download progress, and the queue prints a status table
every `--status-interval` seconds listing each GPU's job, elapsed time, and latest log line. Use
`--stream-logs` to mirror worker output into the console, and `--fetch-all-datasets` to prepare the
whole catalog instead of only the datasets in the matrix.

## Distillation baselines

`distillation-baselines` retrains each completed study's three classical-KD students with DKD, RLD,
and LoCa, keeping the teacher, data, splits, initial weights, schedule, and checkpoint rule, and scores
them on the study's test split against both of the study's students. It reads the matrix output
root, so run it on the machine that holds the Experiment 1 results. Results land in each study's
`baselines/distillation/<method>/`, and a rerun resumes validated artifacts.

```bash
python -m kd distillation-baselines --matrix runs/experiment1-multidataset --dry-run
python -m kd distillation-baselines --matrix runs/experiment1-multidataset
```

To use several GPUs, give each process a disjoint job list; each control's recorded device is
`cuda:0`, so select the physical GPU with `CUDA_VISIBLE_DEVICES`:

```bash
jobs=($(python -c 'import json; print(*[j["id"] for j in json.load(open("runs/experiment1-multidataset/matrix_summary.json"))["jobs"] if j["status"] == "complete"])'))
gpus=(0 1 2 3)
mkdir -p logs
for i in "${!gpus[@]}"; do
  mine=(); for j in "${!jobs[@]}"; do ((j % ${#gpus[@]} == i)) && mine+=("${jobs[j]}"); done
  CUDA_VISIBLE_DEVICES=${gpus[i]} nohup python -m kd distillation-baselines \
    --matrix runs/experiment1-multidataset --jobs "${mine[@]}" \
    > logs/distillation-baselines-gpu${gpus[i]}.log 2>&1 &
done
wait
python scripts/baseline_comparison_table.py --matrix runs/experiment1-multidataset \
  --output docs/distillation-baselines-results.md
```

The last command regenerates [the results table](distillation-baselines-results.md).

## Other studies

Use an indexed device directly:

```bash
python -m kd neuron-surgery-baselines \
  --study-config configs/neuron_surgery/NAME.json \
  --output runs/NAME-baselines \
  --root data \
  --device cuda:0

python -m kd neuron-surgery-study \
  --study-config configs/neuron_surgery/NAME.json \
  --baseline runs/NAME-baselines \
  --output runs/NAME-surgery \
  --device cuda:0
```

For multiple GPUs, start separate processes with distinct output directories:

```bash
CUDA_VISIBLE_DEVICES=0 python -m kd neuron-surgery-baselines ... --device cuda:0
CUDA_VISIBLE_DEVICES=1 python -m kd neuron-surgery-baselines ... --device cuda:0
```

Do not let two jobs write to the same run directory. Start with the protocol batch size of 128 so the
comparison remains matched. Increasing batch size can improve utilization, but it changes the optimizer
trajectory and must be treated as a separate experiment rather than a hardware-only optimization.

Watch `nvidia-smi dmon` during the first full epoch. Low utilization with free memory usually indicates
that loader workers should be increased; sustained memory pressure should be handled by lowering batch
size, not by silently changing precision midway through a run.
