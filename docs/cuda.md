# CUDA execution

The project uses one explicit CUDA device per process. This is intentional for the compact 32x32
models: independent dataset, profile, and seed jobs usually utilize a multi-GPU server better than
splitting one small model with data parallelism.

## Preflight

Install the NVIDIA driver and a CUDA-enabled PyTorch wheel, clone the source repository, fetch the
selected datasets, and run:

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
the `kd` Conda environment (or a local `.venv` when Conda is absent) and installs the CUDA PyTorch
build plus this project. Installs are skipped when the dependencies already import; `--force-bootstrap`
reinstalls them. Long operations are no longer silent — dataset downloads and Git LFS transfers report
transferred bytes, rate, and ETA, pip shows download progress, and the queue prints a status table
every `--status-interval` seconds listing each GPU's job, elapsed time, and latest log line. Use
`--stream-logs` to mirror worker output into the console, and `--fetch-all-datasets` to prepare the
whole catalog instead of only the datasets in the matrix.

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
