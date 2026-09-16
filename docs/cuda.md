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
