#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/cifar10-lt-neuron-campaign}"
STUDY_CONFIG="${STUDY_CONFIG:-configs/neuron_surgery/confirmatory_f100.json}"
REGISTRY_URL="${KD_DATASET_REGISTRY:-ssh://git@gitea.izzus.dev:2222/syafiq/kd-repair.git}"
REGISTRY_REF="${KD_DATASET_REGISTRY_REF:-catalog-v1.2.0}"
DATASET="${DATASET:-cifar10}"
DATASET_VERSION="${DATASET_VERSION:-1.0}"
DATASET_PROFILE="${DATASET_PROFILE:-lt-if100}"

DRY_RUN_ONLY=0
SKIP_FETCH=0
SKIP_TESTS=0
SKIP_PREFLIGHT=0
BASELINE_ONLY=0
STUDY_ONLY=0

usage() {
  cat <<'EOF'
Run Experiment 1 on one CUDA device.

Usage: scripts/run_gpu_experiment1.sh [options]

Options:
  --device DEVICE       CUDA device passed to the project (default: cuda:0)
  --python PATH         Python interpreter (default: python)
  --data-root PATH      Dataset directory (default: data)
  --output-root PATH    Experiment output directory
  --config PATH         Neuron-surgery study configuration
  --registry-url URL    Private dataset-registry SSH URL
  --registry-ref REF    Immutable dataset catalog tag
  --skip-fetch          Reuse already prepared local data
  --skip-tests          Skip the unit-test suite
  --skip-preflight      Skip nvidia-smi and the CUDA runtime check
  --dry-run-only        Validate without starting expensive training
  --baseline-only       Stop after matched baseline training
  --study-only          Reuse existing baselines and run only surgery/KD
  -h, --help            Show this help

Environment variables matching the defaults above are also supported:
PYTHON_BIN, DEVICE, DATA_ROOT, OUTPUT_ROOT, STUDY_CONFIG,
KD_DATASET_REGISTRY, KD_DATASET_REGISTRY_REF, DATASET,
DATASET_VERSION, and DATASET_PROFILE.
EOF
}

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --config) STUDY_CONFIG="$2"; shift 2 ;;
    --registry-url) REGISTRY_URL="$2"; shift 2 ;;
    --registry-ref) REGISTRY_REF="$2"; shift 2 ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --dry-run-only) DRY_RUN_ONLY=1; shift ;;
    --baseline-only) BASELINE_ONLY=1; shift ;;
    --study-only) STUDY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((BASELINE_ONLY && STUDY_ONLY)); then
  printf '%s\n' '--baseline-only and --study-only cannot be combined.' >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ ! -f "$STUDY_CONFIG" ]]; then
  printf 'Study configuration not found: %s\n' "$STUDY_CONFIG" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p logs
STUDY_LABEL="$(basename -- "$STUDY_CONFIG" .json)"
LOG_FILE="logs/experiment1-${STUDY_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Python:     %s\n' "$PYTHON_BIN"
printf 'Device:     %s\n' "$DEVICE"
printf 'Config:     %s\n' "$STUDY_CONFIG"
printf 'Data:       %s\n' "$DATA_ROOT"
printf 'Output:     %s\n' "$OUTPUT_ROOT"
printf 'Log:        %s\n' "$LOG_FILE"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11 or newer is required; found {sys.version.split()[0]}")
PY

if ((SKIP_FETCH == 0)); then
  command -v git >/dev/null 2>&1 || { printf '%s\n' 'git is required.' >&2; exit 1; }
  git lfs version
  "$PYTHON_BIN" -m kd dataset registry-config \
    --url "$REGISTRY_URL" \
    --ref "$REGISTRY_REF"
  "$PYTHON_BIN" -m kd dataset registry-status
  "$PYTHON_BIN" -m kd dataset fetch "$DATASET" \
    --version "$DATASET_VERSION" \
    --profile "$DATASET_PROFILE" \
    --root "$DATA_ROOT"
fi

"$PYTHON_BIN" -m kd dataset verify "$DATASET" \
  --version "$DATASET_VERSION" \
  --profile "$DATASET_PROFILE" \
  --root "$DATA_ROOT"

if ((SKIP_PREFLIGHT == 0)); then
  nvidia-smi
  "$PYTHON_BIN" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))
PY
  "$PYTHON_BIN" -m kd cuda-check --device "$DEVICE" --precision auto
fi

if ((SKIP_TESTS == 0)); then
  "$PYTHON_BIN" -m unittest discover -s tests
fi

BASELINE_DIR="${OUTPUT_ROOT}/baselines/confirmatory"
STUDY_DIR="${OUTPUT_ROOT}/studies/confirmatory"

if ((STUDY_ONLY == 0)); then
  "$PYTHON_BIN" -m kd neuron-surgery-baselines \
    --study-config "$STUDY_CONFIG" \
    --output "$BASELINE_DIR" \
    --root "$DATA_ROOT" \
    --device "$DEVICE" \
    --dry-run

  if ((DRY_RUN_ONLY == 0)); then
    "$PYTHON_BIN" -m kd neuron-surgery-baselines \
      --study-config "$STUDY_CONFIG" \
      --output "$BASELINE_DIR" \
      --root "$DATA_ROOT" \
      --device "$DEVICE"
  fi
fi

if ((BASELINE_ONLY)); then
  printf 'Baseline stage complete. Log: %s\n' "$LOG_FILE"
  exit 0
fi

if ((DRY_RUN_ONLY && STUDY_ONLY == 0)); then
  if [[ ! -f "${BASELINE_DIR}/teacher_f100_confirmatory/config.json" ]]; then
    printf '%s\n' 'Baseline dry-run passed. Study validation requires completed baseline artifacts.'
    printf 'Run again without --dry-run-only to continue. Log: %s\n' "$LOG_FILE"
    exit 0
  fi
fi

"$PYTHON_BIN" -m kd neuron-surgery-study \
  --study-config "$STUDY_CONFIG" \
  --baseline "$BASELINE_DIR" \
  --output "$STUDY_DIR" \
  --device "$DEVICE" \
  --dry-run

if ((DRY_RUN_ONLY == 0)); then
  "$PYTHON_BIN" -m kd neuron-surgery-study \
    --study-config "$STUDY_CONFIG" \
    --baseline "$BASELINE_DIR" \
    --output "$STUDY_DIR" \
    --device "$DEVICE"
fi

printf 'Experiment 1 complete. Results: %s\n' "$STUDY_DIR"
printf 'Log: %s\n' "$LOG_FILE"
