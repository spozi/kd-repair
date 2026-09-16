#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/experiment1-multidataset}"
REGISTRY_URL="${KD_DATASET_REGISTRY:-ssh://git@gitea.izzus.dev:2222/syafiq/kd-repair.git}"
REGISTRY_REF="${KD_DATASET_REGISTRY_REF:-catalog-v1.2.0}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
DATASETS_CSV="${EXPERIMENT_DATASETS:-cinic10,svhn,cifar100,cifar10,gtsrb}"
PROFILES_CSV="${EXPERIMENT_PROFILES:-balanced,lt-if10,lt-if50,lt-if100}"

DRY_RUN_ONLY=0
SKIP_FETCH=0
SKIP_TESTS=0
SKIP_PREFLIGHT=0

usage() {
  cat <<'EOF'
Run the multi-dataset Experiment 1 matrix on four CUDA GPUs.

Usage: scripts/run_gpu_experiment1_4gpu.sh [options]

Options:
  --gpu-ids IDS         Four comma-separated physical GPU IDs (default: 0,1,2,3)
  --datasets NAMES      Comma-separated dataset subset
  --profiles NAMES      Comma-separated profile subset
  --python PATH         Python interpreter (default: python)
  --data-root PATH      Dataset directory (default: data)
  --output-root PATH    Root for plans and isolated study outputs
  --registry-url URL    Private dataset-registry SSH URL
  --registry-ref REF    Immutable dataset catalog tag
  --skip-fetch          Reuse already prepared local datasets
  --skip-tests          Skip the unit-test suite
  --skip-preflight      Skip per-GPU CUDA runtime checks
  --dry-run-only        Validate generated baseline protocols without training
  -h, --help            Show this help

Default matrix: CIFAR-10, CIFAR-100, SVHN, CINIC-10, and GTSRB, each with
balanced, IF10, IF50, and IF100 profiles (20 studies). The queue keeps four
workers active and resumes validated artifacts when rerun with the same output.
EOF
}

while (($#)); do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --datasets) DATASETS_CSV="$2"; shift 2 ;;
    --profiles) PROFILES_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --registry-url) REGISTRY_URL="$2"; shift 2 ;;
    --registry-ref) REGISTRY_REF="$2"; shift 2 ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --dry-run-only) DRY_RUN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
  printf '%s\n' 'Bash 5.1 or newer is required for the dynamic GPU queue.' >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
IFS=',' read -r -a DATASETS_ARRAY <<< "$DATASETS_CSV"
IFS=',' read -r -a PROFILES_ARRAY <<< "$PROFILES_CSV"
if ((${#GPU_IDS_ARRAY[@]} != 4)); then
  printf 'Exactly four GPU IDs are required; received: %s\n' "$GPU_IDS_CSV" >&2
  exit 2
fi
if ((${#DATASETS_ARRAY[@]} == 0 || ${#PROFILES_ARRAY[@]} == 0)); then
  printf '%s\n' 'At least one dataset and profile are required.' >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    printf 'GPU IDs must be nonnegative integers; received: %s\n' "$gpu" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    printf 'GPU IDs must be unique; repeated: %s\n' "$gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

cd "$REPO_ROOT"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" logs/experiment1-4gpu
PLAN_DIR="$OUTPUT_ROOT/matrix-plan"
SUPERVISOR_LOG="logs/experiment1-4gpu/supervisor-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

printf 'Multi-dataset Experiment 1\n'
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'GPUs:       %s\n' "$GPU_IDS_CSV"
printf 'Datasets:   %s\n' "$DATASETS_CSV"
printf 'Profiles:   %s\n' "$PROFILES_CSV"
printf 'Data:       %s\n' "$DATA_ROOT"
printf 'Output:     %s\n' "$OUTPUT_ROOT"
printf 'Log:        %s\n' "$SUPERVISOR_LOG"

"$PYTHON_BIN" -m kd neuron-surgery-matrix-plan \
  --output "$PLAN_DIR" \
  --datasets "${DATASETS_ARRAY[@]}" \
  --profiles "${PROFILES_ARRAY[@]}"

if ((SKIP_FETCH == 0)); then
  command -v git >/dev/null 2>&1 || { printf '%s\n' 'git is required.' >&2; exit 1; }
  git lfs version
  "$PYTHON_BIN" -m kd dataset registry-config \
    --url "$REGISTRY_URL" \
    --ref "$REGISTRY_REF"
  "$PYTHON_BIN" -m kd dataset registry-status
  for dataset in "${DATASETS_ARRAY[@]}"; do
    "$PYTHON_BIN" -m kd dataset fetch "$dataset" \
      --version 1.0 \
      --profile balanced \
      --root "$DATA_ROOT"
  done
fi

for dataset in "${DATASETS_ARRAY[@]}"; do
  for profile in "${PROFILES_ARRAY[@]}"; do
    "$PYTHON_BIN" -m kd dataset verify "$dataset" \
      --version 1.0 \
      --profile "$profile" \
      --root "$DATA_ROOT"
  done
done

if ((SKIP_PREFLIGHT == 0)); then
  nvidia-smi
  for gpu in "${GPU_IDS_ARRAY[@]}"; do
    printf 'CUDA preflight on physical GPU %s\n' "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m kd cuda-check \
      --device cuda:0 \
      --precision auto
  done
fi

if ((SKIP_TESTS == 0)); then
  "$PYTHON_BIN" -m unittest discover -s tests
fi

declare -a JOB_IDS=() JOB_DATASETS=() JOB_VERSIONS=() JOB_PROFILES=()
declare -a JOB_CONFIGS=() JOB_OUTPUTS=()
while IFS=$'\t' read -r id dataset version profile config output; do
  JOB_IDS+=("$id")
  JOB_DATASETS+=("$dataset")
  JOB_VERSIONS+=("$version")
  JOB_PROFILES+=("$profile")
  JOB_CONFIGS+=("$config")
  JOB_OUTPUTS+=("$output")
done < "$PLAN_DIR/jobs.tsv"

if ((${#JOB_IDS[@]} == 0)); then
  printf '%s\n' 'The matrix plan contains no jobs.' >&2
  exit 1
fi

declare -A PID_GPU=() PID_JOB=()
next_job=0
active_jobs=0
failures=0

stop_children() {
  printf '%s\n' 'Stopping active experiment workers...'
  for pid in "${!PID_GPU[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap stop_children INT TERM

launch_job() {
  local index="$1" gpu="$2"
  local id="${JOB_IDS[$index]}"
  local worker_log="logs/experiment1-4gpu/${id}.log"
  local -a command=(
    "$SCRIPT_DIR/run_gpu_experiment1.sh"
    --python "$PYTHON_BIN"
    --device cuda:0
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT/${JOB_OUTPUTS[$index]}"
    --config "${JOB_CONFIGS[$index]}"
    --registry-url "$REGISTRY_URL"
    --registry-ref "$REGISTRY_REF"
    --skip-fetch
    --skip-tests
    --skip-preflight
  )
  if ((DRY_RUN_ONLY)); then
    command+=(--dry-run-only)
  fi
  printf 'Starting %-24s on physical GPU %s (log: %s)\n' "$id" "$gpu" "$worker_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export DATASET="${JOB_DATASETS[$index]}"
    export DATASET_VERSION="${JOB_VERSIONS[$index]}"
    export DATASET_PROFILE="${JOB_PROFILES[$index]}"
    exec "${command[@]}"
  ) >"$worker_log" 2>&1 &
  local pid="$!"
  PID_GPU[$pid]="$gpu"
  PID_JOB[$pid]="$id"
  active_jobs=$((active_jobs + 1))
}

for gpu in "${GPU_IDS_ARRAY[@]}"; do
  if ((next_job >= ${#JOB_IDS[@]})); then
    break
  fi
  launch_job "$next_job" "$gpu"
  next_job=$((next_job + 1))
done

while ((active_jobs)); do
  finished_pid=""
  if wait -n -p finished_pid; then
    exit_code=0
  else
    exit_code=$?
  fi
  gpu="${PID_GPU[$finished_pid]}"
  id="${PID_JOB[$finished_pid]}"
  unset 'PID_GPU[$finished_pid]' 'PID_JOB[$finished_pid]'
  active_jobs=$((active_jobs - 1))
  if ((exit_code == 0)); then
    printf 'Completed: %s\n' "$id"
  else
    printf 'FAILED: %s (exit %s, see logs/experiment1-4gpu/%s.log)\n' \
      "$id" "$exit_code" "$id" >&2
    failures=$((failures + 1))
  fi
  if ((next_job < ${#JOB_IDS[@]})); then
    launch_job "$next_job" "$gpu"
    next_job=$((next_job + 1))
  fi
done

if ((failures)); then
  printf '%s of %s jobs failed. Successful jobs remain resumable.\n' \
    "$failures" "${#JOB_IDS[@]}" >&2
  exit 1
fi

if ((DRY_RUN_ONLY)); then
  printf 'All %s baseline protocols validated without training.\n' "${#JOB_IDS[@]}"
else
  "$PYTHON_BIN" -m kd neuron-surgery-matrix-summary \
    --plan "$PLAN_DIR" \
    --output-root "$OUTPUT_ROOT"
  printf 'All %s jobs finished. Summary: %s/matrix_summary.md\n' \
    "${#JOB_IDS[@]}" "$OUTPUT_ROOT"
fi
printf 'Supervisor log: %s\n' "$SUPERVISOR_LOG"
