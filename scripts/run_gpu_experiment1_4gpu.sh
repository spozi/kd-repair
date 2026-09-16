#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN_SET=0
if [[ -n "${PYTHON_BIN+x}" ]]; then
  PYTHON_BIN_SET=1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-kd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/experiment1-multidataset}"
REGISTRY_URL="${KD_DATASET_REGISTRY:-https://gitea.izzus.dev/syafiq/kd-repair.git}"
REGISTRY_REF="${KD_DATASET_REGISTRY_REF:-catalog-v1.2.0}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
DATASETS_CSV="${EXPERIMENT_DATASETS:-cinic10,svhn,cifar100,cifar10,gtsrb}"
PROFILES_CSV="${EXPERIMENT_PROFILES:-balanced,lt-if10,lt-if50,lt-if100}"
STATUS_INTERVAL="${STATUS_INTERVAL:-30}"

DRY_RUN_ONLY=0
SKIP_FETCH=0
SKIP_TESTS=0
SKIP_PREFLIGHT=0
SKIP_BOOTSTRAP=0
FORCE_BOOTSTRAP=0
FETCH_ALL=0
STREAM_LOGS=0

# Child processes write through a tee pipe; keep their output unbuffered so progress is live.
export PYTHONUNBUFFERED=1
export KD_PROGRESS="${KD_PROGRESS:-1}"

CATALOG_DATASETS=(
  cifar10 cifar100 svhn cinic10 gtsrb fashionmnist pathmnist bloodmnist
  dermamnist organamnist caltech101 eurosat stl10
)

usage() {
  cat <<'EOF'
Run the multi-dataset Experiment 1 matrix on four CUDA GPUs.

Usage: scripts/run_gpu_experiment1_4gpu.sh [options]

Options:
  --gpu-ids IDS         Four comma-separated physical GPU IDs (default: 0,1,2,3)
  --datasets NAMES      Comma-separated dataset subset
  --profiles NAMES      Comma-separated profile subset
  --python PATH         Python interpreter; skips automatic provisioning
  --conda-env NAME      Conda environment to provision when no venv is active (default: kd)
  --pytorch-index URL   PyTorch CUDA wheel index (default: cu132)
  --skip-bootstrap      Use the current/default Python without provisioning anything
  --force-bootstrap     Reinstall dependencies even when they are already importable
  --data-root PATH      Dataset directory (default: data)
  --output-root PATH    Root for plans and isolated study outputs
  --registry-url URL    Private dataset-registry SSH URL
  --registry-ref REF    Immutable dataset catalog tag
  --skip-fetch          Reuse already prepared local datasets
  --fetch-all-datasets  Prepare the whole catalog, not just the matrix datasets
  --skip-tests          Skip the unit-test suite
  --skip-preflight      Skip per-GPU CUDA runtime checks
  --dry-run-only        Validate generated baseline protocols without training
  --status-interval N   Seconds between queue status reports; 0 disables (default: 30)
  --stream-logs         Mirror each worker's output into the console, prefixed by job id
  -h, --help            Show this help

Environment provisioning is automatic: an activated virtualenv is used as-is, a
Conda environment is created when Conda is available, and otherwise a local
.venv is created. Dependencies install only when missing; pass --force-bootstrap
to reinstall.

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
    --python) PYTHON_BIN="$2"; PYTHON_BIN_SET=1; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --pytorch-index) PYTORCH_INDEX_URL="$2"; shift 2 ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --force-bootstrap) FORCE_BOOTSTRAP=1; shift ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --registry-url) REGISTRY_URL="$2"; shift 2 ;;
    --registry-ref) REGISTRY_REF="$2"; shift 2 ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --fetch-all-datasets) FETCH_ALL=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --dry-run-only) DRY_RUN_ONLY=1; shift ;;
    --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
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
if [[ ! "$STATUS_INTERVAL" =~ ^[0-9]+$ ]]; then
  printf 'Status interval must be a nonnegative integer; received: %s\n' "$STATUS_INTERVAL" >&2
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

LOG_DIR="logs/experiment1-4gpu"
STATE_DIR="${LOG_DIR}/.state"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
rm -rf "$STATE_DIR"
mkdir -p "$STATE_DIR"
PLAN_DIR="$OUTPUT_ROOT/matrix-plan"
SUPERVISOR_LOG="${LOG_DIR}/supervisor-$(date -u +%Y%m%dT%H%M%SZ).log"
# Provisioning happens after this point so its output is visible and logged too.
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() { date -u +%H:%M:%SZ; }

log_stage() { printf '\n=== [%s] %s ===\n' "$(timestamp)" "$*"; }

format_elapsed() {
  local seconds="$1"
  printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

pip_progress_flag() {
  # pip hides its bar when stdout is a pipe; "raw" keeps byte counts in the log.
  if [[ -t 1 ]]; then printf '%s' 'on'; else printf '%s' 'raw'; fi
}

torch_supports_local_gpu() {
  # A wheel built for the wrong architecture imports cleanly and only fails at the first
  # kernel launch, so compare each GPU's compute capability against the wheel's arch list.
  "$1" - <<'PY' >/dev/null 2>&1
import sys

try:
    import torch
except Exception:
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(0)
architectures = torch.cuda.get_arch_list()
for index in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(index)
    if f"sm_{major}{minor}" not in architectures:
        raise SystemExit(1)
PY
}

report_gpu_mismatch() {
  "$1" - 2>/dev/null <<'PY' || true
import torch

names = sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())})
caps = sorted({"sm_%d%d" % torch.cuda.get_device_capability(i)
               for i in range(torch.cuda.device_count())})
print(f"  installed torch {torch.__version__} has kernels for: "
      f"{' '.join(torch.cuda.get_arch_list())}")
print(f"  this machine has: {', '.join(names)} ({', '.join(caps)})")
PY
}

python_has_requirements() {
  "$1" - <<'PY' >/dev/null 2>&1 || return 1
import importlib.util
import sys

if sys.version_info < (3, 11):
    raise SystemExit(1)
for module in ("torch", "torchvision", "numpy", "PIL", "scipy", "matplotlib"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(1)
PY
  # Being importable is not enough: the wheel must carry kernels for this machine's GPUs.
  torch_supports_local_gpu "$1"
}

ensure_requirements() {
  local interpreter="$1" progress
  if ((FORCE_BOOTSTRAP == 0)) && python_has_requirements "$interpreter"; then
    printf 'Dependencies already satisfied.\n'
    return
  fi
  progress="$(pip_progress_flag)"
  log_stage "Installing dependencies into ${interpreter}"
  "$interpreter" -m pip install --upgrade pip
  if ((FORCE_BOOTSTRAP)) || ! torch_supports_local_gpu "$interpreter"; then
    if "$interpreter" -c 'import torch' >/dev/null 2>&1; then
      printf 'The installed PyTorch has no kernels for this GPU; reinstalling.\n'
      report_gpu_mismatch "$interpreter"
    fi
    printf 'Installing PyTorch from %s (multi-GB download; progress follows)\n' "$PYTORCH_INDEX_URL"
    # --force-reinstall: pip would otherwise treat a wrong-architecture build as satisfying.
    "$interpreter" -m pip install --progress-bar "$progress" --force-reinstall \
      torch torchvision --index-url "$PYTORCH_INDEX_URL"
  fi
  "$interpreter" -m pip install --progress-bar "$progress" -e "${REPO_ROOT}[reports]"
}

bootstrap_conda() {
  log_stage "Provisioning Conda environment ${CONDA_ENV}"
  if ! conda run -n "$CONDA_ENV" python -c 'import sys' >/dev/null 2>&1; then
    printf 'Creating Conda environment %s with python %s\n' "$CONDA_ENV" "$PYTHON_VERSION"
    # --no-capture-output streams solver and download progress instead of buffering it.
    conda create -n "$CONDA_ENV" "python=$PYTHON_VERSION" -y
  fi
  local resolved
  resolved="$(conda run -n "$CONDA_ENV" --no-capture-output \
    python -c 'import sys; print(sys.executable)' | tr -d '\r' | tail -n 1)"
  if [[ ! -x "$resolved" ]]; then
    printf 'Could not resolve the Conda interpreter for environment %s\n' "$CONDA_ENV" >&2
    exit 1
  fi
  PYTHON_BIN="$resolved"
  printf 'Conda interpreter: %s\n' "$PYTHON_BIN"
  ensure_requirements "$PYTHON_BIN"
}

bootstrap_venv() {
  local venv_dir="${REPO_ROOT}/.venv" base_python=""
  base_python="$(command -v python3 || command -v python || true)"
  if [[ -z "$base_python" ]]; then
    printf '%s\n' 'No python3 found to create a virtual environment.' >&2
    exit 1
  fi
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    log_stage "Creating virtual environment ${venv_dir}"
    "$base_python" -m venv "$venv_dir"
  fi
  PYTHON_BIN="${venv_dir}/bin/python"
  printf 'Virtualenv interpreter: %s\n' "$PYTHON_BIN"
  ensure_requirements "$PYTHON_BIN"
}

bootstrap_environment() {
  if ((SKIP_BOOTSTRAP)); then
    printf 'Provisioning skipped; using %s\n' "$PYTHON_BIN"
    return
  fi
  if ((PYTHON_BIN_SET)); then
    log_stage "Using requested interpreter ${PYTHON_BIN}"
    ensure_requirements "$PYTHON_BIN"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    log_stage "Using the activated virtual environment ${VIRTUAL_ENV}"
    ensure_requirements "$PYTHON_BIN"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    bootstrap_conda
    return
  fi
  bootstrap_venv
}

printf 'Multi-dataset Experiment 1\n'
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'GPUs:       %s\n' "$GPU_IDS_CSV"
printf 'Datasets:   %s\n' "$DATASETS_CSV"
printf 'Profiles:   %s\n' "$PROFILES_CSV"
printf 'Data:       %s\n' "$DATA_ROOT"
printf 'Output:     %s\n' "$OUTPUT_ROOT"
printf 'Log:        %s\n' "$SUPERVISOR_LOG"

bootstrap_environment

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
printf 'Python:     %s\n' "$PYTHON_BIN"

log_stage 'Freezing the job matrix'
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
  if ((FETCH_ALL)); then
    FETCH_LIST=("${CATALOG_DATASETS[@]}")
  else
    FETCH_LIST=("${DATASETS_ARRAY[@]}")
  fi
  log_stage "Preparing ${#FETCH_LIST[@]} dataset(s)"
  fetch_position=0
  for dataset in "${FETCH_LIST[@]}"; do
    fetch_position=$((fetch_position + 1))
    printf '[%s] dataset %s/%s: %s\n' \
      "$(timestamp)" "$fetch_position" "${#FETCH_LIST[@]}" "$dataset"
    "$PYTHON_BIN" -m kd dataset fetch "$dataset" \
      --profile balanced \
      --root "$DATA_ROOT"
  done
fi

log_stage 'Verifying prepared datasets'
for dataset in "${DATASETS_ARRAY[@]}"; do
  for profile in "${PROFILES_ARRAY[@]}"; do
    "$PYTHON_BIN" -m kd dataset verify "$dataset" \
      --version 1.0 \
      --profile "$profile" \
      --root "$DATA_ROOT"
  done
done

if ((SKIP_PREFLIGHT == 0)); then
  log_stage 'CUDA preflight'
  nvidia-smi
  for gpu in "${GPU_IDS_ARRAY[@]}"; do
    printf 'CUDA preflight on physical GPU %s\n' "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m kd cuda-check \
      --device cuda:0 \
      --precision auto
  done
fi

if ((SKIP_TESTS == 0)); then
  log_stage 'Unit tests'
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

declare -A PID_GPU=() PID_JOB=() PID_START=()
next_job=0
active_jobs=0
started_jobs=0
completed_jobs=0
failures=0
MONITOR_PID=""
QUEUE_START="$(date +%s)"

write_progress() {
  printf '%s\t%s\t%s\t%s\n' \
    "$completed_jobs" "$failures" "${#JOB_IDS[@]}" "$started_jobs" > "$STATE_DIR/progress"
}

status_monitor() {
  set +e
  local interval="$1"
  while :; do
    sleep "$interval"
    local completed=0 failed=0 total=0 started=0 now
    if [[ -r "$STATE_DIR/progress" ]]; then
      IFS=$'\t' read -r completed failed total started < "$STATE_DIR/progress"
    fi
    now="$(date +%s)"
    printf '[%s] status: %s/%s finished (%s failed), %s running, elapsed %s\n' \
      "$(timestamp)" "${completed:-0}" "${total:-0}" "${failed:-0}" \
      "$((${started:-0} - ${completed:-0}))" \
      "$(format_elapsed $((now - QUEUE_START)))"
    local state
    for state in "$STATE_DIR"/gpu-*; do
      [[ -e "$state" ]] || continue
      local id start log last=""
      IFS=$'\t' read -r id start log < "$state" || continue
      [[ -n "${start:-}" ]] || continue
      if [[ -r "$log" ]]; then
        last="$(tail -n 40 "$log" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -n 1)"
      fi
      printf '    GPU %-3s %-26s %s  %s\n' \
        "${state##*/gpu-}" "$id" "$(format_elapsed $((now - start)))" "${last:0:90}"
    done
  done
}

cleanup_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_children() {
  printf '%s\n' 'Stopping active experiment workers...'
  cleanup_monitor
  for pid in "${!PID_GPU[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap stop_children INT TERM
trap cleanup_monitor EXIT

launch_job() {
  local index="$1" gpu="$2"
  local id="${JOB_IDS[$index]}"
  local worker_log="${LOG_DIR}/${id}.log"
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
  started_jobs=$((started_jobs + 1))
  printf '[%s] starting %-26s on physical GPU %s (job %s/%s, log: %s)\n' \
    "$(timestamp)" "$id" "$gpu" "$started_jobs" "${#JOB_IDS[@]}" "$worker_log"
  if ((STREAM_LOGS)); then
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export DATASET="${JOB_DATASETS[$index]}"
      export DATASET_VERSION="${JOB_VERSIONS[$index]}"
      export DATASET_PROFILE="${JOB_PROFILES[$index]}"
      exec "${command[@]}"
    ) > >(tee "$worker_log" | awk -v tag="$id" '{print "[" tag "] " $0; fflush()}') 2>&1 &
  else
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export DATASET="${JOB_DATASETS[$index]}"
      export DATASET_VERSION="${JOB_VERSIONS[$index]}"
      export DATASET_PROFILE="${JOB_PROFILES[$index]}"
      exec "${command[@]}"
    ) >"$worker_log" 2>&1 &
  fi
  local pid="$!"
  PID_GPU[$pid]="$gpu"
  PID_JOB[$pid]="$id"
  PID_START[$pid]="$(date +%s)"
  printf '%s\t%s\t%s\n' "$id" "${PID_START[$pid]}" "$worker_log" > "$STATE_DIR/gpu-${gpu}"
  active_jobs=$((active_jobs + 1))
  write_progress
}

log_stage "Running ${#JOB_IDS[@]} studies across ${#GPU_IDS_ARRAY[@]} GPUs"
write_progress

for gpu in "${GPU_IDS_ARRAY[@]}"; do
  if ((next_job >= ${#JOB_IDS[@]})); then
    break
  fi
  launch_job "$next_job" "$gpu"
  next_job=$((next_job + 1))
done

if ((STATUS_INTERVAL > 0)); then
  status_monitor "$STATUS_INTERVAL" &
  MONITOR_PID="$!"
  printf 'Queue status reported every %ss; per-job logs live in %s/\n' \
    "$STATUS_INTERVAL" "$LOG_DIR"
fi

while ((active_jobs)); do
  finished_pid=""
  # Wait only on worker PIDs so the status monitor is never mistaken for a job.
  if wait -n -p finished_pid "${!PID_JOB[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  if [[ -z "$finished_pid" ]]; then
    continue
  fi
  gpu="${PID_GPU[$finished_pid]}"
  id="${PID_JOB[$finished_pid]}"
  started_at="${PID_START[$finished_pid]}"
  unset 'PID_GPU[$finished_pid]' 'PID_JOB[$finished_pid]' 'PID_START[$finished_pid]'
  rm -f "$STATE_DIR/gpu-${gpu}"
  active_jobs=$((active_jobs - 1))
  completed_jobs=$((completed_jobs + 1))
  elapsed="$(format_elapsed $(($(date +%s) - started_at)))"
  if ((exit_code == 0)); then
    printf '[%s] completed %-26s in %s (%s/%s done)\n' \
      "$(timestamp)" "$id" "$elapsed" "$completed_jobs" "${#JOB_IDS[@]}"
  else
    failures=$((failures + 1))
    printf '[%s] FAILED %-26s after %s (exit %s, see %s/%s.log)\n' \
      "$(timestamp)" "$id" "$elapsed" "$exit_code" "$LOG_DIR" "$id" >&2
  fi
  write_progress
  if ((next_job < ${#JOB_IDS[@]})); then
    launch_job "$next_job" "$gpu"
    next_job=$((next_job + 1))
  fi
done

cleanup_monitor
printf '\nQueue finished in %s\n' "$(format_elapsed $(($(date +%s) - QUEUE_START)))"

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
