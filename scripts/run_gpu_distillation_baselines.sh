#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX="${MATRIX:-runs/experiment1-multidataset}"
DATA_ROOT="${DATA_ROOT:-data}"
REGISTRY_URL="${KD_DATASET_REGISTRY:-https://gitea.izzus.dev/syafiq/kd-repair.git}"
REGISTRY_REF="${KD_DATASET_REGISTRY_REF:-catalog-v1.2.0}"
GPU_IDS_CSV="${GPU_IDS:-}"
RUNS_PER_GPU="${RUNS_PER_GPU:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-3072}"
LAUNCH_GAP="${LAUNCH_GAP:-30}"
METHODS_CSV="${METHODS:-dkd,rld,loca}"
JOBS_CSV="${JOBS:-}"
STATUS_INTERVAL="${STATUS_INTERVAL:-60}"
SKIP_PREFLIGHT=0
SKIP_FETCH=0
DRY_RUN_ONLY=0

export PYTHONUNBUFFERED=1
export KD_PROGRESS="${KD_PROGRESS:-1}"
# Returns freed blocks to the driver in growable segments, so co-resident runs fragment less.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Run the DKD, RLD and LoCa distillation baselines with several runs sharing each GPU.

Usage: scripts/run_gpu_distillation_baselines.sh [options]

Options:
  --matrix PATH         Experiment 1 matrix root with matrix_summary.json
                        (default: runs/experiment1-multidataset)
  --gpu-ids IDS         Comma-separated physical GPU IDs (default: every detected GPU)
  --runs-per-gpu N      Concurrent runs allowed on one GPU (default: 4)
  --min-free-mib N      Start another run on a busy GPU only while it has this much
                        free memory (default: 3072)
  --launch-gap N        Seconds between starts on the same GPU, so the previous run's
                        memory is visible before the next is admitted (default: 30)
  --data-root PATH      Dataset directory; replaces the root recorded by the original
                        server, which is not part of any run's data identity (default: data)
  --skip-fetch          Reuse already prepared datasets instead of fetching and verifying
  --registry-url URL    Dataset-registry Git URL (default: the public HTTPS registry)
  --registry-ref REF    Immutable dataset catalog tag (default: catalog-v1.2.0)
  --methods NAMES       Comma-separated subset of dkd,rld,loca (default: all three)
  --jobs IDS            Comma-separated study ids (default: every completed study)
  --python PATH         Python interpreter of the pinned environment (default: python)
  --skip-preflight      Skip the per-GPU CUDA check
  --dry-run-only        Validate every study's baseline protocol without training
  --status-interval N   Seconds between status reports; 0 disables (default: 60)
  -h, --help            Show this help

One run is one (study, method) pair: three students trained in turn, then scored. Runs
write only inside their own study's baselines/distillation/<method>/ directory, so any
number can share a GPU. Every student keeps its control's batch size, schedule and
precision; concurrency changes only wall-clock time. Rerunning resumes finished students
and interrupted epochs. When all runs succeed the results table is regenerated.
EOF
}

while (($#)); do
  case "$1" in
    --matrix) MATRIX="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --registry-url) REGISTRY_URL="$2"; shift 2 ;;
    --registry-ref) REGISTRY_REF="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --runs-per-gpu) RUNS_PER_GPU="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --launch-gap) LAUNCH_GAP="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --jobs) JOBS_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --dry-run-only) DRY_RUN_ONLY=1; shift ;;
    --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in RUNS_PER_GPU MIN_FREE_MIB LAUNCH_GAP STATUS_INTERVAL; do
  if [[ ! "${!name}" =~ ^[0-9]+$ ]]; then
    printf '%s must be a nonnegative integer; received: %s\n' "$name" "${!name}" >&2
    exit 2
  fi
done
if ((RUNS_PER_GPU < 1)); then
  printf '%s\n' '--runs-per-gpu must be at least 1.' >&2
  exit 2
fi
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  printf '%s\n' 'Bash 4.4 or newer is required.' >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || { printf '%s\n' 'nvidia-smi not found.' >&2; exit 1; }

if [[ -z "$GPU_IDS_CSV" ]]; then
  GPU_IDS_CSV="$(nvidia-smi --query-gpu=index --format=csv,noheader \
    | tr -d '[:blank:]' | paste -sd, -)"
fi
IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
if ((${#GPU_IDS_ARRAY[@]} == 0)); then
  printf '%s\n' 'No GPUs selected or detected.' >&2
  exit 2
fi
IFS=',' read -r -a METHODS_ARRAY <<< "$METHODS_CSV"

cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
LOG_DIR="logs/distillation-baselines"
mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="${LOG_DIR}/supervisor-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$SUPERVISOR_LOG") 2>&1

timestamp() { date -u +%H:%M:%SZ; }
log_stage() { printf '\n=== [%s] %s ===\n' "$(timestamp)" "$*"; }
format_elapsed() {
  printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}
free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d '[:blank:]'
}
used_mib() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
    | tr -d '[:blank:]' | tr ',' '/'
}

printf 'Distillation baselines\n'
printf 'Matrix:       %s\n' "$MATRIX"
printf 'GPUs:         %s\n' "$GPU_IDS_CSV"
printf 'Runs per GPU: %s (admitted while >= %s MiB free)\n' "$RUNS_PER_GPU" "$MIN_FREE_MIB"
printf 'Methods:      %s\n' "$METHODS_CSV"
printf 'Data:         %s\n' "$DATA_ROOT"
printf 'Python:       %s\n' "$PYTHON_BIN"
printf 'Log:          %s\n' "$SUPERVISOR_LOG"

log_stage 'Planning runs'
declare -a RUN_JOBS=() RUN_METHODS=() DATASET_PROFILES=()
LOADER_WORKERS=0
while IFS=$'\t' read -r kind first second; do
  case "$kind" in
    workers) LOADER_WORKERS="$first" ;;
    data) DATASET_PROFILES+=("$first $second") ;;
    run) RUN_JOBS+=("$first"); RUN_METHODS+=("$second") ;;
  esac
done < <(MATRIX="$MATRIX" JOBS="$JOBS_CSV" METHODS="$METHODS_CSV" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

matrix = Path(os.environ["MATRIX"])
complete = {job["id"]: job for job in json.loads((matrix / "matrix_summary.json").read_text())["jobs"]
            if job["status"] == "complete"}
selected = [job for job in os.environ["JOBS"].split(",") if job] or list(complete)
missing = sorted(set(selected) - set(complete))
if missing:
    raise SystemExit(f"Jobs without a repaired-teacher comparison: {missing}")
workers = 0
for job_id in selected:
    job = complete[job_id]
    baselines = matrix / job["dataset"] / job["profile"] / "baselines" / "confirmatory"
    for config in baselines.glob("kd_*/config.json"):
        workers = max(workers, json.loads(config.read_text())["train"]["workers"])
    print(f"data\t{job['dataset']}\t{job['profile']}")
print(f"workers\t{workers}\t")
# Method-major order spreads each study's three runs apart, so co-resident runs rarely
# load the same dataset at once.
for method in os.environ["METHODS"].split(","):
    for job_id in selected:
        print(f"run\t{job_id}\t{method}")
PY
)
TOTAL_RUNS=${#RUN_JOBS[@]}
if ((TOTAL_RUNS == 0)); then
  printf '%s\n' 'No runs planned; check the matrix path and job ids above.' >&2
  exit 1
fi
SLOTS=$((${#GPU_IDS_ARRAY[@]} * RUNS_PER_GPU))
printf '%s runs (%s students) across %s GPU slots\n' "$TOTAL_RUNS" "$((TOTAL_RUNS * 3))" "$SLOTS"

CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
NEEDED=$((SLOTS * (LOADER_WORKERS + 1)))
if ((NEEDED > CORES)); then
  printf 'WARNING: %s slots x (%s loader workers + 1) = %s processes on %s CPU cores.\n' \
    "$SLOTS" "$LOADER_WORKERS" "$NEEDED" "$CORES"
  printf '%s\n' '         The CPU may become the bottleneck; lower --runs-per-gpu if GPU utilization stays low.'
fi

if ((SKIP_FETCH == 0)); then
  log_stage 'Preparing datasets'
  git lfs version
  "$PYTHON_BIN" -m kd dataset registry-config --url "$REGISTRY_URL" --ref "$REGISTRY_REF"
  declare -A FETCHED=()
  for pair in "${DATASET_PROFILES[@]}"; do
    dataset="${pair% *}"
    if [[ -z "${FETCHED[$dataset]:-}" ]]; then
      printf '[%s] fetching %s\n' "$(timestamp)" "$dataset"
      "$PYTHON_BIN" -m kd dataset fetch "$dataset" --profile balanced --root "$DATA_ROOT"
      FETCHED[$dataset]=1
    fi
  done
  for pair in "${DATASET_PROFILES[@]}"; do
    "$PYTHON_BIN" -m kd dataset verify "${pair% *}" --profile "${pair#* }" --root "$DATA_ROOT"
  done
fi

log_stage 'Validating baseline protocols'
IFS=',' read -r -a SELECTED_JOBS <<< "$JOBS_CSV"
validate=("$PYTHON_BIN" -m kd distillation-baselines --matrix "$MATRIX" --data-root "$DATA_ROOT" --methods "${METHODS_ARRAY[@]}" --dry-run)
if ((${#SELECTED_JOBS[@]})); then
  validate+=(--jobs "${SELECTED_JOBS[@]}")
fi
"${validate[@]}" > "${LOG_DIR}/dry-run.json"
printf 'All %s protocols resolve; details in %s/dry-run.json\n' "$TOTAL_RUNS" "$LOG_DIR"
if ((DRY_RUN_ONLY)); then
  exit 0
fi

if ((SKIP_PREFLIGHT == 0)); then
  log_stage 'CUDA preflight'
  for gpu in "${GPU_IDS_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m kd cuda-check --device cuda:0 --precision auto
  done
fi

declare -A PID_GPU=() PID_RUN=() PID_START=() GPU_ACTIVE=() GPU_LAST_START=()
for gpu in "${GPU_IDS_ARRAY[@]}"; do
  GPU_ACTIVE[$gpu]=0
  GPU_LAST_START[$gpu]=0
done
next_run=0
finished=0
failures=0
QUEUE_START="$(date +%s)"
last_status="$QUEUE_START"

stop_children() {
  printf '%s\n' 'Stopping active runs; rerun the same command to resume.'
  for pid in "${!PID_GPU[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap stop_children INT TERM

run_log() { printf '%s/%s-%s.log' "$LOG_DIR" "${RUN_JOBS[$1]}" "${RUN_METHODS[$1]}"; }

launch() {
  local index="$1" gpu="$2" log
  log="$(run_log "$index")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m kd distillation-baselines --matrix "$MATRIX" \
    --data-root "$DATA_ROOT" --jobs "${RUN_JOBS[$index]}" --methods "${RUN_METHODS[$index]}" >"$log" 2>&1 &
  local pid="$!"
  PID_GPU[$pid]="$gpu"
  PID_RUN[$pid]="$index"
  PID_START[$pid]="$(date +%s)"
  GPU_ACTIVE[$gpu]=$((GPU_ACTIVE[$gpu] + 1))
  GPU_LAST_START[$gpu]="${PID_START[$pid]}"
  printf '[%s] start  %-22s %-4s on GPU %s (%s running there; run %s/%s)\n' \
    "$(timestamp)" "${RUN_JOBS[$index]}" "${RUN_METHODS[$index]}" "$gpu" \
    "${GPU_ACTIVE[$gpu]}" "$((index + 1))" "$TOTAL_RUNS"
}

# A GPU accepts a run when it has a free slot, its last start has had time to allocate, and
# (unless it is idle) enough memory remains for one more run.
admit() {
  local gpu="$1" now
  now="$(date +%s)"
  ((GPU_ACTIVE[$gpu] < RUNS_PER_GPU)) || return 1
  ((GPU_ACTIVE[$gpu] == 0)) && return 0
  ((now - GPU_LAST_START[$gpu] >= LAUNCH_GAP)) || return 1
  (($(free_mib "$gpu") >= MIN_FREE_MIB))
}

reap() {
  local pid code index gpu elapsed
  for pid in "${!PID_RUN[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    if wait "$pid"; then code=0; else code=$?; fi
    index="${PID_RUN[$pid]}"
    gpu="${PID_GPU[$pid]}"
    elapsed="$(format_elapsed $(($(date +%s) - PID_START[$pid])))"
    unset 'PID_RUN[$pid]' 'PID_GPU[$pid]' 'PID_START[$pid]'
    GPU_ACTIVE[$gpu]=$((GPU_ACTIVE[$gpu] - 1))
    finished=$((finished + 1))
    if ((code == 0)); then
      printf '[%s] done   %-22s %-4s in %s (%s/%s finished)\n' "$(timestamp)" \
        "${RUN_JOBS[$index]}" "${RUN_METHODS[$index]}" "$elapsed" "$finished" "$TOTAL_RUNS"
    else
      failures=$((failures + 1))
      printf '[%s] FAILED %-22s %-4s after %s (exit %s, see %s)\n' "$(timestamp)" \
        "${RUN_JOBS[$index]}" "${RUN_METHODS[$index]}" "$elapsed" "$code" "$(run_log "$index")" >&2
    fi
  done
}

status() {
  local now gpu pid index last
  now="$(date +%s)"
  printf '[%s] status: %s/%s finished (%s failed), %s running, elapsed %s\n' "$(timestamp)" \
    "$finished" "$TOTAL_RUNS" "$failures" "${#PID_RUN[@]}" "$(format_elapsed $((now - QUEUE_START)))"
  for gpu in "${GPU_IDS_ARRAY[@]}"; do
    printf '    GPU %s: %s runs, %s MiB used\n' "$gpu" "${GPU_ACTIVE[$gpu]}" "$(used_mib "$gpu")"
  done
  for pid in "${!PID_RUN[@]}"; do
    index="${PID_RUN[$pid]}"
    last="$(tail -n 20 "$(run_log "$index")" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -n 1 || true)"
    printf '      %-22s %-4s %s  %s\n' "${RUN_JOBS[$index]}" "${RUN_METHODS[$index]}" \
      "$(format_elapsed $((now - PID_START[$pid])))" "${last:0:80}"
  done
}

log_stage "Running ${TOTAL_RUNS} runs"
while ((next_run < TOTAL_RUNS || ${#PID_RUN[@]})); do
  reap
  # Fill at most one slot per GPU per pass, so each start is measured before the next.
  for gpu in "${GPU_IDS_ARRAY[@]}"; do
    if ((next_run < TOTAL_RUNS)) && admit "$gpu"; then
      launch "$next_run" "$gpu"
      next_run=$((next_run + 1))
    fi
  done
  if ((STATUS_INTERVAL > 0 && $(date +%s) - last_status >= STATUS_INTERVAL)); then
    status
    last_status="$(date +%s)"
  fi
  sleep 5
done

printf '\nAll runs ended in %s\n' "$(format_elapsed $(($(date +%s) - QUEUE_START)))"
if ((failures)); then
  printf '%s of %s runs failed. Rerun the same command to resume them.\n' "$failures" "$TOTAL_RUNS" >&2
  exit 1
fi
"$PYTHON_BIN" scripts/baseline_comparison_table.py --matrix "$MATRIX" \
  --output docs/distillation-baselines-results.md
printf 'Results table: docs/distillation-baselines-results.md\n'
