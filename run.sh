#!/usr/bin/env bash
# run_params_sequential.sh  <script_with_PARAMS.sh>
#
# Runs all experiments from PARAMS array sequentially.
# Works on both Mac and Linux.
set -eo pipefail

# ---------------------------------------------------------------------------
# Args / paths
# ---------------------------------------------------------------------------
src="${1:?Usage: $0 path/to/script_with_PARAMS.sh}"
base_dir="${src%%/*}"
run_stamp=$(date +"%d%b%H%M" | tr '[:upper:]' '[:lower:]')

mkdir -p "$base_dir/logs/$run_stamp"
master_log="$base_dir/logs/$run_stamp/master.log"
exec > >(tee -a "$master_log") 2>&1

# ---------------------------------------------------------------------------
# Load PARAMS array
# ---------------------------------------------------------------------------
block=$(sed -n '/^PARAMS[[:space:]]*=(/,/^[[:space:]]*)[[:space:]]*$/p' "$src")
[[ -n "$block" ]] || { echo "ERROR: could not parse PARAMS block from $src" >&2; exit 1; }
eval "$block"
total=${#PARAMS[@]}
[[ $total -gt 0 ]] || { echo "ERROR: PARAMS array is empty after eval" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
# source .venv/bin/activate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_dir_for() { echo "$base_dir/logs/$run_stamp/$(printf '%03d' "$1")"; }

run_one() {
  local task_id="$1"
  local params="${PARAMS[$task_id]}"
  local log_dir; log_dir="$(log_dir_for "$task_id")"
  mkdir -p "$log_dir"

  export SLURM_ARRAY_TASK_ID="$task_id"
  export EXPERIMENT_DIR="$log_dir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶ task $task_id of $total"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] params: $params"

  eval python -m "$base_dir.main" $params \
    >| "$log_dir/stdout.log" \
    2>| "$log_dir/stderr.log" \
  && echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ task $task_id done" \
  || {
    local rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ task $task_id failed (rc=$rc)"
    touch "$log_dir/.failed"
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "========================================"
echo "Run stamp  : $run_stamp"
echo "Base dir   : $base_dir"
echo "Total tasks: $total"
echo "========================================"

trap 'echo ""; echo "Caught signal — exiting."; exit 1' INT TERM

for (( i = 0; i < total; i++ )); do
  run_one "$i"
done

fail=0
while IFS= read -r -d '' _; do
  (( fail++ )) || true
done < <(find "$base_dir/logs/$run_stamp" -name '.failed' -print0 2>/dev/null)

echo "========================================"
echo "Done.  Failures: $fail / $total"
echo "Logs : $base_dir/logs/$run_stamp/"
echo "========================================"
(( fail == 0 ))