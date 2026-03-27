#!/usr/bin/env bash
# run_params_parallel_2numa.sh  <script_with_PARAMS.sh>
#
# Runs N_WORKERS experiments in parallel, distributed evenly across 2 NUMA
# nodes. Each worker is pinned to a non-overlapping slice of physical cores
# on its node (no HT siblings used). Tasks are claimed atomically via mkdir
# so fast workers pick up the slack from slow ones.
#
# Config: set N_WORKERS and THREADS_PER_EXP below.
# Constraint: (N_WORKERS / 2) * THREADS_PER_EXP <= physical cores per NUMA node
# e.g. 4 workers x 12 threads = 24 cores/node  (leaves 8 cores/node headroom)
set -eo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_WORKERS=4          # must be even (split evenly across 2 NUMA nodes)
THREADS_PER_EXP=8   # physical cores per worker (no HT siblings)

# ---------------------------------------------------------------------------
# Args / paths
# ---------------------------------------------------------------------------
src="${1:?Usage: $0 path/to/script_with_PARAMS.sh}"
base_dir="${src%%/*}"
run_stamp=$(date +"%d%b%H%M" | tr '[:upper:]' '[:lower:]')

mkdir -p "$base_dir/logs/$run_stamp"
master_log="$base_dir/logs/$run_stamp/master.log"
exec > >(tee -a "$master_log") 2>&1

(( N_WORKERS % 2 == 0 )) || { echo "ERROR: N_WORKERS must be even." >&2; exit 1; }
WORKERS_PER_NODE=$(( N_WORKERS / 2 ))

# Set Gurobi environment variables
export GUROBI_HOME=~/Software/Gurobi/gurobi1301/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$GUROBI_HOME/licenses/gurobi.lic


# ---------------------------------------------------------------------------
# Thread environment
# ---------------------------------------------------------------------------
export GRB_THREADS="$THREADS_PER_EXP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SLURM_ARRAY_JOB_ID="$run_stamp"

# ---------------------------------------------------------------------------
# Load PARAMS array
# ---------------------------------------------------------------------------
block=$(sed -n '/^PARAMS[[:space:]]*=(/,/^[[:space:]]*)[[:space:]]*$/p' "$src")
[[ -n "$block" ]] || { echo "ERROR: could not parse PARAMS block from $src" >&2; exit 1; }
eval "$block"
total=${#PARAMS[@]}
[[ $total -gt 0 ]] || { echo "ERROR: PARAMS array is empty after eval" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Build a CPU set for a specific slice of physical cores on a NUMA node.
# ---------------------------------------------------------------------------
cpus_for_numa_slice() {
  local node="$1" offset="$2" count="$3"
  lscpu -p=CPU,CORE,SOCKET,NODE | grep -v '^#' \
  | awk -F, -v node="$node" -v offset="$offset" -v count="$count" '
      $4 == node {
        if (!($2 in seen)) seen[$2] = $1
      }
      END {
        n = asorti(seen, cores, "@ind_num_asc")
        out = ""
        for (i = offset+1; i <= offset+count && i <= n; i++)
          out = out (out=="" ? "" : ",") seen[cores[i]]
        print out
      }'
}

declare -a CPUSETS
for (( w = 0; w < N_WORKERS; w++ )); do
  local_node=$(( w / WORKERS_PER_NODE ))
  local_slot=$(( w % WORKERS_PER_NODE ))
  offset=$(( local_slot * THREADS_PER_EXP ))
  cpuset="$(cpus_for_numa_slice "$local_node" "$offset" "$THREADS_PER_EXP")"
  [[ -n "$cpuset" ]] || {
    echo "ERROR: empty CPU set for worker $w (node=$local_node offset=$offset count=$THREADS_PER_EXP)" >&2
    exit 2
  }
  CPUSETS[$w]="$cpuset"
done

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
# source .venv/bin/activate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_dir_for() { echo "$base_dir/logs/$run_stamp/$(printf '%03d' "$1")"; }

run_one() {
  local task_id="$1" params="$2" numa="$3" cpus="$4" worker_id="$5"
  local log_dir; log_dir="$(log_dir_for "$task_id")"
  mkdir -p "$log_dir"

  export SLURM_ARRAY_TASK_ID="$task_id"
  export EXPERIMENT_DIR="$log_dir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶ worker $worker_id | task $task_id of $total | NUMA $numa | cpus: $cpus"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] params: $params"

  numactl --physcpubind="$cpus" --cpunodebind="$numa" --membind="$numa" \
    python -m "$base_dir.main" $params \
    >| "$log_dir/stdout.log" \
    2>| "$log_dir/stderr.log" \
  && echo "[$(date '+%Y-%m-%d %H:%M:%S')] worker $worker_id : task $task_id done" \
  || {
    local rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ task $task_id failed (rc=$rc)"
    touch "$log_dir/.failed"
  }
}

# Claim the next available task using atomic mkdir.
# Pre-created claim dirs act as tickets; mkdir succeeds for exactly one worker.
# Returns the task id via stdout, or returns 1 if nothing left to claim.
claim_next_task() {
  local claim_dir="$base_dir/logs/$run_stamp/.claims"
  for (( tid = 0; tid < total; tid++ )); do
    if mkdir "$claim_dir/$(printf '%03d' "$tid")" 2>/dev/null; then
      echo "$tid"
      return 0
    fi
  done
  return 1
}

# Each worker claims and runs tasks until none are left.
run_queue() {
  local worker_id="$1" numa="$2" cpus="$3"
  local tid
  while tid=$(claim_next_task); do
    run_one "$tid" "${PARAMS[$tid]}" "$numa" "$cpus" "$worker_id"
  done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "========================================"
echo "Run stamp      : $run_stamp"
echo "Base dir       : $base_dir"
echo "Total tasks    : $total"
echo "Workers        : $N_WORKERS ($WORKERS_PER_NODE per NUMA node)"
echo "Threads/worker : $THREADS_PER_EXP"
for (( w = 0; w < N_WORKERS; w++ )); do
  numa=$(( w / WORKERS_PER_NODE ))
  printf "  worker %d -> NUMA %d  cpus: %s\n" "$w" "$numa" "${CPUSETS[$w]}"
done
echo "========================================"

# Trap Ctrl+C or kill -> tear down all worker children cleanly
trap '
  echo ""
  echo "Caught signal — killing all workers..."
  kill -- -$(ps -o pgid= -p $$ | tr -d " ") 2>/dev/null
  exit 1
' INT TERM

# Initialise claims directory
mkdir -p "$base_dir/logs/$run_stamp/.claims"

# Launch all workers in parallel, collect PIDs
declare -a pids
for (( w = 0; w < N_WORKERS; w++ )); do
  numa=$(( w / WORKERS_PER_NODE ))
  run_queue "$w" "$numa" "${CPUSETS[$w]}" &
  pids[$w]=$!
done

# Join all workers
for (( w = 0; w < N_WORKERS; w++ )); do
  wait "${pids[$w]}" || true
done

# Count failures via marker files
fail=0
while IFS= read -r -d '' _; do
  (( fail++ )) || true
done < <(find "$base_dir/logs/$run_stamp" -name '.failed' -print0 2>/dev/null)

echo "========================================"
echo "Done.  Failures: $fail / $total"
echo "Logs : $base_dir/logs/$run_stamp/"
echo "========================================"
(( fail == 0 ))