#!/usr/bin/env bash
# Like run.sh, but runs the experiments of a table in parallel: WORKERS workers, each pinned
# to its own THREADS_PER_WORKER physical cores of socket 0 (NUMA node 0), memory on node 0.
# Every worker takes the next unstarted run from a shared queue, so long runs don't hold up
# the others. The script returns once every run has finished, then builds summary.xlsx.
#
# Usage (from anywhere):  ./run_par.sh <problem>/<table>.sh      e.g.  ./run_par.sh fl_v2/t4.sh
#
# Linux only (numactl, lscpu, flock, pgrep). THREADS in solver_constants.py should equal
# THREADS_PER_WORKER, or Gurobi starts more threads than the worker has cores.
#
# Table format and output are the same as run.sh (see there). Differences:
#   - stats.jsonl lines are in the order runs finish, not by run id
#   - console output of the workers interleaves (each line names its run)
set -eo pipefail

WORKERS=4
THREADS_PER_WORKER=4
NODE=0   # NUMA node 0 = socket 0 on this machine

root="$(cd "$(dirname "$0")" && pwd)"
[[ $# -eq 1 ]] || { echo "Usage: $0 <problem>/<table>.sh" >&2; exit 2; }
[[ -f "$1" ]] || { echo "ERROR: table file not found: $1" >&2; exit 2; }
table="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"   # absolute, before the cd below
for cmd in numactl lscpu flock pgrep; do
  command -v "$cmd" > /dev/null || { echo "ERROR: $cmd not found (run_par.sh needs Linux)" >&2; exit 2; }
done

# ---- Problem package: the top-level folder of the repo that holds the table ----
[[ "$table" == "$root"/*/* ]] || { echo "ERROR: table file must be inside a problem folder of $root" >&2; exit 2; }
relative="${table#"$root"/}"
problem="${relative%%/*}"
[[ -f "$root/$problem/main.py" ]] || { echo "ERROR: $problem has no main.py" >&2; exit 2; }

cd "$root"                                        # so `python -m <problem>.main` works
PYTHON="${PYTHON:-.venv/bin/python}"

# ---- CPUs: one hardware thread per physical core of the node, split between the workers ----
# lscpu -p prints CPU,CORE,NODE; CORE is unique machine-wide, so the first CPU seen per CORE
# skips the hyperthread siblings.
mapfile -t cores < <(lscpu -p=CPU,CORE,NODE | awk -F, -v node="$NODE" '!/^#/ && $3 == node && !seen[$2]++ { print $1 }')
needed=$(( WORKERS * THREADS_PER_WORKER ))
(( ${#cores[@]} >= needed )) || { echo "ERROR: node $NODE has ${#cores[@]} physical cores, need $needed" >&2; exit 2; }
cpusets=()
for (( w = 0; w < WORKERS; w++ )); do
  cpuset="${cores[*]:w * THREADS_PER_WORKER:THREADS_PER_WORKER}"
  cpusets+=("${cpuset// /,}")
done

# ---- Load the table ----
PARAMS=()
SEEDS=()
source "$table"
[[ ${#PARAMS[@]} -gt 0 ]] || { echo "ERROR: no PARAMS in $table" >&2; exit 2; }
if [[ ${#SEEDS[@]} -gt 0 ]]; then
  for params in "${PARAMS[@]}"; do
    if [[ " $params " == *" --seed "* ]]; then
      echo "ERROR: $table sets SEEDS and also --seed in PARAMS; use one or the other" >&2; exit 2
    fi
  done
fi

# Every run's arguments, indexed by run id
runs=()
for params in "${PARAMS[@]}"; do
  if [[ ${#SEEDS[@]} -gt 0 ]]; then
    for seed in "${SEEDS[@]}"; do
      runs+=("$params --seed $seed")
    done
  else
    runs+=("$params")
  fi
done

export EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%m%d_%H%M_%S)}"   # run_all.sh sets its own
export RUN_TOTAL=${#runs[@]}   # for the [k/total] progress in the console
experiment_dir="results/$EXPERIMENT_ID"   # must match EXPERIMENTS_DIR in <problem>/main.py
mkdir -p "$experiment_dir"
echo "$(date '+%Y-%m-%d %H:%M:%S') Running $relative ($WORKERS workers on node $NODE)" > "$experiment_dir/out.stdout"
for (( w = 0; w < WORKERS; w++ )); do
  echo "worker $w: CPUs ${cpusets[w]}" >> "$experiment_dir/out.stdout"
done
cp "$table" "$experiment_dir/"   # the exact PARAMS/SEEDS run; postprocess.py reads the table name from it
echo "Problem: $problem"
echo "Table:   $relative ($RUN_TOTAL runs)"
echo "Logs:    $experiment_dir"
for (( w = 0; w < WORKERS; w++ )); do
  echo "Worker $w: CPUs ${cpusets[w]}"
done

# ---- Queue: a counter file holding the next run id, read and bumped under a lock ----
queue="$experiment_dir/.queue"
echo 0 > "$queue"
next_run() {
  local i
  {
    flock 9
    i=$(< "$queue")
    echo $(( i + 1 )) > "$queue"
  } 9> "$queue.lock"
  echo "$i"
}

# Build summary.xlsx when the script ends: after the last run, or early on Ctrl+C
# (then it covers the runs that finished). Skipped if no run finished.
postprocess() {
  rm -f "$queue" "$queue.lock"
  if [[ -f "$experiment_dir/stats.jsonl" ]]; then
    $PYTHON postprocess.py "$experiment_dir/stats.jsonl" || echo "postprocess failed"
  fi
}
trap postprocess EXIT

# Ctrl+C stops the whole table. Background workers ignore SIGINT (bash does that for `&`
# jobs in a script), so they and their Python runs are stopped here with SIGTERM.
worker_pids=()
descendants() {
  local child
  for child in $(pgrep -P "$1" || true); do
    echo "$child"
    descendants "$child"
  done
}
stop_workers() {
  local pids=() pid
  for pid in "${worker_pids[@]}"; do
    pids+=("$pid" $(descendants "$pid"))   # collected first: a killed worker's children get reparented
  done
  [[ ${#pids[@]} -gt 0 ]] && { kill -TERM "${pids[@]}" 2> /dev/null || true; }   # some may have exited already
  wait || true
}
trap 'echo "Interrupted"; stop_workers; exit 130' INT TERM

# ---- Run ----
# A run that crashes is logged by main.py (out.stdout, err.stderr, stats.jsonl) and the
# worker goes on with the next run.
worker() {
  trap - EXIT INT TERM   # only the main script post-processes and stops workers
  local cpus="$1" i
  while true; do
    i=$(next_run)
    (( i < ${#runs[@]} )) || return 0
    RUN_ID="$i" numactl --physcpubind="$cpus" --membind="$NODE" $PYTHON -m "$problem.main" ${runs[i]} \
      || echo "Run $i failed (see $experiment_dir/err.stderr)"
  done
}

for (( w = 0; w < WORKERS; w++ )); do
  worker "${cpusets[w]}" &
  worker_pids+=($!)
done
wait
