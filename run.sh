#!/usr/bin/env bash
# Run every experiment in a table file, one after another, then build summary.xlsx.
#
# Usage (from anywhere):  ./run.sh <problem>/<table>.sh      e.g.  ./run.sh cds_budget_v2/table1.sh
#
# The problem is the top-level package the table file sits in. It must have a main.py
# (run as `python -m <problem>.main`) that reads EXPERIMENT_ID, RUN_ID and RUN_TOTAL from
# the environment and writes its logs and stats.jsonl to <problem>/experiments/$EXPERIMENT_ID/.
# At the end, postprocess.py (repo root) turns stats.jsonl into summary.xlsx.
#
# The table file defines (see cds_budget_v2/table1.sh):
#   PARAMS=( "--n 20 --k 5 ..." ... )   arguments for main.py, one line per setting
#   SEEDS=( 0 1 2 3 4 )                 optional: every PARAMS line runs once per seed.
#                                       Without SEEDS each line runs once, as written.
# It is sourced (executed by bash), so keep it to PARAMS and SEEDS.
#
# Output goes to <problem>/experiments/<MMDD_HHMM_SS>/:
#   NNN_*.log      per-run logs (NNN = run id 000, 001, ...)
#   console.log    console output of all runs, each line timestamped
#   stats.jsonl    one line of statistics per run
#   tableN.sh      copy of the table file that was run
#   summary.xlsx   built by postprocess.py at the end
set -eo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
[[ $# -eq 1 ]] || { echo "Usage: $0 <problem>/<table>.sh" >&2; exit 2; }
[[ -f "$1" ]] || { echo "ERROR: table file not found: $1" >&2; exit 2; }
table="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"   # absolute, before the cd below

# ---- Problem package: the top-level folder of the repo that holds the table ----
[[ "$table" == "$root"/*/* ]] || { echo "ERROR: table file must be inside a problem folder of $root" >&2; exit 2; }
relative="${table#"$root"/}"
problem="${relative%%/*}"
[[ -f "$root/$problem/main.py" ]] || { echo "ERROR: $problem has no main.py" >&2; exit 2; }

cd "$root"                                        # so `python -m <problem>.main` works
PYTHON="${PYTHON:-.venv/bin/python}"

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
  runs_per_line=${#SEEDS[@]}
else
  runs_per_line=1
fi

export EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%m%d_%H%M_%S)}"   # run_all.sh sets its own
export RUN_TOTAL=$(( ${#PARAMS[@]} * runs_per_line ))   # for the [k/total] progress in the console
experiment_dir="$problem/experiments/$EXPERIMENT_ID"
mkdir -p "$experiment_dir"
echo "$(date '+%Y-%m-%d %H:%M:%S') Running $relative" > "$experiment_dir/console.log"
cp "$table" "$experiment_dir/"   # the exact PARAMS/SEEDS run; postprocess.py reads the table name from it
echo "Problem: $problem"
echo "Table:   $relative ($RUN_TOTAL runs)"
echo "Logs:    $experiment_dir"

# Build summary.xlsx when the script ends: after the last run, or early on Ctrl+C
# (then it covers the runs that finished). Skipped if no run finished.
postprocess() {
  if [[ -f "$experiment_dir/stats.jsonl" ]]; then
    $PYTHON postprocess.py "$experiment_dir/stats.jsonl" || echo "postprocess failed"
  fi
}
trap postprocess EXIT
# Ctrl+C stops the whole table, however the run exited (Gurobi catches Ctrl+C itself, so
# Python may exit like a crash rather than with KeyboardInterrupt).
trap 'echo "Interrupted"; exit 130' INT TERM

# ---- Run ----
# A run that crashes is logged by main.py (console.log, err.stderr, stats.jsonl) and the
# table goes on with the next run.
run_one() {
  export RUN_ID="$run_id"
  $PYTHON -m "$problem.main" "$@" || echo "Run $run_id failed (see $experiment_dir/err.stderr)"
  echo
  run_id=$((run_id + 1))
}

run_id=0
for params in "${PARAMS[@]}"; do
  if [[ ${#SEEDS[@]} -gt 0 ]]; then
    for seed in "${SEEDS[@]}"; do
      run_one $params --seed "$seed"
    done
  else
    run_one $params
  fi
done
