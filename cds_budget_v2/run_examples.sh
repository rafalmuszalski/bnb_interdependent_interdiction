#!/usr/bin/env bash
# Run cds_budget_v2.main on a few example instances, one after another.
# Logs go to cds_budget_v2/experiments/<start time MMDD_HHMM_SS>/<instance index 000, 001, ...>_{bnb,nodes,ccg}.log
# plus console.log: the console output of all runs, each line stamped YYYY-MM-DD HH:MM:SS
# Usage (from anywhere):  ./cds_budget_v2/run_examples.sh
set -eo pipefail

cd "$(dirname "$0")/.."                           # repo root, so `python -m cds_budget_v2.main` works
PYTHON="${PYTHON:-.venv/bin/python}"

PARAMS=(
  "--n 20 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 3 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 3 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 4 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 4 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  # "--n 20 --k 5 --b_p 4 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
)
SEEDS=(0 1 2 3 4)

export EXPERIMENT_ID="$(date +%m%d_%H%M_%S)"
export RUN_TOTAL=$(( ${#PARAMS[@]} * ${#SEEDS[@]} ))   # for the [k/total] progress in the console
echo "Logs: cds_budget_v2/experiments/$EXPERIMENT_ID"

run_id=0
for params in "${PARAMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    export RUN_ID="$run_id"
    $PYTHON -m cds_budget_v2.main $params --seed "$seed"
    echo
    run_id=$((run_id + 1))
  done
done
