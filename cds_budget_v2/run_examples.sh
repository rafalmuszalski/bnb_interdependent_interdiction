#!/usr/bin/env bash
# Run cds_budget_v2.main on a few example instances, one after another.
# Logs go to cds_budget_v2/experiments/<start time MMDD_HHMM_SS>/<instance index 000, 001, ...>_{bnb,nodes,ccg}.log
# Usage (from anywhere):  ./cds_budget_v2/run_examples.sh
set -eo pipefail

cd "$(dirname "$0")/.."                           # repo root, so `python -m cds_budget_v2.main` works
PYTHON="${PYTHON:-.venv/bin/python}"

PARAMS=(
  "--n 20 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 3 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 3 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 4 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 4 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
  "--n 20 --k 5 --b_p 4 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 120"
)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

export EXPERIMENT_ID="$(date +%m%d_%H%M_%S)"
echo "Logs: cds_budget_v2/experiments/$EXPERIMENT_ID"

run_id=0
for params in "${PARAMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== [$run_id] $params --seed $seed"
    export RUN_ID="$run_id"
    $PYTHON -m cds_budget_v2.main $params --seed "$seed"
    echo
    run_id=$((run_id + 1))
  done
done
