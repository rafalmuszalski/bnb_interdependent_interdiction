#!/usr/bin/env bash
# Run cds_budget_v2.main on a few example instances, one after another.
# Usage (from anywhere):  ./cds_budget_v2/run_examples.sh
set -eo pipefail

cd "$(dirname "$0")/.."                           # repo root, so `python -m cds_budget_v2.main` works
PYTHON="${PYTHON:-.venv/bin/python}"

PARAMS=(
  "--n 20 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
)

for params in "${PARAMS[@]}"; do
  echo "=== $params"
  $PYTHON -m cds_budget_v2.main $params
  echo
done
