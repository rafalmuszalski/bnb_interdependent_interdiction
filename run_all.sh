#!/usr/bin/env bash
# Run every table, one after another, then put all raw results in one workbook.
#
# Usage (from anywhere):  ./run_all.sh
#
# Each table runs through run_par.sh (parallel workers on socket 0; Linux only), so its experiment folder is results/<STAMP>_<table>/.
# At the end (also after a crash or Ctrl+C) every table that has results goes into
# results/all_<STAMP>.xlsx, one raw sheet per table (t1, t2, ...).
# A table that fails doesn't stop the others.

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
PYTHON="${PYTHON:-.venv/bin/python}"

TABLES=(
  # cds_budget_v2/t1.sh
  # cds_budget_v2/t2.sh
  # cds_target_v2/t3.sh
  # fl_v2/t4.sh
  # fl_v2/t5.sh
  # cds_budget_v2/t1_small.sh
  # cds_budget_v2/t2_small.sh
  cds_target_v2/t3_small.sh
  # fl_v2/t4_small.sh
  # fl_v2/t5_small.sh
)

stamp="$(date +%m%d_%H%M_%S)"
stats_files=()

combine() {
  local existing=()
  for f in "${stats_files[@]}"; do [[ -f "$f" ]] && existing+=("$f"); done
  [[ ${#existing[@]} -gt 0 ]] && $PYTHON postprocess.py -o "results/all_$stamp.xlsx" "${existing[@]}"
}
trap combine EXIT
trap 'echo "Interrupted"; exit 130' INT TERM   # Ctrl+C stops all tables, not just the current one

for table in "${TABLES[@]}"; do
  name="$(basename "$table" .sh)"
  id="${stamp}_${name}"
  stats_files+=("results/$id/stats.jsonl")
  echo "======== $table  ($(date '+%Y-%m-%d %H:%M:%S'))"
  EXPERIMENT_ID="$id" ./run_par.sh "$table" || echo "$table failed"
done
