#!/usr/bin/env bash
# Run every paper table, one after another, then build one workbook with all of them.
#
# Usage (from anywhere):
#   ./run_all.sh                  all tables, in the order of TABLES below
#   ./run_all.sh table1 table5    only these
#
# Each table runs through run.sh as usual, so its experiment folder (logs, stats.jsonl,
# summary.xlsx) is at <problem>/experiments/<STAMP>_<table>/. At the end, and also on a
# crash or Ctrl+C, every table that has results goes into results/all_<STAMP>.xlsx with
# sheets t1_raw, t1_overview, t1_final, t2_raw, ...
#
# A table whose run fails doesn't stop the others; the failures are listed at the end.
set -o pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
PYTHON="${PYTHON:-.venv/bin/python}"

TABLES=(
  # cds_budget_v2/table1.sh
  # ds_budget_v2/table2.sh
  # ds_budget_v2/table3.sh
  cds_target_v2/table4.sh
  fl_v2/table5.sh
  fl_v2/table6.sh
  # ds_budget_v2/table7.sh
  # ds_budget_v2/table8.sh
)

# ---- Pick the tables ----
selected=()
if [[ $# -eq 0 ]]; then
  selected=("${TABLES[@]}")
else
  for name in "$@"; do
    match=""
    for t in "${TABLES[@]}"; do
      [[ "$(basename "$t" .sh)" == "$name" ]] && match="$t"
    done
    [[ -n "$match" ]] || { echo "ERROR: unknown table $name (known: ${TABLES[*]})" >&2; exit 2; }
    selected+=("$match")
  done
fi

stamp="$(date +%m%d_%H%M_%S)"
out="results/all_$stamp.xlsx"
stats_files=()
failed=()

combine() {
  local existing=()
  for f in "${stats_files[@]}"; do [[ -f "$f" ]] && existing+=("$f"); done
  if [[ ${#existing[@]} -gt 0 ]]; then
    $PYTHON postprocess.py --combine "$out" "${existing[@]}" || echo "combining failed"
  else
    echo "No results to combine"
  fi
  [[ ${#failed[@]} -eq 0 ]] || echo "Failed tables: ${failed[*]}"
}
trap combine EXIT
# Ctrl+C stops everything (run.sh still finishes the interrupted table's summary first);
# without this, bash would see run.sh exit normally and go on to the next table.
trap 'echo "Interrupted"; exit 130' INT TERM

for table in "${selected[@]}"; do
  name="$(basename "$table" .sh)"
  problem="${table%%/*}"
  id="${stamp}_${name}"
  stats_files+=("$problem/experiments/$id/stats.jsonl")
  echo "======== $table  ($(date '+%Y-%m-%d %H:%M:%S'))"
  EXPERIMENT_ID="$id" ./run.sh "$table" || failed+=("$name")
done
