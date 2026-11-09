"""Tables from stats.jsonl:  python -m cds_budget_v2.make_table path/to/stats.jsonl
Saves summary.xlsx next to the stats file."""
import sys
from pathlib import Path

import pandas as pd

GROUP = ["n", "b_p", "b_a"]

SUMMARY_COLUMNS = [
    "bnb_nodes", "bnb_strategies_added", "bnb_structures_added", "bnb_gap", "bnb_time",
    "ccg_iterations", "ccg_cols", "ccg_rows", "ccg_gap", "ccg_time",
]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (n, b_p, b_a): avg and max of each BnB and CCG statistic, and the number of instances."""
    df = df.reindex(columns=GROUP + SUMMARY_COLUMNS)  # missing columns (e.g. --bnb_only runs) become NaN
    groups = df.groupby(GROUP)

    table = groups[SUMMARY_COLUMNS].agg(["mean", "max"])
    table = table.rename(columns={"mean": "avg"}, level=1)
    table[("instances", "")] = groups.size()
    return table


if __name__ == "__main__":
    stats_path = Path(sys.argv[1])
    df = pd.read_json(stats_path, lines=True)

    out_path = stats_path.parent / "summary.xlsx"
    summary_table(df).to_excel(out_path)
    print(f"Saved {out_path}")
