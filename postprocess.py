"""Tables from stats.jsonl:  python postprocess.py path/to/stats.jsonl   (run.sh calls it at the end)
Saves summary.xlsx next to the stats file, with two sheets:
    runs                 every line of stats.jsonl, one row per run
    condensed_overview   summary_table(): avg/max per (n, b_p, b_a)"""
import sys
from pathlib import Path

import pandas as pd

GROUP = ["n", "b_p", "b_a"]

SUMMARY_COLUMNS = [
    "bnb_nodes", "bnb_strategies_added", "bnb_structures_added", "bnb_defender_cache_hits", "bnb_gap",
    "bnb_protector_time", "bnb_attacker_time", "bnb_defender_time", "bnb_heuristic_time", "bnb_time",
    "bnb_time_per_node",
    "ccg_iterations", "ccg_cols", "ccg_rows", "ccg_defender_cache_hits", "ccg_gap",
    "ccg_protector_time", "ccg_attacker_time", "ccg_defender_time", "ccg_time",  # CCG has no heuristic
    "ccg_time_per_iteration",
]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (n, b_p, b_a): avg and max of each BnB and CCG statistic (e.g. bnb_time_avg), and the number of instances."""
    df = df.copy()
    # Ratios are computed per run, then aggregated: avg is the mean of the per-run ratios,
    # not avg time / avg nodes. A run with 0 iterations gets NaN instead of inf.
    df["bnb_time_per_node"] = df.get("bnb_time") / df.get("bnb_nodes")
    if "ccg_time" in df:
        df["ccg_time_per_iteration"] = df["ccg_time"] / df["ccg_iterations"].replace(0, float("nan"))
    df = df.reindex(columns=GROUP + SUMMARY_COLUMNS)  # missing columns (e.g. --bnb_only runs) become NaN
    groups = df.groupby(GROUP)

    table = groups[SUMMARY_COLUMNS].agg(["mean", "max"])
    table = table.rename(columns={"mean": "avg"}, level=1)
    table[("instances", "")] = groups.size()
    table.columns = [f"{stat}_{agg}" if agg else stat for stat, agg in table.columns]  # flat header
    return table.reset_index()


if __name__ == "__main__":
    stats_path = Path(sys.argv[1])
    df = pd.read_json(stats_path, lines=True)

    # Excel cells can't hold lists (policy, attack, structure) and cap text at 32767 chars
    runs = df.apply(lambda col: col.map(lambda v: str(v)[:32767] if isinstance(v, (list, str)) else v))

    out_path = stats_path.parent / "summary.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        runs.to_excel(writer, sheet_name="runs", index=False)
        summary_table(df).to_excel(writer, sheet_name="condensed_overview", index=False)
    print(f"Saved {out_path}")
