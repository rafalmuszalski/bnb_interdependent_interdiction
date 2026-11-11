"""Results workbook from stats.jsonl files.

    python postprocess.py stats.jsonl [more stats.jsonl ...] [-o OUT.xlsx]

Every stats file becomes one sheet with one row per run (every line of stats.jsonl as is).
The sheet is named after the table file run.sh copied next to the stats (e.g. t1_small), or
after the experiment folder if there is none. If the table has a paper table in FINAL
(t1.sh and t1_small.sh both -> t1), that goes into a second sheet, e.g. t1_final.
Without -o the workbook is summary.xlsx next to the first stats file (run.sh uses this);
run_all.sh passes -o to put all tables in one workbook.

Paper conventions (each row is one setting, all its seeds together):
    Time avg   mean time; runs that didn't finish count at the time limit. TL if none finished
    Time max   max time; TL if any run didn't finish
    Gap        mean (UB - LB) / UB in percent over the runs that didn't finish;
               – if all finished, ∞ if one of them has no lower bound
    Speedup    CCG Time avg / BnB Time avg; ∞ if only CCG has unfinished runs, – if BnB has
               unfinished runs or a solver didn't run
A solver that didn't run (e.g. CCG with --bnb_only) gets – everywhere.
"""
import argparse
from pathlib import Path

import pandas as pd

DASH, INF, TL = "–", "∞", "TL"


def table_name(stats_path: Path) -> str:
    """Stem of the table file next to the stats (e.g. t1_small), else the experiment folder."""
    tables = sorted(stats_path.parent.glob("t*.sh"))
    return tables[0].stem if tables else stats_path.parent.name


def raw_runs(df: pd.DataFrame) -> pd.DataFrame:
    # Excel cells can't hold lists (policy, attack, structure) and cap text at 32767 chars
    return df.apply(lambda col: col.map(lambda v: str(v)[:32767] if isinstance(v, (list, str)) else v))


# =============================================================================
# Cells: each takes the runs of one setting and the solver prefix ("bnb" or "ccg")
# =============================================================================

def stat(runs: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` of the runs; all NaN if the stats don't have it."""
    return runs[name] if name in runs else pd.Series(float("nan"), index=runs.index)


def ran(runs, s) -> bool:
    return stat(runs, f"{s}_status").notna().any()


def solved(runs, s) -> pd.Series:
    return stat(runs, f"{s}_status") == "optimal"


def effective_time(runs, s) -> pd.Series:
    """Wall time; a run that didn't finish counts at the time limit."""
    return stat(runs, f"{s}_time").where(solved(runs, s), runs["time_limit"])


def number(x, decimals: int):
    if pd.isna(x):
        return DASH
    return round(x) if decimals == 0 else round(x, decimals)


def mean_of(name: str, decimals: int = 0):
    return lambda runs, s: number(stat(runs, f"{s}_{name}").mean(), decimals)


def gap(runs, s):
    unfinished = runs[~solved(runs, s)]
    if unfinished.empty:
        return DASH
    gaps = stat(unfinished, f"{s}_gap")  # stats.jsonl stores an infinite gap (no lower bound) as null
    return INF if gaps.isna().any() else number(100 * gaps.mean(), 2)


def time_avg(runs, s):
    return TL if not solved(runs, s).any() else number(effective_time(runs, s).mean(), 1)


def time_max(runs, s):
    return TL if not solved(runs, s).all() else number(stat(runs, f"{s}_time").max(), 1)


def speedup(runs):
    if not (ran(runs, "bnb") and ran(runs, "ccg")) or not solved(runs, "bnb").all():
        return DASH
    if not solved(runs, "ccg").all():
        return INF
    return number(effective_time(runs, "ccg").mean() / effective_time(runs, "bnb").mean(), 2)


# =============================================================================
# Paper tables: column name in the sheet -> cell
# =============================================================================

def t1_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 (CDS, budget): BnB vs CCG per (n, b_p, b_a)."""
    columns = {
        "BnB |N|":      ("bnb", mean_of("nodes")),
        "BnB L'_a":     ("bnb", mean_of("strategies_distinct")),
        "BnB L'_d":     ("bnb", mean_of("structures_distinct")),
        "BnB Gap":      ("bnb", gap),
        "BnB Time avg": ("bnb", time_avg),
        "BnB Time max": ("bnb", time_max),
        "CCG It":       ("ccg", mean_of("iterations")),
        "CCG Cols":     ("ccg", mean_of("cols")),
        "CCG Rows":     ("ccg", mean_of("rows")),
        "CCG Gap":      ("ccg", gap),
        "CCG Time avg": ("ccg", time_avg),
        "CCG Time max": ("ccg", time_max),
    }
    rows = []
    for (n, b_p, b_a), runs in df.groupby(["n", "b_p", "b_a"], sort=True):
        row = {"n": n, "b_p": b_p, "b_a": b_a}
        for header, (s, cell) in columns.items():
            row[header] = cell(runs, s) if ran(runs, s) else DASH
        row["Speedup"] = speedup(runs)
        rows.append(row)
    return pd.DataFrame(rows)


PROPAGATION = {"strategy_propagation": {-1: "N", 0: "B"}, "structure_propagation": {-1: "N", 0: "B", 1: "L"}}


def t2_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 (CDS, budget, propagation): BnB time split and subproblem calls per
    (n, b_p, b_a) and propagation type, averaged over the seeds. Times add up:
    Total = O + Strat. + Time P + Time A + Time D. Not built if any run didn't finish."""
    unfinished = df[df["bnb_status"] != "optimal"]
    if not unfinished.empty:
        runs = ", ".join(f"{r.run_id} ({r.bnb_status})" for r in unfinished.itertuples())
        return pd.DataFrame(columns=[
            f"t2_final not generated: {len(unfinished)} of {len(df)} runs did not solve to optimality "
            f"(run_id: {runs}). Every run in this table should finish, so this should not happen; "
            f"the times and counts of unfinished runs could be corrupted, so the table is not generated."])

    df = df.copy()
    df["no_cache"] = df["no_cache"].fillna(False) if "no_cache" in df else False  # runs from before the flag cached
    df["bnb_overhead"] = (stat(df, "bnb_time") - stat(df, "bnb_strategies_time") - stat(df, "bnb_protector_time")
                          - stat(df, "bnb_attacker_time") - stat(df, "bnb_defender_time"))
    df["bnb_defender_solves"] = stat(df, "bnb_defender_calls") - stat(df, "bnb_defender_cache_hits")
    columns = {                                      # header -> (stat, decimals)
        "|N|":      ("bnb_nodes", 0),
        "Total":    ("bnb_time", 1),
        "O":        ("bnb_overhead", 1),
        "Strat.":   ("bnb_strategies_time", 1),
        "Time P":   ("bnb_protector_time", 1),
        "Time A":   ("bnb_attacker_time", 1),
        "Time D":   ("bnb_defender_time", 1),
        "Calls A":  ("bnb_attacker_calls", 0),
        "Calls D":  ("bnb_defender_solves", 0),
        "D cached": ("bnb_defender_cache_hits", 0),
    }
    rows = []
    group = ["n", "b_p", "b_a", "strategy_propagation", "structure_propagation", "no_cache"]
    for (n, b_p, b_a, p, a, no_cache), runs in df.groupby(group, sort=True):  # sorted: NN, NB, NL, BB, BL; cache first
        row = {"n": n, "b_p": b_p, "b_a": b_a,
               "P": PROPAGATION["strategy_propagation"][p], "A": PROPAGATION["structure_propagation"][a],
               "no_cache": no_cache}
        for header, (name, decimals) in columns.items():
            row[header] = number(stat(runs, name).mean(), decimals)
        rows.append(row)
    return pd.DataFrame(rows)


FINAL = {"t1": t1_final, "t2": t2_final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path, nargs="+", help="path(s) to stats.jsonl")
    parser.add_argument("-o", "--out", type=Path, help="output workbook (default: summary.xlsx next to the first stats file)")
    args = parser.parse_args()

    out = args.out or args.stats[0].parent / "summary.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        for path in args.stats:
            df = pd.read_json(path, lines=True)
            name = table_name(path)
            raw_runs(df).to_excel(writer, sheet_name=name[:31], index=False)  # Excel's sheet name limit
            table = name.removesuffix("_small")
            if table in FINAL:
                FINAL[table](df).to_excel(writer, sheet_name=f"{table}_final", index=False)
    print(f"Saved {out}")
