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
    Speedup    Time avg of the slower method / Time avg of the faster one (t1: CCG / BnB,
               t5: no SR / SR); ∞ if only the slower one has unfinished runs, – if the faster
               one has unfinished runs or a solver didn't run
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


def mean_of(name: str, decimals: int = 1):
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


def speedup(slow, slow_s, fast, fast_s):
    """Time avg of `slow` (runs, solver prefix) / Time avg of `fast`. ∞ if only `slow` has
    unfinished runs, – if `fast` has unfinished runs or either solver didn't run."""
    if not (ran(slow, slow_s) and ran(fast, fast_s)) or not solved(fast, fast_s).all():
        return DASH
    if not solved(slow, slow_s).all():
        return INF
    return number(effective_time(slow, slow_s).mean() / effective_time(fast, fast_s).mean(), 2)


# =============================================================================
# Checks: a table is only built if its runs are what the table assumes
# =============================================================================

def run_ids(runs: pd.DataFrame) -> str:
    return ", ".join(str(i) for i in runs["run_id"])


def checks(df: pd.DataFrame, group: list[str], fixed: list[str], bnb_only: bool) -> list[str]:
    """Reasons the table can't be trusted; empty if it can.
    group     parameters that make a row (the runs of a row differ only in the seed)
    fixed     parameters that must have one value in every run (else rows would mix settings)
    bnb_only  CCG must not have run; otherwise it ran in every run or in none"""
    reasons = []
    crashed = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    if not crashed.empty:
        reasons.append(f"{len(crashed)} runs crashed (run_id: {run_ids(crashed)})")

    for p in fixed:
        if p in df and df[p].nunique(dropna=False) > 1:
            values = ", ".join(str(v) for v in df[p].unique())
            reasons.append(f"{p} should be the same in every run but takes the values {values}")

    seeds = df.groupby(group)["seed"].apply(lambda s: tuple(sorted(s)))
    if seeds.nunique() > 1:
        reasons.append("the rows don't all have the same seeds, so they average different instances")
    repeated = df[df.duplicated(group + ["seed"], keep=False)]
    if not repeated.empty:
        reasons.append(f"a setting ran more than once with the same seed (run_id: {run_ids(repeated)})")

    bnb_ran, ccg_ran = stat(df, "bnb_status").notna(), stat(df, "ccg_status").notna()
    if not bnb_ran.all():
        reasons.append(f"BnB has no results for run_id: {run_ids(df[~bnb_ran])}")
    if bnb_only and ccg_ran.any():
        reasons.append(f"CCG ran but this table has no CCG columns (run_id: {run_ids(df[ccg_ran])})")
    if not bnb_only and ccg_ran.any() and not ccg_ran.all():
        reasons.append(f"CCG ran in some runs but not in run_id: {run_ids(df[~ccg_ran])}")
    return reasons


def not_generated(table: str, reasons: list[str]) -> pd.DataFrame:
    """A sheet with a single cell that says why the table wasn't built."""
    return pd.DataFrame(columns=[f"{table} not generated: " + " | ".join(reasons)])


# =============================================================================
# Paper tables: column name in the sheet -> cell
# =============================================================================

def t1_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 (CDS, budget): BnB vs CCG per (n, b_p, b_a), averaged over the seeds."""
    group = ["n", "b_p", "b_a"]
    reasons = checks(df, group, fixed=["k", "protectors", "bridges", "strategy_propagation",
                                       "structure_propagation", "no_cache"], bnb_only=False)
    if reasons:
        return not_generated("t1_final", reasons)

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
    for (n, b_p, b_a), runs in df.groupby(group, sort=True):
        row = {"n": n, "b_p": b_p, "b_a": b_a}
        for header, (s, cell) in columns.items():
            row[header] = cell(runs, s) if ran(runs, s) else DASH
        row["Speedup"] = speedup(runs, "ccg", runs, "bnb")
        rows.append(row)
    return pd.DataFrame(rows)


PROPAGATION = {"strategy_propagation": {-1: "N", 0: "B"}, "structure_propagation": {-1: "N", 0: "B", 1: "L"}}


def t2_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 (CDS, budget, propagation): BnB time split and subproblem calls per
    (n, b_p, b_a), propagation type and caching, averaged over the seeds. Times add up:
    Total = O + Strat. + Time P + Time A + Time D. Not built if any run didn't finish."""
    df = df.copy()
    df["no_cache"] = df["no_cache"].fillna(False) if "no_cache" in df else False  # runs from before the flag cached
    group = ["n", "b_p", "b_a", "strategy_propagation", "structure_propagation", "no_cache"]
    reasons = checks(df, group, fixed=["k", "protectors", "bridges"], bnb_only=True)
    unfinished = df[df["bnb_status"] != "optimal"]
    if not unfinished.empty:
        runs = ", ".join(f"{r.run_id} ({r.bnb_status})" for r in unfinished.itertuples())
        reasons.append(
            f"{len(unfinished)} of {len(df)} runs did not solve to optimality (run_id: {runs}). Every run in "
            f"this table should finish, so this should not happen; the times and counts of unfinished runs "
            f"could be corrupted")
    if reasons:
        return not_generated("t2_final", reasons)

    df["bnb_overhead"] = (stat(df, "bnb_time") - stat(df, "bnb_strategies_time") - stat(df, "bnb_protector_time")
                          - stat(df, "bnb_attacker_time") - stat(df, "bnb_defender_time"))
    df["bnb_defender_solves"] = stat(df, "bnb_defender_calls") - stat(df, "bnb_defender_cache_hits")
    columns = {                                      # header -> (stat, decimals)
        "|N|":      ("bnb_nodes", 1),
        "Total":    ("bnb_time", 1),
        "O":        ("bnb_overhead", 1),
        "Strat.":   ("bnb_strategies_time", 1),
        "Time P":   ("bnb_protector_time", 1),
        "Time A":   ("bnb_attacker_time", 1),
        "Time D":   ("bnb_defender_time", 1),
        "Calls A":  ("bnb_attacker_calls", 1),
        "Calls D":  ("bnb_defender_solves", 1),
        "D cached": ("bnb_defender_cache_hits", 1),
    }
    rows = []
    for (n, b_p, b_a, p, a, no_cache), runs in df.groupby(group, sort=True):  # sorted: NN, NB, NL, BB, BL; cache first
        row = {"n": n, "b_p": b_p, "b_a": b_a,
               "P": PROPAGATION["strategy_propagation"][p], "A": PROPAGATION["structure_propagation"][a],
               "no_cache": no_cache}
        for header, (name, decimals) in columns.items():
            row[header] = number(stat(runs, name).mean(), decimals)
        rows.append(row)
    return pd.DataFrame(rows)


def t3_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3 (CDS, target): BnB per (n, r_p, b_a), averaged over the seeds. b_p in the
    stats is the protector target r_p (percent of the attack damage to remove)."""
    group = ["n", "b_p", "b_a"]
    reasons = checks(df, group, fixed=["k", "protectors", "bridges", "strategy_propagation",
                                       "structure_propagation"], bnb_only=True)
    if reasons:
        return not_generated("t3_final", reasons)

    columns = {
        "|N|":      mean_of("nodes"),
        "L'_a":     mean_of("strategies_distinct"),
        "L'_d":     mean_of("structures_distinct"),
        "Gap":      gap,
        "Time avg": time_avg,
        "Time max": time_max,
    }
    rows = []
    for (n, r_p, b_a), runs in df.groupby(group, sort=True):
        row = {"n": n, "r_p": r_p, "b_a": b_a}
        for header, cell in columns.items():
            row[header] = cell(runs, "bnb")
        rows.append(row)
    return pd.DataFrame(rows)


def t4_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4 (FL): BnB per (|U|, |K|, |R|, b_a), averaged over the seeds. The paper splits
    it by |U| and states |V| and b_p in the caption; here they are columns."""
    group = ["V", "U", "K", "R", "b_p", "b_a"]
    reasons = checks(df, group, fixed=["V", "b_p", "strategy_propagation", "structure_propagation",
                                       "anti_symmetry"], bnb_only=True)
    if reasons:
        return not_generated("t4_final", reasons)

    columns = {
        "|N|":      mean_of("nodes"),
        "L'_a":     mean_of("strategies_distinct"),
        "L'_d":     mean_of("structures_distinct"),
        "Gap":      gap,
        "Time avg": time_avg,
        "Time max": time_max,
    }
    rows = []
    for (V, U, K, R, b_p, b_a), runs in df.groupby(group, sort=True):
        row = {"|V|": V, "|U|": U, "|K|": K, "|R|": R, "b_p": b_p, "b_a": b_a}
        for header, cell in columns.items():
            row[header] = cell(runs, "bnb")
        rows.append(row)
    return pd.DataFrame(rows)


def t5_final(df: pd.DataFrame) -> pd.DataFrame:
    """Table 5 (FL, symmetry reduction): BnB with SR (--anti_symmetry) vs without, averaged
    over the seeds. One parameter is varied at a time around a base setting (the most common
    value of each parameter); one block of rows per varied parameter, in the paper's order,
    each including the base setting. SR and no SR runs of a setting are matched by its
    parameters; the checks make sure both exist with the same seeds."""
    params = {"V": "|V|", "U": "|U|", "K": "|K|", "R": "|R|", "b_a": "b_a"}  # varied, in block order
    setting = list(params) + ["b_p"]
    reasons = checks(df, setting + ["anti_symmetry"], fixed=["b_p", "strategy_propagation", "structure_propagation"],
                     bnb_only=True)
    variants = df.groupby(setting)["anti_symmetry"].nunique()
    if (variants < 2).any():
        reasons.append(f"settings with only SR or only no SR runs, so they can't be compared: "
                       f"{', '.join(str(k) for k in variants[variants < 2].index)} ({', '.join(setting)})")
    base = {p: df[p].mode()[0] for p in params}
    off_base = (df[list(params)] != pd.Series(base)).sum(axis=1)  # parameters each run varies from the base
    if (off_base > 1).any():
        reasons.append(f"runs vary more than one parameter from the base setting "
                       f"{', '.join(f'{params[p]}={v}' for p, v in base.items())} (run_id: {run_ids(df[off_base > 1])})")
    if reasons:
        return not_generated("t5_final", reasons)

    columns = {
        "|N|":      mean_of("nodes"),
        "Gap":      gap,
        "Time avg": time_avg,
        "Time max": time_max,
    }
    rows = []
    for varied in params:
        block = df[(off_base == 0) | (df[varied] != base[varied])]  # base setting + settings varying only `varied`
        for value, runs in block.groupby(varied, sort=True):
            sr, no_sr = runs[runs["anti_symmetry"].astype(bool)], runs[~runs["anti_symmetry"].astype(bool)]
            row = {"Varied": params[varied]}
            row.update({header: value if p == varied else base[p] for p, header in params.items()})
            row["b_p"] = runs["b_p"].iloc[0]
            for header, cell in columns.items():
                row[f"SR {header}"] = cell(sr, "bnb")
            for header, cell in columns.items():
                row[f"no SR {header}"] = cell(no_sr, "bnb")
            row["Speedup"] = speedup(no_sr, "bnb", sr, "bnb")
            rows.append(row)
    return pd.DataFrame(rows)


FINAL = {"t1": t1_final, "t2": t2_final, "t3": t3_final, "t4": t4_final, "t5": t5_final}


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
