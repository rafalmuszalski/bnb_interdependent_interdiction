"""Tables from stats.jsonl:  python postprocess.py path/to/stats.jsonl [--table table1]
(run.sh calls it at the end). Saves summary.xlsx next to the stats file, with three sheets:
(With --combine OUT.xlsx and several stats files, it writes one workbook instead, with sheets
tN_raw / tN_overview / tN_final per table; run_all.sh uses this.)
    runs                 every line of stats.jsonl, one row per run
    condensed_overview   summary_table(): avg/max of every statistic per setting
    paper_table          the table exactly as laid out in the paper (see TABLES)

The paper table is chosen by the table file run.sh copies into the experiment folder
(e.g. table1.sh -> TABLES["table1"]), or by --table for folders without one.

Paper conventions (each row aggregates the runs of one setting, i.e. all its seeds):
    #Opt     runs solved to optimality, e.g. 4/5
    Time     mean wall time in seconds; runs that didn't finish count at the time limit.
             "TL" if no run finished
    Gap      mean (UB - LB) / UB in percent over the runs that didn't finish.
             "–" if all finished, "∞" if one of them has no lower bound
    counts   (|N|, L'_a, L'_d, It, Cols, Rows, Calls) means, rounded
    Speedup  mean time of the first method / mean time of the second. "≥" when only the
             first has unfinished runs (its time is a lower bound), "≤" when only the second
             does, "–" when both do
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

DASH, INF, TL = "–", "∞", "TL"

# =============================================================================
# condensed_overview: every statistic, avg/max per setting
# =============================================================================

# Parameter columns of the stats records (each problem has a subset). A summary row is one
# setting: all parameters equal, every seed together.
PARAMETERS = ["n", "k", "V", "U", "K", "R", "b_p", "b_a", "protectors", "bridges", "anti_symmetry",
              "strategy_propagation", "structure_propagation", "time_limit"]

SUMMARY_COLUMNS = [
    "bnb_nodes", "bnb_strategies_added", "bnb_structures_added", "bnb_defender_cache_hits", "bnb_gap",
    "bnb_protector_time", "bnb_attacker_time", "bnb_defender_time", "bnb_heuristic_time", "bnb_time",
    "bnb_time_per_node",
    "ccg_iterations", "ccg_cols", "ccg_rows", "ccg_defender_cache_hits", "ccg_gap",
    "ccg_protector_time", "ccg_attacker_time", "ccg_defender_time", "ccg_time",  # CCG has no heuristic
    "ccg_time_per_iteration",
]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per setting: avg and max of each BnB and CCG statistic (e.g. bnb_time_avg), and the number of instances.
    Statistics a problem doesn't have (e.g. ccg_* for BnB-only problems) are left out."""
    group = [c for c in PARAMETERS if c in df]
    df = df.copy()
    # Ratios are computed per run, then aggregated: avg is the mean of the per-run ratios,
    # not avg time / avg nodes. A run with 0 iterations gets NaN instead of inf.
    df["bnb_time_per_node"] = df.get("bnb_time") / df.get("bnb_nodes")
    if "ccg_time" in df:
        df["ccg_time_per_iteration"] = df["ccg_time"] / df["ccg_iterations"].replace(0, float("nan"))
    df = df.reindex(columns=group + SUMMARY_COLUMNS)  # missing columns (e.g. --bnb_only runs) become NaN
    groups = df.groupby(group)

    table = groups[SUMMARY_COLUMNS].agg(["mean", "max"])
    table = table.rename(columns={"mean": "avg"}, level=1)
    table[("instances", "")] = groups.size()
    table.columns = [f"{stat}_{agg}" if agg else stat for stat, agg in table.columns]  # flat header
    return table.dropna(axis=1, how="all").reset_index()


# =============================================================================
# Paper columns: the name printed in the paper -> how it's computed from a setting's runs
# =============================================================================
# Each function takes the runs of one setting (all seeds) and the solver prefix ("bnb" or
# "ccg") and returns the cell as it appears in the paper.

def stat(runs: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` of the runs; all NaN if the stats don't have it."""
    return runs[name] if name in runs else pd.Series(float("nan"), index=runs.index)


def solved(runs: pd.DataFrame, s: str) -> pd.Series:
    return stat(runs, f"{s}_status") == "optimal"


def effective_time(runs: pd.DataFrame, s: str) -> pd.Series:
    """Wall time; a run that didn't finish counts at the time limit."""
    return stat(runs, f"{s}_time").where(solved(runs, s), runs["time_limit"])


def number(x: float, decimals: int) -> str:
    return DASH if pd.isna(x) else f"{x:,.{decimals}f}"


def mean_of(name: str, decimals: int = 0) -> Callable:
    """Mean of statistic `<solver>_<name>`."""
    return lambda runs, s: number(stat(runs, f"{s}_{name}").mean(), decimals)


def optimal_count(runs, s):
    return f"{solved(runs, s).sum()}/{len(runs)}"


def time_column(decimals: int) -> Callable:
    def time(runs, s):
        return TL if not solved(runs, s).any() else number(effective_time(runs, s).mean(), decimals)
    return time


def gap(runs, s):
    unfinished = runs[~solved(runs, s)]
    if unfinished.empty:
        return DASH
    gaps = stat(unfinished, f"{s}_gap")  # stats.jsonl stores an infinite gap (no lower bound) as null
    return INF if gaps.isna().any() else number(100 * gaps.mean(), 2)


def overhead(runs, s):
    """Time not spent in the protector, attacker or defender (branching, bounds, propagation)."""
    parts = stat(runs, f"{s}_protector_time") + stat(runs, f"{s}_attacker_time") + stat(runs, f"{s}_defender_time")
    return number((stat(runs, f"{s}_time") - parts).mean(), 1)


def defender_solves(runs, s):
    """Defender calls that needed a Gurobi solve (cache hits excluded)."""
    return number((stat(runs, f"{s}_defender_calls") - stat(runs, f"{s}_defender_cache_hits")).mean(), 0)


@dataclass
class Column:
    header: str        # as printed in the paper
    value: Callable    # (runs, solver prefix) -> cell


COLUMNS = {
    "#Opt":    Column("#Opt", optimal_count),
    "|N|":     Column("|N|", mean_of("nodes")),                  # BnB nodes explored
    "L'_a":    Column("L'_a", mean_of("strategies_distinct")),   # distinct attack strategies (protector cuts)
    "L'_d":    Column("L'_d", mean_of("structures_distinct")),   # distinct defender structures (attacker cuts)
    "Gap":     Column("Gap", gap),
    "Time":    Column("Time", time_column(1)),
    "Time_s":  Column("Time", time_column(0)),                   # Table 5: whole seconds
    "It":      Column("It", mean_of("iterations")),
    "Cols":    Column("Cols", mean_of("cols")),
    "Rows":    Column("Rows", mean_of("rows")),
    # Table 3: time split (raw times, so the parts add up to Total) and defender solves
    "Total":   Column("Total", mean_of("time", 1)),
    "O":       Column("O", overhead),
    "P":       Column("P", mean_of("protector_time", 1)),
    "A":       Column("A", mean_of("attacker_time", 1)),
    "D":       Column("D", mean_of("defender_time", 1)),
    "Calls_D": Column("D", defender_solves),
}

# Row parameters as printed in the paper
ROW_HEADERS = {"n": "n", "k": "κ", "b_p": "b_p", "b_a": "b_a", "V": "|V|", "U": "|U|", "K": "|K|", "R": "|R|",
               "strategy_propagation": "P", "structure_propagation": "A"}


def speedup(slow_runs, slow_s, fast_runs, fast_s) -> str:
    """Mean time of `slow` / mean time of `fast` (see the module docstring for ≥, ≤, –)."""
    if slow_runs.empty or fast_runs.empty:
        return DASH
    slow_open, fast_open = not solved(slow_runs, slow_s).all(), not solved(fast_runs, fast_s).all()
    if slow_open and fast_open:
        return DASH
    ratio = effective_time(slow_runs, slow_s).mean() / effective_time(fast_runs, fast_s).mean()
    return ("≥" if slow_open else "≤" if fast_open else "") + f"{ratio:.2f}"


# =============================================================================
# Paper tables
# =============================================================================

@dataclass
class Side:
    """A group of columns under one top header (e.g. "BnB", "CCG", "BnB SR")."""
    label: str
    solver: str                                  # stats prefix: "bnb" or "ccg"
    columns: list[str]                           # keys of COLUMNS
    where: dict = field(default_factory=dict)    # only runs with these parameter values, e.g. {"anti_symmetry": True}


@dataclass
class PaperTable:
    rows: list[str]                              # parameters that make a row, outermost first
    sides: list[Side]
    speedup: tuple[str, str] | None = None       # (slower side, faster side) labels
    headers: dict = field(default_factory=dict)  # row parameter -> header, overriding ROW_HEADERS
    labels: dict = field(default_factory=dict)   # row parameter -> {value: label}
    base: dict | None = None                     # one-at-a-time design (Table 6): one block per varied
                                                 # parameter (in this order), the others at their base value


BNB = ["#Opt", "|N|", "L'_a", "L'_d", "Gap", "Time"]
CCG = ["#Opt", "It", "Cols", "Rows", "Gap", "Time"]
BNB_VS_CCG = PaperTable(rows=["n", "b_p", "b_a"], sides=[Side("BnB", "bnb", BNB), Side("CCG", "ccg", CCG)],
                        speedup=("CCG", "BnB"))

TABLES = {
    "table1": BNB_VS_CCG,                                                    # CDS, budget
    "table2": BNB_VS_CCG,                                                    # DS, budget
    "table7": BNB_VS_CCG,                                                    # DS, budget, medium (appendix)
    "table8": PaperTable(rows=["n", "b_p", "b_a"], sides=[Side("BnB", "bnb", BNB)]),  # DS, large (appendix)
    "table3": PaperTable(                                                    # constraint propagation
        rows=["n", "b_p", "b_a", "strategy_propagation", "structure_propagation"],
        sides=[Side("", "bnb", ["|N|"]),
               Side("Time", "bnb", ["Total", "O", "P", "A", "D"]),
               Side("Calls", "bnb", ["Calls_D"])],
        labels={"strategy_propagation": {-1: "N", 0: "B"}, "structure_propagation": {-1: "N", 0: "B", 1: "L"}}),
    "table4": PaperTable(rows=["n", "b_p", "b_a"], sides=[Side("", "bnb", BNB)],   # CDS, target
                         headers={"b_p": "r_p"}),
    "table5": PaperTable(rows=["U", "K", "R", "b_a"],                        # FL; one block per |U|
                         sides=[Side("", "bnb", BNB[:-1] + ["Time_s"])]),
    "table6": PaperTable(                                                    # FL, symmetry reduction
        rows=["U", "V", "K", "R", "b_a"],
        sides=[Side("BnB SR", "bnb", ["#Opt", "|N|", "Gap", "Time"], where={"anti_symmetry": True}),
               Side("BnB no SR", "bnb", ["#Opt", "|N|", "Gap", "Time"], where={"anti_symmetry": False})],
        speedup=("BnB no SR", "BnB SR"),
        base={"V": 3, "U": 15, "K": 13, "R": 4, "b_a": 5}),
}


def blocks(df: pd.DataFrame, spec: PaperTable) -> list[pd.DataFrame]:
    """The runs of each block: the whole table, or (one-at-a-time design) per varied parameter
    the runs with every other parameter at its base value, in the order of `base`. The base
    setting is in every block."""
    if spec.base is None:
        return [df]
    out = []
    for varied in spec.base:
        others = [p for p in spec.rows if p != varied]
        mask = pd.Series(True, index=df.index)
        for p in others:
            mask &= df[p] == spec.base[p]
        out.append(df[mask])
    return out


def paper_table(df: pd.DataFrame, spec: PaperTable) -> pd.DataFrame:
    frames = []
    for block in blocks(df, spec):
        records = {}
        for key, runs in block.groupby(spec.rows, sort=True):  # sorted by raw value (N, B, L order for Table 3)
            row, subsets = {}, {}
            for side in spec.sides:
                sub = runs
                for p, v in side.where.items():
                    sub = sub[sub[p] == v]
                subsets[side.label] = (sub, side.solver)
                has_data = not sub.empty and stat(sub, f"{side.solver}_status").notna().any()
                for c in side.columns:
                    column = COLUMNS[c]
                    row[(side.label, column.header)] = column.value(sub, side.solver) if has_data else DASH
            if spec.speedup:
                slow, fast = spec.speedup
                row[("", "Speedup")] = speedup(*subsets[slow], *subsets[fast])
            label = tuple(spec.labels.get(p, {}).get(v, v) for p, v in zip(spec.rows, key))
            records[label] = row
        frames.append(pd.DataFrame.from_dict(records, orient="index"))

    table = pd.concat(frames)
    table.index.names = [spec.headers.get(p, ROW_HEADERS.get(p, p)) for p in spec.rows]
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def find_table(stats_path: Path) -> str | None:
    """Name of the table run.sh copied next to the stats (e.g. table1), if any."""
    copies = sorted(stats_path.parent.glob("table*.sh"))
    return copies[0].stem if copies else None


def write_sheets(writer: pd.ExcelWriter, stats_path: Path, table_name: str | None, names: tuple[str, str, str]):
    """Write the raw runs, condensed overview and (if the table is known) paper table of one
    stats.jsonl into `writer`, under the sheet names `names` = (raw, overview, paper)."""
    df = pd.read_json(stats_path, lines=True)

    # Excel cells can't hold lists (policy, attack, structure) and cap text at 32767 chars
    runs = df.apply(lambda col: col.map(lambda v: str(v)[:32767] if isinstance(v, (list, str)) else v))

    raw, overview, paper = names
    runs.to_excel(writer, sheet_name=raw, index=False)
    summary_table(df).to_excel(writer, sheet_name=overview, index=False)
    if table_name in TABLES:
        paper_table(df, TABLES[table_name]).to_excel(writer, sheet_name=paper)
    else:
        print(f"No paper table for {stats_path}: "
              f"{'unknown table ' + table_name if table_name else 'no table file found'} (known: {', '.join(TABLES)})")


def short_name(table_name: str) -> str:
    """Sheet prefix: table1 -> t1."""
    return table_name.replace("table", "t", 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="With --combine OUT.xlsx, every stats file goes into one workbook with sheets "
               "tN_raw, tN_overview, tN_final per table (run_all.sh does this).")
    parser.add_argument("stats", type=Path, nargs="+", help="path(s) to stats.jsonl")
    parser.add_argument("--table", help=f"paper table to build, one of {', '.join(TABLES)} "
                                        "(default: from the table file next to the stats; single file only)")
    parser.add_argument("--combine", type=Path, metavar="OUT.xlsx",
                        help="write all stats files into this one workbook")
    args = parser.parse_args()

    if args.combine:
        if args.table:
            parser.error("--table only works on a single stats file, not with --combine")
        args.combine.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(args.combine) as writer:
            for path in args.stats:
                table_name = find_table(path)
                if table_name is None:
                    print(f"Skipping {path}: no table*.sh next to it, so no sheet name")
                    continue
                t = short_name(table_name)
                write_sheets(writer, path, table_name, (f"{t}_raw", f"{t}_overview", f"{t}_final"))
        print(f"Saved {args.combine}")
    else:
        if len(args.stats) != 1:
            parser.error("give one stats file, or use --combine for several")
        stats_path = args.stats[0]
        table_name = args.table or find_table(stats_path)
        out_path = stats_path.parent / "summary.xlsx"
        with pd.ExcelWriter(out_path) as writer:
            write_sheets(writer, stats_path, table_name, ("runs", "condensed_overview", "paper_table"))
        print(f"Saved {out_path}" + (f" (paper table: {table_name})" if table_name in TABLES else ""))
