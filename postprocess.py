"""Raw results workbook from stats.jsonl files.

    python postprocess.py stats.jsonl [more stats.jsonl ...] [-o OUT.xlsx]

Every stats file becomes one sheet with one row per run (every line of stats.jsonl as is).
The sheet is named after the table file run.sh copied next to the stats (e.g. t1), or after
the experiment folder if there is none. Without -o the workbook is summary.xlsx next to the
first stats file (run.sh uses this); run_all.sh passes -o to put all tables in one workbook.
"""
import argparse
from pathlib import Path

import pandas as pd


def sheet_name(stats_path: Path) -> str:
    tables = sorted(stats_path.parent.glob("t*.sh"))
    name = tables[0].stem if tables else stats_path.parent.name
    return name[:31]  # Excel's limit


def raw_runs(stats_path: Path) -> pd.DataFrame:
    df = pd.read_json(stats_path, lines=True)
    # Excel cells can't hold lists (policy, attack, structure) and cap text at 32767 chars
    return df.apply(lambda col: col.map(lambda v: str(v)[:32767] if isinstance(v, (list, str)) else v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path, nargs="+", help="path(s) to stats.jsonl")
    parser.add_argument("-o", "--out", type=Path, help="output workbook (default: summary.xlsx next to the first stats file)")
    args = parser.parse_args()

    out = args.out or args.stats[0].parent / "summary.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        for path in args.stats:
            raw_runs(path).to_excel(writer, sheet_name=sheet_name(path), index=False)
    print(f"Saved {out}")
