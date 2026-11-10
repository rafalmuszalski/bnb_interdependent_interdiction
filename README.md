# Branch-and-Bound Framework for a General Class of Interdependent Three-stage Interdiction Problems

This repository contains the code and data used in the following paper:

> PAPER-CITATION

If you use this repository in your research, please cite our paper.

## Usage

### Dependencies

Python 3.14.3 with the packages in `requirements.txt`:

```
gurobipy==13.0.2
networkx==3.6.1
pandas==3.0.6
openpyxl==3.1.5
matplotlib==3.11.1
numpy==2.5.1
```
 A full [Gurobi](https://www.gurobi.com/) license is required.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`run.sh` uses `.venv/bin/python`; to use another interpreter, set `PYTHON`, e.g. `PYTHON=python3 ./run.sh ...`.

### Running the Code

To run the experiments, execute the following command from the root directory:

```shell
bash run.sh <PATH_TO_TABLE>
```

where `PATH_TO_TABLE` is the local path to the instance table.

For example, while in the `bnb_interdependent_interdiction` folder:

```shell
bash run.sh CDS_budget/table1.sh
bash run.sh DS_budget/table2.sh
bash run.sh DS_budget/table3.sh
bash run.sh CDS_target/table4.sh
bash run.sh FL/table5.sh
bash run.sh FL/table6.sh
bash run.sh DS_budget/table7.sh
bash run.sh DS_budget/table8.sh

```
would run the experiments to generate the eight tables that appear in the paper. 

### Interpreting the Results

Each call to `run.sh` creates `<problem>/experiments/<MMDD_HHMM_SS>/` with:

| File | Contents |
|---|---|
| `console.log` | Console output of all runs, timestamped: the table being run, each run's arguments and its BnB/CCG result. |
| `stats.jsonl` | One JSON line per run: the run's arguments and all statistics (below). |
| `summary.xlsx` | Built from `stats.jsonl`. Sheet `runs`: one row per run. Sheet `condensed_overview`: average and maximum of the main statistics per (`n`, `b_p`, `b_a`). |
| `NNN_bnb.log` | BnB progress of run `NNN`: one line per node with its status, value, lower/upper bound, gap and elapsed time. |
| `NNN_nodes.log` | One line per BnB node with the protector's policy, the attacker's strategy and the defender's structure. |
| `NNN_ccg.log` | CCG progress of run `NNN`: one line per iteration with the master size, bounds, gap and elapsed time. |

BnB node statuses: `int` (feasible for the interdependent game), `frac` (policy and structure overlap; branched), `sopt` (attacker over-hedged; branched), `infea` (no feasible structure), `kill` (cut off by the incumbent mid-solve), `prune` (bound no better than the incumbent; not solved).

Besides the run's arguments, `stats.jsonl` has `error` (the traceback if the run crashed, else empty) and these statistics, prefixed `bnb_` or `ccg_`:

| Statistic | Meaning |
|---|---|
| `status` | `optimal`, `time_limit` or `mem_limit` |
| `upper_bound`, `lower_bound`, `gap` | Final bounds and relative gap |
| `policy`, `attack`, `structure` | Best solution found |
| `time` | Wall-clock seconds from model construction to the end of the solve |
| `protector_time`, `attacker_time`, `defender_time` | Gurobi time per model, excluding solves nested in its callbacks; their sum is at most `time` |
| `bnb_heuristic_time` | Time in the BnB heuristic; its solves are also counted in the three times above |
| `bnb_nodes` | Nodes explored (solved), excluding pruned and unexplored nodes |
| `ccg_iterations`, `ccg_cols`, `ccg_rows` | Master problems solved, and the final master's size |
| `strategies_added`, `structures_added` | Cuts added to the protector and attacker models (repeats included); `*_distinct` counts unique ones |
| `defender_calls`, `defender_cache_hits` | Defender subproblems requested, and how many were answered from the cache without a Gurobi solve |

### Problem-Specific Arguments

Arguments for each problem type are defined via `argparse` and documented in the code and paper. The supported problem types are:

- `CDS_budget`
- `CDS_target`
- `DS_budget`
- `FL`
