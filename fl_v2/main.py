"""Run one instance:  python -m fl_v2.main --V 3 --U 15 --K 12 --R 4 --b_p 4 --b_a 5 --anti_symmetry"""
import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .bnb import BranchAndBound
from .instance import random_instance
from .model import ThreeLevelGame

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"   # ISO 8601, e.g. 2026-11-10 14:32:05

console = logging.getLogger("console")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--V",                     type=int,   default=3,   help="Number of facilities")
    p.add_argument("--U",                     type=int,   default=10,  help="Number of survivors")
    p.add_argument("--K",                     type=int,   default=10,  help="Number of crews")
    p.add_argument("--R",                     type=int,   default=4,   help="Number of regions")
    p.add_argument("--b_p",                   type=int,   default=2,   help="Protector budget (patrols)")
    p.add_argument("--b_a",                   type=int,   default=5,   help="Attacker budget (edges cut)")
    p.add_argument("--strategy_propagation",  type=int,   default=0,   choices=[-1, 0],    help="Protector cut propagation: -1 off, 0 constraint")
    p.add_argument("--structure_propagation", type=int,   default=1,   choices=[-1, 0, 1], help="Attacker cut propagation: -1 off, 0 constraint, 1 lazy")
    p.add_argument("--anti_symmetry",         action="store_true",     help="Use crews in index order in the protector")
    p.add_argument("--seed",                  type=int,   default=1,   help="Random seed")
    p.add_argument("--time_limit",            type=float, default=60*60*1, help="Time limit in seconds")
    return p.parse_args()


def log_to_file(name: str, path: Path):
    """Send everything the `name` logger writes to `path`, and nothing to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def setup_console(path: Path, run_id: int):
    """The `console` logger prints to stdout and appends to `path`, every line timestamped and tagged with the run id."""
    console.setLevel(logging.INFO)
    console.propagate = False
    formatter = logging.Formatter(f"%(asctime)s [{run_id:03d}] %(message)s", datefmt=TIMESTAMP_FORMAT)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a")):
        handler.setFormatter(formatter)
        console.addHandler(handler)


def solver_stats(name: str, solver: BranchAndBound, game: ThreeLevelGame) -> dict:
    """BnB statistics, with keys prefixed by `name` (e.g. bnb_status).
    A maximization: lower_bound is the incumbent, upper_bound the best open bound."""
    stats = {
        "status":              solver.status,
        "upper_bound":         solver.upper_bound,
        "lower_bound":         solver.lower_bound,
        "gap":                 solver.gap,
        "time":                solver.time,
        "protector_time":      game.solve_time["protector"],
        "attacker_time":       game.solve_time["attacker"],
        "defender_time":       game.solve_time["defender"],
        "defender_calls":      game.defender_calls,
        "defender_cache_hits": game.defender_cache_hits,
        "strategies_added":    game.strategies_added,
        "strategies_distinct": len(game.distinct_strategies),
        "structures_added":    game.structures_added,
        "structures_distinct": len(game.distinct_structures),
        "policy":              solver.best.policy,
        "attack":              solver.best.attack,
        "allocation":          solver.best.allocation,
        "location":            solver.best.location,
    }
    return {f"{name}_{key}": value for key, value in stats.items()}


def write_stats(path: Path, record: dict):
    """Append `record` to `path` as one JSON line."""
    for key, value in record.items():
        if isinstance(value, float) and math.isinf(value):  # JSON has no infinity
            record[key] = None
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    experiment_id = os.environ.get("EXPERIMENT_ID", datetime.now().strftime("%m%d_%H%M_%S"))
    run_id = int(os.environ.get("RUN_ID", 0))
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    prefix = experiment_dir / f"{run_id:03d}"
    log_to_file("bnb", Path(f"{prefix}_bnb.log"))
    log_to_file("nodes", Path(f"{prefix}_nodes.log"))
    setup_console(experiment_dir / "console.log", run_id)  # shared by all runs of the experiment
    run_total = os.environ.get("RUN_TOTAL")  # set by run.sh; absent for a single run
    progress = f"{run_id + 1}/{run_total} " if run_total else ""
    console.info(f"=== [{progress}run {run_id:03d}] {' '.join(sys.argv[1:])}")
    console.info(f"Logs: {prefix}")

    record = {
        "experiment_id": experiment_id,
        "run_id":        run_id,
        "V":             args.V,
        "U":             args.U,
        "K":             args.K,
        "R":             args.R,
        "b_p":           args.b_p,
        "b_a":           args.b_a,
        "anti_symmetry": args.anti_symmetry,
        "seed":          args.seed,
        "time_limit":    args.time_limit,
        "structure_propagation": args.structure_propagation,
        "strategy_propagation":  args.strategy_propagation,
        "error":         None,
    }

    # The stats line is written even if the solver crashes; whatever it got so far is recorded.
    bnb = None
    try:
        inst = random_instance(num_facilities=args.V, num_survivors=args.U, num_crews=args.K,
                               num_regions=args.R, b_p=args.b_p, b_a=args.b_a, seed=args.seed)

        game = ThreeLevelGame(inst, structure_propagation=args.structure_propagation,
                              strategy_propagation=args.strategy_propagation, anti_symmetry=args.anti_symmetry)
        bnb = BranchAndBound(game, time_limit=args.time_limit)
        sol = bnb.solve()
        console.info(f"BnB  {bnb.status}  value {sol.value:g}  policy {sol.policy}  attack {sol.attack}  "
                     f"({bnb.nodes_explored} nodes explored, {bnb.time:.1f}s)")
    except KeyboardInterrupt:  # Ctrl+C: not an error, so not in err.stderr; marked so postprocess.py drops the run
        record["error"] = "KeyboardInterrupt"
        raise
    except Exception:
        record["error"] = traceback.format_exc()
        console.error(record["error"].rstrip())
        with open(experiment_dir / "err.stderr", "a") as f:  # every error of the experiment, next to its other logs
            f.write(f"{datetime.now().strftime(TIMESTAMP_FORMAT)} run {run_id:03d}: {' '.join(sys.argv[1:])}\n"
                    f"{record['error']}\n")
        raise
    finally:
        if bnb is not None:
            record.update(solver_stats("bnb", bnb, bnb.game))
            record["bnb_nodes"] = bnb.nodes_explored  # explored, not created
            record["bnb_heuristic_time"] = bnb.heuristic_time
        write_stats(experiment_dir / "stats.jsonl", record)


if __name__ == "__main__":
    main()
