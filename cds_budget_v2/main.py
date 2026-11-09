"""Run one instance:  python -m cds_budget_v2.main --n 20 --k 5 --b_p 2 --b_a 2"""
import argparse
import json
import logging
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

from .bnb import BranchAndBound
from .ccg import CCG
from .instance import small_world_clique
from .model import ThreeLevelGame

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",                     type=int,   default=30,  help="Number of nodes in the graph")
    p.add_argument("--k",                     type=int,   default=5,   help="Clique size")
    p.add_argument("--b_p",                   type=int,   default=1,   help="Protector budget")
    p.add_argument("--b_a",                   type=int,   default=3,   help="Attacker budget")
    p.add_argument("--strategy_propagation",  type=int,   default=0,   choices=[-1, 0],    help="Protector cut propagation (BnB): -1 off, 0 constraint")
    p.add_argument("--structure_propagation", type=int,   default=1,   choices=[-1, 0, 1], help="Attacker cut propagation (BnB): -1 off, 0 constraint, 1 lazy")
    p.add_argument("--seed",                  type=int,   default=1,   help="Random seed")
    p.add_argument("--time_limit",            type=float, default=60*60*3, help="Time limit in seconds")
    p.add_argument("--bnb_only",              action="store_true",     help="Run only BnB, skip CCG")
    p.add_argument("--protectors",            type=int,   default=2,   help="Protectors per node within a clique")
    p.add_argument("--bridges",               type=int,   default=4,   help="Max inter-clique bridges per pair of cliques")
    return p.parse_args()


def log_to_file(name: str, path: Path):
    """Send everything the `name` logger writes to `path`, and nothing to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def solver_stats(name: str, solver, game: ThreeLevelGame) -> dict:
    """Statistics shared by BnB and CCG, with keys prefixed by `name` (e.g. bnb_status)."""
    upper_bound = solver.best.weight
    lower_bound = solver.lower_bound
    gap = (upper_bound - lower_bound) / upper_bound if 0 < upper_bound < math.inf else math.inf

    stats = {
        "status":              solver.status,
        "upper_bound":         upper_bound,
        "lower_bound":         lower_bound,
        "gap":                 gap,
        "time":                solver.time,
        "protector_time":      game.solve_time["protector"],
        "attacker_time":       game.solve_time["attacker"],
        "defender_time":       game.solve_time["defender"],
        "strategies_added":    game.strategies_added,
        "strategies_distinct": len(game.distinct_strategies),
        "structures_added":    game.structures_added,
        "structures_distinct": len(game.distinct_structures),
        "policy":              solver.best.policy,
        "attack":              solver.best.attack,
        "structure":           solver.best.structure,
    }
    return {f"{name}_{key}": value for key, value in stats.items()}


def write_stats(path: Path, record: dict):
    """Append `record` to `path` as one JSON line."""
    for key, value in record.items():
        if value in (math.inf, -math.inf):  # JSON has no infinity
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
    log_to_file("ccg", Path(f"{prefix}_ccg.log"))
    print(f"Logs: {prefix}")

    record = {
        "experiment_id": experiment_id,
        "run_id":        run_id,
        "n":             args.n,
        "k":             args.k,
        "b_p":           args.b_p,
        "b_a":           args.b_a,
        "protectors":    args.protectors,
        "bridges":       args.bridges,
        "seed":          args.seed,
        "time_limit":    args.time_limit,
        "structure_propagation": args.structure_propagation,
        "strategy_propagation":  args.strategy_propagation,
        "error":         None,
    }

    # The stats line is written even if a solver crashes; whatever it got so far is recorded.
    bnb = ccg = None
    try:
        inst = small_world_clique(n=args.n, clique_size=args.k, protectors=args.protectors,
                                  bridges=args.bridges, b_p=args.b_p, b_a=args.b_a, seed=args.seed)

        game = ThreeLevelGame(inst, structure_propagation=args.structure_propagation,
                              strategy_propagation=args.strategy_propagation)
        bnb = BranchAndBound(game, time_limit=args.time_limit)
        sol = bnb.solve()
        print(f"BnB  {bnb.status}  value {sol.weight:g}  policy {sol.policy}  attack {sol.attack}  "
              f"structure {sol.structure}  ({len(bnb.nodes)} nodes, {bnb.time:.1f}s)")

        if not args.bnb_only:
            ccg = CCG(inst, time_limit=args.time_limit)
            sol = ccg.solve()
            print(f"CCG  {ccg.status}  value {sol.weight:g}  policy {sol.policy}  attack {sol.attack}  "
                  f"structure {sol.structure}  ({ccg.iterations} iterations, {ccg.time:.1f}s)")
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        if bnb is not None:
            record.update(solver_stats("bnb", bnb, bnb.game))
            record["bnb_nodes"] = len(bnb.nodes)
            record["bnb_heuristic_time"] = bnb.heuristic_time
        if ccg is not None:
            record.update(solver_stats("ccg", ccg, ccg))
            record["ccg_iterations"] = ccg.iterations
            record["ccg_cols"] = ccg.protector_model.NumVars
            record["ccg_rows"] = ccg.protector_model.NumConstrs
        write_stats(experiment_dir / "stats.jsonl", record)


if __name__ == "__main__":
    main()
