"""Run one instance:  python -m cds_budget_v2.main --n 20 --k 5 --b_p 2 --b_a 2"""
import argparse
import logging
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
    p.add_argument("--strategy_propagation",  type=int,   default=0,   choices=[-1, 0],    help="Protector constraint propagation type (not used yet)")
    p.add_argument("--structure_propagation", type=int,   default=1,   choices=[-1, 0, 1], help="Attacker constraint propagation type (not used yet)")
    p.add_argument("--seed",                  type=int,   default=1,   help="Random seed")
    p.add_argument("--time_limit",            type=float, default=60*60*3, help="Time limit in seconds")
    p.add_argument("--bnb_only",              action="store_true",     help="Run only BnB, skip CCG")
    p.add_argument("--protectors",            type=int,   default=2,   help="Protectors per node within a clique")
    p.add_argument("--bridges",               type=int,   default=4,   help="Max inter-clique bridges per pair of cliques")
    p.add_argument("--run_dir",               type=Path,  default=None, help="Folder for this run's logs (default: experiments/MMDD_HHMM_SS/0)")
    return p.parse_args()


def make_run_dir(run_dir: Path | None) -> Path:
    """Use the given folder, or experiments/MMDD_HHMM_SS/0 for a one-off run."""
    if run_dir is None:
        run_dir = EXPERIMENTS_DIR / datetime.now().strftime("%m%d_%H%M_%S") / "0"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_to_file(name: str, path: Path):
    """Send everything the `name` logger writes to `path`, and nothing to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def main():
    args = parse_args()
    run_dir = make_run_dir(args.run_dir)
    log_to_file("bnb", run_dir / "bnb.log")
    log_to_file("ccg", run_dir / "ccg.log")

    inst = small_world_clique(n=args.n, clique_size=args.k, protectors=args.protectors,
                              bridges=args.bridges, b_p=args.b_p, b_a=args.b_a, seed=args.seed)
    print(f"Run dir: {run_dir}")

    # TODO: statistics, CSV output
    bnb = BranchAndBound(ThreeLevelGame(inst), time_limit=args.time_limit)
    sol = bnb.solve()
    print(f"BnB  value {sol.weight:<8g} policy {sol.policy}  attack {sol.attack}  "
          f"structure {sol.structure}  ({len(bnb.nodes)} nodes, {bnb.time:.1f}s)")

    if not args.bnb_only:
        ccg = CCG(inst, time_limit=args.time_limit)
        weight, policy, attack, structure = ccg.solve()
        print(f"CCG  value {weight:<8g} policy {policy}  attack {attack}  "
              f"structure {structure}  ({ccg.iterations} iterations, {ccg.time:.1f}s)")


if __name__ == "__main__":
    main()
