"""Run one instance:  python -m cds_budget_v2.main --n 20 --k 5 --b_p 2 --b_a 2"""
import argparse

from .instance import small_world_clique
from .model import ThreeLevelGame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",                     type=int,   default=30,  help="Number of nodes in the graph")
    p.add_argument("--k",                     type=int,   default=5,   help="Clique size")
    p.add_argument("--b_p",                   type=int,   default=1,   help="Protector budget")
    p.add_argument("--b_a",                   type=int,   default=3,   help="Attacker budget")
    p.add_argument("--strategy_propagation",  type=int,   default=0,   choices=[-1, 0],    help="Protector constraint propagation type (not used yet)")
    p.add_argument("--structure_propagation", type=int,   default=1,   choices=[-1, 0, 1], help="Attacker constraint propagation type (not used yet)")
    p.add_argument("--seed",                  type=int,   default=1,   help="Random seed")
    p.add_argument("--time_limit",            type=float, default=60*60*3, help="Time limit in seconds (not used yet)")
    p.add_argument("--bnb_only",              action="store_true",     help="Run only BnB (not used yet)")
    p.add_argument("--protectors",            type=int,   default=2,   help="Protectors per node within a clique")
    p.add_argument("--bridges",               type=int,   default=4,   help="Max inter-clique bridges per pair of cliques")
    return p.parse_args()


def main():
    args = parse_args()

    inst = small_world_clique(n=args.n, clique_size=args.k, protectors=args.protectors,
                              bridges=args.bridges, b_p=args.b_p, b_a=args.b_a, seed=args.seed)

    # TODO: replace with BnB / CCG runs, logging, and CSV output
    game = ThreeLevelGame(inst)
    policy, attack, structure, value = game.solve_three_level_game()

    print(f"Policy:    {policy}")
    print(f"Attack:    {attack}")
    print(f"Structure: {structure}")
    print(f"Value:     {value}")


if __name__ == "__main__":
    main()
