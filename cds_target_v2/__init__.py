"""Connected Dominating Set (target) protector-attacker-defender game.

The protector fortifies as few nodes as possible so that no attack is worth more than a
target t_p (set by --b_p, a percentage). Solved by branch-and-bound only (no CCG).

Layers, bottom to top:
    instance      the input data (graph, protection relation, budgets)
    game          ThreeLevelGame: solves one level given the others' moves (used by BnB)
    bnb           the solution method
    main          CLI entry point: python -m cds_target_v2.main --n 20 ...
"""
