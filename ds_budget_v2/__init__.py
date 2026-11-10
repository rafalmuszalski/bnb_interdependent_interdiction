"""Dominating Set (budget) protector-attacker-defender game.

Layers, bottom to top:
    instance      the input data (graph, protection relation, budgets)
    formulations  Gurobi models built from an instance (shared by BnB and CCG)
    game          ThreeLevelGame: solves one level given the others' moves (used by BnB)
    bnb, ccg      the two solution methods
    main          CLI entry point: python -m ds_budget_v2.main --n 20 ...
"""
