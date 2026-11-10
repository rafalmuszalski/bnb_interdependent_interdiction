"""Facility location with shared crews: protector-attacker-defender game.

The protector assigns crews to patrol regions, the attacker cuts facility-survivor edges,
and the defender stations the remaining crews at facilities to serve survivors.
A crew can't both patrol and be stationed. Solved by branch-and-bound only (no CCG).

Layers, bottom to top:
    instance      the input data (locations, regions, edges, budgets)
    game          ThreeLevelGame: solves one level given the others' moves (used by BnB)
    bnb           the solution method
    main          CLI entry point: python -m fl_v2.main --V 3 --U 15 ...
"""
