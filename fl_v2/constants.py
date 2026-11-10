"""Numerical constants and solver settings shared by the models and BnB."""

EPS = 1e-3               # tolerance when comparing objective values and bounds
BINARY_THRESHOLD = 0.5   # a binary variable above this counts as 1
BIG_M = 1e6              # big-M in the cuts; also the value of an infeasible protector (as -M)
DISTANCE_PENALTY = 1e3   # defender's reward per survivor served; the distance is subtracted from it
DEFENDER_MIP_GAP = 1e-8  # the defender is solved to (near) proven optimality

GUROBI_SEED = 1          # Gurobi's own random seed: fixed, so --seed only changes the instance
THREADS = 8              # Gurobi threads per model
SOFT_MEM_LIMIT_GB = 16   # Gurobi soft memory limit; a solve that hits it stops with mem_limit
