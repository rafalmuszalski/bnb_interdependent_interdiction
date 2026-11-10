"""Numerical constants and solver settings shared by all problems (models, BnB and CCG)."""

EPS = 1e-5               # tolerance when comparing objective values and bounds
BINARY_THRESHOLD = 0.5   # a binary variable above this counts as 1

# FL only
DISTANCE_PENALTY = 1e3   # defender's reward per survivor served; the distance is subtracted from it
DEFENDER_MIP_GAP = 1e-8  # the defender is solved to (near) proven optimality

GUROBI_SEED = 1          # Gurobi's own random seed: fixed, so --seed only changes the instance
THREADS = 8              # Gurobi threads per model
SOFT_MEM_LIMIT_GB = 16   # Gurobi soft memory limit; a solve that hits it stops with mem_limit
