"""Numerical constants and solver settings shared by all problems (models, BnB and CCG)."""

EPS = 1e-5               # tolerance when comparing objective values and bounds
BINARY_THRESHOLD = 0.5   # a binary variable above this counts as 1

# DS / CDS only
MIN_NODE_WEIGHT = 100    # node weights are drawn uniformly from [MIN_NODE_WEIGHT, MAX_NODE_WEIGHT]
MAX_NODE_WEIGHT = 300

# FL only
DISTANCE_PENALTY = 1e3   # defender's reward per survivor served; the distance is subtracted from it
DEFENDER_MIP_GAP_ABS = 0.5  # integer objective: an absolute gap below 1 proves the defender optimal
OBJ_TOL = 0.5            # objective values are integers: values within this are equal

GUROBI_SEED = 1          # Gurobi's own random seed: fixed, so --seed only changes the instance
THREADS = 8              # Gurobi threads per model
SOFT_MEM_LIMIT_GB = 16   # Gurobi soft memory limit; a solve that hits it stops with mem_limit
