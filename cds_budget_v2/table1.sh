# Table 1: BnB vs CCG on n=20 small-world clique graphs, protector budget x attacker budget.
# Run with:  ./run.sh cds_budget_v2/table1.sh   (from the repo root)
#
# Every PARAMS line runs once per seed in SEEDS. Leave SEEDS out to put --seed in the PARAMS lines.

PARAMS=(
  "--n 20 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 3 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 3 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 4 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 4 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 20 --k 5 --b_p 4 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 3 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 3 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 4 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 4 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
  "--n 30 --k 5 --b_p 4 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 1200"
)

SEEDS=(0 1 2 3 4)
