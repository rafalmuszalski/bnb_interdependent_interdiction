# Connected Dominating Set Budget. General experiments comparing the BnB and CCG method.

# Define experiment parameters (each row corresponds to one experiment)
PARAMS=(
  "--n 20 --k 5 --b_p 2 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 2 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 2 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 20 --k 5 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 100 --k 10 --b_p 3 --b_a 2 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 100 --k 10 --b_p 3 --b_a 3 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 100 --k 10 --b_p 3 --b_a 4 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 100 --k 10 --b_p 3 --b_a 5 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
  "--n 100 --k 10 --b_p 3 --b_a 6 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600"
)