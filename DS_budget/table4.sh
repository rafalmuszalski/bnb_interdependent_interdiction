# Dominating Set Budget. Investigating Constraint propagation effects. BnB only. 

# Define experiment parameters (each row corresponds to one experiment)
PARAMS=(
  "--n 70  --k 5  --b_p 4  --b_a 3 --strategy_propagation -1 --structure_propagation -1 --seed 1 --time_limit 3600 --bnb_only"
  "--n 70  --k 5  --b_p 4  --b_a 3 --strategy_propagation -1 --structure_propagation  0 --seed 1 --time_limit 3600 --bnb_only"
  "--n 70  --k 5  --b_p 4  --b_a 3 --strategy_propagation -1 --structure_propagation  1 --seed 1 --time_limit 3600 --bnb_only"
  "--n 70  --k 5  --b_p 4  --b_a 3 --strategy_propagation  0 --structure_propagation  0 --seed 1 --time_limit 3600 --bnb_only"
  "--n 70  --k 5  --b_p 4  --b_a 3 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600 --bnb_only"
  "--n 100 --k 10 --b_p 3  --b_a 7 --strategy_propagation -1 --structure_propagation -1 --seed 1 --time_limit 3600 --bnb_only"
  "--n 100 --k 10 --b_p 3  --b_a 7 --strategy_propagation -1 --structure_propagation  0 --seed 1 --time_limit 3600 --bnb_only"
  "--n 100 --k 10 --b_p 3  --b_a 7 --strategy_propagation -1 --structure_propagation  1 --seed 1 --time_limit 3600 --bnb_only"
  "--n 100 --k 10 --b_p 3  --b_a 7 --strategy_propagation  0 --structure_propagation  0 --seed 1 --time_limit 3600 --bnb_only"
  "--n 100 --k 10 --b_p 3  --b_a 7 --strategy_propagation  0 --structure_propagation  1 --seed 1 --time_limit 3600 --bnb_only"
)