# Table 3: effect of constraint propagation, DS interdiction, BnB only.
# Run with:  ./run.sh ds_budget_v2/table3.sh   (from the repo root)
#
# Instances (n,b_p,b_a) = (70,4,3) with clique size 5 and (100,3,7) with clique size 10,
# each with all 6 propagation types: protector P in {N=-1, B=0} x attacker A in {N=-1, B=0, L=1}.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--n 50 --k 10 --b_p 3  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation -1 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation  0 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation -1 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  0 --time_limit 3600 --bnb_only"
  # "--n 70  --k 5  --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation -1 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation  0 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation -1 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation -1 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  0 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 3  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
)

SEEDS=(1)
