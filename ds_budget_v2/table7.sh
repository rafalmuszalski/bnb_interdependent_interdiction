# Table 7 (appendix): BnB vs CCG, DS interdiction on medium-sized instances.
# Run with:  ./run.sh ds_budget_v2/table7.sh   (from the repo root)
#
# n in {160,180,200}, clique size 10, b_p in {4,5,6}, b_a in {3,...,7}.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--n 50 --k 10 --b_p 3  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 100 --k 10 --b_p 4  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --bnb_only --time_limit 3600"
  # "--n 100 --k 10 --b_p 4  --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --bnb_only --time_limit 3600"
  # "--n 100 --k 10 --b_p 4  --b_a 5 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --bnb_only --time_limit 3600"
  # "--n 100 --k 10 --b_p 4  --b_a 6 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --bnb_only --time_limit 3600"
  # "--n 100 --k 10 --b_p 4  --b_a 7 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --bnb_only --time_limit 3600"
)

SEEDS=(1)
