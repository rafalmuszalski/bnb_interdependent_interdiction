# Table 8 (appendix): BnB only, DS interdiction on large instances.
# Run with:  ./run.sh ds_budget_v2/table8.sh   (from the repo root)
#
# n in {220,240}, clique size 10, b_p in {3,4,5,6}, b_a in {3,4,5} (the grid of the
# table; the paper's text says {4,5,6} for both).
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--n 50 --k 10 --b_p 3  --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 220 --k 10 --b_p 3  --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
  # "--n 220 --k 10 --b_p 3  --b_a 5 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600 --bnb_only"
)

SEEDS=(1)
