# Table 1: BnB vs CCG, CDS interdiction, budget-constrained protector.
# Run with:  ./run.sh cds_budget_v2/table1.sh   (from the repo root)
#
# n in {20,30,40,50}, clique size 5, b_p and b_a in {2,3,4}.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  # "--n 30  --k 10  --b_p 2  --b_a 2 --bnb_only --time_limit 3600"
  # "--n 30  --k 10  --b_p 2  --b_a 3 --bnb_only --time_limit 3600"
  # "--n 30  --k 10  --b_p 2  --b_a 4 --bnb_only --time_limit 3600"
  # "--n 30  --k 10  --b_p 3  --b_a 2 --bnb_only --time_limit 3600"
  # "--n 30  --k 10  --b_p 3  --b_a 3 --bnb_only --time_limit 3600"
  # "--n 30  --k 10  --b_p 3  --b_a 4 --bnb_only --time_limit 3600"
  "--n 60  --k 10  --b_p 4  --b_a 2 --bnb_only --time_limit 3600"
  "--n 60  --k 10  --b_p 4  --b_a 3 --bnb_only --time_limit 3600"
  "--n 60  --k 10  --b_p 4  --b_a 4 --bnb_only --time_limit 3600"
)

SEEDS=(1)
