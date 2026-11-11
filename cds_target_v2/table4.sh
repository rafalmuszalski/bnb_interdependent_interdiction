# Table 4: BnB, CDS interdiction, target-constrained protector.
# Run with:  ./run.sh cds_target_v2/table4.sh   (from the repo root)
#
# n in {20,30,40,50}, clique size 5, r_p (--b_p, percent of the disruption removed) in
# {20,40,60,80,90}, b_a in {2,3,4}.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--n 20  --k 5  --b_p 20 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 20 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 20 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 40 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 40 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 40 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 60 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 60 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 60 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 80 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 80 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 80 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 90 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 90 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 20  --k 5  --b_p 90 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 20 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 20 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 20 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 40 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 40 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 40 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 60 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 60 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 60 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 80 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 80 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 80 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 90 --b_a 2 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 90 --b_a 3 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
  "--n 30  --k 5  --b_p 90 --b_a 4 --protectors 2 --bridges 4 --strategy_propagation  0 --structure_propagation  1 --time_limit 3600"
)

SEEDS=(1)
