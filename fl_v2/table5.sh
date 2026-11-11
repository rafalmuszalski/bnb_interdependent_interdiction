# Table 5: BnB, FL interdiction, with symmetry reduction.
# Run with:  ./run.sh fl_v2/table5.sh   (from the repo root)
#
# |V| = 3, |U| in {15,20}, |K| in {12,15}, |R| in {4,6,8,9}, b_a in {5,6,7}, b_p = 4.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--V 3 --U 15 --K 12 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 6 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 6 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 8 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 8 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 9 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 12 --R 9 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 6 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 6 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 8 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 8 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 9 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 9 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 6 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 6 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 8 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 8 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 9 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 12 --R 9 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 6 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 6 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 8 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 8 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 9 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 15 --R 9 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
)

SEEDS=(1)
