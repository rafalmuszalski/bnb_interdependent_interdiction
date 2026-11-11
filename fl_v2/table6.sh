# Table 6: BnB on FL interdiction with (SR) and without (no SR) symmetry reduction.
# Run with:  ./run.sh fl_v2/table6.sh   (from the repo root)
#
# One parameter varied at a time from (|V|,|U|,|K|,|R|,b_a) = (3,15,13,4,5): |V| in {3..6},
# |U| in {15,18,20,23}, |K| in {13,15,17,18}, |R| in {4,6,8,9}, b_a in {5,...,8}. b_p = 4.
# Each configuration runs with and without --anti_symmetry; the base configuration is listed once.
# Every PARAMS line runs once per seed in SEEDS. Every parameter is explicit: nothing
# depends on main.py defaults. postprocess.py builds the paper layout (sheet "paper_table").

PARAMS=(
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 4 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 4 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 5 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 5 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 6 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 6 --U 15 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 18 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 18 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 20 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 20 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 23 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 23 --K 13 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 15 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 15 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 17 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 17 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 18 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 18 --R 4 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 6 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 8 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 9 --b_p 4 --b_a 5 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 6 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 7 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 8 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600 --anti_symmetry"
  "--V 3 --U 15 --K 13 --R 4 --b_p 4 --b_a 8 --strategy_propagation 0 --structure_propagation 1 --time_limit 3600"
)

SEEDS=(1)
