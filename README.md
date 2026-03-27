# Branch-and-Bound Framework for a General Class of Interdependent Three-stage Interdiction Problems

This repository contains the code and data used in the following paper:

> PAPER-CITATION

If you use this repository in your research, please cite our paper.

## Usage

### Dependencies

The code was tested with the following versions:

| Package | Version |
|---|---|
| [Gurobi](https://www.gurobi.com/) | 13.0.1 |
| Python | 3.12.3 |
| NetworkX | 3.6.1 |
| Gurobipy | 13.0.1 |
| Matplotlib | 3.10.8 |

### Running the Code

To run the experiments, execute the following command from the root directory:

```shell
bash run.sh <PATH_TO_TABLE>
```

where `PATH_TO_TABLE` is the relative path to the instance table.

#### Parallel Execution (Linux)

On Linux, you can run experiments concurrently using:

```shell
bash run_par.sh CDS_budget/table1.sh
```

Configuration details are documented inside `run_par.sh`. You may need to heavily modify this file to suit your hardware.

#### Running with SLURM

The commands inside each `tableX.sh` file can be wrapped with SLURM directives to run on a computing cluster. To do so:

1. Add the necessary SLURM directives to the top of the script.
2. Set the working directory to the project root.
3. Create a per-task logging directory; redirect `stdout`/`stderr` to it and export `EXPERIMENT_DIR` pointing to that directory. 
4. Activate your Python environment and set any required environment variables (`GUROBI_HOME`, `GRB_LICENSE_FILE`).
5. Add the `PARAMS=()` array in the script — SLURM selects which experiment to run based on `SLURM_ARRAY_TASK_ID`.
6. Submit using `sbatch run_slurm.sh`


### Problem-Specific Arguments

Arguments for each problem type are defined via `argparse` and documented in the code and paper. The supported problem types are:

- `CDS_budget`
- `CDS_target`
- `DS_budget`
- `FL`