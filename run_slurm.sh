#!/bin/bash -l
#SBATCH --cluster=<CLUSTER_NAME>
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --qos=<QOS_NAME>
#SBATCH --account=<ACCOUNT_NAME>
#SBATCH --time=01:00:00              # Time limit per array task
#SBATCH --ntasks=1                   # One task (Gurobi runs as a single process)
#SBATCH --cpus-per-task=16           # Gurobi threads
#SBATCH --mem=16000                  # Memory allocation
#SBATCH --gpus=0
#SBATCH --job-name=<JOB_NAME>
#SBATCH --output=/dev/null           # Bootstrap logs (before redirect)
#SBATCH --error=/dev/null            # Bootstrap errors (before redirect)
#SBATCH --array=0-<N>                # Set range to (number of experiments - 1)
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=ALL

# Move to the working directory
cd <PATH_TO_PROJECT_ROOT> || exit 1

# Job and task IDs
job_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

# Pad the task_id with leading zeros (3 digits: 000, 001, ...)
task_id_padded=$(printf "%03d" "$task_id")

# Define job-specific directory
job_directory="<LOG_DIR>/${job_id}"
mkdir -p "$job_directory"

# Define experiment-specific subdirectory (unique per task)
experiment_directory="$job_directory/${task_id_padded}"
mkdir -p "$experiment_directory"

# Redirect all stdout/stderr for this task into padded directory
exec >"$experiment_directory/out.stdout" 2>"$experiment_directory/errors.stderr"

# Export for use by the experiment
EXPERIMENT_DIR="$experiment_directory"
export EXPERIMENT_DIR

# Activate virtual environment
source <PATH_TO_VENV>/bin/activate

# Set Gurobi environment variables
export GUROBI_HOME=<PATH_TO_GUROBI>
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$GUROBI_HOME/licenses/gurobi.lic
export OMP_NUM_THREADS=$NUM_CORES
export GUROBI_NUM_THREADS=$NUM_CORES

# Print job info
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of Cores: $NUM_CORES"
echo "SLURMTMPDIR=$SLURMTMPDIR"
echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo ""

# Define experiment parameters (each row corresponds to one experiment)
# Update arguments to match your module's argparse interface
PARAMS=(
  "--param1 <VALUE> --param2 <VALUE>"
  "--param1 <VALUE> --param2 <VALUE>"
  # ... add one entry per experiment
)

# Run experiment and store logs per task
srun python -m CDS_budget.main ${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Completion message
echo ""
echo "Job ended at : $(date +"%Y-%m-%d %H:%M:%S")"
echo "Peak memory used: $(cat /proc/$$/status | grep VmPeak | awk '{print $2, $3}')"
sstat -j $SLURM_JOB_ID --format=JobID,MaxRSS,AveCPU,ElapsedRaw 2>/dev/null
seff $SLURM_JOB_ID 2>/dev/null
echo ""