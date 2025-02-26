#!/bin/bash
#SBATCH --partition=EPYC                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=1                    # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G                     # Memory per CPU
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --output=Job_runner_output.log       # Standard output log
#SBATCH --error=Job_runner_error.log         # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# This slurm job is supposed to rerun itself starting from a certain point in the list of tests name
iteration_number="$1" # The element of the test_name to start
test_names=("focusMU0_M2B" "focusMU0_E2B")

# Activate the virtual environment
source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate

# Convert iteration_number to integer and get array length
iteration_number=$((iteration_number))
array_length=${#test_names[@]}

# Check if iteration number exceeds array length
if [ "$iteration_number" -gt "$array_length" ]; then
    echo "Iteration number ($iteration_number) exceeds number of tests ($array_length)"
    exit 0  # Clean exit since this is an expected condition
fi

# Selecting one element of the list
test_name=${test_names[${iteration_number}]}

echo "Starting job for test: $test_name"

# Wait until no jobs are running
while true; do
    # Run squeue command to check running jobs (for current user)
    running_jobs=$(squeue -u $USER -h | wc -l)
    
    # If this is the only job running (count = 1 for this job itself), proceed
    if [ "$running_jobs" -le "1" ]; then
        echo "No other jobs running. Starting Run_jobs.py..."
        break
    else
        echo "$(date): $((running_jobs - 1)) other jobs still running. Waiting 30 seconds..."
        sleep 30
    fi
done

# Run the Python script
python3 Run_jobs.py --test_name "$test_name" --tests_num 10 --SLURM true --iteration $iteration_number
