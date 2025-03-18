#!/bin/bash
#SBATCH --partition=EPYC                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=1                    # CPUs needed for the runner
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --output=job_runner_%j.log           # Standard output log
#SBATCH --error=job_runner_%j.err            # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# This slurm job is supposed to rerun itself starting from a certain point in the list of tests name
iteration_number=${1:-0}  # Default to 0 if no argument provided # The index from which to start inside file_list

# checking the start time
start_time=$(date +%s)

# Read the file line by line and add each line to the array file_list
mapfile -t file_list < "RUNTHIS.txt"

for ((i=iteration_number; i<${#file_list[@]}; i++)); do
    current_job="${file_list[$i]}"
    echo "Processing: ${current_job}"

    # Wait until less then 5 jobs are running
    while true; do

        # Run squeue command to check running jobs (for current user)
        running_jobs=$(squeue -u $USER -h | wc -l)
        
        # If there are less then 5 jobs runnning, proceed
        if [ "$running_jobs" -lt "5" ]; then
            break
        else
            echo "$(date): $((running_jobs - 1)) other jobs still running. Waiting 30 seconds..."
            sleep 30
        fi
    done

    # Sbatch the current job
    sbatch "${current_job}"

    # Wait a bit for the job to be registered)
    sleep 2
    
    # Check that the time passed is less than 1 hour and rerun this job otherwise
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ "$elapsed_time" -ge "3600" ]; then
        sbatch Simple_Job_Runner.sh "$((i+1))"
        exit 0 # Exit with success code
    fi

done
