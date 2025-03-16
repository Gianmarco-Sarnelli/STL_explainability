#!/bin/bash
#SBATCH --partition=lovelace  # Updated partition name
#SBATCH --account=ai-lab      # Updated account name
#SBATCH --ntasks=1            # Number of tasks
#SBATCH --nodes=1             # Use exactly one node
#SBATCH --cpus-per-task=1     # Not meny cpus needed
#SBATCH --mem-per-cpu=2G      # Increased memory
#SBATCH --output=Generate_jobs_output_%j.log # Standard output with job ID
#SBATCH --error=Generate_jobs_error_%j.log   # Standard error with job ID
#SBATCH --get-user-env        # Activating the environment


# Read the file line by line and add each line to the array file_list
mapfile -t file_list < "RUNTHIS.txt"

for current_job in "${file_list[@]}"; do
    # Runnning each job
    sbatch "${current_job}"
done