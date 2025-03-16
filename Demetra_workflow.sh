#!/bin/bash

# Submit the job generation script and capture its job ID using --parsable
generate_job_id=$(sbatch --parsable Demetra_Generate_jobs.sh)
echo "Submitted Generate_jobs with job ID: $generate_job_id"

# Wait for the job to complete using scontrol
scontrol wait jobid=$generate_job_id

echo "Job $generate_job_id completed, now processing RUNTHIS.txt"
# Read the file line by line and add each line to the array file_list
mapfile -t file_list < "RUNTHIS.txt"

for current_job in "${file_list[@]}"; do
    # Runnning each job
    sbatch "$(current_job)"
done
