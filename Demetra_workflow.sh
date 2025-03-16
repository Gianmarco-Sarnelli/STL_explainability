#!/bin/bash

# Submit the job generation script and capture its job ID using --parsable
generate_job_id=$(sbatch --parsable Demetra_Generate_jobs.sh)
echo "Submitted Generate_jobs with job ID: $generate_job_id"

# Check if job exists in the queue, if so wait for it
if squeue -j $generate_job_id &>/dev/null; then
    echo "Job $generate_job_id is running, waiting for completion..."
    scontrol wait jobid=$generate_job_id
else
    echo "Job $generate_job_id already completed or not found."
fi


echo "Job $generate_job_id completed, now processing RUNTHIS.txt"
# Read the file line by line and add each line to the array file_list
mapfile -t file_list < "RUNTHIS.txt"

for current_job in "${file_list[@]}"; do
    # Runnning each job
    sbatch "${current_job}"
done
