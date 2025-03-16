#!/bin/bash

# Submit the job generation script and capture its job ID using --parsable
generate_job_id=$(sbatch --parsable Demetra_Generate_jobs.sh)
echo "Submitted Generate_jobs with job ID: $generate_job_id"

sleep 10


# Check if job exists and wait for completion
if squeue -j $generate_job_id &>/dev/null; then
    scontrol wait jobid=$generate_job_id
fi


sleep 10


echo "Job $generate_job_id completed, now processing RUNTHIS.txt"
# Read the file line by line and add each line to the array file_list
mapfile -t file_list < "RUNTHIS.txt"

for current_job in "${file_list[@]}"; do
    # Runnning each job
    sbatch "${current_job}"
done
