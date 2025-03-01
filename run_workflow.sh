#!/bin/bash

# Submit the job generation script and capture its job ID using --parsable
generate_job_id=$(sbatch --parsable Slurm_Generate_jobs.sh)
echo "Submitted Generate_jobs with job ID: $generate_job_id"

# Submit the job runner script with dependency on the generation script
runner_job_id=$(sbatch --parsable --dependency=afterok:$generate_job_id Simple_Job_Runner.sh 0)
echo "Submitted Job_runner with job ID: $runner_job_id (will start after job $generate_job_id completes successfully)"
