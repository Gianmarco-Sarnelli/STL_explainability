#!/bin/bash
#SBATCH --partition=lovelace                     # Partition name
#SBATCH --account=ai-lab                     # Account name
#SBATCH --ntasks=1                           # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=16                   # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=4G                     # Memory per CPU
#SBATCH --output=output_selfnorm_V_mu1_M2M_9.log         # Standard output log
#SBATCH --error=error_selfnorm_V_mu1_M2M_9.log           # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# Run the Python script
python3 Test_model.py job_files/params_selfnorm_V_mu1_M2M_9.json selfnorm_V_mu1_M2M yes
