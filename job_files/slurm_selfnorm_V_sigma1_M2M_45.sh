#!/bin/bash

#SBATCH --partition=EPYC                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=16                   # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G                     # Memory per CPU
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --output=output_selfnorm_V_sigma1_M2M_45.log         # Standard output log
#SBATCH --error=error_selfnorm_V_sigma1_M2M_45.log           # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# Activate the virtual environment
source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate

# Run the Python script
python3 Test_model.py job_files/params_selfnorm_V_sigma1_M2M_45.json selfnorm_V_sigma1_M2M yes
