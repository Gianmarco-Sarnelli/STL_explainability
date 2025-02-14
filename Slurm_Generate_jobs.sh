#!/bin/bash
#SBATCH --partition=EPYC                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=1                    # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G                     # Memory per CPU
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --get-user-env                       # Activating the environment

# Activate the virtual environment
source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate

# Run the Generate_jobs script
# $1 will be test_name, $2 will be number (20), $3 will be "yes"
python3 Generate_jobs.py "$1" "$2" "$3"