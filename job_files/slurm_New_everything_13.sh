#!/bin/bash
#SBATCH --partition=EPYC        # Partition name
#SBATCH --account=dssc          # Account name
#SBATCH --ntasks=1              # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=2       # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G        # Memory per CPU
#SBATCH --time=2:00:00          # Time limit (2 hours)
#SBATCH --output=output_%j.log  # Standard output log
#SBATCH --error=error_%j.log    # Standard error log (%j is a special SLURM variable that gets replaced with the job ID)
#SBATCH --get-user-env          # Activating the environment

# Activate the virtual environment
source /path_to_your_venv/bin/activate #MODIFY!!!!!

# Run the Python script
python3 Test_distance_slurm.py job_files/params_New_everything_13.json New_everything 
