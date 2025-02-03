import itertools
import numpy as np
import json
import sys
import sqlite3
import os

def initialize_database(test_name):
    """Initialize SQLite database with the required structure"""

    os.makedirs("Databases", exist_ok=True)  # Create directory if needed
    db_path = f"Databases/database_{test_name}.db"
    
    # Creating the database
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        # Create the main results table
        c.execute('''CREATE TABLE IF NOT EXISTS results
(n_psi_added INTEGER,
n_traj INTEGER,
local_std REAL,
n_traj_points INTEGER,
Dist REAL,
Cos_Dist REAL,
Dist_rho REAl,
Norm_glob REAl,
Norm_loc REAl,
Norm_imp REAl,
Pinv_error REAl,
Sum_weights REAL,
Sum_squared_weights REAL,
Elapsed_time REAL,
Process_mem REAL,
PRIMARY KEY (n_psi_added, n_traj, local_std, n_traj_points))''')

        # Optimize database performance

        # Enable WAL mode for better concurrent access
        c.execute('PRAGMA journal_mode=WAL')
        # Optimize for many concurrent readers
        c.execute('PRAGMA cache_size=-2000')  # 2MB cache
        # Batch writes for better performance
        c.execute('PRAGMA synchronous=NORMAL')

        conn.commit()



# Getting the test name
try:
    test_name = sys.argv[1]
    n_jobs = int(sys.argv[2])
except IndexError:
    raise ValueError("No test name or number of jobs provided. Usage: python3 Generate_jobs.py <test_name> <n_jobs>")

# Initialize database
initialize_database(test_name)

# Parameters for the test
list_n_traj_points = [11, 33, 55]
list_stds = [1, 0.8, 0.6]
list_n_traj = [1000, 2000, 3000, 4000]
list_n_psi_added = [350, 600, 1000]

# Saving the parameters in the Params file #NOTE: This is not needed anymore, but could be useful
os.makedirs("Params", exist_ok=True)  # Create directory if needed
with open(f"Params/Params_{test_name}.txt", 'w') as file:
    file.write(','.join(str(x) for x in list_n_psi_added) + '\n')
    file.write(','.join(str(x) for x in list_n_traj) + '\n')
    file.write(','.join(str(x) for x in list_stds) + '\n')
    file.write(','.join(str(x) for x in list_n_traj_points) + '\n')

# Generate all combinations
all_combinations = list(itertools.product(
    list_n_psi_added,
    list_n_traj,
    list_stds,
    list_n_traj_points
))

# Combinations of parameters per job
combinations_per_job = int(np.ceil(len(all_combinations) / n_jobs))

# Distribute combinations across jobs
os.makedirs("job_files", exist_ok=True)  # Create directory if needed
for job_id in range(n_jobs):
    start_idx = job_id * combinations_per_job
    end_idx = min((job_id + 1) * combinations_per_job, len(all_combinations))
    job_combinations = all_combinations[start_idx:end_idx]
    
    # Save combinations for this job
    params_file = f"job_files/params_{test_name}_{job_id}.json"
    with open(params_file, 'w') as f:
        json.dump(job_combinations, f)
    
    # Create SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --partition=EPYC                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=8                    # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G                     # Memory per CPU
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --output=output_{test_name}_{job_id}.log         # Standard output log
#SBATCH --error=error_{test_name}_{job_id}.log           # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# Activate the virtual environment
source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate

# Run the Python script
python3 Test_distance_slurm.py {params_file} {test_name} 
"""

    slurm_file = f"job_files/slurm_{test_name}_{job_id}.sh"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)





# before using databases I saved everything as ndarrays

# Creating the numpy array for the resulting distances 
# The array contains a list of values: (n_psi_added,n_traj,local_std,Dist_mean,Cos_dist_mean,Dist_rho)
#Distances = np.ndarray(shape=(len(list_n_psi_added), len(list_n_traj), len(list_stds), len(list_n_traj_points)), dtype=object)
# Creating the arrays for the norms of the kernels
# The array contains : (n_psi_added,n_traj,local_std,Norm_global,Norm_loc,Norm_imp)
#Norms = np.ndarray(shape=(len(list_n_psi_added), len(list_n_traj), len(list_stds), len(list_n_traj_points)), dtype=object)
# Creating the arrays for the pseudoinverse error
# The array contains : (n_psi_added,n_traj,local_std,Dist_rho)
#Pinv_error = np.ndarray(shape=(len(list_n_psi_added), len(list_n_traj), len(list_stds), len(list_n_traj_points)), dtype=object)

# Filling the arrays with nan values
#for (idx1, n_psi_added) in enumerate(list_n_psi_added):
#    for (idx2, n_traj) in enumerate(list_n_traj):
#        for (idx3, local_std) in enumerate(list_stds):
#            for (idx4, n_traj_points) in enumerate(list_n_traj_points):
#                Distances[idx1, idx2, idx3, idx4] = (n_psi_added,n_traj,local_std,n_traj_points,math.nan,math.nan, math.nan)
#                Norms[idx1, idx2, idx3, idx4] = (n_psi_added,n_traj,local_std,n_traj_points,math.nan,math.nan,math.nan, math.nan)
#                Pinv_error[idx1, idx2, idx3, idx4] = (n_psi_added,n_traj,local_std,n_traj_points,math.nan)

# Saving the arrays
#np.save(f'Distances/Distances_{test_name}.npy', Distances)
#np.save(f'Norms/Norms_{test_name}.npy', Norms)
#np.save(f'Pinv_error/Pinv_error_{test_name}.npy', Pinv_error)
