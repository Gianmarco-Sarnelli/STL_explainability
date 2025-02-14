import itertools
import numpy as np
import json
import sys
import sqlite3
import os
from traj_measure import BaseMeasure, Easy_BaseMeasure, Brownian, Gaussian, SemiBrownian
from phis_generator import StlGenerator
import torch
import pickle
from Local_Matrix import local_matrix

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
phi_id INTEGER,
base_xi_id INTEGER,
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
n_e REAL,
PRIMARY KEY (n_psi_added, n_traj, local_std, n_traj_points, phi_id, base_xi_id))''')

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
    save_all = sys.argv[3]
except IndexError:
    raise ValueError("Wrong number of agrs provided. Usage: python3 Generate_jobs.py <test_name> <n_jobs> <save_all>")

# Initialize database
initialize_database(test_name)

# Parameters for the test
list_n_traj_points = [11] #[11, 55, 99]
list_stds = [0.6] #[1, 0.8, 0.6]
list_n_traj = [4000] #[1000, 2000, 3000, 4000]
list_n_psi_added = [1000] #[350, 600, 1000]
list_phi_id = [x for x in range(100)]
list_base_xi_id = [x for x in range(100)]

# If we want to save all variables we need to initialize them
if save_all=="yes":
    if (len(list_n_traj_points) != 1) or (len(list_stds) != 1):
        raise RuntimeError("To be able to save the trajectories the parameters 'list_n_traj_points' and 'list_stds' must have a single value")
    
    ##  Saving the values of psi, phi, global_xi, local_xi, base_xi, dweights and PHI ##
    
    # Device used
    device: torch.device = torch.device("cpu")  # Force CPU usage
    # Evaluation of formulae
    evaluate_at_all_times = False # TODO: implement for time evaluation
    n_vars = 2
    # NOTE: n_traj_points must be >= max_timespan in phis_generator
    # NOTE: max_timespan must be greater than 11 (for some reason) #TODO: find why this is the case
    #n_traj_points = 11
    # Parameters for the sampling of the formulae
    leaf_probability = 0.5
    time_bound_max_range = 10
    prob_unbound_time_operator = 0.1
    atom_threshold_sd = 1.0

    # Parameters of the process
    n_psi_added = max(list_n_psi_added)  # We will save all the psi needed and then select a subset of them for each test.
    n_traj = max(list_n_traj)
    local_std = list_stds[0]
    n_traj_points = list_n_traj_points[0]
    n_psi = n_traj + n_psi_added 
    local_n_traj = n_traj
    global_n_traj = n_traj

    # Checking if the test name is in the format "global_name2local_name"
    global_name, local_name = "M", "B"
    if "2" in test_name:
        index = test_name.index("2")
        global_name = test_name[index - 1]
        local_name = test_name[index + 1]

    # Initializing the global trajectory distribution and sampling global_xi
    # BaseMeasure = M, Easy_BaseMeasure = E, Brownian = B, Gaussian = G, SemiBrownian = S.
    match global_name:
        case "M":
            global_distr = BaseMeasure(device=device)
        case "E":
            global_distr = Easy_BaseMeasure(device=device)
        case "B":
            global_distr = Brownian(device=device)
        case "G":
            global_distr = Gaussian(device=device)
        case "S":
            global_distr = SemiBrownian(device=device)
        case _:
            raise RuntimeError("Global distribution name is not allowed")
         
    global_xi = global_distr.sample(samples=global_n_traj, 
                                varn=n_vars, 
                                points=n_traj_points)
    if not os.path.exists("Global_xi_dir"):
        os.makedirs("Global_xi_dir")
    torch.save(global_xi, os.path.join("Global_xi_dir",f"{test_name}.pt"))

    # ITERATING ON base_xi_id:

    base_xi_dict = {}
    local_xi_dict = {}
    dweights_dict = {}
    for i in list_base_xi_id:

        # Initializing the base trajectory distribution and sampling base_xi
        base_xi = global_distr.sample(samples=1,
                                varn=n_vars, 
                                points=n_traj_points)
        base_xi_dict[i] = base_xi

        # Initializing the local trajectory distribution and sampling local_xi
        match local_name:
            case "M":
                local_distr = BaseMeasure(base_traj=base_xi[0], sigma1=local_std, device=device)
            case "E":
                local_distr = Easy_BaseMeasure(base_traj=base_xi[0], sigma1=local_std, device=device)
            case "B":
                local_distr = Brownian(base_traj=base_xi[0], std=local_std, device=device)
            case "G":
                local_distr = Gaussian(base_traj=base_xi[0], std=local_std, device=device)
            case "S":
                local_distr = SemiBrownian(base_traj=base_xi[0], std=local_std, device=device)
            case _:
                raise RuntimeError("Local distribution name is not allowed")
            
        local_xi = local_distr.sample(samples=local_n_traj, 
                                        varn=n_vars, 
                                        points=n_traj_points)
        local_xi_dict[i] = local_xi

        # Saving the values of the weights
        converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_psi, 
                                n_traj = global_n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                target_distr = local_distr,
                                proposal_distr = global_distr)
        converter.compute_dweights()
        dweights_dict[i] = converter.dweights
    
    if not os.path.exists("Dweights_dir"):
        os.makedirs("Dweights_dir")
    torch.save(dweights_dict, os.path.join("Local_xi_dir",f"{test_name}.pt"))
    if not os.path.exists("Local_xi_dir"):
        os.makedirs("Local_xi_dir")
    torch.save(local_xi_dict, os.path.join("Local_xi_dir",f"{test_name}.pt"))
    if not os.path.exists("Base_xi_dir"):
        os.makedirs("Base_xi_dir")
    torch.save(base_xi_dict, os.path.join("Base_xi_dir",f"{test_name}.pt"))

    # Sampling the formulae (phi and psi) for the kernels
    # Each formula phi will have its kernel representation
    formulae_distr = StlGenerator(leaf_prob=leaf_probability, 
                            time_bound_max_range=time_bound_max_range,
                            unbound_prob=prob_unbound_time_operator, 
                            threshold_sd=atom_threshold_sd)
    
    # ITERATING ON phi_id:
    phi_bag_dict = {}
    for i in list_phi_id:
        phi_bag = formulae_distr.bag_sample(1, n_vars)
        phi_bag_dict[i] = phi_bag
    # Save with pickle
    if not os.path.exists("phis_dir"):
        os.makedirs("phis_dir")
    with open(os.path.join("phis_dir", f"{test_name}.pkl"), 'wb') as f:
        pickle.dump(phi_bag_dict, f)

    psi_bag = formulae_distr.bag_sample(n_psi, n_vars)
    # Save with pickle
    if not os.path.exists("psis_dir"):
        os.makedirs("psis_dir")
    with open(os.path.join("psis_dir", f"{test_name}.pkl"), 'wb') as f:
        pickle.dump(psi_bag, f)

    # Computing the robustness of each psi over the global_xi
    PHI = torch.empty(n_psi, global_n_traj)
    for (i, formula) in enumerate(psi_bag):
        PHI[i, :] = torch.tanh(formula.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    if not os.path.exists("PHI_dir"):
        os.makedirs("PHI_dir")
    torch.save(PHI, os.path.join("PHI_dir",f"{test_name}.pt"))
else:
    save_all = "no"


# Saving the parameters in the Params file #NOTE: This is not needed anymore, but could be useful
os.makedirs("Params", exist_ok=True)  # Create directory if needed
with open(f"Params/Params_{test_name}.txt", 'w') as file:
    file.write(','.join(str(x) for x in list_n_psi_added) + '\n')
    file.write(','.join(str(x) for x in list_n_traj) + '\n')
    file.write(','.join(str(x) for x in list_stds) + '\n')
    file.write(','.join(str(x) for x in list_n_traj_points) + '\n')
    file.write(','.join(str(x) for x in list_phi_id) + '\n')
    file.write(','.join(str(x) for x in list_base_xi_id) + '\n')

# Generate all combinations
all_combinations = list(itertools.product(
    list_n_psi_added,
    list_n_traj,
    list_stds,
    list_n_traj_points,
    list_phi_id,
    list_base_xi_id,
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
python3 Test_distance.py {params_file} {test_name} {save_all}
"""

    slurm_file = f"job_files/slurm_{test_name}_{job_id}.sh"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)