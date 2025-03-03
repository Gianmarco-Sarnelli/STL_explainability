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
(weight_strategy STRING,
n_psi_added INTEGER,
n_traj INTEGER,
local_std REAL,
global_std REAL,
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
PRIMARY KEY (weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id))''')

        # Optimize database performance

        # Enable WAL mode for better concurrent access
        c.execute('PRAGMA journal_mode=WAL')
        # Optimize for many concurrent readers
        c.execute('PRAGMA cache_size=-2000')  # 2MB cache
        # Batch writes for better performance
        c.execute('PRAGMA synchronous=NORMAL')

        conn.commit()



def save_params(test_name, list_weight_strategy, list_n_traj_points, list_local_std, list_global_std, list_n_traj, list_n_psi_added, list_phi_id, list_base_xi_id):

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

    #Parameters specific for mu_0
    totvar_mult = 1
    sign_ch = 2

    # Parameters of the process
    n_psi_added = max(list_n_psi_added)  # We will save all the psi needed and then select a subset of them for each test.
    n_traj = max(list_n_traj)
    n_psi = n_traj + n_psi_added 

    # Checking if the test name is in the format "global_name2local_name"
    global_name, local_name = "M", "B"
    if "2" in test_name:
        index = test_name.index("2")
        global_name = test_name[index - 1]
        local_name = test_name[index + 1]

    ## Global_xi, Base_xi, Local_xi and Dweights ##
    global_xi_dict = {}
    base_xi_dict = {}
    local_xi_dict = {}
    dweights_dict = {}
    true_dweights_dict = {}
    base_std = 1  # We fix this parameter to make the sampling of the base_xi independent of the global_std
    ## Iteration on n_traj_points ##
    for n_traj_points in list_n_traj_points:
        ## Iteration on global_std ##
        for global_std in list_global_std:
            match global_name:
                case "M":
                    #global_distr = BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                    global_distr = BaseMeasure(sigma0=global_std, device=device)
                case "E":
                    #global_distr = Easy_BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                    global_distr = Easy_BaseMeasure(sigma0=global_std, device=device)
                case "B":
                    global_distr = Brownian(device=device)
                case "G":
                    global_distr = Gaussian(device=device)
                case "S":
                    global_distr = SemiBrownian(device=device)
                case _:
                    raise RuntimeError("Global distribution name is not allowed")
            global_xi = global_distr.sample(samples=n_traj, 
                                            varn=n_vars, 
                                            points=n_traj_points)
            global_xi_dict[(n_traj_points, global_std)] = global_xi

            #some print:
            print(f"Keys in global_xi_dict: {global_xi_dict.keys()}")

        ## Iteration on base_xi_id
        for base_xi_id in list_base_xi_id:
            # Initializing the base_distr
            match global_name:
                case "M":
                    base_distr = BaseMeasure(sigma0=base_std, device=device)
                    #base_distr = BaseMeasure(sigma0=base_std, sigma1=base_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                case "E":
                    base_distr = Easy_BaseMeasure(sigma0=base_std, device=device)
                    #base_distr = Easy_BaseMeasure(sigma0=base_std, sigma1=base_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                case "B":
                    base_distr = Brownian(device=device)
                case "G":
                    base_distr = Gaussian(device=device)
                case "S":
                    base_distr = SemiBrownian(device=device)
                case _:
                    raise RuntimeError("Global distribution name is not allowed")
            
            base_xi = base_distr.sample(samples=1, 
                                        varn=n_vars, 
                                        points=n_traj_points)
            base_xi_dict[(n_traj_points, base_xi_id)] = base_xi

            #some print
            print(f"Keys in base_xi_dict: {base_xi_dict.keys()}")

            ## Iteration on local_std ##
            for local_std in list_local_std:
                # Inizializing the local_distr
                match local_name:
                    case "M":
                        local_distr = BaseMeasure(base_traj=base_xi[0], sigma0=local_std, device=device)
                        #local_distr = BaseMeasure(base_traj=base_xi[0], sigma0=local_std, sigma1=local_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                    case "E":
                        local_distr = Easy_BaseMeasure(base_traj=base_xi[0], sigma0=local_std, device=device)
                        #local_distr = Easy_BaseMeasure(base_traj=base_xi[0], sigma0=local_std, sigma1=local_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                    case "B":
                        local_distr = Brownian(base_traj=base_xi[0], std=local_std, device=device)
                    case "G":
                        local_distr = Gaussian(base_traj=base_xi[0], std=local_std, device=device)
                    case "S":
                        local_distr = SemiBrownian(base_traj=base_xi[0], std=local_std, device=device)
                    case _:
                        raise RuntimeError("Local distribution name is not allowed")
                    
                local_xi = local_distr.sample(samples=n_traj, 
                                                varn=n_vars, 
                                                points=n_traj_points)
                local_xi_dict[(n_traj_points, local_std, base_xi_id)] = local_xi

                #some print:
                print(f"Keys in local_xi_dict: {local_xi_dict.keys()}")

                ## Iteration on weight_strategy ##
                for weight_strategy in list_weight_strategy:
                    ## Iteration again on global_std ##
                    for global_std in list_global_std: # We need to reiterate on the global_std to obtain the correct global_distr and global_xi

                        global_xi = global_xi_dict[(n_traj_points, global_std)] # Reloading the currect global_xi

                        match global_name:
                            case "M":
                                global_distr = BaseMeasure(sigma0=global_std, device=device)
                                #global_distr = BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                            case "E":
                                global_distr = Easy_BaseMeasure(sigma0=global_std, device=device)
                                #global_distr = Easy_BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
                            case "B":
                                global_distr = Brownian(device=device)
                            case "G":
                                global_distr = Gaussian(device=device)
                            case "S":
                                global_distr = SemiBrownian(device=device)
                            case _:
                                raise RuntimeError("Global distribution name is not allowed")
                            
                        # Computing the Dweights
                        converter = local_matrix(n_vars = n_vars, 
                                                n_formulae = n_psi, 
                                                n_traj = n_traj, 
                                                n_traj_points = n_traj_points,
                                                evaluate_at_all_times = evaluate_at_all_times,
                                                target_distr = local_distr,
                                                proposal_distr = global_distr,
                                                weight_strategy = weight_strategy,
                                                proposal_traj = global_xi)
                        converter.compute_dweights()
                        dweights_dict[(weight_strategy, n_traj_points, global_std, local_std, base_xi_id)] = converter.dweights
                        true_dweights_dict[(weight_strategy, n_traj_points, global_std, local_std, base_xi_id)] = converter.true_dweights

                        #some prints:
                        print(f"Keys in dweights_dict: {dweights_dict.keys()}")
                        print(f"Keys in true_dweights_dict: {true_dweights_dict.keys()}")


    # Saving Global_xi_dict
    os.makedirs("Global_xi_dir", exist_ok=True)
    torch.save(global_xi_dict, os.path.join("Global_xi_dir",f"{test_name}.pt"))
    # Saving Base_xi_dict
    os.makedirs("Base_xi_dir", exist_ok=True)
    torch.save(base_xi_dict, os.path.join("Base_xi_dir",f"{test_name}.pt"))
    # Saving local_xi_dict
    os.makedirs("Local_xi_dir", exist_ok=True)
    torch.save(local_xi_dict, os.path.join("Local_xi_dir",f"{test_name}.pt"))
    # Saving Dweights
    os.makedirs("Dweights_dir", exist_ok=True)
    torch.save(dweights_dict, os.path.join("Dweights_dir",f"{test_name}.pt"))
    # Saving True_Dweights
    os.makedirs("True_Dweights_dir", exist_ok=True)
    torch.save(true_dweights_dict, os.path.join("True_Dweights_dir",f"{test_name}.pt"))



    ## psi and phi ##
    formulae_distr = StlGenerator(leaf_prob=leaf_probability, 
                            time_bound_max_range=time_bound_max_range,
                            unbound_prob=prob_unbound_time_operator, 
                            threshold_sd=atom_threshold_sd)
    
    ## Iteration on phi_id ##
    phi_bag_dict = {}
    for i in list_phi_id:
        phi_bag = formulae_distr.bag_sample(1, n_vars)
        phi_bag_dict[i] = phi_bag
    # Save with pickle
    os.makedirs("phis_dir", exist_ok=True)
    with open(os.path.join("phis_dir", f"{test_name}.pkl"), 'wb') as f:
        pickle.dump(phi_bag_dict, f)

    psi_bag = formulae_distr.bag_sample(n_psi, n_vars)
    # Save with pickle
    os.makedirs("psis_dir", exist_ok=True)
    with open(os.path.join("psis_dir", f"{test_name}.pkl"), 'wb') as f:
        pickle.dump(psi_bag, f)
    

    print("params are saved!")





# Getting the test name
try:
    test_name = sys.argv[1]
    n_jobs = int(sys.argv[2])
    save_all = sys.argv[3]
    partition = sys.argv[4]
except IndexError:
    raise ValueError("Wrong number of agrs provided. Usage: python3 Generate_jobs.py <test_name> <n_jobs> <save_all> <partition>")

# Initialize database
initialize_database(test_name)

# check for the right partition:
if (partition != "THIN") and (partition != "EPYC"):
    raise RuntimeError(f"Unable to use the partition: {partition}")

# Parameters for the test
list_weight_strategy = ["self_norm"]#, "standard"]
list_n_traj_points = [11]
list_local_std = [1, 0.6]
list_global_std = [1, 4]
list_n_traj = [2000, 4000]
list_n_psi_added = [500]
list_phi_id = [0] #[x for x in range(3)]
list_base_xi_id = [x for x in range(5)]

# If we want to save all variables we need to initialize them
if save_all=="yes":
    ##  Saving the values of psi, phi, global_xi, local_xi, base_xi and dweights ##
    save_params(test_name, 
                list_weight_strategy, 
                list_n_traj_points, 
                list_local_std, 
                list_global_std, 
                list_n_traj, 
                list_n_psi_added, 
                list_phi_id, 
                list_base_xi_id)
else:
    save_all = "no"


# Saving the parameters in the Params file #NOTE: This is not needed anymore, but could be useful
os.makedirs("Params", exist_ok=True)  # Create directory if needed
with open(f"Params/Params_{test_name}.txt", 'w') as file:
    file.write("list_n_psi_added,"+','.join(str(x) for x in list_n_psi_added) + '\n')
    file.write("list_n_traj,"+','.join(str(x) for x in list_n_traj) + '\n')
    file.write("list_local_std,"+','.join(str(x) for x in list_local_std) + '\n')
    file.write("list_n_traj_points,"+','.join(str(x) for x in list_n_traj_points) + '\n')
    file.write("list_phi_id,"+','.join(str(x) for x in list_phi_id) + '\n')
    file.write("list_base_xi_id,"+','.join(str(x) for x in list_base_xi_id) + '\n')
    file.write("list_weight_strategy,"+','.join(x for x in list_weight_strategy) + '\n')
    file.write("list_global_xi,"+','.join(str(x) for x in list_global_std))

# Generate all combinations
all_combinations = list(itertools.product(
    list_n_psi_added,
    list_n_traj,
    list_local_std,     # NOTE: in Test_distance.py we use FIRST local_std THEN global_std, so alway use this notation!!
    list_global_std,
    list_n_traj_points,
    list_phi_id,
    list_base_xi_id,
    list_weight_strategy
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
#SBATCH --partition={partition}                     # Partition name
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

# Create a file to store all slurm script paths for this test
job_list_file = f"generated_{test_name}.txt"
with open(job_list_file, 'w') as f:
    for job_id in range(n_jobs):
        slurm_file = f"job_files/slurm_{test_name}_{job_id}.sh"
        f.write(f"{slurm_file}\n")