import itertools
import numpy as np
import json
import sys
import sqlite3
import os
from traj_measure import BaseMeasure, Easy_BaseMeasure, Brownian, Gaussian, SemiBrownian, GaussianShift
from phis_generator import StlGenerator
import torch
import math
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
target_std REAL,
proposal_std REAL,
n_traj_points INTEGER,
phi_id INTEGER,
base_xi_id INTEGER,
mu0 REAL,
mu1 REAL,
sigma1 REAL,
q REAL,
q0 REAL,
Dist REAL,
Cos_Dist REAL,
Dist_rho REAl,
Norm_proposal REAl,
Norm_target REAl,
Norm_imp REAl,
Pinv_error REAl,
Sum_weights REAL,
Sum_squared_weights REAL,
Elapsed_time REAL,
Process_mem REAL,
n_e REAL,
overlap_form REAL,
dist_form REAL,
dist_new_kernels REAL,
dist_embed REAL,
model_to_target REAL,
model_to_imp REAL, 
PRIMARY KEY (weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0))''')

        # Optimize database performance

        # Enable WAL mode for better concurrent access
        c.execute('PRAGMA journal_mode=WAL')
        # Optimize for many concurrent readers
        c.execute('PRAGMA cache_size=-2000')  # 2MB cache
        # Batch writes for better performance
        c.execute('PRAGMA synchronous=NORMAL')

        conn.commit()

def get_distribution(name, std, totvar_mult, sign_ch, n_traj_points, device, mu0=0.0, mu1=0.0, sigma1=1.0, q=0.1, q0=0.5):
    match name:
        case "M":
            distr = BaseMeasure(mu0=mu0, sigma0=std, mu1=mu1, sigma1=sigma1, q=q, q0=q0, device=device)
        case "E":
            distr = Easy_BaseMeasure(mu0=mu0, sigma0=std, mu1=mu1, sigma1=sigma1, q=q, q0=q0, device=device)
        case "H": # H = High Variance
            distr = BaseMeasure(sigma0=std, sigma1=std*totvar_mult*math.sqrt(n_traj_points), q=sign_ch/n_traj_points, device=device)
        case "J":
            distr = Easy_BaseMeasure(sigma0=std, sigma1=std*totvar_mult*math.sqrt(n_traj_points), q=sign_ch/n_traj_points, device=device)
        case "B":
            distr = Brownian(std=std, device=device)
        case "T":
            distr = GaussianShift(std=std, device=device)
        case "G":
            distr = Gaussian(std=std, device=device)
        case "S":
            distr = SemiBrownian(std=std, device=device)
        case _:
            raise RuntimeError(f"Distribution name '{name}' is not allowed")
    return distr


def save_params(test_name, list_weight_strategy, list_n_traj_points, list_target_std, list_proposal_std, list_n_traj, 
                list_n_psi_added, list_phi_id, list_base_xi_id, list_mu0, list_mu1, list_sigma1, list_q, list_q0):

    TEST_ON_MODEL = True # This param defines if we need to generate jobs for the script Test_model.py

    # Device used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Evaluation of formulae
    evaluate_at_all_times = False
    n_vars = 3
    # NOTE: n_traj_points must be >= max_timespan in phis_generator
    # NOTE: max_timespan must be greater than 11 (for some reason) #TODO: find why this is the case
    #n_traj_points = 11
    # Parameters for the sampling of the formulae
    leaf_probability = 0.5
    time_bound_max_range = 10
    prob_unbound_time_operator = 0.1
    atom_threshold_sd = 1.0
    #Parameters specific for High_Variance_mu_0
    totvar_mult = 1
    sign_ch = 2

    # Parameters of the process
    n_psi_added = max(list_n_psi_added)  # We will save all the psi needed and then select a subset of them for each test.
    n_traj = max(list_n_traj)
    n_psi = 1000 if TEST_ON_MODEL else n_traj + n_psi_added

    # Checking if the test name is in the format "proposal_name2target_name"
    proposal_name, target_name = "M", "M"
    if "2" in test_name:
        index = test_name.index("2")
        proposal_name = test_name[index - 1]
        target_name = test_name[index + 1]


    ## Proposal_xi, Target_xi and Dweights ##
    proposal_xi_dict = {}
    base_xi_dict = {}
    target_xi_dict = {}
    dweights_dict = {}
    true_dweights_dict = {}
    n_e_dict = {}
    base_std = 1  # We fix this parameter to make the sampling of the base_xi independent of the proposal_std
    ## Iteration on n_traj_points ##
    for n_traj_points in list_n_traj_points:
        ## Iteration on proposal_std ##
        for proposal_std in list_proposal_std: # list_mu0, list_mu1, list_sigma1, list_q, list_q0
            ## Iteration on mu0 ##
            for mu0 in list_mu0:
                ## Iteration on mu1 ##
                for mu1 in list_mu1:
                    ## Iteration on sigma1 ##
                    for sigma1 in list_sigma1:
                        ## Iteration on q ##
                        for q in list_q:
                            ## Iteration on q0 ##
                            for q0 in list_q0:

                                proposal_distr = get_distribution(proposal_name, proposal_std, totvar_mult, sign_ch, n_traj_points, device, mu0=mu0, mu1=mu1, sigma1=sigma1, q=q, q0=q0)
                                proposal_xi = proposal_distr.sample(samples=n_traj, 
                                                                varn=n_vars, 
                                                                points=n_traj_points)
                                proposal_xi_dict[(n_traj_points, proposal_std, mu0, mu1, sigma1, q, q0)] = proposal_xi


        ## Iteration on target_std ##
        for target_std in list_target_std:
            # Inizializing the target_distr

            target_distr = get_distribution(target_name, target_std, totvar_mult, sign_ch, n_traj_points, device)
            target_xi = target_distr.sample(samples=n_traj, 
                                            varn=n_vars, 
                                            points=n_traj_points)
            target_xi_dict[(n_traj_points, target_std)] = target_xi


            ## Iteration on weight_strategy ##
            for weight_strategy in list_weight_strategy:
                
                ## Iteration again on proposal_std ##
                for proposal_std in list_proposal_std: # We need to reiterate on the proposal_std 
                                                        # to obtain the correct proposal_distr and proposal_xi
                    ## Iteration again on mu0 ##
                    for mu0 in list_mu0:
                        ## Iteration again on mu1 ##
                        for mu1 in list_mu1:
                            ## Iteration again on sigma1 ##
                            for sigma1 in list_sigma1:
                                ## Iteration again on q ##
                                for q in list_q:
                                    ## Iteration again on q0 ##
                                    for q0 in list_q0:
                                        # Checking if we are running Test_model.py or not
                                        if TEST_ON_MODEL:
                                            ## Iteration on phi_id ## 
                                            for phi_id in list_phi_id: ## THIS is done so that the weights consider only the first model_n_vars variables of the trajectories
                                                
                                                # Selecting the model
                                                model_list = ["human", "linear", "maritime", "robot2", "robot4", "robot5", "train"]
                                                model_name = model_list[phi_id] if phi_id < 7 else "not_a_model"
                                                if model_name == "maritime":
                                                    n_vars_model = 2
                                                elif model_name in ("robot2", "robot4", "robot5"):
                                                    n_vars_model = 3
                                                elif model_name in ("human", "linear", "train"):
                                                    n_vars_model = 1
                                                else:
                                                    n_vars_model = n_vars
                                                    print(f"The phi_id {phi_id} won't work on a model. Defaulting to a standard test of distance")

                                                proposal_xi = proposal_xi_dict[(n_traj_points, proposal_std, mu0, mu1, sigma1, q, q0)] # Reloading the currect proposal_xi
                                                proposal_distr = get_distribution(proposal_name, proposal_std, totvar_mult, sign_ch, n_traj_points, device, mu0=mu0, mu1=mu1, sigma1=sigma1, q=q, q0=q0)
                                                # Creating the trajectories where we cut some dimensions out for the model
                                                proposal_xi_cut = proposal_xi[:,:n_vars_model,:]

                                                # Computing the Dweights
                                                converter = local_matrix(n_vars = n_vars_model, 
                                                                        n_formulae = n_psi, 
                                                                        n_traj = n_traj, 
                                                                        n_traj_points = n_traj_points,
                                                                        evaluate_at_all_times = evaluate_at_all_times,
                                                                        target_distr = target_distr,
                                                                        proposal_distr = proposal_distr,
                                                                        weight_strategy = weight_strategy,
                                                                        proposal_traj = proposal_xi_cut)
                                                converter.compute_dweights()
                                                dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)] = converter.dweights
                                                true_dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)] = converter.true_dweights
                                                ## Iteration on the actual traj_number to compute n_e ##
                                                for actual_traj_n in list_n_traj:
                                                    # Computing the Dweights AGAIN
                                                    converter = local_matrix(n_vars = n_vars_model, 
                                                                            n_formulae = n_psi, 
                                                                            n_traj = actual_traj_n, 
                                                                            n_traj_points = n_traj_points,
                                                                            evaluate_at_all_times = evaluate_at_all_times,
                                                                            target_distr = target_distr,
                                                                            proposal_distr = proposal_distr,
                                                                            weight_strategy = weight_strategy,
                                                                            proposal_traj = proposal_xi_cut[:actual_traj_n])
                                                    converter.compute_dweights()
                                                    n_e_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id, actual_traj_n)] = converter.n_e


                                        else: # In this case we don't need to change the weights depending on the number of variables of the model

                                            proposal_xi = proposal_xi_dict[(n_traj_points, proposal_std, mu0, mu1, sigma1, q, q0)] # Reloading the currect proposal_xi
                                            proposal_distr = get_distribution(proposal_name, proposal_std, totvar_mult, sign_ch, n_traj_points, device, mu0=mu0, mu1=mu1, sigma1=sigma1, q=q, q0=q0)

                                            # Computing the Dweights
                                            converter = local_matrix(n_vars = n_vars, 
                                                                    n_formulae = n_psi, 
                                                                    n_traj = n_traj, 
                                                                    n_traj_points = n_traj_points,
                                                                    evaluate_at_all_times = evaluate_at_all_times,
                                                                    target_distr = target_distr,
                                                                    proposal_distr = proposal_distr,
                                                                    weight_strategy = weight_strategy,
                                                                    proposal_traj = proposal_xi)
                                            converter.compute_dweights()
                                            dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)] = converter.dweights
                                            true_dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)] = converter.true_dweights
                                            ## Iteration on the actual traj_number to compute n_e ##
                                            for actual_traj_n in list_n_traj:
                                                # Computing the Dweights AGAIN
                                                converter = local_matrix(n_vars = n_vars_model, 
                                                                        n_formulae = n_psi, 
                                                                        n_traj = actual_traj_n, 
                                                                        n_traj_points = n_traj_points,
                                                                        evaluate_at_all_times = evaluate_at_all_times,
                                                                        target_distr = target_distr,
                                                                        proposal_distr = proposal_distr,
                                                                        weight_strategy = weight_strategy,
                                                                        proposal_traj = proposal_xi_cut[:actual_traj_n])
                                                converter.compute_dweights()
                                                n_e_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id, actual_traj_n)] = converter.n_e


    # Saving proposal_xi_dict
    os.makedirs("Proposal_xi_dir", exist_ok=True)
    torch.save(proposal_xi_dict, os.path.join("Proposal_xi_dir",f"{test_name}.pt"))
    # Saving target_xi_dict
    os.makedirs("Target_xi_dir", exist_ok=True)
    torch.save(target_xi_dict, os.path.join("Target_xi_dir",f"{test_name}.pt"))
    # Saving Dweights
    os.makedirs("Dweights_dir", exist_ok=True)
    torch.save(dweights_dict, os.path.join("Dweights_dir",f"{test_name}.pt"))
    # Saving True_Dweights
    os.makedirs("True_Dweights_dir", exist_ok=True)
    torch.save(true_dweights_dict, os.path.join("True_Dweights_dir",f"{test_name}.pt"))
    # Saving n_e
    os.makedirs("n_e_dir", exist_ok=True)
    torch.save(n_e_dict, os.path.join("n_e_dir",f"{test_name}.pt"))


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









### Start of the script ###

# Getting the test name
try:
    test_name = sys.argv[1]
    n_jobs = int(sys.argv[2])
    save_all = sys.argv[3]
    partition = sys.argv[4]
    script = sys.argv[5]
except IndexError:
    raise ValueError("Wrong number of agrs provided. Usage: python3 Generate_jobs.py <test_name> <n_jobs> <save_all> <partition> <script>")

# Initialize database
initialize_database(test_name)

# check for the right partition:
if (partition != "THIN") and (partition != "EPYC") and (partition != "lovelace"):
    raise RuntimeError(f"Unable to use the partition: {partition}")

# Parameters for the test
list_weight_strategy =  ["only_target"]#["self_norm", "only_target", "square_root"] # NOTE: try to assign a single weight strategy to a single job. This will avoid confusion in the results!!
list_n_traj_points = [100]
list_target_std = [1] #[1, 0.6]
list_proposal_std = [1]#[1, 1.2, 1.4, 1.6, 1.8, 2]#[1] #[1, 4]
list_n_traj = [2000, 6000, 10000] #[1000, 10000]#[1000, 4000]
list_n_psi_added = [0] #[-500, 500]
list_phi_id = [0, 1, 2, 5, 6, 10, 11, 12, 13, 14]
list_base_xi_id = [0]#    #NOTE: fix this to a single value
list_mu0 =[0]# [0, 0.2, 0.4, 0.6, 0.8, 1]#[0]#[0, 0.1, 0.2, 0.3, 0.4, 0.5]#[0, 0.1]
list_mu1 = [0]#[0, 0.1]
list_sigma1 = [1]#[1, 1.2, 1.4, 1.6, 1.8, 2]#[1]#[1, 1.1]
list_q = [0.1]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
list_q0 = [0.5]


# If we want to save all variables we need to initialize them
if save_all=="yes":
    ##  Saving the params ##
    save_params(test_name,
                list_weight_strategy,
                list_n_traj_points,
                list_target_std,
                list_proposal_std,
                list_n_traj, 
                list_n_psi_added,
                list_phi_id,
                list_base_xi_id,
                list_mu0, 
                list_mu1, 
                list_sigma1, 
                list_q, 
                list_q0)

else:
    save_all = "no"


# Saving the parameters in the Params file #NOTE: This is not needed anymore, but could be useful
os.makedirs("Params", exist_ok=True)  # Create directory if needed
with open(f"Params/Params_{test_name}.txt", 'w') as file:
    file.write("list_n_psi_added,"+','.join(str(x) for x in list_n_psi_added) + '\n')
    file.write("list_n_traj,"+','.join(str(x) for x in list_n_traj) + '\n')
    file.write("list_target_std,"+','.join(str(x) for x in list_target_std) + '\n')
    file.write("list_n_traj_points,"+','.join(str(x) for x in list_n_traj_points) + '\n')
    file.write("list_phi_id,"+','.join(str(x) for x in list_phi_id) + '\n')
    file.write("list_base_xi_id,"+','.join(str(x) for x in list_base_xi_id) + '\n')
    file.write("list_weight_strategy,"+','.join(x for x in list_weight_strategy) + '\n')
    file.write("list_proposal_std,"+','.join(str(x) for x in list_proposal_std) + '\n')
    file.write("list_mu0,"+','.join(str(x) for x in list_mu0) + '\n')
    file.write("list_mu1,"+','.join(str(x) for x in list_mu1) + '\n')
    file.write("list_sigma1,"+','.join(str(x) for x in list_sigma1) + '\n')
    file.write("list_q,"+','.join(str(x) for x in list_q) + '\n')
    file.write("list_q0,"+','.join(str(x) for x in list_q0))


# Generate all combinations
all_combinations = list(itertools.product(
    list_n_psi_added,
    list_n_traj,
    list_target_std,     # NOTE: in Test_distance.py we use FIRST target_std THEN proposal_std, so alway use this notation
    list_proposal_std,
    list_n_traj_points,
    list_phi_id,
    list_base_xi_id,
    list_weight_strategy,
    list_mu0, 
    list_mu1, 
    list_sigma1, 
    list_q, 
    list_q0)
)

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
    if partition != "lovelace": # This is the case where we run on ORFEO

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
python3 {script} {params_file} {test_name} {save_all}
"""


    else:  # This is the case where we run on DEMETRA

        slurm_script = f"""#!/bin/bash
#SBATCH --partition={partition}                     # Partition name
#SBATCH --account=ai-lab                     # Account name
#SBATCH --ntasks=1                           # Number of tasks (since we're using multiprocessing)
#SBATCH --cpus-per-task=16                   # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=4G                     # Memory per CPU
#SBATCH --output=output_{test_name}_{job_id}.log         # Standard output log
#SBATCH --error=error_{test_name}_{job_id}.log           # Standard error log
#SBATCH --get-user-env                       # Activating the environment

# Run the Python script
python3 {script} {params_file} {test_name} {save_all}
"""


    # Writing job file
    slurm_file = f"job_files/slurm_{test_name}_{job_id}.sh"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)

# Create a file to store all slurm script paths for this test
job_list_file = f"generated_{test_name}.txt"
with open(job_list_file, 'w') as f:
    for job_id in range(n_jobs):
        slurm_file = f"job_files/slurm_{test_name}_{job_id}.sh"
        f.write(f"{slurm_file}\n")