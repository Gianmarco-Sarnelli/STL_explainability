import torch
import multiprocessing as mp
from itertools import product
import numpy as np
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure, Easy_BaseMeasure, Brownian, Gaussian, SemiBrownian
from phis_generator import StlGenerator
import math
from typing import List, Any, Tuple
import time
import psutil
import os
import sys
import json
import sqlite3
import re
import pickle

# Removing the warnings when we use pickle
import warnings
warnings.filterwarnings("ignore")


"""
Evaluates the transformation of kernels when three parameters are varied:
1) The number of trajectories used
2) The number of formulae used (in particoular the number of formulae minus the number of trajectories)
3) The standard deviation of the distribution around the base trajectory

The euclidean and cosine distance between two tipe of kernels is measured:
1) Local kernels obtained by sampling around a local trajectory
2) Kernels transformed using importance sampling (starting from global distribution)

The result is an array (Distances) that contains the distances between the two kernels when the
parameters are changed.

Also computes the norms of the three kernels:
1) Norm of local kernel
2) Norm of global kernel
3) Norm of the importance sampling kernel

This results into an array (Norms) that contains these norms computed while varying the previous parameters.

This script is optimized to run on multiprocess

This script allows to fix the same formulae and trajectories for the testing, and also allows to precompute PHI
"""

def Work_on_process(params, test_name):
    """
    Process a single parameter combination and return the results

    Parameters
    ----------
    params = (n_psi_added, n_traj, local_std, n_traj_points): tuple of parameters used by the iteration
    
    Returns
    ---------
    dn_psi_added, n_traj, local_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem
    """
    # Timing each process
    start_time = time.time()

    # Begin work on process
    print("Begin work on process")

    ## Parameters ##
    # Process ID
    process = psutil.Process(os.getpid())
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
    n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id, weight_strategy = params
    n_psi = n_traj + n_psi_added 

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
            global_distr = BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
        case "E":
            global_distr = Easy_BaseMeasure(sigma0=global_std, sigma1=global_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
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

    # Initializing the base trajectory distribution and sampling base_xi
    # (the local distribution will be created around the base trajectory)
    # (for now the base measure will be the same as the global measure)
    base_xi = global_distr.sample(samples=1,
                                varn=n_vars, 
                                points=n_traj_points)
    
    # Initializing the local trajectory distribution and sampling local_xi
    match local_name:
        case "M":
            local_distr = BaseMeasure(base_traj=base_xi[0], sigma0=local_std, sigma1=local_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
        case "E":
            local_distr = Easy_BaseMeasure(base_traj=base_xi[0], sigma0=local_std, sigma1=local_std*totvar_mult*n_traj_points, q=sign_ch/n_traj_points, device=device)
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

    # Sampling the formulae (phi and psi) for the kernels
    # Each formula phi will have its kernel representation
    formulae_distr = StlGenerator(leaf_prob=leaf_probability, 
                            time_bound_max_range=time_bound_max_range,
                            unbound_prob=prob_unbound_time_operator, 
                            threshold_sd=atom_threshold_sd)
    phi_bag = formulae_distr.bag_sample(1, n_vars)
    psi_bag = formulae_distr.bag_sample(n_psi, n_vars)
    
    ## K_loc ##
    # Computing the robustness of each psi over the local_xi
    rhos_psi_local = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_local[i, :] = torch.tanh(formula.quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over local_xi
    rhos_phi_local = torch.tanh(phi_bag[0].quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the local kernels
    K_loc = torch.tensordot(rhos_psi_local, rhos_phi_local, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 
    # Deleting used tensors
    del rhos_psi_local

    ## K_glob ##
    # Computing the robustness of each psi over the global_xi
    rhos_psi_global = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_global[i, :] = torch.tanh(formula.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over the global_xi
    rhos_phi_global = torch.tanh(phi_bag[0].quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the global kernel
    K_glob = torch.tensordot(rhos_psi_global, rhos_phi_global, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_psi, 
                                n_traj = n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                target_distr = local_distr,
                                proposal_distr = global_distr,
                                weight_strategy=weight_strategy)
    # Computing the matrix Q that converts to a local kernel around the base_xi
    if converter.compute_Q(proposal_traj = global_xi, PHI = rhos_psi_global):
        # returns if there are problems with the pseudoinverse 
        return weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # Computing the importance sampling kernel starting from the global one
    K_imp = converter.convert_to_local(K_glob).type(torch.float32)
    
    
    # Computing the peak memory used
    Process_mem = process.memory_info().rss / 1024 / 1024
    # Saving the goodness metric of the pseudo inverse
    Pinv_error = converter.pinv_error
    # Saving other metrics of the local matrix transformation
    Sum_weights = converter.sum_weights
    Sum_squared_weights = converter.sum_squared_weights
    # Deleting used tensors
    converter.__dict__.clear()  # Removes all instance attributes
    del rhos_psi_global, converter

    #Testing the norms of the kernels
    #print(f"n_psi_added: {n_psi_added}, n_traj = {n_traj}, local_std = {local_std}")
    Norm_glob = torch.norm(K_glob).item()
    #print(f"Testing the norm of K_glob: {Norm_glob}")
    Norm_loc = torch.norm(K_loc).item()
    #print(f"Testing the norm of K_loc: {Norm_loc}")
    Norm_imp = torch.norm(K_imp).item()
    #print(f"Testing the norm of K_imp: {Norm_imp}")

    #Computing the matrix Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_loc - K_imp).item()
    Cos_dist = 1 - torch.dot(K_loc/Norm_loc, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_global - rhos_phi_local).item()/math.sqrt(n_traj)
    #print(f"The distance is: {Dist}")
    #print(f"The cosine distance is : {Cos_Dist}")
    #print(f"The robustness distance is : {Dist_rho}")
    #print(f"The pseudoinverse error is : {pinv_error}")

    # Deleting used tensors
    del K_glob, K_loc, K_imp, rhos_phi_global, rhos_phi_local

    # End timing
    Elapsed_time = time.time() - start_time
    
    #return distances_result, norms_result, pinv_result, total_time, process_mem
    return weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem
    



def Work_on_process_precomp(params, test_name):
    """
    Process a single parameter combination and return the results

    Parameters
    ----------
    params = (n_psi_added, n_traj, local_std, n_traj_points): tuple of parameters used by the iteration
    
    Returns
    ---------
    dn_psi_added, n_traj, local_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem
    """
    # Timing each process
    start_time = time.time()

    # Begin work on process
    print("Begin work on process")

    ## Parameters ##
    # Process ID
    process = psutil.Process(os.getpid())
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
    n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id, weight_strategy = params
    n_psi = n_traj + n_psi_added 

    # Checking if the test name is in the format "global_name2local_name"
    global_name, local_name = "M", "B"
    if "2" in test_name:
        index = test_name.index("2")
        global_name = test_name[index - 1]
        local_name = test_name[index + 1]

        
    # Loading the saved tensors
    global_xi_dict = torch.load(os.path.join("Global_xi_dir", f"{test_name}.pt"))
    global_xi = global_xi_dict[(n_traj_points, global_std)][:n_traj, :, :] # Selecting only the first n_traj elements
    del global_xi_dict

    base_xi_dict = torch.load(os.path.join("Base_xi_dir",f"{test_name}.pt"))
    base_xi = base_xi_dict[(n_traj_points, base_xi_id)][:n_traj, :, :] # Selecting only the first n_traj elements
    del base_xi_dict
    
    local_xi_dict = torch.load(os.path.join("Local_xi_dir",f"{test_name}.pt"))
    local_xi = local_xi_dict[(n_traj_points, local_std, base_xi_id)][:n_traj, :, :] # Selecting only the first n_traj elements
    del local_xi_dict

    dweights_dict = torch.load(os.path.join("Dweights_dir", f"{test_name}.pt"))
    dweights = dweights_dict[(weight_strategy, n_traj_points, global_std, local_std, base_xi_id)][:n_traj] # Selecting only the first n_traj elements
    del dweights_dict

    # Loading the saved formulae
    with open(os.path.join("phis_dir", f"{test_name}.pkl"), 'rb') as f:
        phi_bag_dict = pickle.load(f)
    phi_bag = phi_bag_dict[phi_id]
    del phi_bag_dict
    with open(os.path.join("psis_dir", f"{test_name}.pkl"), 'rb') as f:
        psi_bag_tot = pickle.load(f)
        psi_bag = psi_bag_tot[:n_psi]

    
    ## K_loc ##
    # Computing the robustness of each psi over the local_xi
    rhos_psi_local = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_local[i, :] = torch.tanh(formula.quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over local_xi
    rhos_phi_local = torch.tanh(phi_bag[0].quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the local kernels
    K_loc = torch.tensordot(rhos_psi_local, rhos_phi_local, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 
    # Deleting used tensors
    del rhos_psi_local

    ## K_glob ##
    # Loading PHI, the robustness of each psi over the global_xi#NOPE
    # Computing the robustness of each psi over the global_xi
    rhos_psi_global = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_global[i, :] = torch.tanh(formula.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over the global_xi
    rhos_phi_global = torch.tanh(phi_bag[0].quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the global kernel
    K_glob = torch.tensordot(rhos_psi_global, rhos_phi_global, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_psi, 
                                n_traj = n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                )
    # Computing the matrix Q that converts to a local kernel around the base_xi
    if converter.compute_Q(proposal_traj = global_xi, PHI = rhos_psi_global, dweights=dweights):
        # returns if there are problems with the pseudoinverse 
        return weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # Computing the importance sampling kernel starting from the global one
    K_imp = converter.convert_to_local(K_glob).type(torch.float32)
    
    
    # Computing the peak memory used
    Process_mem = process.memory_info().rss / 1024 / 1024
    # Saving the goodness metric of the pseudo inverse
    Pinv_error = converter.pinv_error
    # Saving other metrics of the local matrix transformation
    Sum_weights = converter.sum_weights
    Sum_squared_weights = converter.sum_squared_weights
    # Deleting used tensors
    converter.__dict__.clear()  # Removes all instance attributes
    del rhos_psi_global, converter

    #Testing the norms of the kernels
    #print(f"n_psi_added: {n_psi_added}, n_traj = {n_traj}, local_std = {local_std}")
    Norm_glob = torch.norm(K_glob).item()
    #print(f"Testing the norm of K_glob: {Norm_glob}")
    Norm_loc = torch.norm(K_loc).item()
    #print(f"Testing the norm of K_loc: {Norm_loc}")
    Norm_imp = torch.norm(K_imp).item()
    #print(f"Testing the norm of K_imp: {Norm_imp}")

    #Computing the matrix Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_loc - K_imp).item()
    Cos_dist = 1 - torch.dot(K_loc/Norm_loc, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_global - rhos_phi_local).item()/math.sqrt(n_traj)
    #print(f"The distance is: {Dist}")
    #print(f"The cosine distance is : {Cos_Dist}")
    #print(f"The robustness distance is : {Dist_rho}")
    #print(f"The pseudoinverse error is : {pinv_error}")

    # Deleting used tensors
    del K_glob, K_loc, K_imp, rhos_phi_global, rhos_phi_local

    # End timing
    Elapsed_time = time.time() - start_time
    
    #return distances_result, norms_result, pinv_result, total_time, process_mem
    return weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem


if __name__ == "__main__":
    # Get the parameter file and test name from command line argument
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
        save_all = sys.argv[3]
        # BaseMeasure = M, Easy_BaseMeasure = E, Brownian = B, Gaussian = G, SemiBrownian = S.
    except IndexError:
        raise ValueError("No test name or params file provided. Usage: python3 Test_distance_slurm.py <params_file> <test_name> <save_all>")
        # Exemple: python3 Test_distance_slurm.py params_basic.json TESTtestTEST

    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Some prints
    print(f"parameter combinations: {parameter_combinations}")
    print(f"test name: {test_name}")
        
    # Creating the inputs to be passed to the process
    inputs = [(param, test_name) for param in parameter_combinations]

    # Process work in parallel
    num_processes = mp.cpu_count() - 1 # Use all available CPUs except one
    #num_processes = 1  # Apply this for a single process at a time

    # working on the base algorithm or on the precomputed one:
    if save_all == "yes":
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(Work_on_process_precomp, inputs)
            print(results)
    else:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(Work_on_process, inputs)
            print(results)

    # Store results in database
    for result in results:
        
        db_path = f"Databases/database_{test_name}.db"
        if not os.path.exists(db_path):
            print(f"Database {db_path} not found")
            exit()
        weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem = result
        
        # Computing n_e
        try:
            n_e = (Sum_weights**2)/Sum_squared_weights
        except:
            n_e = None

        with sqlite3.connect(db_path, timeout=60.0) as conn:  # Increased timeout for concurrent access
            c = conn.cursor()
            # Saving the values in the database
            c.execute('''INSERT OR REPLACE INTO results 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (weight_strategy, n_psi_added, n_traj, local_std, global_std, n_traj_points, phi_id, base_xi_id,
                    Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp,
                    Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e))
            
            conn.commit()
    
    #Extract the informations in the file name
    pattern = r"job_files/params_(.+?)_(\d+)(?:_done)?\.json"
    match = re.match(pattern, params_file)
    if match:
        test_name_file, job_id = match.groups()
    else:
        print(f"Match not valid")

    print(f"Completed job {job_id}")

    # Create the new filename by inserting '_done' before the extension
    base = params_file.rsplit('.', 1)[0]  # Get everything before the .json
    new_filename = f"{base}_done.json"
    # Rename the file
    os.rename(params_file, new_filename)

