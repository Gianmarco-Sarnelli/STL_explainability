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
from model_to_formula import search_from_kernel

# Removing the warnings when we use pickle
import warnings
warnings.filterwarnings("ignore")


#NOTE: This script transforms kernel sampled from a proposal distribution into kernels that approximate the ones from a target distribution.
# Any possible naming mistakes can be from the previous version where we transformed global to local

"""
Evaluates the transformation of kernels when three parameters are varied:
1) The number of trajectories used
2) The number of formulae used (in particoular the number of formulae minus the number of trajectories)
3) The standard deviation of the distributions

The euclidean and cosine distance between two tipe of kernels is measured:
1) Target kernels obtained by sampling from a target distribution
2) Kernels transformed using importance sampling (starting from a proposal distribution)

The result is an array (Distances) that contains the distances between the two kernels when the
parameters are changed.

Also computes the norms of the three kernels:
1) Norm of target kernel
2) Norm of proposal kernel
3) Norm of the importance sampling kernel

This results into an array (Norms) that contains these norms computed while varying the previous parameters.

This script is optimized to run on multiprocess

This script allows to fix the same formulae and trajectories for the testing, and also allows to precompute PHI
"""


def Work_on_process_precomp(params, test_name):
    """
    Process a single parameter combination and returns the results

    Parameters
    ----------
    params = (n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, weight_strategy): tuple of parameters used by the iteration
    
    Returns
    ---------
    weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem
    """
    # Timing each process
    start_time = time.time()

    ## Parameters ##

    # Process ID
    process = psutil.Process(os.getpid())
    # Device used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Evaluation of formulae
    evaluate_at_all_times = False
    n_vars = 2
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
    n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, weight_strategy = params
    n_psi = n_traj + n_psi_added 
    n_phi = 1 #how many formulae are computed at the same time (for now is one)
    n_traj_embedding = 10000 # Ho many trajectories are used to compute the embeddings in the database

    # Checking if the test name is in the format "proposal_name2target_name"
    proposal_name, target_name = "B", "M"
    if "2" in test_name:
        index = test_name.index("2")
        proposal_name = test_name[index - 1]
        target_name = test_name[index + 1]

    # Loading the saved tensors
    proposal_xi_dict = torch.load(os.path.join("Proposal_xi_dir", f"{test_name}.pt"))
    proposal_xi = proposal_xi_dict[(n_traj_points, proposal_std)][:n_traj, :, :] # Selecting only the first n_traj elements
    del proposal_xi_dict
    
    target_xi_dict = torch.load(os.path.join("Target_xi_dir",f"{test_name}.pt"))
    target_xi = target_xi_dict[(n_traj_points, target_std)][:n_traj, :, :] # Selecting only the first n_traj elements
    del target_xi_dict

    dweights_dict = torch.load(os.path.join("Dweights_dir", f"{test_name}.pt"))
    dweights = dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std)][:n_traj] # Selecting only the first n_traj elements
    del dweights_dict

    true_dweights_dict = torch.load(os.path.join("True_Dweights_dir", f"{test_name}.pt"))
    true_dweights = true_dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std)][:n_traj] # Selecting only the first n_traj elements
    del true_dweights_dict


    # Loading the saved formulae
    with open(os.path.join("phis_dir", f"{test_name}.pkl"), 'rb') as f:
        phi_bag_dict = pickle.load(f)
    phi_bag = phi_bag_dict[phi_id]
    del phi_bag_dict
    with open(os.path.join("psis_dir", f"{test_name}.pkl"), 'rb') as f:
        psi_bag_tot = pickle.load(f)
        psi_bag = psi_bag_tot[:n_psi]

    
    ## K_target ##
    # Computing the robustness of each psi over the target_xi
    rhos_psi_target = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_target[i, :] = torch.tanh(formula.quantitative(target_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over target_xi
    rhos_phi_target = torch.tanh(phi_bag[0].quantitative(target_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the target kernels
    K_target = torch.tensordot(rhos_psi_target, rhos_phi_target, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 
    # Deleting used tensors
    del rhos_psi_target

    ## K_proposal ##
    # Computing the robustness of each psi over the proposal_xi
    rhos_psi_proposal = torch.empty(n_psi, n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_proposal[i, :] = torch.tanh(formula.quantitative(proposal_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over the proposal_xi
    rhos_phi_proposal = torch.tanh(phi_bag[0].quantitative(proposal_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the proposal kernel
    K_proposal = torch.tensordot(rhos_psi_proposal, rhos_phi_proposal, dims=([1],[0]) ) / (n_traj * math.sqrt(n_psi)) 

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_psi, 
                                n_traj = n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                )
    # Computing the matrix Q that converts to a target kernel
    if converter.compute_Q(proposal_traj = proposal_xi, PHI = rhos_psi_proposal, dweights=dweights):
        # returns if there are problems with the pseudoinverse 
        return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # Computing the importance sampling kernel starting from the proposal one
    K_imp = converter.convert_to_local(K_proposal).type(torch.float32)  # TODO: Change the name?
    
    
    # Computing the peak memory used
    Process_mem = process.memory_info().rss / 1024 / 1024
    # Saving the goodness metric of the pseudo inverse
    Pinv_error = converter.pinv_error
    # Saving other metrics that are precomputed 
    Sum_weights = max(torch.sum(true_dweights).item(), torch.finfo(true_dweights.dtype).tiny) # Finding the sum of the weights (clipping it at the minimum float value)
    Sum_squared_weights = torch.sum(torch.square(true_dweights)).item()
    # Deleting used tensors
    converter.__dict__.clear()  # Removes all instance attributes
    del rhos_psi_proposal, converter

    #Testing the norms of the kernels
    Norm_proposal = torch.norm(K_proposal).item()
    Norm_target = torch.norm(K_target).item()
    Norm_imp = torch.norm(K_imp).item()

    #Computing the matrix Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_target - K_imp).item()
    Cos_dist = 1 - torch.dot(K_target/Norm_target, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_proposal - rhos_phi_target).item()/math.sqrt(n_traj)

    # Deleting used tensors
    del K_proposal, K_target, K_imp, rhos_phi_proposal, rhos_phi_target

    # End timing
    Elapsed_time = time.time() - start_time
    
    #return distances_result, norms_result, pinv_result, total_time, process_mem
    return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem



if __name__ == "__main__":
    # Get the parameter file and test name from command line argument
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
        save_all = sys.argv[3]
        # BaseMeasure = M, Easy_BaseMeasure = E, Brownian = B, Gaussian = G, SemiBrownian = S.
    except IndexError:
        raise ValueError("No test name or params file provided. Usage: python3 Test_distance_slurm.py <params_file> <test_name> <save_all>")

    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Creating the inputs to be passed to the process
    inputs = [(param, test_name) for param in parameter_combinations]

    # Process work in parallel
    num_processes = mp.cpu_count() - 1 # Use all available CPUs except one
    #num_processes = 1  # Apply this for a single process at a time

    # working on the base algorithm or on the precomputed one:
    if save_all == "yes":
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(Work_on_process_precomp, inputs)
    else:
        raise RuntimeError("This code requires to save all parameters. Set save_all='true'")

    # Store results in database
    for result in results:
        
        db_path = f"Databases/database_{test_name}.db"
        if not os.path.exists(db_path):
            print(f"Database {db_path} not found")
            exit()
        weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem = result
        
        # Computing n_e
        try:
            n_e = (Sum_weights**2)/Sum_squared_weights
        except:
            n_e = None

        print(f"weight_strategy = {weight_strategy}, n_psi_added = {n_psi_added}, n_traj = {n_traj}, target_std = {target_std}, proposal_std = {proposal_std}, n_traj_points = {n_traj_points}, phi_id = {phi_id}, base_xi_id = {base_xi_id}, Dist = {Dist}, Cos_dist = {Cos_dist}, Dist_rho = {Dist_rho}, Norm_proposal = {Norm_proposal}, Norm_target = {Norm_target}, Norm_imp = {Norm_imp}, Pinv_error = {Pinv_error}, Sum_weights = {Sum_weights}, Sum_squared_weights = {Sum_squared_weights}, Elapsed_time = {Elapsed_time}, Process_mem = {Process_mem}, n_e = {n_e}")

        overlap_form = math.nan
        dist_form = math.nan

        with sqlite3.connect(db_path, timeout=60.0) as conn:  # Increased timeout for concurrent access
            c = conn.cursor()
            # Saving the values in the database
            c.execute('''INSERT OR REPLACE INTO results 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id,
                    Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp,
                    Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e, overlap_form, dist_form))
            
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

