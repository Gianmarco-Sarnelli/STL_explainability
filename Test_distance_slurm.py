import torch
import multiprocessing as mp
from itertools import product
import numpy as np
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure, LocalBrownian
from phis_generator import StlGenerator
import math
from typing import List, Any, Tuple
import time
import psutil
import os
import sys
import json
import sqlite3


"""
Evaluates the transformation of kernels when three parameters are varied:
1) The number of trajectories used
2) The number of formulae used (in particoular the number of formulae minus the number of trajectories)
3) The standard deviation of the distribution around the local trajectory

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
"""

def Work_on_process(params: Tuple[int, int, float, int]) -> Tuple[Any, Any, Any, float, float]:
    """
    Process a single parameter combination and return the results

    Parameters
    ----------
    params = (n_psi_added, n_traj, local_std, n_traj_points): tuple of parameters used by the iteration

    Returns
    ---------
    distances_result, norms_result, pinv_result, total_time, process_mem

    """
    # Timing each process
    start_time = time.time()

    ## Parameters ##
    # Process ID
    process = psutil.Process(os.getpid())
    # Device used
    device: torch.device = torch.device("cpu")  # Force CPU usage
    # Evaluation of formulae
    evaluate_at_all_times = False # TODO: implement for time evaluation
    # Parameters for the sampling of the base trajectories (using mu0 distribution)
    base_initial_std = 1.0
    base_total_var_std = 1.0
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
    n_psi_added, n_traj, local_std, n_traj_points = params
    n_psi = n_traj + n_psi_added 
    local_n_traj = n_traj
    global_n_traj = n_traj

    # Initializing the base trajectory distribution and sampling base_xi
    # (the local distribution will be created around the base trajectory)
    base_distr = BaseMeasure(device=device, 
                                sigma0=base_initial_std, 
                                sigma1=base_total_var_std)
    base_xi = base_distr.sample(samples=1,
                                varn=n_vars, 
                                points=n_traj_points)

    # Initializing the global trajectory distribution and sampling global_xi
    global_distr = BaseMeasure(device=device, 
                                    sigma0=base_initial_std, 
                                    sigma1=base_total_var_std)
    global_xi = global_distr.sample(samples=global_n_traj, 
                                varn=n_vars, 
                                points=n_traj_points)
    
    # Initializing the local trajectory distribution and sampling local_xi
    local_distr = LocalBrownian(base_traj=base_xi[0], 
                                std=local_std, 
                                device=device)
    local_xi = local_distr.sample(samples=local_n_traj, 
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
    rhos_psi_local = torch.empty(n_psi, local_n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_local[i, :] = torch.tanh(formula.quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over local_xi
    rhos_phi_local = torch.tanh(phi_bag[0].quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the local kernels
    K_loc = torch.tensordot(rhos_psi_local, rhos_phi_local, dims=([1],[0]) ) / (local_n_traj * math.sqrt(n_psi)) 
    # Deleting used tensors
    del rhos_psi_local

    ## K_glob ##
    # Computing the robustness of each psi over the global_xi
    rhos_psi_global = torch.empty(n_psi, global_n_traj)
    for (i, formula) in enumerate(psi_bag):
        rhos_psi_global[i, :] = torch.tanh(formula.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over the global_xi
    rhos_phi_global = torch.tanh(phi_bag[0].quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the global kernel
    K_glob = torch.tensordot(rhos_psi_global, rhos_phi_global, dims=([1],[0]) ) / (global_n_traj * math.sqrt(n_psi)) 

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_psi, 
                                n_traj = global_n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                target_distr = local_distr,
                                proposal_distr = global_distr)
    # Computing the matrix Q that converts to a local kernel around the base_xi
    if converter.compute_Q(proposal_traj = global_xi, PHI = rhos_psi_global):
        # returns if there are problems with the pseudoinverse 
        return n_psi_added, n_traj, local_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan        
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
    Dist_rho = torch.norm(rhos_phi_global - rhos_phi_local).item()/math.sqrt(global_n_traj)
    #print(f"The distance is: {Dist}")
    #print(f"The cosine distance is : {Cos_Dist}")
    #print(f"The robustness distance is : {Dist_rho}")
    #print(f"The pseudoinverse error is : {pinv_error}")

    # Deleting used tensors
    del K_glob, K_loc, K_imp, rhos_phi_global, rhos_phi_local

    # End timing
    Elapsed_time = time.time() - start_time
    
    #return distances_result, norms_result, pinv_result, total_time, process_mem
    return n_psi_added, n_traj, local_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem




if __name__ == "__main__":
    # Get the parameter file and test name from command line argument
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
    except IndexError:
        raise ValueError("No test name or params file provided. Usage: python3 Test_distance_slurm.py <params_file> <test_name>")

    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Process work in parallel
    #num_processes = mp.cpu_count() - 1 # Use all available CPUs except one
    num_processes = 1  # Apply this for a single process at a time
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(Work_on_process, parameter_combinations)

    # Store results in database
    for result in results:
        
        db_path = f"Databases/database_{test_name}.db"
        
        n_psi_added, n_traj, local_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem = results
        
        with sqlite3.connect(db_path, timeout=60.0) as conn:  # Increased timeout for concurrent access
            c = conn.cursor()
            # Saving the values in the database
            c.execute('''INSERT OR REPLACE INTO results 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (n_psi_added, n_traj, local_std, n_traj_points,
                    Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp,
                    Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem))
            
            conn.commit()

    


"""
if __name__ == "__main__":

    # Get the parameter file and test name from command line argument
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
    except IndexError:
        raise ValueError("No test name or params file provided. Usage: python3 Test_distance_slurm.py <params_file> <test_name>")
    
    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Reading the actual parameters from the txt file:
    with open(f'Params/Params_{test_name}.txt', 'r') as file:
        # Read each line and split by comma to convert back to list
        list_n_psi_added = [int(x) for x in file.readline().strip().split(',')]
        list_n_traj = [int(x) for x in file.readline().strip().split(',')]
        list_stds = [float(x) for x in file.readline().strip().split(',')]
        list_n_traj_points = [int(x) for x in file.readline().strip().split(',')]
    
    #num_processes = mp.cpu_count() - 1 # Use all available CPUs except one
    num_processes = 1 # Apply this for a single process at a time
    
    # Process work in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(Work_on_process, parameter_combinations)

    # Store results in arrays and saving the elapsed time
    for result, params in zip(results, parameter_combinations):
        distances_result, norms_result, pinv_result, elapsed_time, process_mem = result
        n_psi_added, n_traj, local_std, n_traj_points = params
        
        # Find indices for storing results
        idx1 = list_n_psi_added.index(n_psi_added)
        idx2 = list_n_traj.index(n_traj)
        idx3 = list_stds.index(local_std)
        idx4 = list_n_traj_points.index(n_traj_points)

        # Loading the arrays. implement in parallel 
        Distances = np.load(f'Distances/Distances_{test_name}.npy', allow_pickle=True)
        Norms = np.load(f'Norms/Norms_{test_name}.npy', allow_pickle=True)
        Pinv_error = np.load(f'Pinv_error/Pinv_error_{test_name}.npy', allow_pickle=True)

        # Filling the result arrays
        Distances[idx1, idx2, idx3, idx4] = distances_result
        Norms[idx1, idx2, idx3, idx4] = norms_result
        Pinv_error[idx1, idx2, idx3, idx4] = pinv_result

        # Saving the arrays. implement in parallel
        np.save(f'Distances/Distances_{test_name}.npy', Distances)
        np.save(f'Norms/Norms_{test_name}.npy', Norms)
        np.save(f'Pinv_error/Pinv_error_{test_name}.npy', Pinv_error)

        # Saving the csv. implement in parallel
        with open(f'Resources/Resources_{test_name}.csv', 'a') as file:
            # The components are: n_psi, n_traj, n_traj_points, total_time, process_mem
            file.write(f"{n_psi_added+n_traj},{n_traj},{n_traj_points},{elapsed_time},{process_mem}\n")
"""


