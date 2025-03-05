import torch
import multiprocessing as mp
from itertools import product
import numpy as np
import math
import time
import psutil
import os
import sys
import json
import sqlite3
import re
import pickle
from model_to_formula import search_from_kernel, quantitative_model
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure, Easy_BaseMeasure, Brownian, Gaussian, SemiBrownian

# Removing the warnings when we use pickle
import warnings
warnings.filterwarnings("ignore")

"""
Evaluates the transformation of kernels when using a model's robustness function:
1) The local kernel is obtained by sampling from a local distribution
2) The global kernel is obtained by sampling from a global distribution
3) The importance sampling kernel is created by transforming the global kernel

The script then measures the Euclidean and cosine distance between the local and 
importance sampling kernels. It also uses FAISS to find similar formulae to both kernels.
"""

def Work_on_process(params, test_name, model_path):
    """
    Process a single parameter combination and return the results
    
    Parameters
    ----------
    params = (n_psi, n_traj, local_std, global_std, n_traj_points, weight_strategy): tuple of parameters
    test_name: string indicating the distributions to use (format "global_name2local_name")
    model_path: path to the saved model state dict
    
    Returns
    ---------
    (various measurements including distances, norms, and timing information)
    """
    # Timing each process
    start_time = time.time()

    # Begin work on process
    print("Begin work on process")

    ## Parameters ##
    # Process ID
    process = psutil.Process(os.getpid())
    # Device used
    device = torch.device("cpu")  # Force CPU usage
    # Evaluation of formulae
    evaluate_at_all_times = False 
    n_vars = 2  # Number of variables in trajectories

    # Parameters of the process
    n_psi, n_traj, local_std, global_std, n_traj_points, weight_strategy = params

    # Checking if the test name is in the format "global_name2local_name"
    global_name, local_name = "M", "B"  # Default values
    if "2" in test_name:
        index = test_name.index("2")
        global_name = test_name[index - 1]
        local_name = test_name[index + 1]

    # Initializing the global trajectory distribution and sampling global_xi
    # BaseMeasure = M, Easy_BaseMeasure = E, Brownian = B, Gaussian = G, SemiBrownian = S.
    match global_name:
        case "M":
            global_distr = BaseMeasure(sigma0=global_std, device=device)
        case "E":
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

    # Initializing the base trajectory and local distribution
    base_xi = global_distr.sample(samples=1,
                                 varn=n_vars, 
                                 points=n_traj_points)
    
    # Initializing the local trajectory distribution and sampling local_xi
    match local_name:
        case "M":
            local_distr = BaseMeasure(base_traj=base_xi[0], sigma0=local_std, device=device)
        case "E":
            local_distr = BaseMeasure(base_traj=base_xi[0], sigma0=local_std, device=device)
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

    # Initialize model for robustness computation
    model = quantitative_model(model_path=model_path, nvars=n_vars)
    
    # Generate random trajectories as psi features
    psi_xi = global_distr.sample(samples=n_psi, 
                                varn=n_vars, 
                                points=n_traj_points)
    
    ## K_loc ##
    # Computing the robustness of each psi over the local_xi
    rhos_psi_local = torch.empty(n_psi, n_traj)
    for i in range(n_psi):
        rhos_psi_local[i, :] = model.robustness(local_xi)
    
    # Computing the robustness of model over local_xi as phi
    rhos_phi_local = model.robustness(local_xi)
    
    # Computing the local kernels
    K_loc = torch.tensordot(rhos_psi_local, rhos_phi_local, dims=([1],[0])) / (n_traj * math.sqrt(n_psi))
    
    # Deleting used tensors
    del rhos_psi_local

    ## K_glob ##
    # Computing the robustness of each psi over the global_xi
    rhos_psi_global = torch.empty(n_psi, n_traj)
    for i in range(n_psi):
        rhos_psi_global[i, :] = model.robustness(global_xi)
    
    # Computing the robustness of model over global_xi as phi
    rhos_phi_global = model.robustness(global_xi)
    
    # Computing the global kernel
    K_glob = torch.tensordot(rhos_psi_global, rhos_phi_global, dims=([1],[0])) / (n_traj * math.sqrt(n_psi))

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars=n_vars, 
                             n_formulae=n_psi, 
                             n_traj=n_traj, 
                             n_traj_points=n_traj_points, 
                             evaluate_at_all_times=evaluate_at_all_times,
                             target_distr=local_distr,
                             proposal_distr=global_distr,
                             weight_strategy=weight_strategy)
    
    # Computing the matrix Q that converts to a local kernel around the base_xi
    if converter.compute_Q(proposal_traj=global_xi, PHI=rhos_psi_global):
        # returns if there are problems with the pseudoinverse
        return weight_strategy, n_psi, n_traj, local_std, global_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    
    # Computing the importance sampling kernel starting from the global one
    K_imp = converter.convert_to_local(K_glob).type(torch.float32)
    
    # Computing peak memory used
    Process_mem = process.memory_info().rss / 1024 / 1024
    
    # Saving the goodness metric of the pseudo inverse
    Pinv_error = converter.pinv_error
    
    # Saving other metrics of the local matrix transformation
    Sum_weights = converter.sum_weights
    Sum_squared_weights = converter.sum_squared_weights
    
    # Deleting used tensors
    converter.__dict__.clear()  # Removes all instance attributes
    del rhos_psi_global, converter

    # Testing the norms of the kernels
    Norm_glob = torch.norm(K_glob).item()
    Norm_loc = torch.norm(K_loc).item()
    Norm_imp = torch.norm(K_imp).item()

    # Computing the matrix Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_loc - K_imp).item()
    Cos_dist = 1 - torch.dot(K_loc/Norm_loc, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_global - rhos_phi_local).item()/math.sqrt(n_traj)

    # Using FAISS to retrieve formulae
    if n_psi >= 1000:  # Only use FAISS if we have enough features
        # Rescaling the kernels for the search
        K_loc_scaled = K_loc * n_traj * math.sqrt(n_psi)
        K_imp_scaled = K_imp * n_traj * math.sqrt(n_psi)
        
        # Stack multiple kernels together
        kernels = torch.stack([K_loc_scaled, K_imp_scaled], dim=0)
        
        # k is the number of closest formulae to retrieve
        k = 5
        
        # Search for closest formulae to both kernels at once
        formulae_lists, distances = search_from_kernel(
            kernels=kernels,
            nvar=n_vars,
            k=k,
            n_neigh=64
        )
        
        # Access results for each kernel
        loc_formulae = formulae_lists[0]
        imp_formulae = formulae_lists[1]
        
        loc_dists = distances[0]
        imp_dists = distances[1]
        
        # Computing the overlap between the formulae retrieved
        common_formulae = set([str(f) for f in loc_formulae]).intersection([str(f) for f in imp_formulae])
        overlap_form = len(common_formulae) / k
    else:
        overlap_form = math.nan

    # Deleting used tensors
    del K_glob, K_loc, K_imp, rhos_phi_global, rhos_phi_local

    # End timing
    Elapsed_time = time.time() - start_time
    
    return weight_strategy, n_psi, n_traj, local_std, global_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, overlap_form


def setup_database(db_path):
    """Create the database and tables if they don't exist"""
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS model_results
                    (weight_strategy TEXT,
                     n_psi INTEGER,
                     n_traj INTEGER,
                     local_std REAL,
                     global_std REAL,
                     n_traj_points INTEGER,
                     Dist REAL,
                     Cos_dist REAL,
                     Dist_rho REAL,
                     Norm_glob REAL,
                     Norm_loc REAL,
                     Norm_imp REAL,
                     Pinv_error REAL,
                     Sum_weights REAL,
                     Sum_squared_weights REAL,
                     Elapsed_time REAL,
                     Process_mem REAL,
                     overlap_form REAL,
                     n_e REAL,
                     PRIMARY KEY (weight_strategy, n_psi, n_traj, local_std, global_std, n_traj_points))''')
        conn.commit()


if __name__ == "__main__":
    # Get the parameters from command line arguments
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
        model_path = sys.argv[3]  # Path to the model state dict
    except IndexError:
        raise ValueError("Missing arguments. Usage: python3 Test_model.py <params_file> <test_name> <model_path>")

    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Create database directory if it doesn't exist
    os.makedirs("Databases", exist_ok=True)
    
    # Setup database
    db_path = f"Databases/database_model_{test_name}.db"
    setup_database(db_path)

    # Creating the inputs to be passed to the process
    inputs = [(param, test_name, model_path) for param in parameter_combinations]

    # Process work in parallel
    num_processes = mp.cpu_count() - 1  # Use all available CPUs except one
    # num_processes = 1  # Uncomment for single process debugging

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(Work_on_process, inputs)

    # Store results in database
    for result in results:
        weight_strategy, n_psi, n_traj, local_std, global_std, n_traj_points, Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, overlap_form = result
        
        print(f"weight_strategy = {weight_strategy}, n_psi = {n_psi}, n_traj = {n_traj}, local_std = {local_std}, global_std = {global_std}, n_traj_points = {n_traj_points}, Dist = {Dist}, Cos_dist = {Cos_dist}, Dist_rho = {Dist_rho}, Norm_glob = {Norm_glob}, Norm_loc = {Norm_loc}, Norm_imp = {Norm_imp}, Pinv_error = {Pinv_error}, Sum_weights = {Sum_weights}, Sum_squared_weights = {Sum_squared_weights}, Elapsed_time = {Elapsed_time}, Process_mem = {Process_mem}, overlap_form = {overlap_form}")

        # Computing n_e (effective sample size)
        try:
            n_e = (Sum_weights**2)/Sum_squared_weights
        except:
            n_e = None

        with sqlite3.connect(db_path, timeout=60.0) as conn:
            c = conn.cursor()
            # Save values in database
            c.execute('''INSERT OR REPLACE INTO model_results 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (weight_strategy, n_psi, n_traj, local_std, global_std, n_traj_points,
                       Dist, Cos_dist, Dist_rho, Norm_glob, Norm_loc, Norm_imp,
                       Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, 
                       overlap_form, n_e))
            conn.commit()

    # Extract information from the file name
    pattern = r"job_files/params_(.+?)_(\d+)(?:_done)?\.json"
    match = re.match(pattern, params_file)
    if match:
        test_name_file, job_id = match.groups()
        print(f"Completed job {job_id}")
    else:
        print("Match not valid")

    # Rename the file to indicate completion
    base = params_file.rsplit('.', 1)[0]  # Get everything before the .json
    new_filename = f"{base}_done.json"
    os.rename(params_file, new_filename)