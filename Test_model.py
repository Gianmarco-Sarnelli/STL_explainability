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
from model_to_formula import search_from_kernel, quantitative_model
from IR.phisearch import similarity_based_relevance, search_from_embeddings
from IR.utils import load_pickle

# Removing the warnings when we use pickle
import warnings
warnings.filterwarnings("ignore")


#NOTE: This script is similar to Test_distance.py but additionally uses search_from_kernel 
# to find the closest STL formulae to the target and importance sampling kernels
# and computes overlap and similarity metrics between these sets of formulae.

"""
Evaluates the transformation of kernels when three parameters are varied:
1) The number of trajectories used
2) The number of formulae used (in particoular the number of formulae minus the number of trajectories)
3) The standard deviation of the distributions

In addition to the distances computed in Test_distance.py, this script:
1) Finds the k closest STL formulae to both K_target and K_imp kernels
2) Computes the overlap between these two sets of formulae
3) Computes a distance metric between the formulae using similarity_based_relevance

The results are stored in the same database structure with additional columns for the 
formula overlap and distance metrics.
"""


def Work_on_process_precomp(params, test_name):
    """
    Process a single parameter combination and returns the results

    Parameters
    ----------
    params = (n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, weight_strategy): tuple of parameters used by the iteration
    
    Returns
    ---------
    weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, overlap_form, dist_form
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
    n_train_phis = 1000
    n_traj_embedding = 10000 # Ho many trajectories are used to compute the embeddings in the database
    # Number of closest formulae to retrieve
    k = 5
    # Maximum number of variables in the saved formulae. 
    max_n_vars = 3
    # A formula can work with trajectories with higher dimension but not lower dimension, so n_vars = 3
    n_vars = 3

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

    # Initializing the model
    model_list = ["human", "linear", "maritime", "robot2", "robot4", "robot5", "train"]
    model_name = model_list[phi_id]

    print(f"model name: {model_name}", flush=True)

    if model_name == "maritime":
        n_vars_model = 2
    elif model_name in ("robot2", "robot4", "robot5"):
        n_vars_model = 6
    else:
        n_vars_model = 1
    model_path  = f'IR/data/data/{model_name}/model_state_dict.pth'
    quant_model = quantitative_model(model_path=model_path, nvars=n_vars_model)
    # Creating the trajectories where we cut some dimensions out for the model
    target_xi_cut = target_xi[:,:n_vars_model,:] 
    proposal_xi_cut = proposal_xi[:,:n_vars_model,:]


    # Loading the saved formulae (1000 formulae)
    folder_index = os.path.join("IR", "index")  # Update with actual path
    train_phis = load_pickle(folder_index, 'train_phis_{}_vars.pickle'.format(max_n_vars))
    

    ## K_target ##
    # Computing the robustness of each train_phis over the target_xi
    rhos_psi_target = torch.empty(n_train_phis, n_traj)
    for (i, formula) in enumerate(train_phis):
        rhos_psi_target[i, :] = torch.tanh(formula.quantitative(target_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of the model over target_xi
    rhos_phi_target = torch.tanh(quant_model.robustness(traj=target_xi_cut))
    # Computing the target kernels
    K_target = torch.tensordot(rhos_psi_target, rhos_phi_target, dims=([1],[0]) ) / (n_traj * math.sqrt(n_train_phis)) 
    # Deleting used tensors
    del rhos_psi_target

    ## K_proposal ##
    # Computing the robustness of each train_phis over the proposal_xi
    rhos_psi_proposal = torch.empty(n_train_phis, n_traj)
    for (i, formula) in enumerate(train_phis):
        rhos_psi_proposal[i, :] = torch.tanh(formula.quantitative(proposal_xi, evaluate_at_all_times=evaluate_at_all_times))
    # Computing the robustness of phi over the proposal_xi
    rhos_phi_proposal = torch.tanh(quant_model.robustness(traj=proposal_xi_cut))
    # Computing the proposal kernel
    K_proposal = torch.tensordot(rhos_psi_proposal, rhos_phi_proposal, dims=([1],[0]) ) / (n_traj * math.sqrt(n_train_phis)) 

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_train_phis, 
                                n_traj = n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                )
    # Computing the matrix Q that converts to a target kernel
    if converter.compute_Q(proposal_traj = proposal_xi, PHI = rhos_psi_proposal, dweights=dweights):
        # returns if there are problems with the pseudoinverse 
        return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # Computing the importance sampling kernel starting from the proposal one
    K_imp = converter.convert_to_local(K_proposal).type(torch.float32)
    
    # Computing the peak memory used
    Process_mem = process.memory_info().rss / 1024 / 1024
    # Saving the goodness metric of the pseudo inverse
    Pinv_error = converter.pinv_error
    # Saving other metrics that are precomputed 
    Sum_weights = max(torch.sum(true_dweights).item(), torch.finfo(true_dweights.dtype).tiny) # Finding the sum of the weights (clipping it at the minimum float value)
    Sum_squared_weights = torch.sum(torch.square(true_dweights)).item()
    # Deleting used tensors
    converter.__dict__.clear()  # Removes all instance attributes

    #Testing the norms of the kernels
    Norm_proposal = torch.norm(K_proposal).item()
    Norm_target = torch.norm(K_target).item()
    Norm_imp = torch.norm(K_imp).item()

    #Computing the matrix Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_target - K_imp).item()
    Cos_dist = 1 - torch.dot(K_target/Norm_target, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_proposal - rhos_phi_target).item()/math.sqrt(n_traj)

    #try:

    #Path to the index forlder
    folder_index = os.path.join("IR", "index")  # Update with actual path

    # Rescaling the kernels for the search
    K_target_scaled = K_target * n_traj * math.sqrt(n_train_phis)
    K_imp_scaled = K_imp * n_traj * math.sqrt(n_train_phis)

    start_search_time1 = time.time()
    # Search for closest formulae to each kernel
    target_formulae_list, target_dists = search_from_embeddings(embeddings=K_target_scaled.unsqueeze(0),
                                                                nvar=n_vars_model,
                                                                folder_index=folder_index,
                                                                k=5,
                                                                n_neigh=32,
                                                                n_pc=-1,
                                                                timespan=None,
                                                                nodes=None)
    print(f"target dists: {target_dists}", flush=True)
    total_search_time1 = time.time()-start_search_time1
    print(f"total_search_time1 = {total_search_time1}")
    
    start_search_time2 = time.time()
    imp_formulae_list, imp_dists = search_from_embeddings(embeddings=K_imp_scaled.unsqueeze(0),
                                                                nvar=max_n_vars,
                                                                folder_index=folder_index,
                                                                k=5,
                                                                n_neigh=32,
                                                                n_pc=-1,
                                                                timespan=None,
                                                                nodes=None)
    print(f"imp dists: {imp_dists}", flush=True)
    total_search_time2 = time.time()-start_search_time2
    print(f"total_search_time2 = {total_search_time2}")
    
    # Extract the formulae (first element is the list of formulae for the first/only kernel)
    target_formulae = target_formulae_list[0]
    imp_formulae = imp_formulae_list[0]
    print(f"formulae are there! There are {len(imp_formulae)} imp_formulae and {len(target_formulae)} target_formulae", flush=True)
    print(f"imp_formulae: {imp_formulae}, target_formulae: {target_formulae}")
    
    # Compute overlap between the two sets of formulae
    common_formulae = set([str(f) for f in target_formulae]).intersection([str(f) for f in imp_formulae])
    overlap_form = len(common_formulae) / k
    print(f"overlap_form = {overlap_form}", flush=True)
    





    
    #TODO: Note that the result of the search is a string of the formula, not the formula itself, you must find a way to convert it back!!
    






    # Compute distance between the formulae using similarity_based_relevance
    # Create a BaseMeasure for generating test trajectories (if needed)
    mu0 = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
    test_trajectories = mu0.sample(1000, n_vars_model)  # Sample trajectories for comparison
    
    # Get the first formula from target_formulae to use as reference
    # We could use any formula as reference, but using the top match makes sense
    phi_reference = target_formulae[0]
    
    # Compute similarity between reference formula and all formulae in imp_formulae
    cosine_similarity, sat_diff = similarity_based_relevance(
        phi_reference, 
        imp_formulae,
        n_vars_model, 
        device,
        boolean=True,
        test_trajectories=test_trajectories
    )
    
    # Use the average cosine similarity as a distance metric (1 - avg_similarity)
    dist_form = 1.0 - cosine_similarity.mean().item()
    print(f"dist_form = {dist_form}", flush=True)
    #except Exception as e:
    #    print(f"Error in formula search and comparison: {e}", flush=True)
    #    overlap_form = math.nan
    #    dist_form = math.nan

    # Deleting used tensors
    del K_proposal, K_target, K_imp, rhos_psi_proposal, rhos_phi_proposal, rhos_phi_target

    # End timing
    Elapsed_time = time.time() - start_time
    
    return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, overlap_form, dist_form



if __name__ == "__main__":
    # Get the parameter file and test name from command line argument
    try:
        params_file = sys.argv[1]
        test_name = sys.argv[2]
        save_all = sys.argv[3]
        # BaseMeasure = M, Easy_BaseMeasure = E, Brownian = B, Gaussian = G, SemiBrownian = S.
    except IndexError:
        raise ValueError("No test name or params file provided. Usage: python3 Test_model.py <params_file> <test_name> <save_all>")

    # Load the parameters for this job
    with open(params_file, 'r') as f:
        parameter_combinations = json.load(f)

    # Creating the inputs to be passed to the process
    inputs = [(param, test_name) for param in parameter_combinations[:1]]

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
        weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, overlap_form, dist_form = result
        
        # Computing n_e
        try:
            n_e = (Sum_weights**2)/Sum_squared_weights
        except:
            n_e = None

        print(f"weight_strategy = {weight_strategy}, n_psi_added = {n_psi_added}, n_traj = {n_traj}, target_std = {target_std}, proposal_std = {proposal_std}, n_traj_points = {n_traj_points}, phi_id = {phi_id}, base_xi_id = {base_xi_id}, Dist = {Dist}, Cos_dist = {Cos_dist}, Dist_rho = {Dist_rho}, Norm_proposal = {Norm_proposal}, Norm_target = {Norm_target}, Norm_imp = {Norm_imp}, Pinv_error = {Pinv_error}, Sum_weights = {Sum_weights}, Sum_squared_weights = {Sum_squared_weights}, Elapsed_time = {Elapsed_time}, Process_mem = {Process_mem}, n_e = {n_e}, overlap_form = {overlap_form}, dist_form = {dist_form}, overlap_form = {overlap_form}, dist_form = {dist_form}")

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