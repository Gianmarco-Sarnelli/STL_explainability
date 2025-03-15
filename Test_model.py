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
from model_to_formula import quantitative_model, new_kernel_to_embedding, kernel_to_new_kernel
from IR.phisearch import similarity_based_relevance, search_from_embeddings
from IR.utils import load_pickle, from_string_to_formula

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
    params = (n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, weight_strategy, mu0, mu1, sigma1, q, q0): tuple of parameters used by the iteration
   
    Returns
    ---------
    weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e, overlap_form, dist_form, dist_new_kernels, dist_embed,  model_to_target, model_to_imp
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
    n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, weight_strategy, mu0, mu1, sigma1, q, q0 = params
    print(f"n_psi_added = {n_psi_added}, n_traj = {n_traj}, target_std = {target_std}, proposal_std = {proposal_std}, n_traj_points = {n_traj_points}, phi_id = {phi_id}, base_xi_id = {base_xi_id}, weight_strategy = {weight_strategy}, mu0 = {mu0}, mu1 = {mu1}, sigma1 = {sigma1}, q = {q}, q0 = {q0}")
    n_train_phis = 1000
    n_traj_embedding = 10000 # How many trajectories are used to compute the embeddings in the database
    # Number of closest formulae to retrieve
    k = 10
    # Maximum number of variables in the saved formulae. 
    max_n_vars = 3
    # A formula can work with trajectories with higher dimension but not lower dimension, so n_vars = 3
    n_vars = 3

    # Checking if the test name is in the format "proposal_name2target_name"
    proposal_name, target_name = "M", "M"
    if "2" in test_name:
        index = test_name.index("2")
        proposal_name = test_name[index - 1]
        target_name = test_name[index + 1]

    # Loading the saved tensors
    proposal_xi_dict = torch.load(os.path.join("Proposal_xi_dir", f"{test_name}.pt"))
    proposal_xi = proposal_xi_dict[(n_traj_points, proposal_std, mu0, mu1, sigma1, q, q0)][:n_traj, :, :] # Selecting only the first n_traj elements
    del proposal_xi_dict
    
    target_xi_dict = torch.load(os.path.join("Target_xi_dir",f"{test_name}.pt"))
    target_xi = target_xi_dict[(n_traj_points, target_std)][:n_traj, :, :] # Selecting only the first n_traj elements
    del target_xi_dict

    dweights_dict = torch.load(os.path.join("Dweights_dir", f"{test_name}.pt"))
    dweights = dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)][:n_traj] # Selecting only the first n_traj elements
    del dweights_dict

    true_dweights_dict = torch.load(os.path.join("True_Dweights_dir", f"{test_name}.pt"))
    true_dweights = true_dweights_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id)][:n_traj] # Selecting only the first n_traj elements
    del true_dweights_dict

    n_e_dict = torch.load(os.path.join("n_e_dir", f"{test_name}.pt"))
    n_e = n_e_dict[(weight_strategy, n_traj_points, proposal_std, target_std, mu0, mu1, sigma1, q, q0, phi_id, n_traj)] # NOTE: This n_e is only correctt for the max n_traj !!!!!
    del n_e_dict

    # Initializing the model
    if phi_id < 7:
        model_list = ["human", "linear", "maritime", "robot2", "robot4", "robot5", "train"]
        model_name = model_list[phi_id]

        print(f"model name: {model_name}")

        if model_name == "maritime":
            n_vars_model = 2
        elif model_name in ("robot2", "robot4", "robot5"):
            n_vars_model = 3
        else:
            n_vars_model = 1
        model_path  = f'IR/data/data/{model_name}/model_state_dict.pth'
        quant_model = quantitative_model(model_path=model_path, nvars=n_vars_model)
        # Creating the trajectories where we cut some dimensions out for the model
        target_xi_cut = target_xi[:,:n_vars_model,:] 
        target_xi_filled = torch.zeros((target_xi.shape[0], 3, target_xi.shape[2]), device=device)
        target_xi_filled[:, :n_vars_model, :] = target_xi_cut

        proposal_xi_cut = proposal_xi[:,:n_vars_model,:]
        proposal_xi_filled = torch.zeros((proposal_xi.shape[0], 3, proposal_xi.shape[2]), device=device)
        proposal_xi_filled[:, :n_vars_model, :] = proposal_xi_cut
    else: # If the id is greter than 6 then load a formula
        # Loading the saved formulae
        with open(os.path.join("phis_dir", f"{test_name}.pkl"), 'rb') as f:
            phi_bag_dict = pickle.load(f)
        phi_bag = phi_bag_dict[phi_id]
        n_vars_model = max_n_vars
        proposal_xi_filled = proposal_xi
        target_xi_filled = target_xi
        proposal_xi_cut = proposal_xi
        target_xi_cut = target_xi
        del phi_bag_dict

    # Loading the saved formulae (1000 formulae)
    folder_index = os.path.join("IR", "index")  # Update with actual path
    train_phis = load_pickle(folder_index, 'train_phis_{}_vars.pickle'.format(max_n_vars))
    if n_train_phis != len(train_phis):
        raise RuntimeError(f"The number of train_phis is not 1000, it's {len(train_phis)}")
    

    ## K_target ##
    # Computing the robustness of each train_phis over the target_xi
    rhos_psi_target = torch.zeros(n_train_phis, n_traj)
    rhos_psi_target_norm = torch.zeros(n_train_phis)
    for (i, formula) in enumerate(train_phis):
        rhos_psi_target[i, :] = torch.tanh(formula.quantitative(target_xi_filled, evaluate_at_all_times=evaluate_at_all_times))
        rhos_psi_target_norm[i] = torch.norm(rhos_psi_target[i, :])
    # Computing the robustness of the model (or formula) over target_xi
    if phi_id < 7:
        rhos_phi_target = quant_model.robustness(traj=target_xi_cut)
    else:
        rhos_phi_target = torch.tanh(phi_bag[0].quantitative(target_xi_filled, evaluate_at_all_times=evaluate_at_all_times))

    rhos_phi_target_norm = torch.norm(rhos_phi_target)
    print(f"the norm of rhos_phi_target is : {rhos_phi_target_norm.item()}")
    # Computing the target kernels
    K_target = torch.tensordot(rhos_psi_target, rhos_phi_target, dims=([1],[0]) ) / (n_traj * math.sqrt(n_train_phis)) 
    print(f"the norm of K_target is : {torch.norm(K_target).item()}")
    # Computing the new target kernels
    New_K_target = kernel_to_new_kernel(K_target, norm_psis=rhos_psi_target_norm, norm_phi=rhos_phi_target_norm, n_traj=n_traj, n_formulae=n_train_phis)
    print(f"the norm of New_K_target is : {torch.norm(New_K_target).item()}")
    # Deleting used tensors
    del rhos_psi_target, rhos_psi_target_norm, rhos_phi_target_norm


    ## K_proposal ##
    # Computing the robustness of each train_phis over the proposal_xi
    rhos_psi_proposal = torch.zeros(n_train_phis, n_traj)
    rhos_psi_proposal_norm = torch.zeros(n_train_phis)
    for (i, formula) in enumerate(train_phis):
        rhos_psi_proposal[i, :] = torch.tanh(formula.quantitative(proposal_xi_filled, evaluate_at_all_times=evaluate_at_all_times))
        rhos_psi_proposal_norm[i] = torch.norm(rhos_psi_proposal[i, :])
    # Computing the robustness of the model (or formula) over the proposal_xi
    if phi_id < 7:
        rhos_phi_proposal = quant_model.robustness(traj=proposal_xi_cut)
    else:
        rhos_phi_proposal = torch.tanh(phi_bag[0].quantitative(proposal_xi_filled, evaluate_at_all_times=evaluate_at_all_times))

    rhos_phi_proposal_norm = torch.norm(rhos_phi_proposal)
    print(f"the norm of rhos_phi_proposal is : {rhos_phi_proposal_norm.item()}")
    # Computing the proposal kernel
    K_proposal = torch.tensordot(rhos_psi_proposal, rhos_phi_proposal, dims=([1],[0]) ) / (n_traj * math.sqrt(n_train_phis))
    print(f"the norm of K_proposal is : {torch.norm(K_proposal).item()}") 
    # Computing the new proposal kernels
    New_K_proposal = kernel_to_new_kernel(K_proposal, norm_psis=rhos_psi_proposal_norm, norm_phi=rhos_phi_proposal_norm, n_traj=n_traj, n_formulae=n_train_phis)
    print(f"the norm of New_K_proposal is : {torch.norm(New_K_proposal).item()}")
    # Deleting used tensors

    ## K_imp ##
    # Initializing the converter class
    converter = local_matrix(n_vars = n_vars, 
                                n_formulae = n_train_phis, 
                                n_traj = n_traj, 
                                n_traj_points = n_traj_points, 
                                evaluate_at_all_times = evaluate_at_all_times,
                                )
    # Computing the matrix Q that converts to a target kernel
    if converter.compute_Q(proposal_traj=proposal_xi_cut, PHI=rhos_psi_proposal, dweights=dweights):
        # returns if there are problems with the pseudoinverse 
        return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    # Computing the importance sampling kernel starting from the proposal one
    K_imp = converter.convert_to_local(K_proposal).type(torch.float32)
    print(f"The norm of K_imp is : {torch.norm(K_imp).item()}")
    # TODO: check that this is correct:
    New_K_imp = kernel_to_new_kernel(K_imp, norm_psis=rhos_psi_proposal_norm, norm_phi=rhos_phi_proposal_norm, n_traj=n_traj, n_formulae=n_train_phis)
    print(f"the norm of New_K_imp is : {torch.norm(New_K_imp).item()}")

    
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

    #Computing Dist and Cos_Dist and Dist_rho
    Dist = torch.norm(K_target - K_imp).item()
    Cos_dist = 1 - torch.dot(K_target/Norm_target, K_imp/Norm_imp).item()
    Dist_rho = torch.norm(rhos_phi_proposal - rhos_phi_target).item()/math.sqrt(n_traj)

    # Computing the distance between the new kernels:
    dist_new_kernels = torch.norm(New_K_target - New_K_imp).item()

    #Path to the index forlder
    folder_index = os.path.join("IR", "index")  # Update with actual path

    # Transforming kernels into the embeddings of the database
    target_embedding = new_kernel_to_embedding(New_K_target)
    print(f"the norm of target_embedding is : {torch.norm(target_embedding).item()}")
    imp_embedding = new_kernel_to_embedding(New_K_imp)
    print(f"the norm of imp_embedding is : {torch.norm(imp_embedding).item()}")
    dist_embed = torch.norm(target_embedding-imp_embedding).item()

    start_search_time1 = time.time()
    # Search for closest formulae to each kernel
    target_formulae_list, target_dists = search_from_embeddings(embeddings=target_embedding.unsqueeze(0),
                                                                nvar=n_vars_model,
                                                                folder_index=folder_index,
                                                                k=k,
                                                                n_neigh=64,
                                                                n_pc=-1,
                                                                timespan=None,
                                                                nodes=None)
    print(f"target dists: {target_dists}")
    total_search_time1 = time.time()-start_search_time1
    print(f"total_search_time1 = {total_search_time1}")
    
    start_search_time2 = time.time()
    imp_formulae_list, imp_dists = search_from_embeddings(embeddings=imp_embedding.unsqueeze(0),
                                                                nvar=max_n_vars,
                                                                folder_index=folder_index,
                                                                k=k,
                                                                n_neigh=64,
                                                                n_pc=-1,
                                                                timespan=None,
                                                                nodes=None)
    print(f"imp dists: {imp_dists}")
    total_search_time2 = time.time()-start_search_time2
    print(f"total_search_time2 = {total_search_time2}")
    
    # Extract the formulae (first element is the list of formulae for the first/only kernel)
    target_formulae_str = target_formulae_list[0]
    imp_formulae_str = imp_formulae_list[0]
    print(f"formulae are there! There are {len(imp_formulae_str)} imp_formulae and {len(target_formulae_str)} target_formulae")
    print(f"imp_formulae: {imp_formulae_str}, target_formulae: {target_formulae_str}")
    
    # Compute overlap between the two sets of formulae
    common_formulae = set([str(f) for f in target_formulae_str]).intersection([str(f) for f in imp_formulae_str])
    overlap_form = len(common_formulae) / k
    print(f"overlap_form = {overlap_form}")

    # Transform the formulae from string version to actual formulae
    target_formulae = [from_string_to_formula(x) for x in target_formulae_str]
    imp_formulae = [from_string_to_formula(x) for x in imp_formulae_str]

    # Compute distance between the formulae using similarity_based_relevance
    # Create a BaseMeasure for generating test trajectories (if needed)
    Test_measure = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
    test_trajectories = Test_measure.sample(1000, max_n_vars)  # Sample trajectories for comparison
    
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
    print(f"cosine similarity: {cosine_similarity}")
    dist_form = 1.0 - cosine_similarity.mean().item()
    print(f"dist_form = {dist_form}")

    # Deleting used tensors
    del K_proposal, K_target, K_imp, rhos_psi_proposal, rhos_phi_proposal, rhos_phi_target

    # Computing the cosine distance between the model and the two retrieved formulas
    test_trajectories_filled = torch.zeros_like(test_trajectories)
    test_trajectories_cut = test_trajectories[:, :n_vars_model, :]
    test_trajectories_filled[:,:n_vars_model, :] = test_trajectories_cut

    if phi_id < 7:
        rhos_model = quant_model.robustness(traj=proposal_xi_cut)
    else: 
        rhos_model = torch.tanh(phi_bag[0].quantitative(proposal_xi_filled, evaluate_at_all_times=evaluate_at_all_times))
        
    rhos_model_norm = torch.norm(rhos_model)
    rhos_target = torch.tanh(target_formulae[0].quantitative(proposal_xi_filled, evaluate_at_all_times=evaluate_at_all_times))
    rhos_target_norm = torch.norm(rhos_target)
    rhos_imp = torch.tanh(imp_formulae[0].quantitative(proposal_xi_filled, evaluate_at_all_times=evaluate_at_all_times))
    rhos_imp_norm = torch.norm(rhos_imp)

    model_to_target = 1.0 - ( torch.tensordot(rhos_model, rhos_target, dims=([0],[0]))/(rhos_model_norm*rhos_target_norm) ).item()
    model_to_imp = 1.0 - ( torch.tensordot(rhos_model, rhos_imp, dims=([0],[0]) )/(rhos_model_norm*rhos_imp_norm)).item()

    # End timing
    Elapsed_time = time.time() - start_time
    
    # flushing output
    sys.stdout.flush()

    return weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e, overlap_form, dist_form, dist_new_kernels, dist_embed, model_to_target, model_to_imp



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
        weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0, Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp, Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e, overlap_form, dist_form, dist_new_kernels, dist_embed, model_to_target, model_to_imp = result
        
        # Computing n_e
        try:
            n_e = (Sum_weights**2)/Sum_squared_weights
        except:
            n_e = None

        print(f"weight_strategy = {weight_strategy}, n_psi_added = {n_psi_added}, n_traj = {n_traj}, target_std = {target_std}, proposal_std = {proposal_std}, n_traj_points = {n_traj_points}, phi_id = {phi_id}, base_xi_id = {base_xi_id}, mu0 = {mu0}, mu1 = {mu1}, sigma1 = {sigma1}, q = {q}, q0 = {q0}, Dist = {Dist}, Cos_dist = {Cos_dist}, Dist_rho = {Dist_rho}, Norm_proposal = {Norm_proposal}, Norm_target = {Norm_target}, Norm_imp = {Norm_imp}, Pinv_error = {Pinv_error}, Sum_weights = {Sum_weights}, Sum_squared_weights = {Sum_squared_weights}, Elapsed_time = {Elapsed_time}, Process_mem = {Process_mem}, n_e = {n_e}, overlap_form = {overlap_form}, dist_form = {dist_form}, overlap_form = {overlap_form}, dist_form = {dist_form}, dist_new_kernels = {dist_new_kernels}, dist_embed = {dist_embed},  model_to_target = {model_to_target}, model_to_imp = {model_to_imp}")

        with sqlite3.connect(db_path, timeout=60.0) as conn:  # Increased timeout for concurrent access
            c = conn.cursor()
            # Saving the values in the database
            c.execute('''INSERT OR REPLACE INTO results 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (weight_strategy, n_psi_added, n_traj, target_std, proposal_std, n_traj_points, phi_id, base_xi_id, mu0, mu1, sigma1, q, q0,
                    Dist, Cos_dist, Dist_rho, Norm_proposal, Norm_target, Norm_imp,
                    Pinv_error, Sum_weights, Sum_squared_weights, Elapsed_time, Process_mem, n_e, overlap_form, dist_form, dist_new_kernels, dist_embed, model_to_target, model_to_imp))
            
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