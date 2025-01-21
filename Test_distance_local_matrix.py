import torch
import numpy as np
from matplotlib import pyplot as plt
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure, LocalBrownian
from phis_generator import StlGenerator
import math
from typing import List, Any
import time

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

The results the averaged over many phi and many local distributions.

"""


def compute_robustness_tensor(formula_bag: List[Any], 
                              trajectories: torch.Tensor, 
                              evaluate_at_all_times: bool) -> torch.Tensor:
    """
    Computes the robustness tensor

    Parameters:
    ----------
    formula_bag (List): List of formulae of which we will compute the robustness
    trajectories (torch.Tensor): trajectories tensor of shape [samples, varn, points]
        used to compute the robustness
    evaluate_at_all_times (bool): decides if there should be evaluation at all time
         (currently not available)
    
    Returns:
    --------
    rhos (torch.Tensor): robustness tensor of shape [n_formulae, n_trajectories]
    """
    rhos = torch.empty((len(formula_bag), trajectories.shape[0]))
    for (i, formula) in enumerate(formula_bag):
        rhos[i, :] = torch.tanh(formula.quantitative(trajectories, evaluate_at_all_times=evaluate_at_all_times))
    return rhos


# Device used
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation of formulae
evaluate_at_all_times = False # TODO: implement for time evaluation

# Parameters for the sampling of the base trajectories (using mu0 distribution)
base_initial_std = 1.0
base_total_var_std = 1.0
n_vars = 2
# NOTE: n_traj_points must be >= max_timespan in phis_generator
# NOTE: max_timespan must be greater than 11 (for some reason) #TODO: find why this is the case
n_traj_points = 11

# Parameters for the sampling of the formulae
leaf_probability = 0.5
time_bound_max_range = 10
prob_unbound_time_operator = 0.1
atom_threshold_sd = 1.0

# Parameters for the repetition of the results (for now always set to 1)
base_n_traj = 1
n_phi = 1

# Parameters for the test
list_stds = list(np.arange(1, 0.75, -0.05))#list(np.arange(1, 0.45, -0.05))
list_n_traj = list(range(2000, 5000, 1000))#list(range(100, 1100, 100))
list_n_psi_added = list(range(100, 900, 200))#list(range(100, 1100, 100))

# Creating the numpy array for the resulting distances 
# The array contains a list of values: (n_psi_added,n_traj,local_std,Dist_mean,Cos_dist_mean)
Distances = np.ndarray(shape=(len(list_n_psi_added), len(list_n_traj), len(list_stds)), dtype=object)
# Creating the arrays for the norms of the kernels
# The arrauy contains : (n_psi_added,n_traj,local_std,Norm_global,Norm_loc,Norm_imp)
Norms = np.ndarray(shape=(len(list_n_psi_added), len(list_n_traj), len(list_stds)), dtype=object)

# Initializing the base trajectory distribution. The local distributions will be centered around each base trajectory
base_distr = BaseMeasure(device=device, 
                                sigma0=base_initial_std, 
                                sigma1=base_total_var_std)

# Initializing the global trajectory distribution
global_distr = BaseMeasure(device=device, 
                                sigma0=base_initial_std, 
                                sigma1=base_total_var_std)

# Initializing the formulae distribution
formulae_distr = StlGenerator(leaf_prob=leaf_probability, 
                                    time_bound_max_range=time_bound_max_range,
                                    unbound_prob=prob_unbound_time_operator, 
                                    threshold_sd=atom_threshold_sd)


#cos_dist_list = [] # Easy way of saving the cosine dists
#dist_list = [] # Easy way of saving the dist


# Iteration over the parameters
for (idx1, n_psi_added) in enumerate(list_n_psi_added):
    for (idx2, n_traj) in enumerate(list_n_traj):
        for (idx3, local_std) in enumerate(list_stds):

            # Timing each internal loop
            start_time = time.time()

            n_psi = n_traj + n_psi_added 
            local_n_traj = n_traj
            global_n_traj = n_traj

            # Sampling of the base trajectories (the local distributions will be created around these base trajectories)
            base_xi = base_distr.sample(samples=base_n_traj, 
                                        varn=n_vars, 
                                        points=n_traj_points)

            # Sampling the global trajectories
            global_xi = global_distr.sample(samples=global_n_traj, 
                                        varn=n_vars, 
                                        points=n_traj_points)

            # Sampling the formulae (phi and psi) for the kernels
            # Each formula phi will have its kernel representation
            phi_bag = formulae_distr.bag_sample(n_phi, n_vars)
            psi_bag = formulae_distr.bag_sample(n_psi, n_vars)

            # Iterating over each base_xi and each phi
            
            # Defining the kernel tensors
            K_loc = torch.empty(n_phi, base_n_traj, n_psi, device=device)
            K_global = torch.empty(n_phi, base_n_traj, n_psi, device=device)
            K_imp = torch.empty(n_phi, base_n_traj, n_psi, device=device)

            # Computing the robustness of each psi over the global_xi
            rhos_psi_global = compute_robustness_tensor(psi_bag, global_xi, evaluate_at_all_times)
            
            for i in range(base_n_traj): # Iteration over each base_xi (so we iterate over different local trajectories)
                
                # Initializing the local trajectory distribution and sampling local_xi
                local_distr = LocalBrownian(base_traj=base_xi[i], std=local_std, device=device)
                local_xi = local_distr.sample(samples=local_n_traj, varn=n_vars, points=n_traj_points)

                # Computing the robustness of each psi over the local_xi
                rhos_psi_local = compute_robustness_tensor(psi_bag, local_xi, evaluate_at_all_times)
                
                # Initializing the converter class
                converter = local_matrix(n_vars = n_vars, 
                                            n_formulae = n_psi, 
                                            n_traj = global_n_traj, 
                                            n_traj_points = n_traj_points, 
                                            evaluate_at_all_times = evaluate_at_all_times,
                                            target_distr = local_distr,
                                            proposal_distr = global_distr
                                            )
                # Computing the matrix Q that converts to a local kernel around the base_xi
                converter.compute_Q(proposal_traj = global_xi,
                                    PHI = rhos_psi_global)

                for j in range(n_phi):

                    # Computing the robustness of phi over local_xi
                    rhos_phi_local = torch.tanh(phi_bag[j].quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
                    # Computing the local kernels
                    K_loc[j,i,:] = torch.tensordot(rhos_psi_local, rhos_phi_local, dims=([1],[0]) ) / (local_n_traj * math.sqrt(n_psi))  

                    # Computing the robustness of phi over the global_xi
                    rhos_phi_global = torch.tanh(phi_bag[j].quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))
                    # Computing the global kernel
                    K_global[j,i,:] = torch.tensordot(rhos_psi_global, rhos_phi_global, dims=([1],[0]) ) / (global_n_traj * math.sqrt(n_psi)) 

                    # Computing the importance sampling kernel starting from the global one
                    K_imp[j,i,:] = converter.convert_to_local(K_global[j,i,:])


            #Testing the norms of the kernels
            #print(f"n_psi_added: {n_psi_added}, n_traj = {n_traj}, local_std = {local_std}")
            Norms_global = torch.norm((K_global), dim=2)
            Norm_global = Norms_global.mean().item()
            #print(f"Testing the norm of K_global: {Norm_global}")
            Norms_loc = torch.norm((K_loc), dim=2)
            Norm_loc = Norms_loc.mean().item()
            #print(f"Testing the norm of K_loc: {Norm_loc}")
            Norms_imp = torch.norm((K_imp), dim=2)
            Norm_imp = Norms_imp.mean().item()
            #print(f"Testing the norm of K_imp: {Norm_imp}")

            #Computing the matrix Dist and Cos_Dist
            Dist = torch.norm((K_loc - K_imp), dim=2)
            Cos_Dist = 1 - torch.tensordot(K_loc/(Norms_loc.unsqueeze(-1)), K_imp/(Norms_imp.unsqueeze(-1)), dims=([2],[2]) )
            Dist_mean = Dist.mean().item()
            Cos_dist_mean = Cos_Dist.mean().item()   #TODO: This doesn't work if n_phi and base_n_traj are not 1!
            #print(f"The mean distance is: {Dist_mean}")
            #print(f"The mean cosine distance is : {Cos_dist_mean} \n")


            #cos_dist_list.append(Cos_dist_mean)
            #dist_list.append(Dist_mean)


            # Filling the result arrays
            Distances[idx1, idx2, idx3] = (n_psi_added,n_traj,local_std,Dist_mean,Cos_dist_mean, converter.pinv_error)
            Norms[idx1, idx2, idx3] = (n_psi_added,n_traj,local_std,Norm_global,Norm_loc,Norm_imp, converter.pinv_error)

            # End timing
            total_time = time.time() - start_time
            #print(f"The time elapsed with n_psi={n_psi}, n_traj={n_traj} is:{total_time}")
            with open('Elapsed_time.txt', 'a') as file:
                file.write(f"Time elapsed with n_psi={n_psi}, n_traj={n_traj} is:{total_time}\n")


# Saving the arrays
np.save('Distances_big_new.npy', Distances)
np.save('Norms_big_new.npy', Norms)


# Loading the arrays back
#Distances = np.load('Distances.npy', allow_pickle=True)
#Norms = np.load('Norms.npy', allow_pickle=True)


#Printing the results:
#somma_cos = 0
#elementi_cos = 0
#for x in cos_dist_list:
#    if not math.isnan(x):
#        somma_cos += x
#        elementi_cos += 1

#print(f"The cos distance results are: {somma_cos/elementi_cos}")

#Printing the results:
#somma_dist = 0
#elementi_dist = 0
#for x in dist_list:
#    if not math.isnan(x):
#        somma_dist += x
#        elementi_dist += 1

#print(f"The distance results are: {somma_dist/elementi_dist}")
