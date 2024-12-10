import torch
import numpy as np
from matplotlib import pyplot as plt
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure
from phis_generator import StlGenerator
try:
    import A
    module_used = A
except ImportError:
    import kernel
    module_used = kernel


"""
Samples some base trajectories (base_xi) and forumulae (psi) and confronts 
the kernel of some formulae (phi) obtained in two different ways:
# The classic kernel K_cl(phi) measured on some local trajectories (local_xi)
    around each base_xi
# The kernel K_imp(phi) measured on some global trajectories (global_xi)
    weighted according to immportance sampling

The result is a matrix (Dist) that contains the euclidean distance between the two kernels
of the same formula around the same trajectory
"""

# Device used
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation of formulae
evaluate_at_all_times = False # TODO: implement for time evaluation

# Parameters for the sampling of the base trajectories (using mu0 distribution)
base_initial_std = 1.0
base_total_var_std = 1.0
base_n_traj = 1
n_vars = 2
# NOTE: n_traj_points must be >= max_timespan in phis_generator
# NOTE: max_timespan must be greater than 11 (for some reason) #TODO: find why this is the case
n_traj_points = 11

# Parameters for the sampling of the local trajectories
#local_std = 1
#local_n_traj = 1000
local_n_traj_points = n_traj_points

# Parameters for the sampling of the global trajectories
#global_n_traj = local_n_traj # Using the same number of trajectories for global and local
global_n_traj_points = n_traj_points

# Parameters for the sampling of the formulae
leaf_probability = 0.5
time_bound_max_range = 10
prob_unbound_time_operator = 0.1
atom_threshold_sd = 1.0
#n_psi = 1050
n_phi = 1

# Parameters for the test
list_stds = list(np.arange(1, 0.5, -0.05))
list_n_traj = list(range(100, 1100, 100))
list_n_psi_added = list(range(100, 1100, 100))

# Creating the numpy array for the resulting distances (the array contains a list of values: (n_traj, n_psi_added, local_std, mean distance))
Distances = np.ndarray(shape=(len(list_n_traj), len(list_n_psi_added), len(list_stds)), dtype=object)
Cos_distances = np.ndarray(shape=(len(list_n_traj), len(list_n_psi_added), len(list_stds)), dtype=object)
# Creating the arrays for the norms of the kernels (K_global, K_cl, K_imp)
Norms = np.ndarray(shape=(len(list_n_traj), len(list_n_psi_added), len(list_stds)), dtype=object)


for (idx1, n_psi_added) in enumerate(list_n_psi_added):
    for (idx2, n_traj) in enumerate(list_n_traj):

        n_psi = n_traj + n_psi_added # The number of psi formulae must always be greater than the number of traj
        local_n_traj = n_traj
        global_n_traj = n_traj

        # Sampling of the base trajectories
        base_mu0 = BaseMeasure(device=device, 
                                sigma0=base_initial_std, 
                                sigma1=base_total_var_std)
        base_xi = base_mu0.sample(samples=base_n_traj, 
                                    varn=n_vars, 
                                    points=n_traj_points)

        # Sampling the global trajectories (using the same distr of the base trajectories)
        global_xi = base_mu0.sample(samples=global_n_traj, 
                                    varn=n_vars, 
                                    points=n_traj_points)

        # Sampling the formulae (phi and psi) for the kernels
        formulae = StlGenerator(leaf_prob=leaf_probability, 
                                    time_bound_max_range=time_bound_max_range,
                                    unbound_prob=prob_unbound_time_operator, 
                                    threshold_sd=atom_threshold_sd)
        phi_bag = formulae.bag_sample(n_phi, n_vars)  # we want the kernel representation of each phi
        psi_bag = formulae.bag_sample(n_psi, n_vars)

        # Initializing the robustness tensors
        rhos_phi_local = torch.empty(n_phi, local_n_traj, device=device) # robustness of each phi on the local_xi
        rhos_psi_local = torch.empty(n_psi, local_n_traj, device=device) # robustness of each psi on the local_xi

        rhos_phi_global = torch.empty(n_phi, global_n_traj, device=device) # robustness of each phi on the global_xi
        rhos_psi_global = torch.empty(n_psi, global_n_traj, device=device) # robustness of each psi on the global_xi

        # Computing the global kernels that will be converted into local ones
        for (j, psi) in enumerate(psi_bag):
            rhos_psi_global[j, :] = torch.tanh(psi.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))

        for (j, phi) in enumerate(phi_bag):
            rhos_phi_global[j, :] = torch.tanh(phi.quantitative(global_xi, evaluate_at_all_times=evaluate_at_all_times))

        K_global = torch.tensordot(rhos_phi_global, rhos_psi_global, dims=([1],[1]) ) # TODO: this doesn't work if there's time evaluation!

        for (idx3, local_std) in enumerate(list_stds): # Iteration over different local standard deviations

            # Defining the two kernel tensors
            K_cl = torch.empty(n_phi, base_n_traj, n_psi, device=device) # tensor of the classical kernels of phi
            K_imp = torch.empty(n_phi, 1, n_psi, device=device) # tensor of the kernels of phi obtained with importance sampling 
                                                                # (the second dimension is one but it will be expanded later)
            # Computing the local kernels in the classical way
            for i in range(base_xi.size(0)):
                # Sampling the local trajectories (this is equivalent to adding
                # a brownian motion on the base trajectory)
                noise = local_std*torch.cumsum(torch.randn(local_n_traj, n_vars, n_traj_points, device=device), 2)

                local_xi = noise + base_xi[i]
                # Computing the robustness of each psi on the local_xi
                for (j, psi) in enumerate(psi_bag):
                    rhos_psi_local[j, :] = torch.tanh(psi.quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))
                # Computing the robustness of each phi on the local_xi
                for (j, phi) in enumerate(phi_bag):
                    rhos_phi_local[j, :] = torch.tanh(phi.quantitative(local_xi, evaluate_at_all_times=evaluate_at_all_times))

                # Computing the classical kernels
                K_cl[:,i,:] = torch.tensordot(rhos_phi_local, rhos_psi_local, dims=([1],[1]) ) # TODO: this doesn't work if there's time evaluation!
            
            # Computing the local kernels using importance sampling (first computing the global kernel then converting it to local)
            K_imp = K_global.unsqueeze(1).repeat(1, base_xi.size(0), 1)
            # Initializing the converter class
            converter = local_matrix(n_vars = n_vars, 
                                        n_formulae = n_psi, 
                                        n_traj = global_n_traj, 
                                        n_traj_points = n_traj_points, 
                                        evaluate_at_all_times = evaluate_at_all_times
                                        )
            
            for i in range(base_xi.size(0)):

                # Computing the matrix Q that converts to a local kernel around the base_xi
                converter.compute_Q(local_traj = base_xi[i], 
                                    global_traj = global_xi,
                                    std_local = local_std,
                                    total_var_std = base_total_var_std,
                                    initial_std = base_initial_std,
                                    PHI = rhos_psi_global,
                                    formula_bag = psi_bag)

                # Converting every global kernel to local
                for (j, phi) in enumerate(phi_bag):
                    K_imp[j,i,:] = converter.convert_to_local(K_imp[j,i,:])

            #Testing the norms of the kernels
            print(f"n_psi_added: {n_psi_added}, n_traj = {n_traj}, local_std = {local_std}")
            Norm_global = torch.norm((K_global), dim=1)
            print(f"Testing the norm of K_global: {Norm_global}")
            Norm_cl = torch.norm((K_cl), dim=2)
            print(f"Testing the norm of K_cl: {Norm_cl}")
            #print(f"K_cl :{K_cl}")
            Norm_imp = torch.norm((K_imp), dim=2)
            print(f"Testing the norm of K_imp: {Norm_imp}")
            #print(f"K_imp : {K_imp}")

            #Computing the matrix Dist and Cos_Dist
            Dist = torch.norm((K_cl - K_imp), dim=2)
            Cos_Dist = torch.norm((K_cl/Norm_cl - K_imp/Norm_imp), dim=2)
            #print(f"The matrix Dist is: {Dist}")
            Dist_same_formula = torch.mean(Dist, dim=1)
            Dist_same_traj = torch.mean(Dist, dim=0)
            Dist_mean = Dist.mean()
            Cos_dist_mean = Cos_Dist.mean()
            print(f"The mean distance on a single formula is: {Dist_same_formula}")
            print(f"The mean distance on a single trajectory is: {Dist_same_traj}")
            print(f"The mean distance is: {Dist_mean}")
            print(f"The mean cosine distance is : {Cos_dist_mean} \n")

            # Filling the result arrays
            Distances[idx1, idx2, idx3] = (n_psi_added,n_traj,local_std,Dist_mean)
            Cos_distances[idx1, idx2, idx3] = (n_psi_added,n_traj,local_std,Cos_dist_mean)
            Norms[idx1, idx2, idx3] = (n_psi_added,n_traj,local_std,Norm_global,Norm_cl,Norm_imp)

# Saving the arrays
np.save('Distances.npy', Distances)
np.save('Cos_distances.npy', Cos_distances)
np.save('Norms.npy', Norms)

# Loading the arrays back
#Distances = np.load('Distances.npy')
#Cos_distances = np.load('Cos_distances.npy')
#Norms = np.load('Norms.npy')

#TODO: plot results



    




