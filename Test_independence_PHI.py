import torch
import numpy as np
from matplotlib import pyplot as plt
import time
from Local_Matrix import local_matrix
from traj_measure import BaseMeasure
from phis_generator import StlGenerator

# Device used
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation of formulae
evaluate_at_all_times = False # TODO: implement for time evaluation

# Parameters for the sampling of the trajectories
initial_std = 1.0
total_var_std = 1.0
#n_traj = 1000
n_vars = 2
n_traj_points = 11 

# Parameters for the sampling of the formulae
leaf_probability = 0.5
time_bound_max_range = 10
prob_unbound_time_operator = 0.1
atom_threshold_sd = 1.0
#n_psi = 1050
n_phi = 1  # The test will be done on a single formula phi

# Trajectory generator
mu0 = BaseMeasure(device=device, 
                          sigma0=initial_std, 
                          sigma1=total_var_std)

# Formulae generator
formula = StlGenerator(leaf_prob=leaf_probability, 
                               time_bound_max_range=time_bound_max_range,
                               unbound_prob=prob_unbound_time_operator, 
                               threshold_sd=atom_threshold_sd)


# number of repetition of the test
Reps_traj = 20
N_traj = [10*n for n in range(1, Reps_traj+1) ] # from 10 to 200 trajectories
Reps_psi = 10
N_psi = [2*n for n in range(1, Reps_psi+1) ] # the number of formulae is from 2 to 20 plus the number of trajectories
n_reps = 100 # number of repetition of the indep test for each set of parameters

# Initializing the converter class
converter = local_matrix(n_vars = n_vars,
                    n_traj_points = n_traj_points, 
                    evaluate_at_all_times = evaluate_at_all_times
                    )

Results = torch.empty(Reps_psi, Reps_traj, dtype=torch.float64, device=device) # Result tensor

for (i, n_psi_added) in enumerate(N_psi):
    print(f"Number of psi added: {n_psi_added}")
    start = time.time()
    for (j, n_traj) in enumerate(N_traj):
        print(j)
        n_psi = n_traj + n_psi_added
        indeps = 0
        for k in range(n_reps):
            # Sampling the formulae
            bag = formula.bag_sample(n_psi, 
                                      n_vars)
            converter.n_formulae = n_psi
            converter.bag = bag
            # Sampling the trajectories
            traj = mu0.sample(samples=n_traj, 
                               varn=n_vars, 
                               points=n_traj_points)
            converter.n_traj = n_traj
            converter.traj = traj
            # Computing PHI
            converter.generate_PHI()
            if converter.check_independence():
                indeps += 1
        Results[i,j] = indeps/n_reps
    print(f"Time for {n_psi_added} psi added : {time.time()-start}")

# Saving the Results tensor
torch.save(Results, 'Indep_Results.pt')

# Load the tensor
#Results = torch.load('Indep_Results.pt')

# Printing the results:
#plt.figure(figsize=(10, 8))
#plt.imshow(Results, cmap='viridis') 
#plt.colorbar(label='Value')  # Add a color scale
#plt.title('Percentage of indep col in PHI')
#plt.xlabel('n_traj')
#plt.ylabel('n_formulae added')

# Annotation on the axis
#plt.xticks(range(len(N_traj)), N_traj)
#plt.yticks(range(len(N_psi)), N_psi)

#plt.show()

