import torch
import numpy as np
import pickle
from kernel import StlKernel, KernelRegression, GramMatrix
#from generator import BaseMeasure, StlGenerator
from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from scipy.stats import norm

# device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {device}".format(device=device))

## STL FORMULAE GENERATOR ##
# stl generator parameters
prob_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound o the type [0,infty]
leaf_probability = 0.5  # probability of generating a leaf (always zero for root)
time_bound_max_range = 10  # maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
atom_threshold_sd = 1.0  # std for the normal distribution of the thresholds of atoms

formula = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=time_bound_max_range,
                       unbound_prob=prob_unbound_time_operator, threshold_sd=atom_threshold_sd)

## TRAJECTORY GENERATOR ##
# trajectory sample parameters
initial_std = 1.0  # standard deviation of normal distribution of initial state
total_var_std = 1.0  # standard deviation of normal distribution of total variation
n_vars = 2

mu0 = BaseMeasure(device=device, sigma0=initial_std, sigma1=total_var_std)

## CREATING THE MATRIX Q ##

n_formulae = 1050 # number of formulae used to create the matrix P
n_traj = 1000 # number of trajectories used to create the matrix P
n_traj_points: int = 11 # number of points in a trajectory
# NOTE: n_traj_points must be >= max_timespan in phis_generator
# NOTE: max_timespan must be greater than 11 (for some reason)
evaluate_at_all_times = False


# Trajectory for the local explanation
r0 = torch.zeros([n_vars, n_traj_points], dtype=torch.float64) # list of points of the local trajectory (for now all zeros)
std0 = 10 * torch.ones([n_vars, n_traj_points], dtype=torch.float64) # list of std of the local trajectory (no covariance matrix for now)

P = torch.zeros([n_formulae, n_traj], dtype=torch.float64)
bag = formula.bag_sample(n_formulae, n_vars)
traj = mu0.sample(samples=n_traj, varn=n_vars, points=n_traj_points)
print(f"The shape of the trajectory is : {traj.shape}")

for i in range(n_formulae):
    print(i)
    quant = bag[i].quantitative(traj, evaluate_at_all_times=evaluate_at_all_times)
    # quantitative returns the robustness of the formula for a certain sample
    # quantitative needs in input a tensor x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
    for j in range(n_traj):
        P[i][j] = quant[j].item()

P_daga = torch.linalg.pinv(P)

dprob = torch.ones(n_traj)
# TODO: Find a better definition for stab_fact
stab_fact = 1/norm.pdf(r0[0][0], loc=r0[0][0], scale=std0[0][0])
# stab_fact is a stability factor (otherwise dprob would be 0)
for i in range(n_traj):
    for j in range(n_vars):
        for k in range(n_traj_points):
            dprob[i] = dprob[i] * norm.pdf(traj[i][j][k], loc=r0[j][k], scale=std0[j][k]) * stab_fact
            print(norm.pdf(traj[i][j][k], loc=r0[j][k], scale=std0[j][k]), stab_fact)
            # Probability values on the diagonal of D
    print(f"dprob[{i}] = {dprob[i]}")

print(f"dprob = {dprob}")
print(torch.diag(dprob).type(torch.float64).dtype)
print(f"P : {P}")
print(f"D : {torch.diag(dprob).type(torch.float64)}")
print(f"P_daga : {P_daga}")
Q = torch.matmul(P, torch.diag(dprob).type(torch.float64) )
print(f"P * D : {Q}")
Q = torch.matmul(Q, P_daga)

print(f"Q : {Q}")

## CHECKING THE INDEPENDENCE OF P ##

rank = torch.linalg.matrix_rank(P)

print(rank)

if rank < n_traj:
    print("columns are not independent")
else:
    print("columns are independent!!")
# About 50 new rows (formulae) are enough to have independent columns