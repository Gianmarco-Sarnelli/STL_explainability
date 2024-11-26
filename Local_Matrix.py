import torch
import numpy as np
import pickle
import math
from kernel import StlKernel, KernelRegression, GramMatrix
#from generator import BaseMeasure, StlGenerator
from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from scipy.stats import norm

class local_matrix:
    def __init__(self, prob_unbound_time_operator = 0.1, 
                 leaf_probability = 0.5, 
                 time_bound_max_range = 10, 
                 atom_threshold_sd = 1.0,
                 initial_std = 1.0, 
                 total_var_std = 1.0, 
                 n_vars = 2, 
                 n_formulae = 1050, 
                 n_traj = 1000, 
                 n_traj_points: int = 11, 
                 evaluate_at_all_times = False ):

        # Local distribution (TODO: to be implemented later)
        self.local_distr = "normal"

        # stl generator parameters
        self.prob_unbound_time_operator = prob_unbound_time_operator  # probability of a temporal operator to have a time bound o the type [0,infty]
        self.leaf_probability = leaf_probability  # probability of generating a leaf (always zero for root)
        self.time_bound_max_range = time_bound_max_range  # maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        self.atom_threshold_sd = atom_threshold_sd  # std for the normal distribution of the thresholds of atoms

        # trajectory sample parameters
        self.initial_std = initial_std  # standard deviation of normal distribution of initial state
        self.total_var_std = total_var_std  # standard deviation of normal distribution of total variation
        self.n_vars = n_vars # number of variables of each trajectory

        # local matrix parameters
        self.n_formulae = n_formulae # number of formulae used to create the matrix P
        self.n_traj = n_traj # number of trajectories used to create the matrix P
        self.n_traj_points: int = n_traj_points # number of points in a trajectory
        # NOTE: n_traj_points must be >= max_timespan in phis_generator
        # NOTE: max_timespan must be greater than 11 (for some reason)
        self.evaluate_at_all_times = evaluate_at_all_times

        # device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # formulae and trajectories
        self.bag = None
        self.traj = None

        # Important matrices and values
        self.PHI = None
        self.PHI_daga = None
        self.dprob = None
        self.norm_factor = None
        self.Q = None
        self.rank = None

    def generate_formulae(self):
        formula = StlGenerator(leaf_prob=self.leaf_probability, 
                               time_bound_max_range=self.time_bound_max_range,
                               unbound_prob=self.prob_unbound_time_operator, 
                               threshold_sd=self.atom_threshold_sd)
        self.bag = formula.bag_sample(self.n_formulae, 
                                      self.n_vars)

    def generate_traj(self):
        mu0 = BaseMeasure(device=self.device, 
                          sigma0=self.initial_std, 
                          sigma1=self.total_var_std)
        self.traj = mu0.sample(samples=self.n_traj, 
                               varn=self.n_vars, 
                               points=self.n_traj_points)
        
    def compute_Q(self):
        # Trajectory for the local explanation
        r0 = torch.zeros([self.n_vars, self.n_traj_points], dtype=torch.float64, device=self.device) # list of points of the local trajectory (for now all zeros)
        std0 = 10 * torch.ones([self.n_vars, self.n_traj_points], dtype=torch.float64, device=self.device) # list of std of the local trajectory (no covariance matrix for now)

        # Generating the formulas and trajectories
        self.generate_formulae()
        self.generate_traj()

        # Computing PHI and PHI_daga
        self.PHI = torch.zeros([self.n_formulae, self.n_traj], dtype=torch.float64, device=self.device)
        for i in range(self.n_formulae):
            quant = self.bag[i].quantitative(self.traj, evaluate_at_all_times=self.evaluate_at_all_times)
            # quantitative returns the robustness of the formula for a certain sample
            # quantitative needs in input a tensor x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            for j in range(self.n_traj):
                self.PHI[i][j] = quant[j].item()
        self.PHI_daga = torch.linalg.pinv(self.PHI)

        # Computing dprob (probability values on the diagonal of D)
        self.dprob = torch.ones(self.n_traj, device=self.device)
        if self.evaluate_at_all_times:
            raise("The normalization constant for the evaluation on all time points is not built yet")
            #Here the norm_factor should be the inverse of the sum over all the trajectories AND all times
        else:
            # norm_factor is the (negative log of the) normalization constant
            self.norm_factor = 0
            for i in range(self.n_traj):
                prod = 1
                for j in range(self.n_vars):
                    for k in range(self.n_traj_points):
                        if self.local_distr == "normal":
                            prod *= norm.pdf(self.traj[i][j][k], loc=r0[j][k], scale=std0[j][k])
                        else:
                            raise("Other local distributions are not implemented yet!")
                self.norm_factor += prod
            #norm_factor is the inverse of the sum of the probabilities of each trajectory
            self.norm_factor = - math.log(self.norm_factor)
            #Here we do the opposite of the log for numerical stability

            # To have a more stable code, we add the log of the probabilities and then do the exponentiation
            for i in range(self.n_traj):
                log_prob = self.norm_factor
                for j in range(self.n_vars):
                    for k in range(self.n_traj_points):
                        if self.local_distr == "normal":
                            log_prob = log_prob + math.log(norm.pdf(self.traj[i][j][k], loc=r0[j][k], scale=std0[j][k]))
                        else:
                            raise("Other local distributions are not implemented yet!")
                self.dprob[i] = math.exp(log_prob)

        # Computing Q
        self.Q = torch.matmul(self.PHI, torch.diag(self.dprob).type(torch.float64))
        self.Q = torch.matmul(self.Q, self.PHI_daga)
    
    def check_independence(self):
        ## CHECKING THE INDEPENDENCE OF P ##
        self.rank = torch.linalg.matrix_rank(self.PHI)
        if self.PHI is not None:
            if self.rank < self.n_traj:
                print("Columns of PHI are not independent!!")
            else:
                print("Columns of PHI are independent!!")
        else:
            print("No matrix PHI is found")


# About 1050 rows (formulae) are enough to have 1000 independent columns