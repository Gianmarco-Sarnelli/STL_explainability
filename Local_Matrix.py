import torch
import numpy as np
from matplotlib import pyplot as plt
import math
from kernel import StlKernel, KernelRegression, GramMatrix
from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from scipy.stats import norm

class local_matrix:
    """
    Class to convert a global kernel into a local one
    It can reiceve in input the parameters local_traj, global_traj, std_local, std_global, formula_bag, PHI
    (if already computed) or it can generate them in a standard way 
    """


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
                 evaluate_at_all_times = False,
                 std_local = 10,
                 std_global = 10,
                 global_traj = None,
                 local_traj = None,
                 formula_bag=None,
                 PHI = None ):
        
        self.local_distr = "Brownian"  # The local distribution is considered as a brownian motion around the local trajectory
        self.global_distr = "mu0"    # NOTE: for now these are the only available distributions

        self.normalize_weights = True # The diagonal elements of D will be normalized (divided by the sum of the weights)

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
        self.std_local = std_local
        self.std_global = std_global
        self.n_formulae = n_formulae # number of formulae used to create the matrix P
        self.n_traj = n_traj # number of trajectories used to create the matrix P
        self.n_traj_points: int = n_traj_points # number of points in a trajectory
        # NOTE: n_traj_points must be >= max_timespan in phis_generator
        # NOTE: max_timespan must be greater than 11 (for some reason)
        self.evaluate_at_all_times = evaluate_at_all_times

        # device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # formulae and trajectories
        self.formula_bag = formula_bag
        self.local_traj = local_traj # trajectory in the center of the local distribution
        self.global_traj = global_traj # trajectories used to compute PHI

        # Important matrices and values
        self.PHI = PHI
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
        self.formula_bag = formula.bag_sample(self.n_formulae, 
                                      self.n_vars)

    def generate_global_traj(self):
        if self.global_distr == "mu0": # Then dividing by the global distribution
            mu0 = BaseMeasure(device=self.device, 
                            sigma0=self.initial_std, 
                            sigma1=self.total_var_std)
            self.global_traj = mu0.sample(samples=self.n_traj, 
                                varn=self.n_vars, 
                                points=self.n_traj_points)
        else:
            raise("Other global distributions are not implemented yet!")
        
    def generate_PHI(self):
        # Computing PHI and PHI_daga
        if self.formula_bag is None:
            self.generate_formulae()
        if self.global_traj is None:
            self.generate_global_traj()

        self.PHI = torch.empty([self.n_formulae, self.n_traj], dtype=torch.float64, device=self.device)
        for i in range(self.n_formulae):
            quant = self.formula_bag[i].quantitative(self.global_traj, evaluate_at_all_times=self.evaluate_at_all_times)
            # quantitative returns the robustness of the formula for a certain sample
            # quantitative needs in input a tensor x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            for j in range(self.n_traj):
                self.PHI[i][j] = quant[j].item()

    

    def compute_dprob(self):
        # Computing dprob (probability values on the diagonal of D)
        self.dprob = torch.empty(self.n_traj, device=self.device)

        if self.evaluate_at_all_times:
            raise("The normalization constant for the evaluation on all time points is not built yet")  # TODO: Implementa
        
        else:
            # To have a more stable code, we add the log of the probabilities and then do the exponentiation
            for i in range(self.n_traj):
                log_prob = 0
                local_log_error = False # This signals that the local probability is zero
                global_log_error = False # This signals that the global probability is zero

                if self.local_distr == "Brownian": # Multiplying first by the local distribution
                    for k in range(self.n_traj_points):
                        for j in range(self.n_vars):
                            try:
                                log_prob +=  math.log(norm.pdf(self.global_traj[i][j][k], loc=self.local_traj[j][k], scale=math.sqrt(k+1)*self.std_local))
                            except ValueError:
                                local_log_error = True
                else:
                            raise("Other local distributions are not implemented yet!")
                
                if self.global_distr == "mu0": # Then dividing by the global distribution
                    # computing increments
                    increments = self.global_traj[i, :, 1:] - self.global_traj[i, :, :-1]
                    # summing the absolute values of the increments
                    var_sum = torch.sum(torch.abs(increments), dim=1)
                    for j in range(self.n_vars):
                        try:
                            log_prob -= math.log(norm.pdf(self.global_traj[i][j][0], loc=0, scale=self.initial_std)) # probability density of the first point
                            log_prob -= math.log(2 * norm.pdf(math.sqrt(var_sum[j]), loc=0, scale=self.total_var_std)) # probability density of the total variation
                            log_prob += (self.n_traj_points-1)*math.log(2) # each trajectory has one of 2^n possible combinations of direction of increments
                        except ValueError:
                            global_log_error = True
                else:
                            raise("Other global distributions are not implemented yet!")

                if global_log_error:
                    print("##global_log_error##")
                    self.dprob = torch.zeros(self.n_traj, device=self.device)
                    # This is the case when the traj is too extreme for mu0 and so we get a division by zero
                    # TODO: rivedi meglio che succede se la prob globale è zero
                    break
                elif local_log_error:
                    self.dprob[i] = 0
                    # This is the case where the traj is too extreme for the local distrib and we obtain a simple multiplication by zero
                else:
                    self.dprob[i] = math.exp(log_prob) # standard case
            if self.normalize_weights:
                sum_weights = torch.sum(self.dprob)
                self.dprob /= sum_weights
                print(f"The sum of the weights is: {sum_weights}")


    def compute_Q(self, local_traj=None, global_traj = None, std_local=None, initial_std=None, total_var_std=None, formula_bag=None, PHI=None ):
        # Parameters of the local explanation (computed in a stadard way or with a given value)
        if local_traj is not None:
            self.local_traj = local_traj
        if self.local_traj is None:
            self.local_traj = torch.zeros([self.n_vars, self.n_traj_points], dtype=torch.float64, device=self.device) # Center of the local trajectories (for now all zeros)
        if std_local is not None:
            self.std_local = std_local # std of the local trajectory (no covariance matrix for now)
        if initial_std is not None:
            self.initial_std = initial_std
        if total_var_std is not None:
            self.total_var_std = total_var_std
        
        # Generating the trajectories if not given
        if global_traj is not None:
            self.global_traj = global_traj
        if self.global_traj is None:
            self.generate_global_traj()

        # Generating the formulae if not given
        if formula_bag is not None:
            self.formula_bag = formula_bag
        if self.formula_bag is None:
            self.generate_formulae()

        # Generating PHI/PHI_daga if not given
        if PHI is not None:
            self.PHI = PHI
        if self.PHI is None:
            self.generate_PHI()
        self.PHI_daga = torch.linalg.pinv(self.PHI)
    
        # Computing dprob (probability values on the diagonal of D)
        self.compute_dprob()

        # Computing Q # TODO: scegli quale dtype usare
        M = torch.matmul(self.PHI.type(torch.float64), torch.diag(self.dprob).type(torch.float64))
        self.Q = torch.matmul(M, self.PHI_daga.type(torch.float64))
    
    def check_independence(self):
        ## CHECKING THE INDEPENDENCE OF P ##
        if self.PHI is not None:
            self.rank = torch.linalg.matrix_rank(self.PHI)
            if self.rank < self.n_traj:
                #print("Columns of PHI are not independent!!")
                return False
            else:
                #print("Columns of PHI are independent!!")
                return True
        else:
            print("No matrix PHI is found")

    def convert_to_local(self, global_kernel):
        if self.Q is None:
            print("No matrix Q is found, computing it with given settings")
            self.compute_Q()
        return torch.matmul(self.Q, global_kernel.type(torch.float64))


# About 1050 rows (formulae) are enough to have 1000 independent columns