import torch
import numpy as np
from matplotlib import pyplot as plt
import math
from kernel import StlKernel, KernelRegression, GramMatrix
from phis_generator import StlGenerator
from traj_measure import BaseMeasure, Brownian, Gaussian, SemiBrownian
import sys

class local_matrix:
    """
    Class used to convert a global kernel into a local one
    """


    def __init__(self, prob_unbound_time_operator = 0.1, # Parameters for the sampling of formulae
                 leaf_probability = 0.5, 
                 time_bound_max_range = 10, 
                 atom_threshold_sd = 1.0,

                 formula_bag=None, # Can receive formula bag and formula generator if they are already computed
                 formula_generator = None,

                 n_vars = 2,  # Parameters for the local matrix Q
                 n_formulae = 1050, 
                 n_traj = 1000, 
                 n_traj_points: int = 11, 
                 evaluate_at_all_times = False,

                 PHI = None,  # Can receive PHI if it's already computed

                 target_distr = None,  # Target and proposal distributions (required)
                 proposal_distr = None, 
                 
                 proposal_traj = None,  # Can receive proposal trajectories is they're already computed
                 weight_strategy = "self_norm" # Defines which kind of function to apply to the weights. Can be: ["self_norm", "standard"]
                 ):
        
        # Parameter for the normalization of the weights
        self.weight_strategy = weight_strategy

        # stl generator parameters
        self.prob_unbound_time_operator = prob_unbound_time_operator  # probability of a temporal operator to have a time bound o the type [0,infty]
        self.leaf_probability = leaf_probability  # probability of generating a leaf (always zero for root)
        self.time_bound_max_range = time_bound_max_range  # maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        self.atom_threshold_sd = atom_threshold_sd  # std for the normal distribution of the thresholds of atoms
        self.formula_bag = formula_bag # Formulae already sampled
        self.formula_generator = formula_generator # Generator of the stl formulae

        # Target and proposal distributions
        self.target_distr = target_distr
        self.proposal_distr = proposal_distr
        #self.target_distr_name = self.target_distr.name
        #self.proposal_distr_name = self.proposal_distr.name

        # Trajectories from proposal distribution
        self.proposal_traj = proposal_traj

        # local matrix parameters
        self.n_vars = n_vars # number of variables in the trajectories
        self.n_formulae = n_formulae # number of formulae used to create the matrix P
        self.n_traj = n_traj # number of trajectories used to create the matrix P
        self.n_traj_points: int = n_traj_points # number of points in a trajectory
        # NOTE: n_traj_points must be >= max_timespan in phis_generator
        # NOTE: max_timespan must be greater than 11 (for some reason)
        self.evaluate_at_all_times = evaluate_at_all_times

        # device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Important matrices and values
        self.PHI = PHI
        self.PHI_daga = None
        self.dweights = None
        self.norm_factor = None
        self.Q = None
        self.rank = None
        self.pinv_error = None
        self.true_dweights = None
        self.sum_weights = None
        self.sum_squared_weights = None

    def init_formula_generator(self):
        self.formula_generator = StlGenerator(leaf_prob=self.leaf_probability, 
                               time_bound_max_range=self.time_bound_max_range,
                               unbound_prob=self.prob_unbound_time_operator, 
                               threshold_sd=self.atom_threshold_sd)

    def generate_formulae(self):
        if self.formula_generator is None:
            self.init_formula_generator()

        self.formula_bag = self.formula_generator.bag_sample(self.n_formulae, 
                                      self.n_vars)

    def generate_proposal_traj(self):
        self.proposal_traj = self.proposal_distr.sample(samples=self.n_traj, 
                            varn=self.n_vars, 
                            points=self.n_traj_points)
      
        
    def generate_PHI(self):
        # Computing PHI
        if self.formula_bag is None:
            self.generate_formulae()
        if self.proposal_traj is None:
            self.generate_proposal_traj()

        self.PHI = torch.empty([self.n_formulae, self.n_traj], dtype=torch.float64, device=self.device)
        for i in range(self.n_formulae):
            self.PHI[i, :] = torch.tanh(self.formula_bag[i].quantitative(self.proposal_traj, evaluate_at_all_times=self.evaluate_at_all_times))
        

    def compute_dweights(self):
        # Computing dweights (weights on the diagonal of D)
        self.dweights = torch.empty(self.n_traj, device=self.device, dtype=torch.float64)

        # Sample the proposal trajectories if they're not given (NO! we want to specify the correct proposal_traj)
        if self.proposal_traj is None:
            raise RuntimeError("No proposal_traj found!!")

        if self.evaluate_at_all_times:
            raise RuntimeError("Evaluation on all time points is not built yet")  # TODO: Implementa
        else:
            # To have a more stable code, we add the log of the probabilities and then do the exponentiation
            for i in range(self.n_traj):
                log_prob = 0
                target_log_error = False # This signals that the target probability is zero
                proposal_log_error = False # This signals that the proposal probability is zero

                # Computing the log probability of the target distribution
                target_log_prob, target_log_error =  self.target_distr.compute_pdf_trajectory(trajectory=self.proposal_traj[i, :, :].unsqueeze(0), log=True)
                log_prob += target_log_prob.item()

                if self.weight_strategy != "only_target":
                    # Computing the log probability of the proposal distribution
                    proposal_log_prob, proposal_log_error = self.proposal_distr.compute_pdf_trajectory(trajectory=self.proposal_traj[i, :, :].unsqueeze(0), log=True)
                    log_prob -= proposal_log_prob.item()
                
                # Handling the possible errors
                if proposal_log_error:
                    #print("##proposal_log_error##")
                    self.dweights = torch.zeros(self.n_traj, device=self.device)
                    # This is the case when the traj is too extreme for the proposal distr and so we get a division by zero
                    # TODO: rivedi meglio che succede se la prob globale è zero
                    break
                elif target_log_error:
                    self.dweights[i] = 0
                    # This is the case where the traj is too extreme for the target distrib and we obtain a simple multiplication by zero
                else:
                    try:
                        self.dweights[i] = math.exp(log_prob) # standard case
                    except OverflowError:
                        print(f"Overflow error: log_prob = {log_prob}, target_log_prob = {target_log_prob.item()}, proposal_log_prob = {proposal_log_prob.item()}")

            self.true_dweights = self.dweights.clone() # Saving a copy of the unnormalized weights for later
            self.sum_weights = max(torch.sum(self.dweights).item(), torch.finfo(self.dweights.dtype).tiny) # Finding the sum of the weights (clipping it at the minimum float value)
            self.sum_squared_weights = torch.sum(torch.square(self.dweights)).item()
            match self.weight_strategy:
                case "self_norm":
                    self.dweights /= self.sum_weights
                    self.dweights *= self.n_traj
                case "standard":
                    pass
                case "square_root":
                    self.dweights = torch.sqrt(self.dweights)
                    self.dweights /= self.sum_weights
                    self.dweights *= self.n_traj
                case "only_target":
                    self.dweights /= self.sum_weights
                    self.dweights *= self.n_traj
                case _: # The default case will be the self normalization
                    self.dweights /= self.sum_weights
                    self.dweights *= self.n_traj
            
            #n_e = self.sum_weights**2/self.sum_squared_weights
            #print(f"\n #Inside the local_matrix class#")
            #print(f"The sum of the weights is: {self.sum_weights}")
            #print(f"The sum of squares of the weights is: {self.sum_squared_weights}")
            #print(f"n_e is: {n_e}")
            #print(f"n/n_e = {self.n_traj / n_e}")
            

    def compute_Q(self, proposal_traj=None, PHI=None, dweights=None):

        # Generating the trajectories if not given
        if proposal_traj is not None:
            self.proposal_traj = proposal_traj
        if self.proposal_traj is None:
            print("##There's no proposal_traj!!##")
            self.generate_proposal_traj()

        # Generating PHI/PHI_daga if not given
        if PHI is not None:
            self.PHI = PHI#.type(torch.float64) # Evrything goes bad if I convert here to float64 (Perchè la pseudoinversa più precisa funziona male)
        if self.PHI is None:
            print("##There's no PHI!!##")
            self.generate_PHI()
        try:
            self.PHI_daga = torch.linalg.pinv(self.PHI)#, atol=torch.finfo(self.PHI.dtype).tiny)#.type(torch.float64) 
        except RuntimeError:
            print("## PROBLEM WITH THE PSEUDO INVERSE ##")
            # TODO: Fix the problem with mkl here (Intel MKL ERROR: Parameter 12 was incorrect on entry to SGESDD.)
            return True

        # Computing the error of the pseudo inverse
        self.pinv_error = self.check_goodness_pinv()
    
        # Computing dweights if not given
        if dweights is not None:
            self.dweights = dweights
        if self.dweights is None:
            self.compute_dweights()

        M = self.PHI.type(torch.float64) * self.dweights.type(torch.float64).unsqueeze(0)
        #M = torch.matmul(self.PHI.type(torch.float64), torch.diag(self.dweights).type(torch.float64)) # Old version
        self.Q = torch.matmul(M, self.PHI_daga.type(torch.float64))
    
    def check_independence(self):
        # Checks the independence of PHI
        if self.PHI is not None:
            self.rank = torch.linalg.matrix_rank(self.PHI)
            if self.rank < self.n_traj:
                print("Columns of PHI are not independent!!")
                return False
            else:
                print("Columns of PHI are independent!!")
                return True
        else:
            raise RuntimeError("No matrix PHI is found")
        
    def check_goodness_pinv(self):
        # Checks how good is the pseudoinverse of the matrix PHI
        I_M = torch.eye(self.n_traj) # Identity matrix
        # Computing the distance between the product PHI_daga*PHI and the identity matrix
        error = torch.norm(torch.matmul(self.PHI_daga, self.PHI) - I_M)
        return error.item()

    def convert_to_local(self, global_kernel):
        if self.Q is None:
            raise RuntimeError("No matrix Q is found")
        return torch.matmul(self.Q, global_kernel.type(torch.float64))


# About 1050 rows (formulae) are enough to have 1000 independent columns
