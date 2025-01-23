#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================


import torch
import copy
import math
from scipy.stats import norm
import sys

# TODO: reason about density of sampling etc (both here and in formulae generation)

class Measure:
    def sample(self, samples=100000, varn=2, points=100):
        # Must be overridden
        pass


class BaseMeasure(Measure):
    def __init__(
        self, mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.02, q0=0.5, device="cpu", density=1,
    ):
        """

        Parameters
        ----------
        mu0 : mean of normal distribution of initial state, optional
            The default is 0.0.
        sigma0 : standard deviation of normal distribution of initial state, optional
            The default is 1.0.
        mu1 : DOUBLE, optional
            mean of normal distribution of total variation. The default is 0.0.
        sigma1 : standard deviation of normal distribution of total variation, optional
            The default is 1.0.
        q : DOUBLE, optional
            probability of change of sign in derivative. The default is 0.02.
        q0 : DOUBLE, optional
            probability of initial sign of  derivative. The default is 0.5.
        device : 'cpu' or 'cuda', optional
            device on which to run the algorithm. The default is 'cpu'..
        density : INT, optional
            desity-1 is the number of points to be added (on a line) between the
            non-dense points of the trajectory

        Returns
        -------
        None.

        """
        self.name = "BaseMeasure"
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.q = q
        self.q0 = q0
        self.device = device
        self.density = density

    def sample(self, samples=100000, varn=2, points=100):
        """
        Samples a set of trajectories from the basic measure space, with parameters
        passed to the sampler

        Parameters
        ----------
        points : INT, optional
            number of points per trajectory, including initial one. The default is 1000.
        samples : INT, optional
            number of trajectories. The default is 100000.
        varn : INT, optional
            number of variables per trajectory. The default is 2.


        Returns
        -------
        signal : samples x varn x points pytorch tensor
            The sampled signals.

        """
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available!")

        # generate unif RN
        signal = torch.rand(samples, varn, points, device=self.device)
        # first point is special - set to zero for the moment, and set one point to 1
        signal[:, :, 0] = 0.0
        signal[:, :, -1] = 1.0
        # sorting each trajectory
        signal, _ = torch.sort(signal, 2)
        # computing increments and storing them in points 1 to end
        signal[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
        # generate initial state, according to a normal distribution
        signal[:, :, 0] = self.mu0 + self.sigma0 * torch.randn(signal[:, :, 0].size())

        # sampling change signs from bernoulli in -1, 1
        derivs = (1 - self.q) * torch.ones(samples, varn, points, device=self.device)
        derivs = 2 * torch.bernoulli(derivs) - 1
        # sampling initial derivative
        derivs[:, :, 0] = self.q0
        derivs[:, :, 0] = 2 * torch.bernoulli(derivs[:, :, 0]) - 1
        # taking the cumulative product along axis 2
        derivs = torch.cumprod(derivs, 2)

        # sampling total variation
        totvar = torch.pow(
            self.mu1 + self.sigma1 * torch.randn(samples, varn, 1, device=self.device),
            2,
         )
        # multiplying total variation and derivatives and making initial point non-invasive
        derivs = derivs * totvar
        derivs[:, :, 0] = 1.0

        # computing trajectories by multiplying and then doing a cumulative sum
        signal = signal * derivs
        signal = torch.cumsum(signal, 2)
        # add point density
        if self.density > 1:
            dense_signal = torch.zeros(samples, varn, (points - 1) * self.density + 1)
            dense_signal[:, :, ::self.density] = signal
            diff = dense_signal[:, :, self.density::self.density] - dense_signal[:, :, 0:-self.density:self.density]
            for i in range(self.density - 1):
                dense_signal[:, :, i + 1::self.density] = dense_signal[:, :, 0:-self.density:self.density] + \
                                                          (diff / self.density) * (i + 1)
            signal = copy.deepcopy(dense_signal)
        return signal
    
    def compute_pdf_trajectory(self, trajectory: torch.Tensor,
                               log: bool = False) -> torch.Tensor:
        """
        Computes the probability density function of a trajectory sampled using mu0.
        The computations are done in log space for numerical stability

        Parameters:
        ----------
        trajectory (torch.Tensor): The trajectory tensor of shape [samples, varn, points]
        log (bool): decides if the output will be the log of the pdf or the pdf itself
        
        Returns:
        --------
        log_pdf(torch.Tensor): The tensor containing the log of the pdf of the trajectories
        pdf (torch.Tensor): The tensor containing the pdf of the trajectories

        """
        log_error = False

        # Getting the shape of the trajectory
        samples, varn, points = trajectory.shape

        log_pdf = torch.empty(samples, device=self.device, dtype=torch.float64)
        for i in range(samples):
            # Computing increments
            increments = trajectory[i, :, 1:] - trajectory[i, :, :-1]

            # Computing the total variation
            totvar = torch.sum(torch.abs(increments), dim=1)

            try:
                # Computing the probability density function of the first point
                initial_pdf = torch.distributions.Normal(self.mu0, self.sigma0).log_prob(trajectory[i, :, 0])

                # The probability density function of the total variation is 
                # prob(totvar) = (normal_prob(sqrt(totvar)) + normal_prob(-sqrt(totvar))) / ( 2 * sqrt(totvar) )
                sqroot_totvar = torch.sqrt(totvar)
                sum_normal = torch.distributions.Normal(self.mu1, self.sigma1).log_prob(sqroot_totvar).exp() + torch.distributions.Normal(self.mu1, self.sigma1).log_prob(-sqroot_totvar).exp()
                totvar_pdf = torch.log(sum_normal / (2 * sqroot_totvar) )

                # Computing the Bernoulli probabilities of the change of sign in the increments
                change_sign = increments[:, :-1] * increments[:, 1:]
                bernoulli_pdf = math.log(self.q) * (change_sign < 0) + math.log(1 - self.q) * (change_sign >= 0)

                # The sign of the derivative at the first point is positive with probability: p = ( q0*(1-q) + (1-q0)*q )
                # This is because we consider the case where the derivative is positive and stays positive plus the case where it changes
                p = self.q0*(1-self.q) + (1-self.q0)*self.q
                initial_bernoulli_pdf = math.log(p)*(increments[:, 0] >= 0) + math.log(1-p)*(increments[:, 0] < 0)

                # The probability distribution of the uniform spacings is (n-2)! where n is the number of points in the
                # trajectory. This is the same pdf of the ordered vactor of n-2 points sampled from the uniform distribution.
                # In pytorch we can compute log(n+1) = lgamma(n)
                uniform_spacing_pdf = torch.ones(varn) * torch.lgamma(torch.tensor(points - 1, dtype=torch.float))

                # Computing the log of the jacobian term
                log_jacobian = torch.log(totvar) * (2-points)

                # Computing the log pdf
                log_pdf[i] = torch.sum(initial_pdf)
                log_pdf[i] += torch.sum(totvar_pdf)
                log_pdf[i] += torch.sum(bernoulli_pdf)
                log_pdf[i] += torch.sum(initial_bernoulli_pdf)
                log_pdf[i] += torch.sum(uniform_spacing_pdf)
                log_pdf[i] += torch.sum(log_jacobian)

            except ValueError: # If there's a value error then I considere that the log prob is too low
                log_error = True

        if log:    
            return (log_pdf, log_error)
        if not log:
            return (torch.exp(log_pdf), log_error)
        
    def compute_pdf_trajectory_old(self, trajectory: torch.Tensor,
                               log: bool = False) -> tuple[torch.Tensor, bool]:
        """
        Old version of the pdf
        """
        # Getting the shape of the trajectory
        samples, varn, points = trajectory.shape
        log_error = False
        log_pdf = torch.zeros(samples, device=self.device, dtype=torch.float64)

        for i in range(samples):
            # computing increments
            increments = trajectory[i, :, 1:] - trajectory[i, :, :-1]
            # summing the absolute values of the increments
            var_sum = torch.sum(torch.abs(increments), dim=1)
            for j in range(varn):
                try:
                    log_pdf[i] += math.log(norm.pdf(trajectory[i][j][0], loc=0, scale=self.sigma0)) # probability density of the first point
                    sqrt_var_sum = max(math.sqrt(var_sum[j]), sys.float_info.min) # Computing the sqrt of the variance and clipping it to not be zero
                    log_pdf[i] += math.log(norm.pdf(sqrt_var_sum, loc=0, scale=self.sigma1) / sqrt_var_sum) # probability density of the total variation
                    log_pdf[i] -= (points-1)*math.log(2) # each trajectory has one of 2^n possible combinations of direction of increments
                except ValueError:
                    log_error = True
        
        if log:    
            return (log_pdf, log_error)
        if not log:
            return (torch.exp(log_pdf), log_error)

        

class LocalBrownian(Measure):
    def __init__(
        self, base_traj, std = 1.0, device="cpu"
    ):
        """

        Parameters
        ----------
        base_traj (torch.Tensor): A tensor of shape [varn, points]
            - `varn` represents the number of variables in each trajectory.
            - `points` represents the number points for each trajectory.
        std : standard deviation of normal around the base_traj, optional
            The default is 1.0.
        device : 'cpu' or 'cuda', optional
            device on which to run the algorithm. The default is 'cpu'
        
        Returns
        -------
        None.

        """
        self.name = "LocalBrownian"
        self.device = device
        self.base_traj = base_traj
        self.std = std
        if base_traj.dim() != 2:
            raise ValueError(f"`base_traj` must have 2 dimensions, but got {base_traj.dim()} dimensions.")
        self.base_traj_varn = base_traj.shape[0]     
        self.base_traj_points = base_traj.shape[1]

    def sample(self, samples=100000, varn=2, points=100):
        """
        Samples a set of trajectories around a base trajectory, with parameters
        passed to the sampler

        Parameters
        ----------
        points : INT, optional
            number of points per trajectory, including initial one. The default is 100.
        samples : INT, optional
            number of trajectories. The default is 100000.
        varn : INT, optional
            number of variables per trajectory. The default is 2.


        Returns
        -------
        signal : samples x varn x points pytorch tensor
            The sampled signals.

        """
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available!")

        # generate normal distr of points
        noise = self.std*torch.cumsum(torch.randn(samples, varn, points, device=self.device), 2)
        # Adding the noise to the base traectory
        signal = noise + self.base_traj
        
        return signal
    
    def compute_pdf_trajectory(self, trajectory: torch.Tensor, 
                               log: bool = False) -> torch.Tensor:
        """
        Computes the probability density function of a trajectory sampled using local brownian.
        The computations are done in log space for numerical stability

        Parameters:
        ----------
        trajectory (torch.Tensor): The trajectory tensor of shape [samples, varn, points]
        log (bool): decides if the output will be the log of the pdf or the pdf itself
        
        Returns:
        --------
        log_pdf(torch.Tensor): The tensor containing the log of the pdf of the trajectories
        pdf (torch.Tensor): The tensor containing the pdf of the trajectories

        """
        log_error = False

        # Getting the shape of the trajectory
        samples, varn, points = trajectory.shape
        noise = trajectory - self.base_traj
        diff = noise[:, :, 1:] - noise[:, :, :-1]
        noise[:, :, 1:] = diff

        try:
            #all_log_pdf = torch.distributions.Normal(self.base_traj, self.std).log_prob(noise).type(torch.float64) # BIG MISTAKE!!!!!!!!!!!!!!!!!!!!
            all_log_pdf = torch.distributions.Normal(torch.zeros(samples,varn,points), self.std).log_prob(noise).type(torch.float64) 
            log_pdf = torch.sum(all_log_pdf, dim=(1,2))

        except ValueError: # If there's a value error then it means that the log prob is too low
            log_error = True
        
        if log:    
            return (log_pdf, log_error)
        if not log:
            return (torch.exp(log_pdf), log_error)


    def compute_pdf_trajectory_old(self, trajectory: torch.Tensor, 
                               log: bool = False) -> tuple[torch.Tensor, bool]:
        
        # Getting the shape of the trajectory
        samples, varn, points = trajectory.shape
        log_error = False
        log_pdf = torch.zeros(samples, device=self.device, dtype=torch.float64)
        for i in range(samples):
            for k in range(points):
                for j in range(varn):
                    try:
                        log_pdf[i] +=  math.log(norm.pdf(trajectory[i][j][k], loc=self.base_traj[j][k], scale=math.sqrt(k+1)*self.std))
                    except ValueError:
                        log_error = True

        if log:    
            return (log_pdf, log_error)
        if not log:
            return (torch.exp(log_pdf), log_error)



class LocalGaussian(Measure):
    def __init__(
        self, base_traj, std = 1.0, device="cpu"
    ):
        """
        Applies a gassuian shift to the base trajectory (equal shift at
        every point in time)

        Parameters
        ----------
        base_traj (torch.Tensor): A tensor of shape [varn, points]
            - `varn` represents the number of variables in each trajectory.
            - `points` represents the number points for each trajectory.
        std : standard deviation of normal around the base_traj, optional
            The default is 1.0.
        device : 'cpu' or 'cuda', optional
            device on which to run the algorithm. The default is 'cpu'
        
        Returns
        -------
        None.

        """
        self.name = "LocalGaussian"
        self.device = device
        self.base_traj = base_traj
        self.std = std
        if base_traj.dim() != 2:
            raise ValueError(f"`base_traj` must have 2 dimensions, but got {base_traj.dim()} dimensions.")
        self.base_traj_varn = base_traj.shape[0]     
        self.base_traj_points = base_traj.shape[1]

    def sample(self, samples=100000, varn=2, points=100):
        """
        Samples a set of trajectories around a base trajectory, with parameters
        passed to the sampler

        Parameters
        ----------
        points : INT, optional
            number of points per trajectory, including initial one. The default is 100.
        samples : INT, optional
            number of trajectories. The default is 100000.
        varn : INT, optional
            number of variables per trajectory. The default is 2.


        Returns
        -------
        signal : samples x varn x points pytorch tensor
            The sampled signals.

        """
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available!")

        # generate normal distr of points
        noise = self.std*torch.randn(samples, varn, device=self.device).unsqueeze(-1)
        # Adding the noise to the base trajectory
        signal = self.base_traj + noise
        
        return signal
    
    def compute_pdf_trajectory(self, trajectory: torch.Tensor, 
                               log: bool = False) -> torch.Tensor:
        """
        Computes the probability density function of a trajectory sampled using local brownian.
        
        Parameters:
        ----------
        trajectory (torch.Tensor): The trajectory tensor of shape [samples, varn, points]
        log (bool): decides if the output will be the log of the pdf or the pdf itself
        
        Returns:
        --------
        log_pdf(torch.Tensor): The tensor containing the log of the pdf of the trajectories
        pdf (torch.Tensor): The tensor containing the pdf of the trajectories

        """
        log_error = False

        # Computing the shape of the trajectory
        samples, varn, points = trajectory.shape

        # Computing the noise (at the first trajectory point)
        noise = trajectory[:,:,0] - self.base_traj[:,:,0]

        try:
            log_pdf = torch.distributions.Normal(torch.zeros(samples, varn), self.std).log_prob(noise).type(torch.float64)
        except ValueError: # If there's a value error then it means that the log prob is too low
            log_error = True
        
        if log:    
            return (log_pdf, log_error)
        if not log:
            return (torch.exp(log_pdf), log_error)
