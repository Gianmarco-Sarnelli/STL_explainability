#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""Tools to generate (via sampling) STL logical formulae"""

from typing import Union
import torch
import stl
from stl import Node
import numpy.random as rnd
import copy


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

        Returns
        -------
        None.

        """
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
        signal : samples x varn x points double pytorch tensor
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
        # multiplying total variation and derivatives an making initial point non-invasive
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


# TODO: add density of both traj points and temporal operators
class StlGenerator:
    def __init__(
        self,
        leaf_prob=0.3,
        inner_node_prob=None,
        threshold_mean=0.0,
        threshold_sd=1.0,
        unbound_prob=0.1,
        time_bound_max_range=20,
        adaptive_unbound_temporal_ops=True,
        time_density=1,
        max_timespan=100,
    ):
        """
        leaf_prob
            probability of generating a leaf (always zero for root)
        node_types = ["not", "and", "or", "always", "eventually", "until"]
            Inner node types
        inner_node_prob
            probability vector for the different types of internal nodes
        threshold_mean
        threshold_sd
            mean and std for the normal distribution of the thresholds of atoms
        unbound_prob
            probability of a temporal operator to have a time bound o the type [0,infty]
        time_bound_max_range
            maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        adaptive_unbound_temporal_ops
            if true, unbounded temporal operators are computed from current point to the end of the signal, otherwise
            they are evaluated only at time zero.
        max_timespan
            maximum time depth of a formula.
        """

        # Address the mutability of default arguments
        if inner_node_prob is None:
            inner_node_prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.threshold_mean = threshold_mean
        self.threshold_sd = threshold_sd
        self.unbound_prob = unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.node_types = ["not", "and", "or", "always", "eventually", "until"]
        self.max_timespan = max_timespan
        self.time_density = time_density

    def sample(self, nvars):
        """
        Samples a random formula with distribution defined in class instance parameters

        Parameters
        ----------
        nvars : number of variables of input signals
            how many variables the formula is expected to consider.

        Returns
        -------
        TYPE
            A random formula.

        """
        return self._sample_internal_node(nvars)

    def bag_sample(self, bag_size, nvars):
        """
        Samples a bag of bag_size formulae

        Parameters
        ----------
        bag_size : INT
            number of formulae.
        nvars : INT
            number of vars in formulae.

        Returns
        -------
        a list of formulae.

        """
        formulae = []
        for _ in range(bag_size):
            phi = self.sample(nvars)
            formulae.append(phi)
        return formulae

    def _sample_internal_node(self, nvars):
        # Declare & dummy-assing "idiom"
        node: Union[None, Node]
        node = None
        # choose node type
        nodetype = rnd.choice(self.node_types, p=self.inner_node_prob)
        while True:
            if nodetype == "not":
                n = self._sample_node(nvars)
                node = stl.Not(n)
            elif nodetype == "and":
                n1 = self._sample_node(nvars)
                n2 = self._sample_node(nvars)
                node = stl.And(n1, n2)
            elif nodetype == "or":
                n1 = self._sample_node(nvars)
                n2 = self._sample_node(nvars)
                node = stl.Or(n1, n2)
            elif nodetype == "always":
                n = self._sample_node(nvars)
                unbound, time_bound = self._get_temporal_parameters()
                node = stl.Globally(
                    n, unbound, self.time_density*time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "eventually":
                n = self._sample_node(nvars)
                unbound, time_bound = self._get_temporal_parameters()
                node = stl.Eventually(
                    n, unbound, self.time_density*time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "until":
                raise NotImplementedError(
                    "Node for STL 'Until' operator not yet implemented!"
                )

            if (node is not None) and (node.time_depth() < self.max_timespan):
                return node

    def _sample_node(self, nvars):
        if rnd.rand() < self.leaf_prob:
            # sample a leaf
            var, thr, lte = self._get_atom(nvars)
            return stl.Atom(var, thr, lte)
        else:
            return self._sample_internal_node(nvars)

    def _get_temporal_parameters(self):
        if rnd.rand() < self.unbound_prob:
            return True, 0
        else:
            return False, rnd.randint(self.time_bound_max_range) + 1

    def _get_atom(self, nvars):
        variable = rnd.randint(nvars)
        lte = rnd.rand() > 0.5
        threshold = rnd.normal(self.threshold_mean, self.threshold_sd)
        return variable, threshold, lte


class StlUnifGenerator:
    def __init__(
        self,
        leaf_prob=0.3,
        inner_node_prob=None,
        threshold_bounds=None,
        unbound_prob=0.1,
        time_bound_max_range=20,
        adaptive_unbound_temporal_ops=True,
        max_timespan=100,
    ):
        """
        leaf_prob
            probability of generating a leaf (always zero for root)
        node_types = ["not", "and", "or", "always", "eventually", "until"]
            Inner node types
        inner_node_prob
            probability vector for the different tpes of internal nodes
        threshold_mean
        threshold_sd
            mean and std for the normal distribution of the thresholds of atoms
        unbound_prob
            probability of a temporal operator to have a time bound o the type [0,infty]
        time_bound_max_range
            maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        adaptive_unbound_temporal_ops
            if true, unbounded temporal operators are computed from current point to the end of the signal, otherwise
            they are evaluated only at time zero.
        max_timespan
            maximum time depth of a formula.


        """

        # Address the mutability of default arguments
        if inner_node_prob is None:
            inner_node_prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

        if threshold_bounds is None:
            threshold_bounds = [-3.0, 3.0]

        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.threshold_bounds = threshold_bounds
        self.unbound_prob = unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.node_types = ["not", "and", "or", "always", "eventually", "until"]
        self.max_timespan = max_timespan

    def sample(self, nvars):
        """
        Samples a random formula with distribution defined in class instance parameters

        Parameters
        ----------
        nvars : number of variables of input signals
            how many variables the formula is expected to consider.

        Returns
        -------
        TYPE
            A random formula.

        """
        return self._sample_internal_node(nvars)

    def bag_sample(self, bag_size, nvars):
        """
        Samples a bag of bag_size formulae

        Parameters
        ----------
        bag_size : INT
            number of formulae.
        nvars : INT
            number of vars in formulae.

        Returns
        -------
        a list of formulae.

        """
        formulae = []
        for _ in range(bag_size):
            phi = self.sample(nvars)
            formulae.append(phi)
        return formulae

    def _sample_internal_node(self, nvars):
        # Declare & dummy-assing "idiom"
        node: Union[None, Node]
        node = None
        # choose node type
        nodetype = rnd.choice(self.node_types, p=self.inner_node_prob)
        while True:
            if nodetype == "not":
                n = self._sample_node(nvars)
                node = stl.Not(n)
            elif nodetype == "and":
                n1 = self._sample_node(nvars)
                n2 = self._sample_node(nvars)
                node = stl.And(n1, n2)
            elif nodetype == "or":
                n1 = self._sample_node(nvars)
                n2 = self._sample_node(nvars)
                node = stl.Or(n1, n2)
            elif nodetype == "always":
                n = self._sample_node(nvars)
                unbound, time_bound = self._get_temporal_parameters()
                node = stl.Globally(
                    n, unbound, time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "eventually":
                n = self._sample_node(nvars)
                unbound, time_bound = self._get_temporal_parameters()
                node = stl.Eventually(
                    n, unbound, time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "until":
                raise NotImplementedError(
                    "Node for STL 'Until' operator not yet implemented!"
                )

            if (node is not None) and (node.time_depth() < self.max_timespan):
                return node

    def _sample_node(self, nvars):
        if rnd.rand() < self.leaf_prob:
            # sample a leaf
            var, thr, lte = self._get_atom(nvars)
            return stl.Atom(var, thr, lte)
        else:
            return self._sample_internal_node(nvars)

    def _get_temporal_parameters(self):
        if rnd.rand() < self.unbound_prob:
            return True, 0
        else:
            return False, rnd.randint(self.time_bound_max_range) + 1

    def _get_atom(self, nvars):
        variable = rnd.randint(nvars)
        lte = rnd.rand() > 0.5
        threshold = rnd.uniform(self.threshold_bounds[0], self.threshold_bounds[1])
        return variable, threshold, lte
