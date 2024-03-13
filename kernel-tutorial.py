#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pickle

from kernel import StlKernel, KernelRegression, GramMatrix
from generator import BaseMeasure, StlGenerator


def dump_pickle(name, thing):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(thing, f)

# device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {device}".format(device=device))

## STL FORMULAE GENERATOR ##
# stl generator parameters
prob_unbound_time_operator = 0.1  # probability of a temporal operator to have a time bound o the type [0,infty]
leaf_probability = 0.5  # probability of generating a leaf (always zero for root)
time_bound_max_range = 10  # maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
atom_threshold_sd = 1.0  # std for the normal distribution of the thresholds of atoms

sampler = StlGenerator(leaf_prob=leaf_probability, time_bound_max_range=time_bound_max_range,
                       unbound_prob=prob_unbound_time_operator, threshold_sd=atom_threshold_sd)

## TRAJECTORY GENERATOR ##
# trajectory sample parameters
initial_std = 1.0  # standard deviation of normal distribution of initial state
total_var_std = 1.0  # standard deviation of normal distribution of total variation

mu0 = BaseMeasure(device=device, sigma0=initial_std, sigma1=total_var_std)

## EXPERIMENTAL PARAMETERS ##
# experiment parameters and dataset generation
n_experiments = 50
n_trajectories = 1000  # number of trajectories for average robustness / satisfaction
resample_formulae = True  # whether to run independent experiments
norm_rob = [True, False]  # whether to use normalized robustness or standard robustness
train_size = 1000  # number of formulae in the training set
val_size = 250  # number of formulae in the validation set
test_size = 1000  # number of formulae in the test set
n_vars = 3  # max number of variables allowed in formulae
n_traj_points = 100  # number of points in each trajectory
# cv parameters
alpha_min = -3
alpha_max = 1
cv_steps = 17
ridge_alpha_min = -3
ridge_alpha_max = 1
ridge_cv_steps = 10

boolean = True  # kernel with qualitative semantics also

# parameters for statistical tables
quantiles = torch.tensor([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
# containers to store results and statistics
kreg_MRE, kreg_MSE, kreg_MAE, kreg_MRE_norm, kreg_MSE_norm, kreg_MAE_norm = \
    [torch.zeros(n_experiments) for _ in range(6)]
kreg_acc, kreg_acc_norm = [torch.zeros(n_experiments) for _ in range(2)]
kreg_stat_MRE, kreg_stat_MAE, kreg_stat_MSE, kreg_stat_acc, kreg_stat_MRE_norm, kreg_stat_MAE_norm, \
        kreg_stat_MSE_norm, kreg_stat_acc_norm = [torch.zeros((n_experiments, len(quantiles))) for _ in range(8)]
if boolean:
    bool_MRE, bool_MSE, bool_MAE, bool_MRE_norm, bool_MSE_norm, bool_MAE_norm = \
        [torch.zeros(n_experiments) for _ in range(6)]
    bool_acc, bool_acc_norm = [torch.zeros(n_experiments) for _ in range(2)]
    bool_stat_MRE, bool_stat_MAE, bool_stat_MSE, bool_stat_acc, bool_stat_MRE_norm, bool_stat_MAE_norm, \
    bool_stat_MSE_norm, bool_stat_acc_norm = [torch.zeros((n_experiments, len(quantiles))) for _ in range(8)]
for i in range(n_experiments):
    print(i)
    # test trajectories of current experiment
    # if samples = 1 --> single trajectory predictions, else avg robustness
    traj = mu0.sample(samples=n_trajectories, varn=n_vars, points=n_traj_points)
    if i == 0 or (i > 0 and resample_formulae):
        # instantiate the kernel
        # if you have specific trajectories t of shape [#n_vars, #samples, #n_traj_points] --> signals = t
        kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_vars)
        # kernel ridge regressor
        regressor = KernelRegression(kernel, cross_validate=True)
        if boolean:
            # kernel evaluated with qualitative semantics
            boolean_kernel = StlKernel(mu0, samples=10000, sigma2=0.44, varn=n_vars, boolean=True)
            # and corresponding ridge regressor
            boolean_regressor = KernelRegression(boolean_kernel, cross_validate=True)
            # generating the STL dataset
            phi_train = sampler.bag_sample(bag_size=train_size, nvars=n_vars)
            phi_test = sampler.bag_sample(bag_size=test_size, nvars=n_vars)
            phi_val = sampler.bag_sample(bag_size=val_size, nvars=n_vars)
            if boolean:
                bool_gram_train = GramMatrix(boolean_kernel, phi_train)
                bool_gram_test = boolean_kernel.compute_bag_bag(phi_test, phi_train)
                bool_gram_val = boolean_kernel.compute_bag_bag(phi_val, phi_train)

    for norm in norm_rob:
        # compute dataset (formula, robustness)
        obs_train = torch.zeros(len(phi_train))
        for j, phi in enumerate(phi_train):
            obs_train[j] = torch.mean(phi.quantitative(traj, norm))

        obs_test = torch.zeros(len(phi_test))
        for j, phi in enumerate(phi_test):
            obs_test[j] = torch.mean(phi.quantitative(traj, norm))

        obs_val = torch.zeros(len(phi_val))
        for j, phi in enumerate(phi_val):
            obs_val[j] = torch.mean(phi.quantitative(traj, norm))

        # train the regressor
        regressor.train(phi_train, obs_train, phi_val, obs_val)
        if boolean:
            boolean_regressor.train(phi_train, obs_train, validate_phis=phi_val, validate_obs=obs_val,
                                    gram=bool_gram_train, validate_kernel_vector=bool_gram_val)
        # test the kernel regressor on current dataset
        kreg_mae, kreg_mse, p = regressor.test(phi_test, obs_test)
        kreg_ae = torch.abs(p - obs_test)
        kreg_re = torch.abs(p - obs_test) / torch.abs(obs_test)
        kreg_se = (p - obs_test) * (p - obs_test)
        kreg_mre = torch.mean(kreg_re)
        # accuracy makes senses only if n_trajectories = 1 (i.e. single trajectories)
        acc_kreg = (torch.sign(p) == torch.sign(obs_test)).type(torch.FloatTensor)
        acc_i = torch.sum(acc_kreg).item() / test_size
        if boolean:
            bool_mae, bool_mse, bool_p = boolean_regressor.test(phi_test, obs_test,
                                                                kernel_vector=bool_gram_test)
            bool_ae = torch.abs(bool_p - obs_test)
            bool_re = torch.abs(bool_p - obs_test) / torch.abs(obs_test)
            bool_se = (bool_p - obs_test) * (bool_p - obs_test)
            bool_mre = torch.mean(bool_re)
            acc_bool = (torch.sign(bool_p) == torch.sign(obs_test)).type(torch.FloatTensor)
            bool_acc_i = torch.sum(acc_bool).item() / test_size
        if norm:
            kreg_MRE_norm[i], kreg_MAE_norm[i], kreg_MSE_norm[i], kreg_acc_norm[i] = \
                [kreg_mre, kreg_mae, kreg_mse, acc_i]
            kreg_stat_MRE_norm[i, :], kreg_stat_MAE_norm[i, :], kreg_stat_MSE_norm[i, :], \
                    kreg_stat_acc_norm[i, :] = [torch.quantile(mes, quantiles)
                                                for mes in [kreg_re, kreg_ae, kreg_se, acc_kreg]]
            if boolean:
                bool_MRE_norm[i], bool_MAE_norm[i], bool_MSE_norm[i], bool_acc_norm[i] = \
                    [bool_mre, bool_mae, bool_mse, bool_acc_i]
                bool_stat_MRE_norm[i, :], bool_stat_MAE_norm[i, :], bool_stat_MSE_norm[i, :], \
                    bool_stat_acc_norm[i, :] = [torch.quantile(mes, quantiles)
                                                    for mes in [bool_re, bool_ae, bool_se, acc_bool]]
        else:
            kreg_MRE[i], kreg_MAE[i], kreg_MSE[i], kreg_acc[i] = [kreg_mre, kreg_mae, kreg_mse, acc_i]
            kreg_stat_MRE[i, :], kreg_stat_MAE[i, :], kreg_stat_MSE[i, :], kreg_stat_acc[i, :] = \
                [torch.quantile(mes, quantiles) for mes in [kreg_re, kreg_ae, kreg_se, acc_kreg]]
            if boolean:
                bool_MRE[i], bool_MAE[i], bool_MSE[i], bool_acc[i] = [bool_mre, bool_mae, bool_mse, bool_acc_i]
                bool_stat_MRE[i, :], bool_stat_MAE[i, :], bool_stat_MSE[i, :], bool_stat_acc[i, :] = \
                    [torch.quantile(mes, quantiles) for mes in [bool_re, bool_ae, bool_se, acc_bool]]

files = ['kreg_mae', 'kreg_mse', 'kreg_mre', 'kreg_mae_norm', 'kreg_mse_norm', 'kreg_mre_norm', 'kreg_acc',
         'kreg_acc_norm', 'kreg_mae_stat', 'kreg_mre_stat', 'kreg_mse_stat', 'kreg_mae_stat_norm',
         'kreg_mre_stat_norm', 'kreg_mse_stat_norm', 'kreg_acc_stat', 'kreg_acc_stat_norm']
things = [kreg_MAE, kreg_MSE, kreg_MRE, kreg_MAE_norm, kreg_MSE_norm, kreg_MRE_norm, kreg_acc, kreg_acc_norm,
          kreg_stat_MAE, kreg_stat_MRE, kreg_stat_MSE, kreg_stat_MAE_norm, kreg_stat_MRE_norm, kreg_stat_MSE_norm,
          kreg_stat_acc, kreg_stat_acc_norm]
if boolean:
    bool_files = ['bool_mae', 'bool_mse', 'bool_mre', 'bool_mae_norm', 'bool_mse_norm', 'bool_mre_norm',
                  'bool_acc', 'bool_acc_norm', 'bool_mae_stat', 'bool_mre_stat', 'bool_mse_stat',
                  'bool_acc_stat', 'bool_mae_stat_norm', 'bool_mre_stat_norm', 'bool_mse_stat_norm',
                  'bool_acc_stat_norm']
    bool_things = [bool_MAE, bool_MSE, bool_MRE, bool_MAE_norm, bool_MSE_norm, bool_MRE_norm, bool_acc, bool_acc_norm,
                   bool_stat_MAE, bool_stat_MRE, bool_stat_MSE, bool_stat_acc, bool_stat_MAE_norm, bool_stat_MRE_norm,
                   bool_stat_MSE_norm, bool_stat_acc_norm]
    files = files + bool_files
    things = things + bool_things
for file, thing in zip(files, things):
    dump_pickle(file, thing)
