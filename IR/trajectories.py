import os
import csv
import torch
from torch.utils.data import Dataset
import math
import numpy as np
import copy

from utils import from_string_to_formula


def get_dataset(dataname, datafolder='data', indexes=None):
    # TODO: add times if available
    # load dataset
    with open(datafolder + os.path.sep + dataname + os.path.sep + 'labels.csv', 'r') as f:
        label_reader = csv.reader(f)
        labels = next(label_reader)
        labels = [int(i) for i in labels]

    data = []
    with open(datafolder + os.path.sep + dataname + os.path.sep + 'data.csv', 'r') as f:
        data_reader = csv.reader(f)
        header = next(data_reader)
        n = len(header)

        for _, row in enumerate(data_reader):
            sublists = [[] for _ in range(n)]
            for i, item in enumerate(row):
                sublists[i % n].append(float(item))
            data.append(sublists)
    if indexes is not None:
        return torch.tensor(data)[:, indexes, :], torch.tensor(labels)
    return torch.tensor(data), torch.tensor(labels)


class TrajectoryDataset(Dataset):
    def __init__(self, device, data_fn=None, dataname=None, indexes=None, x=None, y=None):
        if (x is None) or (y is None):
            x, y = data_fn(dataname, indexes=indexes)
        self.trajectories = x.to(device)
        self.labels = y.to(device)
        self.nvars = x.shape[1]
        self.npoints = x.shape[-1]
        self.mean = torch.zeros(self.nvars).to(device)
        self.std = torch.zeros(self.nvars).to(device)
        self.normalized = False

    def reshape_mean_std(self):
        rep_mean = torch.cat([self.mean[i].repeat(
            self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1) for i in range(self.nvars)], dim=1)
        rep_std = torch.cat([self.std[i].repeat(
            self.trajectories.shape[0], self.trajectories.shape[-1]).unsqueeze(1) for i in range(self.nvars)], dim=1)
        return rep_mean.to(self.trajectories.device), rep_std.to(self.trajectories.device)

    def normalize(self):
        self.mean = torch.tensor([self.trajectories[:, i, :].mean() for i in range(self.nvars)])
        self.std = torch.tensor([self.trajectories[:, i, :].std() for i in range(self.nvars)])
        rep_mean, rep_std = self.reshape_mean_std()
        self.trajectories = (self.trajectories - rep_mean) / rep_std
        self.normalized = True

    def inverse_normalize(self):
        rep_mean, rep_std = self.reshape_mean_std()
        self.trajectories = (self.trajectories * rep_std) + rep_mean
        self.normalized = False

    def time_scaling(self, phi, phi_timespan=100):
        # npoints is the number of points in the original trajectory (hence the original formulae)
        current_one_percent = self.npoints/phi_timespan  # in npoints
        phi_str = str(phi)
        temporal_start_idx = [i for i in range(len(phi_str)) if phi_str.startswith('[', i)]
        temporal_middle_idx = [i for i in range(len(phi_str)) if phi_str.startswith(',', i)]
        temporal_end_idx = [i for i in range(len(phi_str)) if phi_str.startswith(']', i)]
        start_idx = temporal_start_idx[0] if len(temporal_start_idx) > 0 else None
        str_list = [phi_str[:start_idx]]
        new_intervals_list = []
        for i, s, m, e in zip(range(len(temporal_start_idx)), temporal_start_idx, temporal_middle_idx,
                              temporal_end_idx):
            right_unbound = True if phi_str[e-1] == 'f' else False
            right_bound = -1. if right_unbound else float(phi_str[m+1:e])
            current_time_interval = [float(phi_str[s+1:m]), right_bound]  # this is the original interval
            # these are hte changes I was doing (so this is the main part that should be changed)
            current_percentage = 0 if right_unbound else current_time_interval[1] - current_time_interval[0]
            new_left = math.floor(current_time_interval[0]*current_one_percent)
            new_time_interval = [new_left, min([new_left + math.ceil(current_percentage*current_one_percent),
                                                self.npoints])]
            new_right_str = 'inf' if right_unbound else str(new_time_interval[1])
            # from now on it is changing the formula parameters
            new_intervals_list += ['[' + str(new_time_interval[0]) + ',' + new_right_str + ']']
            idx = temporal_start_idx[i+1] if i < len(temporal_start_idx) - 1 else None
            str_list.append(phi_str[e+1:idx])
        new_phi_str = ''
        for i in range(len(new_intervals_list)):
            new_phi_str += str_list[i]
            new_phi_str += new_intervals_list[i]
        new_phi_str += str_list[-1]
        return from_string_to_formula(new_phi_str)

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]


# dataset = TrajectoryDataset(data_fn=get_dataset, dataname='robot4', indexes=None, device='cpu')
# print(dataset.trajectories.shape, dataset.labels.shape)
# train_size = int(0.8 * len(dataset))
# test_size = int(0.5 * (len(dataset) - int(0.5 * 0.8 * len(dataset))))
# val_size = len(dataset) - train_size - test_size
# train_subset, test_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
# train_dataset = TrajectoryDataset(x=dataset.trajectories[train_subset.indices],
# y=dataset.labels[train_subset.indices])
# train_dataset.normalize()
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_dataset = TrajectoryDataset(x=dataset.trajectories[test_subset.indices], y=dataset.labels[test_subset.indices])
# test_dataset.normalize()
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
# validation_dataset = TrajectoryDataset(x=dataset.trajectories[val_subset.indices],
# y=dataset.labels[val_subset.indices])
# validation_dataset.normalize()
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)

# train_dataset.inverse_normalize()
# test_dataset.inverse_normalize()
# validation_dataset.inverse_normalize()
