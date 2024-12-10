import torch
global_traj = torch.ones(1, 2, 5)
global_traj = torch.cumsum(global_traj, dim = 2)
global_traj = torch.cumsum(global_traj, dim = 2)
print(f"global_traj = {global_traj}")
global_traj[0, :, 1:] = global_traj[0, :, 1:] - global_traj[0, :, :-1]
print(global_traj)
print(global_traj[0,:,:].shape)

