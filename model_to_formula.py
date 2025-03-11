import torch
import torch.nn as nn
import torch.nn.functional as F
from IR.phisearch import search_from_embeddings

import sys
import os



class SimpleRNN(nn.Module):
    """
    Gated Recurrent Unit model for time series classification
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=True):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.directions, 2)  # Binary classification
        
    def forward(self, x):
        
        x_reor = x.permute(0, 2, 1) # Now the shape is [samples, n_traj_points, n_vars]
        
        # GRU returns output, hidden
        _, hidden = self.gru(x_reor)
        
        # Use the last hidden state from the last layer
        #if self.bidirectional:
        #    # Concatenate both directions
        #    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        #else:
        #    hidden = hidden[-1]

        # Get the final forward and backward hidden states
        forward = hidden[2*self.num_layers-2]  # Last forward direction
        backward = hidden[2*self.num_layers-1]  # Last backward direction
        hidden = torch.cat((forward, backward), dim=1)
            
        # Pass through the fully connected layer
        out = self.fc(hidden)
        return out
    


class quantitative_model:
    def __init__(self, model_path=None, nvars=None): #ex: f'IR/data/data/linear/model_state_dict.pth'
        if model_path is None:
            raise RuntimeError("No model dict found!")
        self.model_path = model_path
        self.nvars = nvars

        # loading the model
        self.model = SimpleRNN(input_dim=self.nvars)
        self.model.load_state_dict(torch.load(self.model_path))
    
    def robustness(self, traj):
        """
        input: traj : torch.Tensor of shape (samples, n_vars, n_traj_points)
        """

        self.model.eval()  # Put in evaluation mode
        with torch.no_grad():
            logits = self.model(traj) 
            probabilities = F.softmax(logits, dim=1)
            rhos = 2 * probabilities[:,1] - 1
        return rhos
    
    def new_robustness(self, traj):
        """
        input: traj : torch.Tensor of shape (samples, n_vars, n_traj_points)

        Computes the new robustness vector (no tanh normalization and norm equal to 1)
        """

        rhos_unnorm = self.robustness(traj)
        #rhos_unnorm = torch.atanh(old_rhos)
        rhos_norm = torch.norm(rhos_unnorm).item()
        rhos = rhos_unnorm / rhos_norm
        return rhos


def new_kernel_to_embedding(new_kernel, sigma2=0.44 ):
    """
    input: new_kernel : torch.Tensor of shape (n_vars, n_traj_points)
    
    This function transforms the new kernel representation into 
    the embedding vector used inside the semantic vector database
    """
    if len(new_kernel.shape) != 2:
        raise RuntimeError(f"new_kernel should have two dimensions. Got shape: {new_kernel.shape}")
    
    embedding = torch.exp( (2 * new_kernel -2) / (2 * sigma2))

    return embedding
    

def compute_bag_bag_manual(phis1, phis2, mu0, max_n_vars, sigma2=0.44, samples=10000):
    """
    This is the expanded version of the funcion 'compute_bag_bag' inside kernel
    This is substantially different from the classical way of computing the kernel
    Differences: 
    * No tanh normalization (!!!)
    * Normalizes each kernel by the sqrt of the l2 norm of the robustness
    * Gaussian kernel instead of classical one (!!!)
    """
    # Initialize signals (normally this would be done in StlKernel constructor)
    signals = mu0.sample(points=100, samples=samples, varn=max_n_vars)
    
    # Compute robustness for both sets of formulas
    n = samples
    k1 = len(phis1)
    k2 = len(phis2)
    
    # Compute robustness for first set
    rhos1 = torch.zeros((k1, n), device=mu0.device)
    selfk1 = torch.zeros((k1, 1), device=mu0.device)
    for i, phi in enumerate(phis1):
        rho = phi.quantitative(signals, evaluate_at_all_times=False)
        selfk1[i] = rho.dot(rho) / n
        rhos1[i, :] = rho
    
    # Compute robustness for second set
    rhos2 = torch.zeros((k2, n), device=mu0.device)
    selfk2 = torch.zeros((k2, 1), device=mu0.device)
    for i, phi in enumerate(phis2):
        rho = phi.quantitative(signals, evaluate_at_all_times=False)
        selfk2[i] = rho.dot(rho) / n
        rhos2[i, :] = rho
    
    # Compute kernel matrix
    kernel_matrix = torch.tensordot(rhos1, rhos2, [[1], [1]])
    kernel_matrix = kernel_matrix / n
    
    # Normalize
    normalize_factor = torch.sqrt(torch.matmul(selfk1, torch.transpose(selfk2, 0, 1)))
    kernel_matrix = kernel_matrix / normalize_factor
    
    # Exponentiate
    # Apply Gaussian kernel formula: exp(-||x-y||²/(2σ²))
    # Where ||x-y||² = ||x||² + ||y||² - 2<x,y>
    k1 = selfk1.size()[0]
    k2 = selfk2.size()[0]
    selfk = 2.0  # Because we're normalized
    return torch.exp(-(selfk - 2 * kernel_matrix) / (2 * sigma2))

    
#def search_from_kernel(kernels, nvar, k=5, n_neigh=64, n_pc=-1, timespan=None, nodes=None):
    """
    Search for closest STL formulae based on kernel embeddings
    
    Parameters:
    -----------
    kernels : torch.Tensor
        Kernel matrices to use for search. Shape can be either:
        - Single kernel: [n_formulae, n_formulae]
        - Multiple kernels stacked: [n_kernels, n_formulae, n_formulae]
    nvar : int
        Number of variables in the STL formulae
    k : int, default=5
        Number of closest formulae to retrieve
    n_neigh : int, default=64
        Number of neighbors for FAISS search
    n_pc : int, default=-1
        Number of principal components (-1 means no PCA)
    timespan : int, optional
        Maximum timespan of the STL formulae
    nodes : int, optional
        Maximum number of nodes in the STL formulae
    
    Returns:
    --------
    tuple
        (formulae_list, distances)
        formulae_list: List of lists of k closest formulae for each kernel
        distances: Matrix of distances to k closest formulae for each kernel
    """
    """
    #Path to the index forlder
    folder_index = os.path.join("IR", "index")  # Update with actual path

    # Check if we have a single kernel or multiple kernels
    if len(kernels.shape) == 2:
        # Single kernel, add a dimension to make it [1, n_formulae, n_formulae]
        kernels = kernels.unsqueeze(0)   # TODO; Rivedi, forse non funziona
    
    # Convert kernel matrices to flattened embeddings
    # Each kernel becomes a row vector
    n_kernels = kernels.shape[0]
    kernel_flat = kernels.view(n_kernels, -1)
    
    # Search for closest formulae
    formulae_list, distances = search_from_embeddings(
        embeddings=kernel_flat,
        nvar=nvar,
        folder_index=folder_index,
        k=k,
        n_neigh=n_neigh,
        n_pc=n_pc,
        timespan=timespan,
        nodes=nodes
    )
    
    return formulae_list, distances
    """

'''# Example usage within the Work_on_process function:
def modified_Work_on_process(params, test_name)::
    # ... existing code ...
    
    # After computing K_loc and K_imp
    
    # Rescaling the kernels for the search:
    K_loc_scaled = K_loc * n_traj * math.sqrt(n_psi)
    K_imp_scaled = K_imp * n_traj * math.sqrt(n_psi)

    # Stack multiple kernels together
    kernels = torch.stack([K_loc_scaled, K_imp_scaled], dim=0)
    
    # Search for closest formulae to both kernels at once
    formulae_lists, distances = search_from_kernel(
        kernels=kernels,
        nvar=n_vars,
        k=5,
        n_neigh=64
    )
    
    # Access results for each kernel
    loc_formulae = formulae_lists[0:n_kernels-1]
    imp_formulae = formulae_lists[n_kernels:]
    
    loc_dists = distances[0:n_kernels-1]
    imp_dists = distances[n_kernels:]
    
    # Alternatively, search for each kernel individually:
    loc_formulae_list, loc_dists = search_from_kernel(
        kernels=K_loc,
        nvar=n_vars,
        folder_index=folder_index,
        k=5,
        n_neigh=64
    )
    
    imp_formulae_list, imp_dists = search_from_kernel(
        kernels=K_imp,
        nvar=n_vars,
        folder_index=folder_index,
        k=5,
        n_neigh=64
    )
    
    # ... rest of the existing code ...
'''