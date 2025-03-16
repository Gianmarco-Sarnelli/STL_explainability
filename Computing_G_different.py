import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

def compute_G_different_S(N: int, s: float, max_rank: int, S_matrix: torch.Tensor, S_prime: torch.Tensor,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the G matrix with different S and S' matrices while maintaining memory efficiency.
    
    Parameters:
    -----------
    N : int
        Determines the size of the matrices (2N x 2N)
    s : float
        The parameter s in the equation
    max_rank : int
        Maximum rank for the expansion (number of terms to compute)
    S_matrix : torch.Tensor
        The S matrix
    S_prime : torch.Tensor
        The S' matrix
    device : str
        Device to use for computation ('cuda' or 'cpu')
        
    Returns:
    --------
    G : torch.Tensor
        The resulting G matrix
    P_list : list
        List of all P^(n) matrices computed
    """
    # Matrix size
    size = 2 * N
    
    # Set device
    torch_device = torch.device(device)
    
    # Move S matrices to the correct device
    S_matrix = S_matrix.to(torch_device)
    S_prime = S_prime.to(torch_device)
    
    # Initialize list to store P matrices
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix times s
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device) * s
    P_list.append(P1)
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        start_time = time.time()
        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
            # Compute efficiently by block regions
            for i in range(size):
                for j in range(size):
                    # First term: P^(k)_{i,j} · ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})
                    term1_sum = 0.0
                    
                    # Calculate the sum for the first term
                    # We can't make the same optimizations as before since S and S' are different
                    # But we can still be memory-efficient by computing on-the-fly
                    for x in range(size):
                        for z in range(size):
                            term1_sum += P_nk[x, z] * (1 + S_matrix[i, x] * S_prime[j, z])
                    
                    P_n[i, j] += P_k[i, j] * term1_sum
                    
                    # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
                    term2_sum = 0.0
                    
                    # We compute this term by directly implementing the summation
                    for x in range(size):
                        for z in range(size):
                            term2_sum += P_k[i, z] * P_nk[x, j] * (1 + S_matrix[i, x] * S_prime[j, z])
                    
                    P_n[i, j] += term2_sum
        
        P_list.append(P_n)
        elapsed = time.time() - start_time
        print(f"Computed P^({n}) matrix in {elapsed:.3f} seconds")
        
        # Print norm for convergence checking
        norm = torch.norm(P_n).item()
        print(f"Norm of P^({n}): {norm:.10f}")
    
    # Compute G by summing all P matrices (s^n factors already included)
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n]
    
    return G, P_list


def compute_G_different_S_optimized(N: int, s: float, max_rank: int, S_matrix: torch.Tensor, S_prime: torch.Tensor,
                                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    A more optimized version for different S and S' matrices that uses batched operations
    where possible while still being memory-efficient.
    
    Parameters are the same as compute_G_different_S.
    """
    # Matrix size
    size = 2 * N
    
    # Set device
    torch_device = torch.device(device)
    
    # Move S matrices to the correct device
    S_matrix = S_matrix.to(torch_device)
    S_prime = S_prime.to(torch_device)
    
    # Initialize list to store P matrices
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix times s
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device) * s
    P_list.append(P1)
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        start_time = time.time()
        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
            # Process by rows to be memory efficient but still use some vectorization
            for i in range(size):
                # First term: P^(k)_{i,j} · ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})
                # We'll process this row-by-row to avoid creating the full 4D tensor
                
                # Compute the term1_sum for all j values in row i
                term1_sums = torch.zeros(size, dtype=torch.float32, device=torch_device)
                
                # This computes all term1_sums for row i
                for j in range(size):
                    # For each position (i,j), calculate ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})
                    # We can vectorize the inner calculation a bit for each j
                    
                    # First, create matrices for the multipliers (1 + S_{i,x} * S'_{j,z})
                    S_i_row = S_matrix[i, :].unsqueeze(1)  # Shape: [size, 1]
                    S_prime_j_row = S_prime[j, :].unsqueeze(0)  # Shape: [1, size]
                    
                    # Compute (1 + S_{i,x} * S'_{j,z}) for all x,z - Shape: [size, size]
                    multipliers = 1 + S_i_row * S_prime_j_row
                    
                    # Multiply with P_nk and sum all elements
                    term1_sums[j] = torch.sum(P_nk * multipliers)
                
                # Now multiply with P_k[i, :] and add to P_n
                P_n[i, :] += P_k[i, :] * term1_sums
                
                # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
                for j in range(size):
                    # Calculate the second term for position (i,j)
                    # We can vectorize this a bit too
                    
                    # We need P_k[i, z] for all z and P_nk[x, j] for all x
                    P_k_i_row = P_k[i, :]  # Shape: [size]
                    P_nk_col_j = P_nk[:, j]  # Shape: [size]
                    
                    # Generate all (x,z) pairs
                    x_indices = torch.arange(size, device=torch_device)
                    
                    # Calculate multipliers for each x
                    term2_sum = 0.0
                    
                    for x in range(size):
                        # Compute (1 + S_{i,x} * S'_{j,z}) for this x and all z
                        multipliers = 1 + S_matrix[i, x] * S_prime[j, :]
                        
                        # Multiply with P_k[i, z] and sum
                        term2_sum += P_nk[x, j] * torch.sum(P_k_i_row * multipliers)
                    
                    P_n[i, j] += term2_sum
        
        P_list.append(P_n)
        elapsed = time.time() - start_time
        print(f"Computed P^({n}) matrix in {elapsed:.3f} seconds (optimized)")
        
        # Print norm for convergence checking
        norm = torch.norm(P_n).item()
        print(f"Norm of P^({n}): {norm:.10f}")
    
    # Compute G by summing all P matrices (s^n factors already included)
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n]
    
    return G, P_list


def visualize_matrix(matrix, title="Matrix Visualization", zero_diagonal=True):
    """
    Visualize a matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        The matrix to visualize
    title : str
        The title for the plot
    zero_diagonal : bool
        Whether to set diagonal values to 0 before visualization
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()
    
    # Create a copy to avoid modifying the original
    viz_matrix = matrix.copy()
    
    # Set diagonal values to 0 if requested
    if zero_diagonal:
        np.fill_diagonal(viz_matrix, 0)
        
    plt.figure(figsize=(8, 6))
    plt.imshow(viz_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    N = 2       # Matrix size will be 2N x 2N
    s = 0.1     # Parameter s
    max_rank = 5  # Maximum rank for the expansion
    
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create example S and S' matrices
    size = 2 * N
    
    # Example S matrix (antisymmetric)
    S_matrix = torch.zeros((size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            if i > j:
                S_matrix[i, j] = 1
            elif i < j:
                S_matrix[i, j] = -1
    
    # Example S' matrix (different from S, also antisymmetric)
    S_prime = torch.zeros((size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            if i > j:
                S_prime[i, j] = 0.8
            elif i < j:
                S_prime[i, j] = -0.8
    
    print("Running optimized implementation for different S matrices...")
    start_time = time.time()
    G, P_list = compute_G_different_S_optimized(N, s, max_rank, S_matrix, S_prime, device=device)
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.3f} seconds")
    
    # Move results to CPU for visualization
    G_cpu = G.cpu()
    
    # Display the results
    print(f"G matrix for s = {s}, up to rank {max_rank}:")
    print(G_cpu)
    
    # Visualize G matrix with diagonal set to zero
    visualize_matrix(G_cpu, f"G matrix for s = {s}, up to rank {max_rank}", zero_diagonal=True)