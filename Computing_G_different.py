import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# TODO: Compute the compacted version of the matrix G!!


def compute_G_different_S(N: int, s: float, max_rank: int, S_matrix: torch.Tensor, S_prime: torch.Tensor, eps: float,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the G matrix with different S and S' matrices:
    
    G_{i,j}(s) = ∑_{n=1}^∞ P_{i,j}^{(n)} · s^(n-1) 
    
    where:
    P_{i,j}^{(n)} = ∑_{k=1}^{n-1} [ P^{(k)}_{i,j} · ∑_{x,z} P^{(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z}) +
                                  + ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})]
    
    and P_{i,j}^{(1)} = δ_{i,j}

    So also:
    P_{i,j}^{(n)} · s^(n-1)  = s · ∑_{k=1}^{n-1} [ P^{(k)}_{i,j}·s^(k-1) · ∑_{x,z} P^{(n-k)}_{x,z}·s^(n-k-1) · (1+S_{i,x} S'_{j,z}) +
                                                + ∑_{x,z} P^{(k)}_{i,z}·s^(k-1) · P^{(n-k)}_{x,j}·s^(n-k-1)  · (1+S_{i,x} S'_{j,z})]
    
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
    erros : float
        The error of this matrix computation
    """
    # Saving the starting time
    start_time = time.time()

    # Matrix size
    size = 2 * N
    
    # Set device
    torch_device = torch.device(device)
    
    # Move S matrices to the correct device
    S_matrix = S_matrix.to(torch_device)
    S_prime = S_prime.to(torch_device)
    
    # Initialize list to store P matrices (actually we consider the matrices P_{i,j}^{(n)} · s^(n-1))
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device)
    P_list.append(P1)
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        current_rank = n

        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
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
                    
                    P_n[i, j] += P_k[i, j] * term1_sum * s # We multiply by s each term
                    
                    # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
                    term2_sum = 0.0
                    
                    # We compute this term by directly implementing the summation
                    for x in range(size):
                        for z in range(size):
                            term2_sum += P_k[i, z] * P_nk[x, j] * (1 + S_matrix[i, x] * S_prime[j, z])
                    
                    P_n[i, j] += term2_sum * s # We multiply by s each term
        
        P_list.append(P_n)
        
        # Compute the norm for convergence checking
        error = torch.norm(P_n).item()
        if error < eps:
            break # Breaking out of the for loop if the error is small

    # Compute G by summing all P matrices (s^n factors already included)
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n]

    elapsed = time.time() - start_time
    print(f"Computed G matrix in {elapsed:.3f} seconds")
    
    return G, error, current_rank


def compute_G_different_S_optimized(N: int, s: float, max_rank: int, S_matrix: torch.Tensor, S_prime: torch.Tensor, eps: float,
                                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    A more optimized version for different S and S' matrices that uses batched operations
    where possible while still being memory-efficient.
    
    Parameters are the same as compute_G_different_S.
    """
    # Saving the starting time
    start_time = time.time()

    # Matrix size
    size = 2 * N
    
    # Set device
    torch_device = torch.device(device)
    
    # Move S matrices to the correct device
    S_matrix = S_matrix.to(torch_device)
    S_prime = S_prime.to(torch_device)
    
    # Initialize list to store P matrices
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device)
    P_list.append(P1)
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        current_rank = n
        
        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
            for i in range(size):
                for j in range(size):
                    # Here we will be multiplying a tensor [size, 1] with a tensor [1, size]
                    z_i_j = 1 + S_matrix[i, :].unsqueeze(1) * S_prime[j, :].unsqueeze(0)

                    # First term: P^(k)_{i,j} · ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})                    
                    # Compute the tensor dot product
                    term1_sum = torch.tensordot(P_nk, z_i_j, dims=([0,1], [0,1]))    
                    P_n[i, j] += P_k[i, j] * term1_sum * s # We multiply by s each term
              
                    # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
                    term2_sum = torch.einsum('x,xz,z->', P_nk[:, j], z_i_j, P_k[i, :])
                    P_n[i, j] += term2_sum * s # We multiply by s each term
            
        P_list.append(P_n)
        
        # Compute the norm for convergence checking
        error = torch.norm(P_n).item()
        if error < eps:
            break # Breaking out of the for loop if the error is small

    
    # Compute G by summing all P matrices (s^n factors already included)
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n]

    elapsed = time.time() - start_time
    print(f"Computed G matrix in {elapsed:.3f} seconds")
    
    return G, error, current_rank


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
    N = 20      # Matrix size will be 2N x 2N
    s = 1/(32*N)     # Parameter s
    max_rank = 5  # Maximum rank for the expansion
    eps = 1e-6

    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create example S and S' matrices
    size = 2 * N
    
    # Example S matrix (antisymmetric)
    S_matrix = torch.zeros((size, size), dtype=torch.float32)
    # Create a vector
    vector = torch.arange(size)    
    indices = torch.randperm(size)
    a = vector[indices]
    print(a)
    for i in range(size):
        for j in range(size):
            if a[i] > a[j]:
                S_matrix[i, j] = 1
            elif a[i] < a[j]:
                S_matrix[i, j] = -1
    
    # Example S' matrix (different from S, also antisymmetric)
    S_prime = torch.zeros((size, size), dtype=torch.float32)
    indices_prime = torch.randperm(size)
    a = vector[indices_prime]
    print(a)
    for i in range(size):
        for j in range(size):
            if a[i] > a[j]:
                S_matrix[i, j] = 1
            elif a[i] < a[j]:
                S_matrix[i, j] = -1
    
    """
    print("Running normal implementation for different S matrices...")
    start_time = time.time()
    G, error, current_rank = compute_G_different_S(N, s, max_rank, S_matrix, S_prime, eps, device=device)
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.3f} seconds with error: {error} and current_rank: {current_rank}")
    """


    # The optimized version is mo much better! Takes just 1.733 seconds for N = 20
    print("Running optimized implementation for different S matrices...")
    start_time = time.time()
    G_other, error_other, current_rank_other = compute_G_different_S_optimized(N, s, max_rank, S_matrix, S_prime, eps, device=device)
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.3f} seconds with error: {error_other}, and current_rank : {current_rank_other}")

    #if not torch.allclose(G, G_other, rtol=1e-6, atol=1e-8):
    #    raise RuntimeError(f"The matrices are not equal!! Error: {torch.norm(G - G_other)}")

    # Move results to CPU for visualization
    G_cpu = G_other.cpu()
    
    # Display the results
    print(f"G matrix for s = {s}, up to rank {max_rank}:")
    print(G_cpu)
    
    # Visualize G matrix with diagonal set to zero
    visualize_matrix(G_cpu, f"G matrix for s = {s}, up to rank {max_rank}", zero_diagonal=True)