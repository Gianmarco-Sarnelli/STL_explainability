import torch
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Optional

# TODO: instad of computing  P_{i,j}^{(n)}, compute  P_{i,j}^{(n)} * s^n
# TODO: Save the matrix in a special folder
# TODO: Implement the function that compresses G into G^*
# TODO: Save G^*
# TODO: Implement the function that multiplies a* G a'*
# TODO: Slice the indeces x, z depending on their position relative to i, j to optimize the code!

def compute_G_matrix(N: int, s: float, max_rank: int, S_matrix: Optional[torch.Tensor] = None,  
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the G matrix using the simplified iterative formula:
    
    G_{i,j}(s) = ∑_{n=1}^∞ P_{i,j}^{(n)} · s^n 
    
    where:
    P_{i,j}^{(n)} = ∑_{k=1}^{n-1} [ P^{(k)}_{i,j} · ∑_{x,z} P^{(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z}) +
                                  + ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})]
    
    and P_{i,j}^{(1)} = δ_{i,j}
    
    Parameters:
    -----------
    N : int
        Determines the size of the matrices (2N x 2N)
    s : float
        The parameter s in the equation
    max_rank : int
        Maximum rank for the expansion (number of terms to compute)
    S_matrix : torch.Tensor, optional
        The S matrix. If None, it will be created using the sign function
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
    
    # Create S matrix if not provided
    if S_matrix is None:
        S_matrix = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        for i in range(size):
            for j in range(size):
                if i > j:
                    S_matrix[i, j] = 1
                elif i < j:
                    S_matrix[i, j] = -1
                # i = j case already set to 0
    else:
        S_matrix = S_matrix.to(torch_device)
    
    # Since S and S' are the same, we can simplify
    S_prime = S_matrix
    
    # Initialize list to store P matrices
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device)
    P_list.append(P1)
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        start_time = time.time()
        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
            # First term: P^(k)_{i,j} · ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})
            for i in range(size):
                for j in range(size):
                    term1_sum = 0.0
                    for x in range(size):
                        for z in range(size):
                            term1_sum += P_nk[x, z] * (1 + S_matrix[i, x] * S_prime[j, z])
                    
                    P_n[i, j] += P_k[i, j] * term1_sum
            
            # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
            for i in range(size):
                for j in range(size):
                    for x in range(size):
                        for z in range(size):
                            P_n[i, j] += P_k[i, z] * P_nk[x, j] * (1 + S_matrix[i, x] * S_prime[j, z])
        
        P_list.append(P_n)
        elapsed = time.time() - start_time
        print(f"Computed P^({n}) matrix in {elapsed:.3f} seconds")
    
    # Compute G(s) using the formula G_{i,j}(s) = ∑_{n=1}^max_rank P_{i,j}^{(n)} · s^n
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n] * (s ** n)
    
    return G, P_list

# TODO: S_S_prime_products cannot be computed! Remove it!

def compute_G_matrix_optimized(N: int, s: float, max_rank: int, S_matrix: Optional[torch.Tensor] = None, 
                              device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Optimized version using PyTorch's tensor operations and broadcasting 
    for the simplified formula:
    
    P_{i,j}^{(n)} = ∑_{k=1}^{n-1} [ P^{(k)}_{i,j} · ∑_{x,z} P^{(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z}) +
                                  + ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})]
    """
    
    # Matrix size
    size = 2 * N
    
    # Set device
    torch_device = torch.device(device)
    
    # Create S matrix if not provided
    if S_matrix is None:
        # Create indices for all pairs
        i_indices, j_indices = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        
        # Create S matrix using the sign rule directly
        S_matrix = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        S_matrix[i_indices > j_indices] = 1.0
        S_matrix[i_indices < j_indices] = -1.0
    else:
        S_matrix = S_matrix.to(torch_device)
    
    # Since S and S' are the same in this problem
    S_prime = S_matrix
    
    # Initialize list to store P matrices
    P_list = [None]  # P^(0) is not used, so we start with None
    
    # P^(1) is the identity matrix
    P1 = torch.eye(size, dtype=torch.float32, device=torch_device)
    P_list.append(P1)
    
    # Precompute the S * S' terms for broadcasting
    # Create tensors for broadcasting
    S_expanded = S_matrix.unsqueeze(2).unsqueeze(3)  # Shape: [size, size, 1, 1]
    S_prime_expanded = S_prime.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, size, size]
    
    # Compute S_{i,x} * S'_{j,z} for all combinations - Shape: [size, size, size, size]
    S_S_prime_products = S_expanded * S_prime_expanded
    
    # Precompute (1 + S_{i,x} * S'_{j,z}) terms - Shape: [size, size, size, size]
    one_plus_S_S_prime = 1 + S_S_prime_products
    
    # Compute P^(n) for n = 2 to max_rank
    for n in range(2, max_rank + 1):
        start_time = time.time()
        P_n = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
        
        for k in range(1, n):
            P_k = P_list[k]
            P_nk = P_list[n-k]
            
            # First term: P^(k)_{i,j} · ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z})
            
            # Sum of P_nk elements - scalar
            P_nk_sum = P_nk.sum()
            
            # Compute ∑_{x,z} P^(n-k)}_{x,z} · (1+S_{i,x} S'_{j,z}) for all i,j
            # Shape: [size, size]
            term1_sums = torch.einsum('xy,ijxy->ij', P_nk, one_plus_S_S_prime)
            
            # Add contribution from first term
            P_n += P_k * term1_sums
            
            # Second term: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} S'_{j,z})
            
            # Using einsum for efficient tensor contraction
            # This computes: ∑_{x,z} P^{(k)}_{i,z} · P^{(n-k)}_{x,j} · (1+S_{i,x} * S'_{j,z})
            term2 = torch.einsum('iz,xj,ixjz->ij', P_k, P_nk, one_plus_S_S_prime)
            
            # Add contribution from second term
            P_n += term2
        
        P_list.append(P_n)
        elapsed = time.time() - start_time
        print(f"Computed P^({n}) matrix in {elapsed:.3f} seconds (optimized)")
    
    # Compute G(s) using the formula G_{i,j}(s) = ∑_{n=1}^max_rank P_{i,j}^{(n)} · s^n
    G = torch.zeros((size, size), dtype=torch.float32, device=torch_device)
    for n in range(1, max_rank + 1):
        G += P_list[n] * (s ** n)
    
    return G, P_list



# TODO: Remove the diagonal from the matrix visualization
# TODO: Maybe visualize the log of the result

def visualize_matrix(matrix, title="Matrix Visualization"):
    """
    Visualize a matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        The matrix to visualize
    title : str
        The title for the plot
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()
        
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# TODO: Probably not useful

def check_convergence(P_list, s, tolerance=1e-6):
    """
    Check the convergence of the series expansion.
    
    Parameters:
    -----------
    P_list : list
        List of P matrices
    s : float
        Parameter value
    tolerance : float
        Convergence tolerance
    
    Returns:
    --------
    is_converged : bool
        Whether the series has converged
    error : float
        Estimated error of the last term
    """
    if len(P_list) < 3:
        return False, float('inf')
    
    n = len(P_list) - 1
    last_term = P_list[n] * (s ** n)
    norm = torch.norm(last_term).item()
    
    print(f"Norm of term {n}: {norm:.10f}")
    
    return norm < tolerance, norm


# Example usage
if __name__ == "__main__":
    # Parameters
    N = 3       # Matrix size will be 2N x 2N (6x6)
    s = 0.1     # Parameter s
    max_rank = 7  # Maximum rank for the expansion
    
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use the optimized implementation
    start_time = time.time()
    G, P_list = compute_G_matrix_optimized(N, s, max_rank, device=device)
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.3f} seconds")
    
    # Move results to CPU for visualization
    G_cpu = G.cpu()
    P_list_cpu = [P.cpu() if P is not None else None for P in P_list]
    
    # Check convergence
    is_converged, error = check_convergence(P_list_cpu, s)
    print(f"Series convergence: {'Yes' if is_converged else 'No'}, estimated error: {error:.10f}")
    
    # Display the results
    print(f"G matrix for s = {s}, up to rank {max_rank}:")
    print(G_cpu)
    
    # Visualize G matrix
    visualize_matrix(G_cpu, f"G matrix for s = {s}, up to rank {max_rank}")
    
    # Visualize the first few P matrices
    for n in range(1, min(4, max_rank + 1)):
        visualize_matrix(P_list_cpu[n], f"P^({n}) matrix")