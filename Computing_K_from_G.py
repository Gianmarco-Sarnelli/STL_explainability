import torch
import time
from typing import Tuple, List, Optional
from Computing_G_different import compute_G_different_S, compute_G_different_S_optimized

import os

"""Computes the matrix K starting from for the new embedding"""

def Compute_K(N, M, s_multiplier, save_G_compact=False):


    # TODO: Multiply by the correct constant!!
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_rank = 7
    s = s_multiplier / (16 * N) # 1/16N is the maximum value of s
    eps = 1e-5 # fixing some value for the epsilon

    #Load the rhos_anchor_psis
    rhos_anchor_psis = torch.load(os.path.join("Special_kernel", f"rhos_anchor_psis_{N}_{M}.pt"))

    G_compact_dict = {}
    Special_K = torch.zeros(M,M)
    for I in range(M):
        a_I = Compute_a(rhos_anchor_psis[:,I])
        S_matrix = Compute_S_matrix(a_I)
        for J in range(M):
            a_J = Compute_a(rhos_anchor_psis[:,J])
            S_prime = Compute_S_matrix(a_J)
            G, error = compute_G_different_S(N=N, s=s, max_rank=max_rank, S_matrix=S_matrix, S_prime=S_prime, eps=eps, device=device)
            
            # Printing the error if needed
            if error > eps:
                print(f"The error of the matrix G in {I} {J} after {max_rank} iterations is {error}")

            G_compact = G[:N, :N] + G[N:, N:] - G[:N, N:] - G[N:, :N]
            if save_G_compact:
                G_compact_dict[(I, J)] = G_compact
                os.makedirs(os.path.join("Special_kernel", "G_compact_dir"), exist_ok=True) # Creating the directory if not present
                torch.save(G_compact_dict, os.path.join("Special_kernel", "G_compact_dir", f"{N}_{M}_{s_multiplier}.pt"))
            # Saving a sample of the matrix G
            if I==1 and J==2:
                os.makedirs(os.path.join("Special_kernel", "G_sample_dir"), exist_ok=True) # Creating the directory if not present
                torch.save(G, os.path.join("Special_kernel", "G_sample_dir", f"{N}_{M}_{s_multiplier}.pt"))
            K = torch.tensordot(rhos_anchor_psis[:,I], torch.tensordot(G_compact, rhos_anchor_psis[:,J], dims=([1],[0]) ) , dims=([0],[0]))
            Special_K[I,J] = K


    # Saving the matrix Special_K
    os.makedirs(os.path.join("Special_kernel", "Special_K_dir"), exist_ok=True) # Creating the directory if not present
    torch.save(Special_K, os.path.join("Special_kernel", "Special_K_dir", f"{N}_{M}_{s_multiplier}.pt"))

    # loading the dict G_compact
    #G_compact_dict = torch.load(os.path.join("Special_kernel", "G_compact_dir", f"{N}_{M}_{s_multiplier}.pt"))





def Compute_a(vector):

    if len(vector.shape)!=1:
        raise RuntimeError(f"the vector a cannot be created from a tensor of dimension {len(vector.shape)}")
    a = torch.cat([vector, vector*(-1)], dim=0)
    return a


def Compute_S_matrix(a):

    size = a.shape[0]
    S = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if (a[i] > a[j]+1e-6):
                S[i,j] = 1
            elif (a[j] > a[i]+1e-6):
                S[i,j] = -1
            else:
                S[i,j] = 0
