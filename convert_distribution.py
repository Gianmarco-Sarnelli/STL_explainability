import torch
from Local_Matrix import local_matrix

def convert_to_local():
    converter = local_matrix()
    converter.compute_Q()
    Q = converter.Q
    print(Q)
    return 0

convert_to_local()