import torch
import os
import numpy as np
import faiss
import time

from utils import load_pickle, from_str_to_n_nodes, execution_time
from kernel import StlKernel
from traj_measure import BaseMeasure
from phis_generator import StlGenerator


def recover_indexes(nvar, folder_index, timespan=None, nodes=None):
    index_info_names = [f for f in os.listdir(folder_index) if f.startswith('info')]
    index_infos = [load_pickle(folder_index, f) for f in index_info_names]
    index_nvar = [i for i, info in enumerate(index_infos) if info['nvars'] <= nvar]
    index_ok = index_nvar
    if timespan is not None:
        index_time = [i for i, info in enumerate(index_infos) if info['timespan'] <= timespan]
        index_ok = list(set(index_ok) & set(index_time))
    if nodes is not None:
        index_nodes = [i for i, info in enumerate(index_infos) if info['max_nodes'] == nodes]
        index_ok = list(set(index_ok) & set(index_nodes))
    index_numbers = [int(index.split('_')[-1][:-7]) for index in index_info_names]
    return [index_numbers[i] for i in index_ok]


def search_from_embeddings(embeddings, nvar, folder_index, k, n_neigh, n_pc=-1, timespan=None, nodes=None):
    print('embedding computed')
    index_numbers = recover_indexes(nvar, folder_index, timespan=timespan, nodes=nodes)
    all_phis, n_phis, dists, ids = [[] for _ in range(4)]
    n_phis.append(0)
    for index_n in index_numbers:                                   
        index_name = 'index_{}.bin'.format(index_n) if n_pc == -1 else 'kpca_{}_index_{}.bin'.format(n_pc, index_n)
        current_index = faiss.read_index(folder_index + os.path.sep + index_name)
        current_index.nprobe = n_neigh
        current_dist, current_ids = current_index.search(x=embeddings.cpu().numpy(), k=k)
        dists.append(current_dist)
        ids.append(current_ids)
        current_phis = load_pickle(folder_index, 'phi_list_{}.pickle'.format(index_n))
        all_phis += current_phis
        n_phis.append(len(current_phis))
    result_heap = faiss.ResultHeap(nq=embeddings.shape[0], k=k)
    offsets = np.cumsum(n_phis)
    for w in range(len(index_numbers)):
        result_heap.add_result(D=dists[w], I=ids[w] + offsets[w])
    result_heap.finalize()  # result_heap.I result_heap.D
    return [[all_phis[j] for j in row] for row in result_heap.I], result_heap.D


def get_embeddings(folder_index, max_n_vars, device, phis, n_pc=-1):
    train_phis = load_pickle(folder_index, 'train_phis_{}_vars.pickle'.format(max_n_vars))
    mu0 = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
    kernel = StlKernel(mu0, varn=max_n_vars, sigma2=0.44, samples=10000)
    gram_phis = kernel.compute_bag_bag(phis, train_phis)
    embeddings = gram_phis
    if n_pc != -1:
        pca_matrix = torch.from_numpy(load_pickle(folder_index, 'pca_proj_{}_vars.pickle'.format(
            max_n_vars))).to(device)
        embeddings = torch.matmul(gram_phis.to(device).float(), pca_matrix[:, :n_pc].float())
    return embeddings


def search_from_formulae(phis, max_n_vars, nvar, folder_index, k, n_neigh, device, n_pc=-1, timespan=None, nodes=None):
    embeddings = get_embeddings(folder_index, max_n_vars, device, phis, n_pc=n_pc)
    return search_from_embeddings(embeddings, nvar, folder_index, k, n_neigh, device, n_pc=n_pc, timespan=timespan,
                                  nodes=nodes)

def similarity_based_relevance(phi_searched, phi_retrieved_list, max_n_vars, device, boolean=True,
                               test_trajectories=None):
    if test_trajectories is None:
        mu0 = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
        test_trajectories = mu0.sample(10000, max_n_vars)
    rob_vector_searched = phi_searched.quantitative(test_trajectories, normalize=True).unsqueeze(0)
    rob_vector_others = torch.cat([phi.quantitative(test_trajectories, normalize=True).unsqueeze(0)
                                   for phi in phi_retrieved_list], dim=0)
    rob_vector_searched_norm = rob_vector_searched / rob_vector_searched.norm(dim=1)[:, None]
    rob_vector_others_norm = rob_vector_others / rob_vector_others.norm(dim=1)[:, None]
    cosine_similarity = torch.mm(rob_vector_searched_norm, rob_vector_others_norm.transpose(0, 1))
    if boolean:
        sat_vector_searched = (rob_vector_searched >= 0).float()
        sat_vector_others = (rob_vector_others >= 0).float()
        sat_diff = torch.mean(torch.abs(sat_vector_searched - sat_vector_others))
        return cosine_similarity, sat_diff
    return cosine_similarity


# dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Running on {device}".format(device=dev))
# base_folder = os.getcwd()
# index_folder = base_folder + os.path.sep + 'index' trajectories under analysis
# sampler = StlGenerator(leaf_prob=0.5)
# all_p = sampler.bag_sample(1000, nvars=3)
# all_phis_str = list(map(str, all_p))
# nodes_phis = np.array(list(map(from_str_to_n_nodes, all_phis_str)))
# search_phis = [all_p[i] for i in np.where(nodes_phis <= 5)[0]]
# print('Searching {} phis', len(search_phis))
# topk = 10
# neigh = 32
# l_of_l = search_from_formulae(search_phis, 3, 3, index_folder, topk, neigh, dev)
# print('ok index')
# search_from_formulae(search_phis, 3, 3, index_folder, 5, 64, dev, n_pc=25)
# print('ok index 25')
# search_from_formulae(search_phis, 3, 3, index_folder, 5, 64, dev, n_pc=50)
# print('ok index 50')
# search_from_formulae(search_phis, 3, 3, index_folder, 5, 64, dev, n_pc=100)
# print('ok index 100')
# search_from_formulae(search_phis, 3, 3, index_folder, 5, 64, dev, n_pc=250)
# print('ok index 250')
# search_from_formulae(search_phis, 3, 3, index_folder, 5, 64, dev, n_pc=500)
# print('ok index 500')
