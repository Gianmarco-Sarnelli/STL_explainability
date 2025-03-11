import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import itertools
import random
import faiss

from utils import load_pickle, dump_pickle, from_string_to_formula, get_leaves_idx, from_str_to_n_nodes
from traj_measure import BaseMeasure
from phis_generator import StlGenerator
from kernel import StlKernel


def split_phis(n_nodes, max_vars, time_list=None):
    all_n_nodes = load_pickle(os.getcwd(), '{}_nodes.pickle'.format(n_nodes))
    def get_max_var_idx(phi): return max(get_leaves_idx(phi)[1])
    def get_time_depth(phi): return from_string_to_formula(phi).time_depth()
    prev_used_indexes = set()
    for nvar in range(max_vars):
        current_var_idx_all = [i for i in range(len(all_n_nodes)) if get_max_var_idx(all_n_nodes[i]) <= nvar]
        current_var_idx = set(current_var_idx_all).difference(prev_used_indexes)
        current_n_var = [all_n_nodes[i] for i in list(current_var_idx)]
        prev_used_indexes.update(current_var_idx)
        if time_list is not None:
            time_list.insert(0, 0)
            current_time_depth = list(map(get_time_depth, current_n_var))
            for start_time, end_time in zip(time_list[:-1], time_list[1:]):
                time_idx = np.append(np.where(np.array(current_time_depth) <= end_time)[0],
                                     np.where(np.array(current_time_depth) > start_time)[0])
                time_phis = [current_n_var[i] for i in time_idx]
                dump_pickle('{}_nodes_{}_vars_{}_time'.format(n_nodes, nvar + 1, end_time), time_phis)
        else:
            dump_pickle('{}_nodes_{}_vars'.format(n_nodes, nvar + 1), current_n_var)


def batch_subset_phis(phis, max_vars, index_n, max_index_length, folder_index, timespan):
    for x in range(0, len(phis), max_index_length):
        current_batch = phis[x:x + max_index_length]
        max_nodes_batch = max(list(map(from_str_to_n_nodes, current_batch)))
        dump_pickle(folder_index + os.path.sep + 'phi_list_{}'.format(index_n), current_batch)
        index_info = {'nvars': max_vars, 'timespan': timespan, 'max_nodes': max_nodes_batch}
        dump_pickle(folder_index + os.path.sep + 'info_index_{}'.format(index_n), index_info)
        index_n += 1
    return index_n


def batch_phis(dim_range, max_vars, max_index_length, folder_index, folder_phis, time_list=None, first_idx=0,
               max_time=100):
    current_index = first_idx
    for nvars in range(max_vars):
        current_vars = nvars + 1
        if time_list is not None:
            for time in time_list:
                current_phis = list(itertools.chain(*[load_pickle(
                    folder_phis, '{}_nodes_{}_vars_{}_time.pickle'.format(n, current_vars, time))
                    for n in range(dim_range[0], dim_range[1]+1)]))
                random.shuffle(current_phis)
                # TODO: time for info when time_list is None
                current_index = batch_subset_phis(current_phis, nvars, current_index, max_index_length, folder_index,
                                                  time)
        else:
            current_phis = list(itertools.chain(*[load_pickle(
                folder_phis, '{}_nodes_{}_vars.pickle'.format(n, current_vars))
                for n in range(dim_range[0], dim_range[1] + 1)]))
            random.shuffle(current_phis)
            current_index = batch_subset_phis(current_phis, nvars, current_index, max_index_length, folder_index,
                                              max_time)
    return current_index


def build_index_from_formulae(cell_list, train_list, proj_matrix, index_n, pca_list, nvars, device, folder_index,
                              max_gram=50000):
    meas = BaseMeasure(device=device, sigma0=1.0, sigma1=1.0, q=0.1)
    stlkernel = StlKernel(meas, varn=nvars, sigma2=0.44, samples=10000)
    nlist = int(np.sqrt(len(cell_list)))
    m, nbits = [5, 8]
    d_list = [len(train_list)] + pca_list
    quantizers = [faiss.IndexFlatL2(d) for d in d_list]
    index_list = [faiss.IndexIVFPQ(q, d, nlist, m, nbits) for q, d in zip(quantizers, d_list)]
    faiss_dev = None
    # if device != 'cpu':
    #    faiss_dev = faiss.StandardGpuResources()
    for start_idx in range(0, len(cell_list), max_gram):
        current_index_list = [faiss.IndexIVFPQ(q, d, nlist, m, nbits) for q, d in zip(quantizers, d_list)]
        current_phis = list(map(from_string_to_formula, cell_list[start_idx:start_idx + max_gram]))
        current_gram = stlkernel.compute_bag_bag(current_phis, train_list)
        if start_idx > 0:
            # current_index_list[0] = faiss.index_cpu_to_gpu(faiss_dev, 0, current_index_list[0]) \
            #    if faiss_dev is not None else current_index_list[0]
            current_index_list[0].train(current_gram.cpu().numpy())
            current_index_list[0].add(current_gram.cpu().numpy())
            faiss.merge_into(index_list[0], current_index_list[0], True)
        else:
            # index_list[0] = faiss.index_cpu_to_gpu(faiss_dev, 0, index_list[0]) if faiss_dev is not None \
            #    else index_list[0]
            index_list[0].train(current_gram.cpu().numpy())
            index_list[0].add(current_gram.cpu().numpy())
        current_pca_full = torch.matmul(current_gram.float().to(device), proj_matrix.float().to(device))
        for i, n_pc in enumerate(pca_list):
            if start_idx > 0:
                # current_index_list[i+1] = faiss.index_cpu_to_gpu(faiss_dev, 0, current_index_list[i+1]) \
                #    if faiss_dev is not None else current_index_list[i+1]
                current_index_list[i+1].train(current_pca_full[:, :n_pc].cpu().numpy())
                current_index_list[i+1].add(current_pca_full[:, :n_pc].cpu().numpy())
                faiss.merge_into(index_list[i+1], current_index_list[i+1], True)
            else:
                # index_list[i + 1] = faiss.index_cpu_to_gpu(faiss_dev, 0, index_list[i + 1]) if faiss_dev is not None \
                #    else index_list[i + 1]
                index_list[i + 1].train(current_pca_full[:, :n_pc].cpu().numpy())
                index_list[i + 1].add(current_pca_full[:, :n_pc].cpu().numpy())
    names = ['index_{}'.format(index_n)] + ['kpca_{}_index_{}'.format(n_pc, index_n) for n_pc in pca_list]
    for index, name in zip(index_list, names):
        faiss.write_index(index, folder_index + os.path.sep + name + '.bin')


# GPU support for index
# faiss_dev = faiss.StandardGpuResources()
# signature_index = faiss.index_cpu_to_gpu(faiss_dev, 0, signature_index)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {device}".format(device=dev))
n_vars = 3
timespan_list = [50, 100]
base_folder = os.getcwd()
phis_folder = base_folder + os.path.sep + 'all_phis_{}_vars'.format(n_vars)
index_folder = base_folder + os.path.sep + 'index'
if not os.path.isdir(index_folder):
    os.mkdir(index_folder)
min_n_nodes = 2
max_n_nodes = 5
fix_formulae_list = False
if fix_formulae_list:
    os.chdir(phis_folder)
    for dim in range(min_n_nodes, max_n_nodes + 1):
        split_phis(dim, n_vars, time_list=None)
max_length = 1000000
n_stored_lists = len([f for f in os.listdir(index_folder) if f.startswith('phi_list')])
batch_formulae_list = False
if batch_formulae_list:
    batch_phis([min_n_nodes, max_n_nodes], n_vars, max_length, index_folder, phis_folder, first_idx=n_stored_lists,
               time_list=None, max_time=timespan_list[-1])
pc_list = [25, 50, 100, 250, 500]
if not os.path.isfile(index_folder + os.path.sep + 'train_phis_{}_vars.pickle'.format(n_vars)):
    sampler = StlGenerator(leaf_prob=0.5, time_bound_max_range=70, max_timespan=timespan_list[-1])
    train_phis = sampler.bag_sample(1000, nvars=n_vars)
    dump_pickle(index_folder + os.path.sep + 'train_phis_{}_vars'.format(n_vars), train_phis)
    mu0 = BaseMeasure(device=dev, sigma0=1.0, sigma1=1.0, q=0.1)
    kernel = StlKernel(mu0, varn=n_vars, sigma2=0.44)
    gram_train = kernel.compute_bag_bag(train_phis, train_phis)
    pca = PCA(n_components=pc_list[-1])
    pca_matrix = pca.fit_transform(gram_train)  # [len(train_phis), pc_list[-1]]
    dump_pickle(index_folder + os.path.sep + 'pca_proj_{}_vars'.format(n_vars), pca_matrix)
else:
    train_phis = load_pickle(index_folder, 'train_phis_{}_vars.pickle'.format(n_vars))
    print('Loaded training formulae')
    pca_matrix = load_pickle(index_folder, 'pca_proj_{}_vars.pickle'.format(n_vars))
    print('Loaded PCA projection matrix', pca_matrix.shape)
n_stored_indexes = len([f for f in os.listdir(index_folder) if f.startswith('index')])
print('number of stored indexes: ', n_stored_indexes)
pca_matrix = torch.from_numpy(pca_matrix).to(dev)
for index_id in range(n_stored_indexes, n_stored_lists):
    batch_phis = load_pickle(index_folder, 'phi_list_{}.pickle'.format(index_id))
    print(index_id, len(batch_phis))
    build_index_from_formulae(batch_phis, train_phis, pca_matrix, index_id, pc_list, n_vars, dev, index_folder,
                              max_gram=50000)
    idx_names = ['index_{}.bin'.format(index_id)] + ['kpca_{}_index_{}.bin'.format(n_pc, index_id) for n_pc in pc_list]
    list_indexes = [faiss.read_index(index_folder + os.path.sep + n) for n in idx_names]
    print([i.ntotal for i in list_indexes])
