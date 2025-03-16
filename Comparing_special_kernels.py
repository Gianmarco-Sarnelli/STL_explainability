import torch
import os

from model_to_formula import quantitative_model, new_kernel_to_embedding, kernel_to_new_kernel
from IR.utils import from_string_to_formula
from phis_generator import StlGenerator
from traj_measure import BaseMeasure
import pickle
from Computing_K_from_G import Compute_K




def Compute_robustness_to_embed(model_name, G_trajectories):

    # Device used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "maritime":
        n_vars_model = 2
    elif model_name in ("robot2", "robot4", "robot5"):
        n_vars_model = 3
    else:
        n_vars_model = 1

    # Choosing the right formula for the model:
    match model_name:
        case "human":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "linear":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "maritime":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "robot2":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "robot4":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "robot5":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
        case "train":
            best_phi_str = ""    #TODO: add the string form of each of the best formulae
            best_phi = from_string_to_formula(best_phi_str)
    

    # Creating the model
    model_path  = f'IR/data/data/{model_name}/model_state_dict.pth'
    quant_model = quantitative_model(model_path=model_path, nvars=n_vars_model)

    # Cutting and filling the trajectories
    G_trajectories_cut = G_trajectories[:,:n_vars_model,:] 
    G_trajectories_filled = torch.zeros((G_trajectories.shape[0], 3, G_trajectories.shape[2]), device=device)
    G_trajectories_filled[:, :n_vars_model, :] = G_trajectories_cut

    # Computing the two robustness
    rho_model = quant_model.robustness(traj=G_trajectories_cut)
    rho_formula = torch.tanh(best_phi.quantitative(G_trajectories_filled, evaluate_at_all_times=False))

    # Saving the robustness of the model
    os.makedirs("Special_kernel/rho_model", exist_ok=True)
    torch.save(rho_model, os.path.join("Special_kernel", "rho_model", f"{model_name}.pt"))
    # Saving the robustness of the formula
    os.makedirs("Special_kernel/rho_formula", exist_ok=True)
    torch.save(rho_formula, os.path.join("Special_kernel", "rho_formula", f"{model_name}.pt"))
    

def Compute_robustness_psi(N, M):
    """Computes the robustness matrix of the anchor_psis on the basis_trajectories"""

    # Loading the saved trajectories
    basis_trajectories = torch.load(os.path.join("Special_kernel", f"basis_trajectories_{M}.pt"))
    # Loading the saved formulae
    with open(os.path.join("Special_kernel", f"anchor_psis_{N}.pkl"), 'rb') as f:
        anchor_psis = pickle.load(f)

    rhos_anchor_psis= torch.zeros(N, M)
    for (i, formula) in enumerate(anchor_psis):
        rhos_anchor_psis[i, :] = torch.tanh(formula.quantitative(basis_trajectories, evaluate_at_all_times=False))
      
    # Saving the robustness matrix
    torch.save(rhos_anchor_psis, os.path.join("Special_kernel", f"rhos_anchor_psis_{N}_{M}.pt"))

def Save_Trajectories(n_traj=40):

    device= "cpu"

    mu0 = BaseMeasure()
    basis_trajectories = mu0.sample(samples=n_traj, varn=3, points=100)

    os.makedirs("Special_kernel", exist_ok=True)
    torch.save(basis_trajectories, os.path.join("Special_kernel", f"basis_trajectories_{n_traj}.pt"))


def Save_anchor_formulas(n_psi=20):

    # Parameters for the sampling of the formulae
    leaf_probability = 0.5
    time_bound_max_range = 10
    prob_unbound_time_operator = 0.1
    atom_threshold_sd = 1.0
    n_vars = 3

    formulae_distr = StlGenerator(leaf_prob=leaf_probability, 
                            time_bound_max_range=time_bound_max_range,
                            unbound_prob=prob_unbound_time_operator, 
                            threshold_sd=atom_threshold_sd)
    
    anchor_psis = formulae_distr.bag_sample(n_psi, n_vars)
    # Save with pickle
    os.makedirs("Special_kernel", exist_ok=True)
    with open(os.path.join("Special_kernel", f"anchor_psis_{n_psi}.pkl"), 'wb') as f:
        pickle.dump(anchor_psis, f)



if __name__ == "__main__":

    N = 20 # Fixing the number of formulae to 20
    M = 40 # Fixing the number of trajectories to 40

    # Chosing the model to use
    model_list = ["human", "linear", "maritime", "robot2", "robot4", "robot5", "train"]
    model_name = ""

    print(f"model name: {model_name}")

    Save_anchor_formulas(n_psi=N)
    Save_Trajectories(n_traj=M)
    Compute_robustness_psi(N, M)

    print("basic stuff are computed!")

