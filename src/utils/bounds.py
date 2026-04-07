import torch

import numpy as np

from collections import defaultdict

from auto_LiRPA_repo.auto_LiRPA.bound_general import BoundedModule
from auto_LiRPA_repo.auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA_repo.auto_LiRPA.perturbations import PerturbationLpNorm



def compute_linear_bounds_CROWN(x_L, x_U, t_model, device='cpu'):
    y_L = torch.tensor(x_L, dtype=torch.float32).reshape(-1, 1) # (2, 1)
    y_U = torch.tensor(x_U, dtype=torch.float32).reshape(-1, 1) # (2, 1)

    y_center = 0.5 * (y_L + y_U)


    bounded_model = BoundedModule(t_model, y_center.T, device=device)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=y_L.T, x_U=y_U.T)
    y_t = BoundedTensor(y_center.T, ptb)

    with torch.no_grad():
        pred = bounded_model(y_t)

    outn = bounded_model.output_name[0]
    inn = bounded_model.input_name[0]
    need = defaultdict(set)
    need[outn].add(inn)

    lb, ub, A_mtx = bounded_model.compute_bounds(x=(y_t,), method="backward", return_A=True, needed_A_dict=need)

    phi = A_mtx[outn][inn]['lA'][0]
    beta = A_mtx[outn][inn]['lbias'][0]

    psi = A_mtx[outn][inn]['uA'][0]
    alpha = A_mtx[outn][inn]['ubias'][0]

    phi = phi.reshape(-1, y_center.shape[0])
    alpha = alpha.reshape(-1, 1)
    psi = psi.reshape(-1, y_center.shape[0])
    beta = beta.reshape(-1, 1)

    return psi, phi, alpha, beta
