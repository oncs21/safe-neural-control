import torch

import numpy as np

from .bounds import compute_linear_bounds_CROWN
from .optimization import solve_reachability_lp_multi_agent, solve_linear_reachability_lp_multi_agent



def compute_collision_set(
        torch_model,
        facet_dirs,
        A_dyn,
        B_dyn,
        eps,
        A_next=None,
        B_next=None,
        x_bounds=(-5, 5),
        y_bounds=(-1, 1),
        u_bounds=(-1, 1)
):
    Psi, Phi, alpha, beta = compute_linear_bounds_CROWN(
        x_L=[x_bounds[0], y_bounds[0]],
        x_U=[x_bounds[1], y_bounds[1]],
        t_model=torch_model
    )
    
    c_star_store = []
    for a in facet_dirs:
        
        c_star = solve_reachability_lp_multi_agent(
            torch_model,
            A_dyn,
            B_dyn,
            Phi,
            beta,
            Psi,
            alpha,
            a,
            eps,
            A_next=A_next,
            b_next=B_next,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            u_bounds=u_bounds
        )


        c_star_store.append(c_star)

    A_p = torch.stack([a.clone().detach() for a in facet_dirs], dim=0)
    b_p = torch.tensor(c_star_store, dtype=torch.float32)
    return A_p, b_p



def partition_box(x_bounds, y_bounds, N):
    xs = np.linspace(x_bounds[0], x_bounds[1], N+1)
    ys = np.linspace(y_bounds[0], y_bounds[1], N+1)

    cells = []
    for i in range(N):
        for j in range(N):
            xL = xs[i]
            xU = xs[i+1]
            yL = ys[j]
            yU = ys[j+1]
            cells.append(([xL, yL], [xU, yU]))
    return cells




def compute_collision_set_with_partitioning(
    torch_model,
    facet_dirs,
    A_dyn,
    B_dyn,
    eps,
    A_next=None,
    B_next=None,
    x_bounds=(-5, 5),
    y_bounds=(-1, 1),
    u_bounds=(-1, 1)
):
    cells = partition_box(x_bounds, y_bounds, N=10)
    
    c_star_store = []
    for a in facet_dirs:
        c_candidates = []

        for (xL_cell, xU_cell) in cells:

            Psi, Phi, alpha, beta = compute_linear_bounds_CROWN(
                x_L=xL_cell,
                x_U=xU_cell,
                t_model=torch_model
            )

            c_k = solve_reachability_lp_multi_agent(
                torch_model,
                A_dyn,
                B_dyn,
                Phi,
                beta,
                Psi,
                alpha,
                a,
                eps,
                A_next=A_next,
                b_next=B_next,
                x_bounds=(xL_cell[0], xU_cell[0]),
                y_bounds=(xL_cell[1], xU_cell[1]),
                u_bounds=u_bounds
            )

            c_candidates.append(c_k)

        c_star_store.append(min(c_candidates))

    A_p = torch.stack([a.clone().detach() for a in facet_dirs], dim=0)
    b_p = torch.tensor(c_star_store, dtype=torch.float32)
    return A_p, b_p







