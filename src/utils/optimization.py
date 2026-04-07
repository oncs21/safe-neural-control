import torch
import torch.nn as nn

import numpy as np

import gurobipy as gp
from gurobipy import GRB

def solve_reachability_lp_multi_agent(
        torch_model,
        A_dyn,
        B_dyn,
        Phi, beta,
        Psi, alpha,       
        a_vec,
        eps,
        A_next,
        b_next,
        x_bounds=(-5, 5),
        y_bounds=(-1, 1),
        u_bounds=(-1, 1)
):
    A_dyn = A_dyn.detach().cpu().numpy()
    B_dyn = B_dyn.detach().cpu().numpy()
    Phi   = Phi.detach().cpu().numpy()
    beta  = beta.detach().cpu().numpy()
    Psi   = Psi.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    a     = a_vec.detach().cpu().numpy()


    m = gp.Model("facet_LP")
    m.params.OutputFlag = 0

    # Variables at time t-1
    x1 = m.addVar(lb=x_bounds[0], ub=x_bounds[1], name="x1")
    y1 = m.addVar(lb=y_bounds[0], ub=y_bounds[1], name="y1")
    x2 = m.addVar(lb=x_bounds[0], ub=x_bounds[1], name="x2")
    y2 = m.addVar(lb=y_bounds[0], ub=y_bounds[1], name="y2")

    p1 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p1")
    p2 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p2")
    m.addConstr(p1 == x1 - x2)
    m.addConstr(p2 == y1 - y2)

    # Next-state variables
    x1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x1_t")
    y1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y1_t")
    x2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x2_t")
    y2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y2_t")

    p1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p1_t")
    p2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p2_t")
    m.addConstr(p1_t == x1_t - x2_t)
    m.addConstr(p2_t == y1_t - y2_t)

    u1_L = float(Phi[0,0])*x1 + float(Phi[0,1])*y1 + float(beta[0][0])
    u1_U = float(Psi[0,0])*x1 + float(Psi[0,1])*y1 + float(alpha[0][0])

    u2_L = float(Phi[0,0])*x2 + float(Phi[0,1])*y2 + float(beta[0][0])
    u2_U = float(Psi[0,0])*x2 + float(Psi[0,1])*y2 + float(alpha[0][0])

    
    m.addConstr(u1_L >= u_bounds[0])
    m.addConstr(u1_U <= u_bounds[1])
    m.addConstr(u2_L >= u_bounds[0])
    m.addConstr(u2_U <= u_bounds[1])

    B_plus  = np.maximum(B_dyn, 0.0)
    B_minus = np.minimum(B_dyn, 0.0)

    # Agent 1 next-state bounds
    m.addConstr(
        x1_t <= A_dyn[0,0]*x1 + A_dyn[0,1]*y1
              + B_plus[0,0]*u1_U + B_minus[0,0]*u1_L
    )
    m.addConstr(
        x1_t >= A_dyn[0,0]*x1 + A_dyn[0,1]*y1
              + B_plus[0,0]*u1_L + B_minus[0,0]*u1_U
    )

    m.addConstr(
        y1_t <= A_dyn[1,0]*x1 + A_dyn[1,1]*y1
              + B_plus[1,0]*u1_U + B_minus[1,0]*u1_L
    )
    m.addConstr(
        y1_t >= A_dyn[1,0]*x1 + A_dyn[1,1]*y1
              + B_plus[1,0]*u1_L + B_minus[1,0]*u1_U
    )

    # Agent 2 next-state bounds
    m.addConstr(
        x2_t <= A_dyn[0,0]*x2 + A_dyn[0,1]*y2
              + B_plus[0,0]*u2_U + B_minus[0,0]*u2_L
    )
    m.addConstr(
        x2_t >= A_dyn[0,0]*x2 + A_dyn[0,1]*y2
              + B_plus[0,0]*u2_L + B_minus[0,0]*u2_U
    )

    m.addConstr(
        y2_t <= A_dyn[1,0]*x2 + A_dyn[1,1]*y2
              + B_plus[1,0]*u2_U + B_minus[1,0]*u2_L
    )
    m.addConstr(
        y2_t >= A_dyn[1,0]*x2 + A_dyn[1,1]*y2
              + B_plus[1,0]*u2_L + B_minus[1,0]*u2_U
    )

    if A_next is None:
        # collision strip at time t
        m.addConstr(p1_t <= eps)
        m.addConstr(p1_t >= -eps)
        

    else:
        A_next_np = A_next.detach().cpu().numpy()
        b_next_np = b_next.detach().cpu().numpy()

        for i in range(A_next_np.shape[0]):
            m.addConstr(
                A_next_np[i, 0] * p1_t + A_next_np[i, 1] * p2_t
                >= b_next_np[i]
            )

    

    m.setObjective(a[0]*p1 + a[1]*p2, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP can't be solved, status={m.Status}")

    return float(m.ObjVal)




def solve_linear_reachability_lp_multi_agent(
    torch_model: nn.Sequential,
    A_dyn,
    B_dyn,
    W_eq,
    b_eq,
    a_vec,
    eps,
    A_next,
    b_next,
    x_bounds,
    y_bounds,
    u_bounds
):
    W_eq = W_eq.detach().cpu().numpy()
    b_eq = b_eq.detach().cpu().numpy()
    A_dyn = A_dyn.detach().cpu().numpy()
    B_dyn = B_dyn.detach().cpu().numpy()
    a = a_vec.detach().cpu().numpy()

    m = gp.Model("facet_LP")
    m.params.OutputFlag = 0

    # Variables at time t-1
    x1 = m.addVar(lb=x_bounds[0], ub=x_bounds[1], name="x1")
    y1 = m.addVar(lb=y_bounds[0], ub=y_bounds[1], name="y1")
    x2 = m.addVar(lb=x_bounds[0], ub=x_bounds[1], name="x2")
    y2 = m.addVar(lb=y_bounds[0], ub=y_bounds[1], name="y2")

    p1 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p1")
    p2 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p2")

    m.addConstr(p1 == x1 - x2)
    m.addConstr(p2 == y1 - y2)

    # Controls
    u1 = m.addVar(lb=u_bounds[0], ub=u_bounds[1])
    u2 = m.addVar(lb=u_bounds[0], ub=u_bounds[1])

    m.addConstr(u1 == W_eq[0,0]*x1 + W_eq[0,1]*y1 + b_eq[0])
    m.addConstr(u2 == W_eq[0,0]*x2 + W_eq[0,1]*y2 + b_eq[0])

    # Variables at time t
    x1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x1_t")
    y1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y1_t")
    x2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x2_t")
    y2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y2_t")

    m.addConstr(x1_t == A_dyn[0,0]*x1 + A_dyn[0,1]*y1 + B_dyn[0,0]*u1)
    m.addConstr(y1_t == A_dyn[1,0]*x1 + A_dyn[1,1]*y1 + B_dyn[1,0]*u1)

    m.addConstr(x2_t == A_dyn[0,0]*x2 + A_dyn[0,1]*y2 + B_dyn[0,0]*u2)
    m.addConstr(y2_t == A_dyn[1,0]*x2 + A_dyn[1,1]*y2 + B_dyn[1,0]*u2)

    p1_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p1_t")
    p2_t = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p2_t")

    m.addConstr(p1_t == x1_t - x2_t)
    m.addConstr(p2_t == y1_t - y2_t)

    if A_next is None:
        m.addConstr(p1_t <= eps)
        m.addConstr(p2_t >= -eps)
    else:
        A_next_np = A_next.detach().cpu().numpy()
        b_next_np = b_next.detach().cpu().numpy()

        for i in range(A_next_np.shape[0]):
            m.addConstr(
                A_next_np[i, 0] * p1_t + A_next_np[i, 1] * p2_t
                >= b_next_np[i]
            )

    m.setObjective(a[0]*p1 + a[1]*p2, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("LP can't be solved")
    
    return float(m.ObjVal)