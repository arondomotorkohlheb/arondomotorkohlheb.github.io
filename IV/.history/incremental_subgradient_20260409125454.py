import numpy as np
import cvxpy as cp
from system_definition import *
from node_functions import make_bar_A, make_bar_B, make_tilde_B
import matplotlib.pyplot as plt


def Ji_xf(Ai, Bi, x0i, T, u_max, xf):
    n, m = Bi.shape
    ui = cp.Variable((m * T, 1))
    bar_Ai = make_bar_A(Ai, T)
    bar_Bi = make_bar_B(Ai, Bi, T)
    tilde_Bi = make_tilde_B(Ai, Bi, T)
    xiT = np.linalg.matrix_power(Ai, T) @ x0i + tilde_Bi @ ui
    Ji = cp.sum_squares(bar_Ai @ x0i + bar_Bi @ ui) + cp.sum_squares(ui)
    obji = Ji
    u_bound = np.asarray(u_max).reshape(-1)
    if u_bound.size == 1:
        u_bound = np.repeat(u_bound.item(), m)
    u_bound = np.tile(u_bound, T).reshape(-1, 1)
    constraintsi = [cp.abs(ui) <= u_bound, xiT == xf]
    prob = cp.Problem(cp.Minimize(obji), constraintsi)
    prob.solve()
    J = prob.value
    return J

def dJi_dxf(Ai, Bi, x0i, T, u_max, xf, dxf = 0.1):
    n = Ai.shape[0]
    grad = np.zeros((n, 1))
    for j in range(n):
        delta = np.zeros((n, 1))
        delta[j, 0] = dxf
        J_plus = Ji_xf(Ai, Bi, x0i, T, u_max, xf + delta)
        J_minus = Ji_xf(Ai, Bi, x0i, T, u_max, xf - delta)
        grad[j, 0] = (J_plus - J_minus) / (2 * dxf)
    return grad

def local_xf_update(i, alpha, xf, T = T, u_max = u_max):
    Ai = A_list[i]
    Bi = B_list[i]
    x0i = x0_list[i]
    nabla = dJi_dxf(Ai, Bi, x0i, T, u_max, xf)
    xf = xf - alpha * nabla
    return xf

def incremental_subgradient_step(alpha, xf_array, W, consensus_steps):
    # get the xf update for each node
    xf_array_new = np.zeros(xf_array.shape)
    n_nodes = len(A_list)
    for i in range(n_nodes):
        xf_array_new[i,:,:] = local_xf_update(i, alpha, xf_array[i])
    # merge xf results with W
    xf_array_new = np.tensordot(np.linalg.matrix_power(W, consensus_steps), xf_array_new, axes=(1, 0))
    return xf_array_new

def incremental_subgradient(consensus_steps = 30):
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    W = np.array([
        [0.75, 0.25, 0.0, 0.0],
        [0.25, 0.5, 0.25, 0.0],
        [0.0, 0.25, 0.5, 0.25],
        [0.0, 0.0, 0.25, 0.75]
    ])
    if W.shape[0] != n_nodes:
        raise ValueError("W must match the number of nodes in A_list/B_list/x0_list")
    # starting values for xf
    xf_array = np.array([np.zeros((n,1)) for _ in range(n_nodes)])
    xf_array_hist = []
    # descend step size
    alpha = 0.0001
    max_iterations = 50
    for iteration in range(max_iterations):
        xf_prev = xf_array.copy()
        xf_array = incremental_subgradient_step(alpha, xf_array, W, consensus_steps)
        xf_array_hist.append(xf_array)
        if np.mean(np.square(xf_array - xf_prev)) < 1e-8:
            break

    return xf_array, np.array(xf_array_hist)

def error(xf_array_hist):
    # Consensus error over adjacent node pairs, averaged per iteration.
    n_nodes = xf_array_hist.shape[1]
    pairwise = []
    for i in range(n_nodes - 1):
        diff_i = xf_array_hist[:, i, :, :] - xf_array_hist[:, i + 1, :, :]
        pairwise.append(np.mean(np.square(diff_i), axis=(1, 2)))
    return np.mean(np.array(pairwise), axis=0)

def error_vs_cons_steps():
    from simulate_distributed_opt import logplot
    fig, ax = plt.subplots(figsize=(8, 5))
    for consensus_step in np.linspace(1, 50, 10):
        xf_array, xf_array_hist = incremental_subgradient(consensus_steps = int(consensus_step))
        error_ = error(xf_array_hist)
        logplot(error_, ax, label=r"c = " + f"{int(consensus_step)}")
    
    print(xf_array)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    from simulate_distributed_opt import plot_xT_evolution
    xf_array, xf_array_hist = incremental_subgradient(consensus_steps = 50)
    plot_xT_evolution(xf_array_hist[:,:,:,0])