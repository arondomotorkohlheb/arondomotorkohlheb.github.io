import numpy as np
import cvxpy as cp
from system_definition import *
from node_functions import make_bar_A, make_bar_B, make_tilde_B
from simulate_distributed_opt import logplot, plot_xT_evolution
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
    constraintsi = [cp.abs(ui) <= u_max, xiT == xf]
    prob = cp.Problem(cp.Minimize(obji), constraintsi)
    prob.solve()
    J = prob.value
    return J

def dJi_dxf(Ai, Bi, x0i, T, u_max, xf, dxf = 0.1):
    xf1 = xf + np.array([[dxf], [0]])
    xf2 = xf - np.array([[dxf], [0]])
    xf3 = xf + np.array([[0],[dxf]])
    xf4 = xf - np.array([[0],[dxf]])

    J1 = Ji_xf(Ai, Bi, x0i, T, u_max, xf1)
    J2 = Ji_xf(Ai, Bi, x0i, T, u_max, xf2)
    J3 = Ji_xf(Ai, Bi, x0i, T, u_max, xf3)
    J4 = Ji_xf(Ai, Bi, x0i, T, u_max, xf4)

    return np.array([
        [(J1 - J2) / (2*dxf)],
        [(J3 - J4) / (2*dxf)]
    ])

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
    for i in range(4):
        xf_array_new[i,:,:] = local_xf_update(i, alpha, xf_array[i])
    # merge xf results with W
    xf_array_new = np.tensordot(np.linalg.matrix_power(W, consensus_steps), xf_array_new, axes=(1, 0))
    return xf_array_new

def incremental_subgradient(consensus_steps = 30):
    W = np.array([
        [0.75, 0.25, 0.0, 0.0],
        [0.25, 0.5, 0.25, 0.0],
        [0.0, 0.25, 0.5, 0.25],
        [0.0, 0.0, 0.25, 0.75]
    ])
    # starting values for xf
    n = 3
    xf_array = np.array([np.zeros((n,1)) for i in range(4)])
    xf_array_hist = []
    # descend step size
    alpha = 0.02
    xf_true = np.array([[-0.1078],[0.2258]])
    max_iterations = 50
    for iteration in range(max_iterations):
        xf_array = incremental_subgradient_step(alpha, xf_array, W, consensus_steps)
        xf_array_hist.append(xf_array)
        if iteration > 2:
            if np.mean(np.square(xf_array_hist[-1]-xf_true)) < 1e-8:
                break

    return xf_array, np.array(xf_array_hist)

def error(xf_array_hist):
    print("xf hist shape",xf_array_hist.shape)
    xf_true = np.array([[-0.1078],[0.2258], []])
    mean = xf_true
    diff = np.mean(np.array([np.mean(np.square(xf_array_hist[:, i, :, :] - mean), axis = (1,2)) for i in range(4)]), axis=0)
    return diff

def error_vs_cons_steps():
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
    xf_array, xf_array_hist = incremental_subgradient(consensus_steps = 50)
    plot_xT_evolution(xf_array_hist[:,:,:,0])