import numpy as np
from node_functions import *
from system_definition import *
from incremental_subgradient import dJi_dxf
from incremental_subgradient import error
import matplotlib.pyplot as plt
from simulate_distributed_opt import logplot, plot_xT_evolution

def nodei_xfi(Ai, Bi, x0i, T, u_max, lambdai, z, rho, xf):
    nabla = dJi_dxf(Ai, Bi, x0i, T, u_max, xf)
    xfi = z - 1/rho * (nabla+lambdai)
    return xfi

def z_update(xf_array, lambda_array, rho):
    z = np.mean(xf_array + lambda_array/rho, axis = 0)
    return z

def lambda_arrray_update(lambda_array, xf_array, z, rho):
    n_nodes = lambda_array.shape[0]
    z_array = np.array([z for _ in range(n_nodes)])
    return lambda_array + rho * (xf_array - z_array)

def ADMM_update(rho, lambda_array, z, xf_array):
    n_nodes = len(A_list)
    for i in range(n_nodes):
        xf_array[i, :, :] = nodei_xfi(A_list[i], B_list[i], x0_list[i], T, u_max, lambda_array[i,:,:], z, rho, xf_array[i, :, :])

    z = z_update(xf_array, lambda_array, rho)

    lambda_array = lambda_arrray_update(lambda_array, xf_array, z, rho)

    return xf_array, z, lambda_array

def ADMM(rho = 100, max_iter = 20):
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    z = np.ones((n,1))
    lambda_array = np.ones((n_nodes, n, 1))
    xf_array = np.ones((n_nodes, n, 1))
    xf_array_hist = [xf_array.copy()]
    for iter in range(max_iter):
        try:
            xf_array, z, lambda_array = ADMM_update(rho, lambda_array, z, xf_array)
            xf_array_hist.append(xf_array.copy())
        except:
            print("divregent behaviour")
            break

    xf_array_hist = np.array(xf_array_hist)
    return xf_array, xf_array_hist

def error_vs_rho():
    fig, ax = plt.subplots(figsize=(8, 5))
    for rho in np.linspace(36, 50, 5):
        xf_array, xf_array_hist = ADMM(int(rho))
        error_ = error(xf_array_hist)
        logplot(error_, ax, label=r"$\rho = $" + f"{int(rho)}")
        

    plt.legend()
    plt.tight_layout()
    plt.show()

def solving_evolution_ADMM(A_list, B_list, x0_list, T, u_max, xf):
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    zero_lambda = np.zeros(n)
    u = [
        nodei_opt(A_list[i], B_list[i], x0_list[i], T, u_max, zero_lambda, zero_lambda)
        for i in range(n_nodes)
    ]
    x_140T = []
    for i in range(n_nodes):
        bar_Ai = make_bar_A(A_list[i], T)
        bar_Bi = make_bar_B(A_list[i], B_list[i], T)
        x_i0T= bar_Ai @ x0_list[i] + bar_Bi @ u[i]
        x_i0T = x_i0T.flatten().reshape(-1, n)
        x_140T.append(x_i0T)
    x_140T = np.array(x_140T)

    return x_140T


if __name__ == "__main__":
    # _, xf_array_hist = ADMM(rho=1, max_iter=20)
    # plot_xT_evolution(xf_array_hist[:, :, :, 0])
    error_vs_rho()
