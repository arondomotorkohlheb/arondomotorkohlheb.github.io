import numpy as np
import cvxpy as cp
from system_definition import *

def make_bar_A(A, T):
    n = A.shape[0]
    bar_A = np.zeros((n * (T), n))
    for i in range(T):
        bar_A[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A, i + 1)
    return bar_A

def make_bar_B(A, B, T):
    n = A.shape[0]
    m = B.shape[1]
    bar_B = np.zeros((n * (T), m * T))
    for i in range(T):
        for j in range(i + 1):
            bar_B[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - j) @ B
    return bar_B

def make_tilde_B(A, B, T):
    n = A.shape[0]
    m = B.shape[1]
    tilde_B = np.zeros((n, m * T))
    for i in range(T):
        tilde_B[:, i * m:(i + 1) * m] = np.linalg.matrix_power(A, T - i - 1) @ B
    return tilde_B

def nodei_opt(Ai, Bi, x0i, T, u_max, lambda_plus, lambda_minus):
    n, m = Bi.shape
    lambda_plus = np.asarray(lambda_plus).reshape(n, 1)
    lambda_minus = np.asarray(lambda_minus).reshape(n, 1)
    ui = cp.Variable((m * T, 1))
    bar_Ai = make_bar_A(Ai, T)
    bar_Bi = make_bar_B(Ai, Bi, T)
    tilde_Bi = make_tilde_B(Ai, Bi, T)
    xiT = np.linalg.matrix_power(Ai, T) @ x0i + tilde_Bi @ ui
    Ji = cp.sum_squares(bar_Ai @ x0i + bar_Bi @ ui) + cp.sum_squares(ui)
    obji = cp.sum(Ji + lambda_plus.T @ xiT - lambda_minus.T @ xiT)
    constraintsi = [cp.abs(ui) <= u_max]
    prob = cp.Problem(cp.Minimize(obji), constraintsi)
    prob.solve()
    return ui.value

def update_lambda(lambdas, alpha, A_list, B_list, x0_list, T, u_max):
    # lambdas: shape (N-1, n)
    # A_list, B_list, x0_list: lists of length N
    # Returns updated lambdas (N-1, n)
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    xT = []
    for i in range(n_nodes):
        lambda_plus = lambdas[i] if i < n_nodes - 1 else np.zeros(n)
        lambda_minus = lambdas[i - 1] if i > 0 else np.zeros(n)
        ui = nodei_opt(A_list[i], B_list[i], x0_list[i], T, u_max, lambda_plus, lambda_minus)
        tilde_Bi = make_tilde_B(A_list[i], B_list[i], T)
        xTi = np.linalg.matrix_power(A_list[i], T) @ x0_list[i] + tilde_Bi @ ui
        xT.append(xTi)

    v = np.vstack([(xT[i] - xT[i + 1]).reshape(1, -1) for i in range(n_nodes - 1)])
    lambdas_new = lambdas + alpha * v
    return lambdas_new, xT

def update_lambda_nesterov(lambdas, y, alpha, beta, A_list, B_list, x0_list, T, u_max):
    # lambdas: shape (N-1, n)
    # A_list, B_list, x0_list: lists of length N
    # Returns updated lambdas (N-1, n)
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    xT = []
    for i in range(n_nodes):
        lambda_plus = lambdas[i] if i < n_nodes - 1 else np.zeros(n)
        lambda_minus = lambdas[i - 1] if i > 0 else np.zeros(n)
        ui = nodei_opt(A_list[i], B_list[i], x0_list[i], T, u_max, lambda_plus, lambda_minus)
        tilde_Bi = make_tilde_B(A_list[i], B_list[i], T)
        xTi = np.linalg.matrix_power(A_list[i], T) @ x0_list[i] + tilde_Bi @ ui
        xT.append(xTi)

    v = np.vstack([(xT[i] - xT[i + 1]).reshape(1, -1) for i in range(n_nodes - 1)])
    lambdas_new = y + alpha * v
    y_new = lambdas_new + beta * (lambdas_new - lambdas)
    return lambdas_new, y_new, xT

if __name__ == "__main__":
    # Example usage (requires system_definition.py to define A_list, B_list, x0_list, T, u_max)
    # Initialize lambdas
    pass
    exit()
    lambdas = np.ones((3, 3))
    alpha = 0.1
    lambdas_new, xT = update_lambda(lambdas, alpha, A_list, B_list, x0_list, T, u_max)
    print("Updated lambdas:\n", lambdas_new)
    print("Terminal states:\n", xT)