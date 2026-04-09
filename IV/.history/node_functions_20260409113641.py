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
    ui = cp.Variable((m * T, 1))
    bar_Ai = make_bar_A(Ai, T)
    bar_Bi = make_bar_B(Ai, Bi, T)
    tilde_Bi = make_tilde_B(Ai, Bi, T)
    xiT = np.linalg.matrix_power(Ai, T) @ x0i + tilde_Bi @ ui
    Ji = cp.sum_squares(bar_Ai @ x0i + bar_Bi @ ui) + cp.sum_squares(ui)
    obji = cp.sum(Ji + lambda_plus.T @ xiT - lambda_minus.T @ xiT)
    # print("obji", obji.shape)
    # exit()
    constraintsi = [cp.abs(ui) <= u_max]
    prob = cp.Problem(cp.Minimize(obji), constraintsi)
    prob.solve()
    return ui.value

def update_lambda(lambdas, alpha, A_list, B_list, x0_list, T, u_max):
    # lambdas: shape (3, n)
    # A_list, B_list, x0_list: lists of length 4, each element is (n, n), (n, m), (n, 1)
    # Returns updated lambdas (3, n)
    u_stars = []
    xT = []
    # Node 1
    u1 = nodei_opt(A_list[0], B_list[0], x0_list[0], T, u_max, lambdas[0], np.zeros_like(lambdas[0]))
    u_stars.append(u1)
    tilde_B1 = make_tilde_B(A_list[0], B_list[0], T)
    xT1 = np.linalg.matrix_power(A_list[0], T) @ x0_list[0] + tilde_B1 @ u1
    xT.append(xT1)

    # Node 2
    u2 = nodei_opt(A_list[1], B_list[1], x0_list[1], T, u_max, lambdas[1], lambdas[0])
    u_stars.append(u2)
    tilde_B2 = make_tilde_B(A_list[1], B_list[1], T)
    xT2 = np.linalg.matrix_power(A_list[1], T) @ x0_list[1] + tilde_B2 @ u2
    xT.append(xT2)

    # Node 3
    u3 = nodei_opt(A_list[2], B_list[2], x0_list[2], T, u_max, lambdas[2], lambdas[1])
    u_stars.append(u3)
    tilde_B3 = make_tilde_B(A_list[2], B_list[2], T)
    xT3 = np.linalg.matrix_power(A_list[2], T) @ x0_list[2] + tilde_B3 @ u3
    xT.append(xT3)

    # Node 4
    u4 = nodei_opt(A_list[3], B_list[3], x0_list[3], T, u_max, np.zeros_like(lambdas[2]), lambdas[2])
    u_stars.append(u4)
    tilde_B4 = make_tilde_B(A_list[3], B_list[3], T)
    xT4 = np.linalg.matrix_power(A_list[3], T) @ x0_list[3] + tilde_B4 @ u4
    xT.append(xT4)

    # v(u*) vector
    v = np.vstack([
        xT[0].T - xT[1].T,
        xT[1].T - xT[2].T,
        xT[2].T - xT[3].T
    ])
    # Update lambdas
    
    lambdas_new = lambdas + alpha * v
    return lambdas_new, xT

def update_lambda_nesterov(lambdas, y, alpha, beta, A_list, B_list, x0_list, T, u_max):
    # lambdas: shape (3, n)
    # A_list, B_list, x0_list: lists of length 4, each element is (n, n), (n, m), (n, 1)
    # Returns updated lambdas (3, n)
    u_stars = []
    xT = []
    # Node 1
    u1 = nodei_opt(A_list[0], B_list[0], x0_list[0], T, u_max, lambdas[0], np.zeros_like(lambdas[0]))
    u_stars.append(u1)
    tilde_B1 = make_tilde_B(A_list[0], B_list[0], T)
    xT1 = np.linalg.matrix_power(A_list[0], T) @ x0_list[0] + tilde_B1 @ u1
    xT.append(xT1)

    # Node 2
    u2 = nodei_opt(A_list[1], B_list[1], x0_list[1], T, u_max, lambdas[1], lambdas[0])
    u_stars.append(u2)
    tilde_B2 = make_tilde_B(A_list[1], B_list[1], T)
    xT2 = np.linalg.matrix_power(A_list[1], T) @ x0_list[1] + tilde_B2 @ u2
    xT.append(xT2)

    # Node 3
    u3 = nodei_opt(A_list[2], B_list[2], x0_list[2], T, u_max, lambdas[2], lambdas[1])
    u_stars.append(u3)
    tilde_B3 = make_tilde_B(A_list[2], B_list[2], T)
    xT3 = np.linalg.matrix_power(A_list[2], T) @ x0_list[2] + tilde_B3 @ u3
    xT.append(xT3)

    # Node 4
    u4 = nodei_opt(A_list[3], B_list[3], x0_list[3], T, u_max, np.zeros_like(lambdas[2]), lambdas[2])
    u_stars.append(u4)
    tilde_B4 = make_tilde_B(A_list[3], B_list[3], T)
    xT4 = np.linalg.matrix_power(A_list[3], T) @ x0_list[3] + tilde_B4 @ u4
    xT.append(xT4)

    # v(u*) vector
    v = np.vstack([
        xT[0].T - xT[1].T,
        xT[1].T - xT[2].T,
        xT[2].T - xT[3].T
    ])
    # Update lambdas
    
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