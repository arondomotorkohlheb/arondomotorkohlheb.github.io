import cvxpy as cp
import numpy as np
from system_definition import *
from node_functions import make_bar_A, make_bar_B


def centralized_lqr():
    # Problem dimensions
    I = len(A_list)  # number of agents
    n = A_list[0].shape[0]  # state dimension
    m = B_list[0].shape[1]  # input dimension



    # Optimization variables
    u = [cp.Variable((T * m, 1)) for _ in range(I)]
    x_f = cp.Variable((n, 1))  # final state is now a decision variable

    u_bound = np.asarray(u_max).reshape(-1)
    if u_bound.size == 1:
        u_bound = np.repeat(u_bound.item(), m)
    u_bound = np.tile(u_bound, T).reshape(-1, 1)

    objective = 0
    constraints = []

    for i in range(I):
        A = A_list[i]
        B = B_list[i]
        x0 = x0_list[i]
        
        barA = make_bar_A(A, T)
        barB = make_bar_B(A, B, T)

        # Compute x_{i,1:T} as a function of u_i
        # print(x0.shape)
        # print(barA.shape, barB.shape)
        x = barA @ x0 + barB @ u[i]

        objective += cp.sum_squares(x) + cp.sum_squares(u[i])

        # Input constraints
        constraints.append(cp.abs(u[i]) <= u_bound)

        # Final state constraint: x_i(T) == x_f
        x_T = x[-n:]
        constraints.append(x_T == x_f)

    # Solve the problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return [u[i].value for i in range(I)]
    # Output
    # print("Optimal final state x_f:", x_f.value)
