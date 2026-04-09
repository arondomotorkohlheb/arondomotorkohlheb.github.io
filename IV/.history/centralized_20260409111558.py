import cvxpy as cp
import numpy as np
from system_definition import *
from node_functions import make_bar_A, make_bar_B


def centralized_lqr():
    # Problem dimensions
    I = 4         # number of agents      # time horizon
    n = 3         # state dimension
    m = 1         # input dimension



    # Optimization variables
    u = [cp.Variable(T * m) for _ in range(I)]
    x_f = cp.Variable(n)  # final state is now a decision variable

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
        x = (barA @ x0)[:,0] + barB @ u[i]

        objective += cp.sum_squares(x) + cp.sum_squares(u[i])

        # Input constraints
        constraints.append(cp.abs(u[i]) <= u_max)

        # Final state constraint: x_i(T) == x_f
        x_T = x[38:]
        constraints.append(x_T == x_f)

    # Solve the problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return np.array([u[i].value for i in range(I)])
    # Output
    # print("Optimal final state x_f:", x_f.value)
