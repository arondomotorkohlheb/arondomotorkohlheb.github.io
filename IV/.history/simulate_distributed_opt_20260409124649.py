import numpy as np
import matplotlib.pyplot as plt
from system_definition import *
from node_functions import *
import matplotlib as mpl
from matplotlib.ticker import LogLocator


def run_lambda_iterations(lambdas_init, alpha, A_list, B_list, x0_list, T, u_max, max_steps=200, tol=1e-4):
    lambdas_hist = [lambdas_init.copy()]
    xT_hist = []
    lambdas = lambdas_init.copy()
    for step in range(max_steps):
        lambdas_new, xT = update_lambda(lambdas, alpha, A_list, B_list, x0_list, T, u_max)
        xT_hist.append([x.flatten() for x in xT])
        lambdas_hist.append(lambdas_new.copy())
        if np.linalg.norm(lambdas_new - lambdas) < tol:
            break
        lambdas = lambdas_new
    
    xT_hist = np.array(xT_hist)  # shape: (steps, 4, n)
    return lambdas, xT_hist

def run_lambda_iterations_Nesterov(lambdas_init, alpha, beta, A_list, B_list, x0_list, T, u_max, max_steps=200, tol=1e-4):
    lambdas_hist = [lambdas_init.copy()]
    xT_hist = []
    lambdas = lambdas_init.copy()
    y =  lambdas_init.copy()
    for step in range(max_steps):
        lambdas_new, y, xT = update_lambda_nesterov(lambdas, y, alpha, beta, A_list, B_list, x0_list, T, u_max)
        xT_hist.append([x.flatten() for x in xT])
        lambdas_hist.append(lambdas_new.copy())
        if np.linalg.norm(lambdas_new - lambdas) < tol:
            break
        lambdas = lambdas_new
    
    xT_hist = np.array(xT_hist)  # shape: (steps, 4, n)
    return lambdas, xT_hist

def run_lambda_iterations_dyn_alpha(lambdas_init, alpha0, r, A_list, B_list, x0_list, T, u_max, max_steps=200, tol=1e-4):
    lambdas_hist = [lambdas_init.copy()]
    xT_hist = []
    lambdas = lambdas_init.copy()
    for step in range(max_steps):
        alpha = alpha0 * r ** step
        lambdas_new, xT = update_lambda(lambdas, alpha, A_list, B_list, x0_list, T, u_max)
        xT_hist.append([x.flatten() for x in xT])
        lambdas_hist.append(lambdas_new.copy())
        if np.linalg.norm(lambdas_new - lambdas) < tol:
            break
        lambdas = lambdas_new
    
    xT_hist = np.array(xT_hist)  # shape: (steps, 4, n)
    return lambdas, xT_hist

def plot_xT_evolution(xT_hist):
    """
    Plots the phase diagram (x1 vs x2) for each node's state trajectory at final time T over the iterations.
    xT_hist: numpy array of shape (steps, 4, 2)
    """
    steps, n_nodes, n = xT_hist.shape
    if n > 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    base_colors = [
        (0, 0, 0),        # Black (start)
        (1, 0, 0),        # Red
        (0, 0.5, 0),      # Green
        (0, 0, 1),        # Blue
        (1, 0.5, 0),      # Orange
    ]
    # For each node, create a colormap from black to its base color to white
    node_colors = [
        mpl.colors.LinearSegmentedColormap.from_list(
            f'node{node}_cmap', [base_colors[node+1], (1, 1, 1)]
        )
        for node in range(n_nodes)
    ]

    for node in range(n_nodes):
        x1 = xT_hist[:, node, 0]
        x2 = xT_hist[:, node, 1]
        if n > 2:
            x3 = xT_hist[:, node, 2]
            for i in range(steps - 1):
                color = node_colors[node](i / (steps - 1))
                ax.plot(
                    [x1[i], x1[i+1]],
                    [x2[i], x2[i+1]],
                    [x3[i], x3[i+1]],
                    color=color,
                    linewidth=2,
                    label=f'Plant {node+1}' if i == 0 else None
                )
            ax.scatter(x1[0], x2[0], x3[0], color='k', marker='o', s=40)
            ax.scatter(x1[-1], x2[-1], x3[-1], color='w', edgecolor='k', marker='o', s=60)
        else:
            for i in range(steps - 1):
                color = node_colors[node](i / (steps - 1))
                ax.plot(
                    [x1[i], x1[i+1]],
                    [x2[i], x2[i+1]],
                    color=color,
                    linewidth=2,
                    label=f'Plant {node+1}' if i == 0 else None
                )
            ax.scatter(x1[0], x2[0], color='k', marker='o', s=40)
            ax.scatter(x1[-1], x2[-1], color='w', edgecolor='k', marker='o', s=60)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    if n > 2:
        ax.set_zlabel(r'$x_3$')
        ax.set_title('3D Phase Diagram of xT Evolution (first 3 states)')
    else:
        ax.set_title('Phase Diagram of xT Evolution for Each Node')
    ax.legend()
    plt.tight_layout()
    plt.show()

def solving_evolution(A_list, B_list, x0_list, T, u_max, lambdas):
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    u = []
    for i in range(n_nodes):
        lambda_plus = lambdas[i] if i < n_nodes - 1 else np.zeros(n)
        lambda_minus = lambdas[i - 1] if i > 0 else np.zeros(n)
        u.append(nodei_opt(A_list[i], B_list[i], x0_list[i], T, u_max, lambda_plus, lambda_minus))

    x_140T = []
    for i in range(n_nodes):
        bar_Ai = make_bar_A(A_list[i], T)
        bar_Bi = make_bar_B(A_list[i], B_list[i], T)
        x_i0T= bar_Ai @ x0_list[i] + bar_Bi @ u[i]
        x_i0T = x_i0T.flatten().reshape(-1, n)
        x_140T.append(x_i0T)
    x_140T = np.array(x_140T)

    return x_140T 

def solving_evolution_us(A_list, B_list, x0_list, T, u_max, u):
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    x_140T = []
    for i in range(n_nodes):
        bar_Ai = make_bar_A(A_list[i], T)
        bar_Bi = make_bar_B(A_list[i], B_list[i], T)
        x_i0T= (bar_Ai @ x0_list[i])[:,0] + bar_Bi @ u[i]
        x_i0T = x_i0T.flatten().reshape(-1, n)
        x_140T.append(x_i0T)
    x_140T = np.array(x_140T)

    return x_140T 

def consensus_error_from_hist(x_hist):
    # x_hist shape: (steps, n_nodes, n)
    steps, n_nodes, _ = x_hist.shape
    error = np.zeros(steps)
    for k in range(steps):
        pairwise = [x_hist[k, i] - x_hist[k, i + 1] for i in range(n_nodes - 1)]
        v = np.vstack(pairwise)
        error[k] = np.mean(np.square(v))
    return error

def logplot(sequence, ax, label = None):
    if label is not None:
        ax.plot(sequence, label = label)
    else:
        ax.plot(sequence)

    ax.set_yscale('log')

    # Change grid resolution:
    # Major ticks at powers of 10
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

    # Minor ticks at custom intervals between major ticks
    # E.g., subs=range(2, 10) means: draw ticks at 2,3,...,9 * 10^n
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10, 2), numticks=100))

    # Grid lines
    ax.grid(True, which='major',  linestyle='--', linewidth=.4, color='gray')
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, color='gray')

    ax.set_xlabel('steps')
    ax.set_ylabel('error (log scale)')
    ax.set_title('Evolution of error on a log scale')

def plot_error():
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    alpha = 5
    lambdas_init = np.ones((n_nodes - 1, n))
    lambdas, xT_hist = run_lambda_iterations(
        lambdas_init, alpha, A_list, B_list, x0_list, T, u_max
    )

    error = consensus_error_from_hist(xT_hist)
    fig, ax = plt.subplots(figsize=(8, 5))
    logplot(error, ax)
    plt.tight_layout()
    plt.show()

def plot_error_xT(xT_hist):
    error = consensus_error_from_hist(xT_hist)
    fig, ax = plt.subplots(figsize=(8, 5))
    logplot(error, ax)
    plt.tight_layout()
    plt.show()

def plot_error_nesterov():
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    alpha = 7
    fig, ax = plt.subplots(figsize=(8, 5))
    beta_range = np.linspace(0.15,0.45,10)
    for beta in beta_range:
        lambdas_init = np.ones((n_nodes - 1, n))
        lambdas, xT_hist = run_lambda_iterations_Nesterov(
            lambdas_init, alpha, beta, A_list, B_list, x0_list, T, u_max
        )

        error = consensus_error_from_hist(xT_hist)
        logplot(error, ax, label=r"$\beta = $" + f"{round(beta,2)}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_4_evolutions(x_140T, x0_list=None):
    n_nodes, _, n = x_140T.shape

    if n > 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    base_colors = [
        (0, 0, 0),        # Black (start)
        (1, 0, 0),        # Red
        (0, 0.5, 0),      # Green
        (0, 0, 1),        # Blue
        (1, 0.5, 0),      # Orange
    ]
    # For each node, create a colormap from black to its base color to white
    node_colors = [
        mpl.colors.LinearSegmentedColormap.from_list(
            f'node{node}_cmap', [base_colors[node+1], (1, 1, 1)]
        )
        for node in range(n_nodes)
    ]

    for node in range(n_nodes):
        if x0_list is not None:
            x0 = np.asarray(x0_list[node]).reshape(1, -1)
            traj = np.vstack([x0, x_140T[node]])
        else:
            traj = x_140T[node]

        steps = traj.shape[0]
        x1 = traj[:, 0]
        x2 = traj[:, 1]
        if n > 2:
            x3 = traj[:, 2]
            for i in range(steps - 1):
                color = node_colors[node](i / (steps - 1))
                ax.plot(
                    [x1[i], x1[i+1]],
                    [x2[i], x2[i+1]],
                    [x3[i], x3[i+1]],
                    color=color,
                    linewidth=2,
                    label=f'Plant {node+1}' if i == 0 else None
                )
            ax.scatter(x1[0], x2[0], x3[0], color='k', marker='o', s=40)  # Start
            ax.scatter(x1[-1], x2[-1], x3[-1], color='w', edgecolor='k', marker='o', s=60)  # End
        else:
            for i in range(steps - 1):
                color = node_colors[node](i / (steps - 1))
                ax.plot(
                    [x1[i], x1[i+1]],
                    [x2[i], x2[i+1]],
                    color=color,
                    linewidth=2,
                    label=f'Plant {node+1}' if i == 0 else None
                )
            ax.scatter(x1[0], x2[0], color='k', marker='o', s=40)  # Start
            ax.scatter(x1[-1], x2[-1], color='w', edgecolor='k', marker='o', s=60)  # End

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    if n > 2:
        ax.set_zlabel(r'$x_3$')
        ax.set_title('3D Phase Diagram of xT Evolution (first 3 states)')
    else:
        ax.set_title('Phase Diagram of xT Evolution for Each Node')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test_static_alpha():
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    # Example system matrices and initial conditions
    lambdas_init = np.ones((n_nodes - 1, n))
    alpha_range = np.linspace(1,10,10)
    for alpha in alpha_range:
        lambdas, xT_hist = run_lambda_iterations(
            lambdas_init, alpha, A_list, B_list, x0_list, T, u_max
        )
        error = consensus_error_from_hist(xT_hist)
        logplot(error, ax, label = r"$\alpha = $" + f"{alpha}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_dynamic_alpha():
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    # Example system matrices and initial conditions
    lambdas_init = np.ones((n_nodes - 1, n))
    r_range = np.linspace(0.95,1,10)
    alpha0 = 7
    for r in r_range:
        lambdas, xT_hist = run_lambda_iterations_dyn_alpha(
            lambdas_init, alpha0, r, A_list, B_list, x0_list, T, u_max
        )
        error = consensus_error_from_hist(xT_hist)
        logplot(error, ax, label = r"$r= $" + f"{round(r,2)}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def solve_with_admm(rho=100, max_iter=50, plot=True, x0_list_admm=None):
    # Local import avoids circular imports at module load time.
    from admm import ADMM

    xf_array, xf_array_hist = ADMM(rho=rho, max_iter=max_iter, x0_list_local=x0_list_admm)
    if plot:
        plot_xT_evolution(xf_array_hist[:, :, :, 0])
    return xf_array, xf_array_hist

if __name__ == "__main__":
    # test_static_alpha()
    # lambdas_init = np.ones((6, 3))
    # lambdas, _ = run_lambda_iterations(lambdas_init, 5, A_list, B_list, x0_list, T, u_max)
    # x_140T = solving_evolution(A_list, B_list, x0_list, T, u_max, lambdas)

    # plot_4_evolutions(x_140T)
    n_nodes = len(A_list)
    n = A_list[0].shape[0]
    lambdas_init = np.ones((n_nodes - 1, n))
    lambdas, _ = run_lambda_iterations(lambdas_init, 5, A_list, B_list, x0_list, T, u_max)
    x_140T = solving_evolution(A_list, B_list, x0_list, T, u_max, lambdas)
    plot_4_evolutions(x_140T, x0_list=x0_list)
    # solve_with_admm(rho=100, max_iter=50, plot=True)
    # test_dynamic_alpha()
    # plot_error_nesterov()