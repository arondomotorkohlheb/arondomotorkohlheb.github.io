import numpy as np
import matplotlib.pyplot as plt
file_path = r'speedgoat_185_yaw_steering.npy'

data = np.load(file_path)
data = data[:, :, :]

labels = [r'$\beta_ref$', r'$\gamma_{ref}$', r'$\omega_{ref}$', r'$\beta$', r'$\gamma$', r'$\omega$', r'$c_p$', r'$P_{ref}$', r'$P_{gen}$' , r'$c_p$', r'c_t']


# Signal indices
IDX_BETA_REF  = 0
IDX_OMEGA_REF = 2
IDX_BETA      = 3
IDX_OMEGA     = 5
IDX_P_REF     = 7
IDX_P_GEN     = 8

t = np.linspace(0, data.shape[0]*2, data.shape[0])

fig, axs = plt.subplots(3, 1, figsize=(8.5, 11.5), sharex=True)
 
# --- β vs β_ref ---
for k in range(data.shape[1]):
    line, = axs[0].plot(t, data[:, k, IDX_BETA])        # achieved (solid)
    axs[0].plot(
        t,
        data[:, k, IDX_BETA_REF],
        linestyle='--',
        color=line.get_color()                          # same color
    )

axs[0].set_ylabel(r'$\beta$ $[\degree]$')
axs[0].set_title(r'$(\beta$ vs $\beta_{ref})_{1:N}$')
axs[0].grid(True)

# Add legend-like box explaining line styles
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='grey', linewidth=2, label='Achieved'),
                   Line2D([0], [0], color='grey', linewidth=2, linestyle='--', label='Reference')]
axs[0].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.4,1), fontsize=11, ncol=1)


# --- ω vs ω_ref ---
for k in range(data.shape[1]):
    line, = axs[1].plot(t, data[:, k, IDX_OMEGA], label=f'Turbine {k+1}')       # achieved (solid)
    axs[1].plot(
        t,
        np.abs(data[:, k, IDX_OMEGA_REF]),
        linestyle='--',
        color=line.get_color()                          # same color
    )

axs[1].set_ylabel(r'$\omega$ [rad/s]')
axs[1].set_title(r'($\omega$ vs $\omega_{ref})_{1:N}$')
axs[1].grid(True)
axs[1].legend(loc='center left', bbox_to_anchor=(-0.4, 1.6), fontsize=11, ncol=1)
# --- Power for each turbine (faded) and average power ---
P_gen = data[:, :, IDX_P_GEN]
P_ref = data[:, :, IDX_P_REF]
lines_turbines = []
for k in range(data.shape[1]):
    line, = axs[2].plot(t, P_gen[:, k], alpha=0.6)  # faded lines for each turbine
    lines_turbines.append(line)
    # Add reference power as dashed line with same color
    axs[2].plot(t, P_ref[:, k], linestyle='--', color=line.get_color(), alpha=0.6, linewidth=1)

# Calculate and plot average power
P_gen_avg = P_gen.mean(axis=1)
P_ref_avg = P_ref.mean(axis=1)
color_11th = plt.cm.tab20(10)  # 11th color (0-indexed)
line_mean, = axs[2].plot(t, P_gen_avg, linewidth=2, color=color_11th, label=r'$P_{\Sigma}/N$')
axs[2].plot(t, P_ref_avg, linestyle='--', color=color_11th, linewidth=2, label=r'$P_{ref,\Sigma}/N$')

axs[2].set_ylabel(r'$P$ [W]')
axs[2].set_title(r'$P_{1:N}$, $P_{\Sigma}/N$')
axs[2].set_xlabel(r'Time [s]')
axs[2].grid(True)
axs[2].legend(loc='upper left', bbox_to_anchor=(-0.4, 2.4), fontsize=11, ncol=1)


# Create legend on the subplot
plt.subplots_adjust(left=0.3, right=0.95, top=0.97, bottom=0.04)
plt.show()