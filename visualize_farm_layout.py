import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Farm layout data
layout_x = [1347.6, 1640.9, 2248, 1934.2, 3149.3, 2227.5, 4218.2, 2520.8, 4097, 4036.4]
layout_y = [919, 1662.5, 1001, 2406, 1083.9, 3149.5, 1564.4, 3893, 2666.3, 3711.2]

# Farm parameters
diameter = 284.0
radius = diameter / 2.0
boundaries_xyz = [0.0, 5000.0, 0.0, 5000.0, 0.0, 1000.0]
x_min, x_max, y_min, y_max, z_min, z_max = boundaries_xyz

# Create figure
fig, ax = plt.subplots(figsize=(8.5, 8.5))

# Get default matplotlib colors
colors = plt.cm.tab10(np.linspace(0, 1, len(layout_x)))

# Plot turbine positions with 3-blade rotors
blade_radius = radius * 0.8  # Blade length proportional to rotor radius
for i, (x, y) in enumerate(zip(layout_x, layout_y)):
    color = colors[i]
    
    # Draw hollow circle for nacelle
    nacelle = Circle((x, y), radius * 0.1, fill=False, edgecolor=color, linewidth=2.5, zorder=3)
    ax.add_patch(nacelle)
    
    # Draw 3 blades at 120-degree intervals
    for blade_angle in [0, 120, 240]:
        angle_rad = np.radians(blade_angle)
        blade_x = x + blade_radius * np.cos(angle_rad)
        blade_y = y + blade_radius * np.sin(angle_rad)
        ax.plot([x, blade_x], [y, blade_y], color=color, linewidth=2.5, zorder=3)
    
    # Add turbine indices as labels
    ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=11, ha='left', zorder=4)

# Set domain boundaries
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Labels and title
ax.set_ylabel('Layout North to South (m)', fontsize=11)
ax.set_xlabel('Layout West to East (m)', fontsize=11)
ax.set_title(f'Wind Farm Layout (Turbine Positions, D={diameter}m)', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend with turbine colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], linewidth=2, label=f'Turbine {i+1}') 
                   for i in range(len(layout_x))]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, ncol=1)

# Set equal aspect ratio to show accurate spatial relationships
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
