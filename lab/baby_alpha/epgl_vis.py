import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from epgl_grid import EPGLGrid

# 1. Setup the Simulation Environment
GRID_SIZE = 50
STEPS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

brain = EPGLGrid(size=GRID_SIZE, device=DEVICE)

# 2. Setup the Plotting Canvas
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('EPGL 2D Skin Brain (Cellular Token Automata)', fontsize=16)

# Titles for the 4 dimensions
axes[0, 0].set_title('E: Energy/Negentropy (Blue)')
axes[0, 1].set_title('P: Pressure/Stress (Red)')
axes[1, 0].set_title('G: Growth/Phase (Green)')
axes[1, 1].set_title('L: Link/Affinity (Purple)')

# Initialize image objects for the animation
im_E = axes[0, 0].imshow(brain.h[0].cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
im_P = axes[0, 1].imshow(brain.h[1].cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
im_G = axes[1, 0].imshow(brain.h[2].cpu().numpy(), cmap='Greens', vmin=0, vmax=1)
im_L = axes[1, 1].imshow(brain.h[3].cpu().numpy(), cmap='Purples', vmin=0, vmax=1)

# Turn off axis ticks for clean visual
for ax in axes.flat:
    ax.axis('off')

# 3. Animation Update Function
def update(frame):
    # Create an empty stimulus grid
    stimulus = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE)
    
    # ENVIRONMENT INTERVENTION (The "Task")
    # Food/Nutrients dropped in top-left periodically
    if frame % 20 < 10:
        stimulus[5:15, 5:15] = torch.rand(10, 10, device=DEVICE) * 0.8
        
    # Constant toxic stress/pressure dropped in bottom-right
    stimulus[35:45, 35:45] = torch.rand(10, 10, device=DEVICE) * 0.4
    
    # Tick the thermodynamic physics engine!
    h_new = brain.step(stimulus, death_pressure=0.85)
    
    # Update matplotlib visual frames
    im_E.set_array(h_new[0].cpu().numpy())
    im_P.set_array(h_new[1].cpu().numpy())
    im_G.set_array(h_new[2].cpu().numpy())
    im_L.set_array(h_new[3].cpu().numpy())
    
    return [im_E, im_P, im_G, im_L]

# 4. Run Animation
print(f"Starting EPGL Vis Simulation on {DEVICE} for {STEPS} steps...")
ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=50, blit=True)

# Save the animation so user can easily view it on Windows
output_file = 'epgl_skin_brain.gif'
ani.save(output_file, writer='pillow', fps=15)
print(f"Animation saved to {output_file}!")
