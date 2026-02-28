import torch
import torch.nn.functional as F

class EPGLGrid:
    """
    2D Cellular Grid Engine natively using PyTorch Tensor operations.
    H: (Energy E, Pressure P, Growth G, Link L)
    No deep learning parameters, solely thermodynamic updates and localized topological convolution.
    """
    def __init__(self, size=50, device='cpu'):
        self.size = size
        self.device = device
        
        # h tensor: [Channels=4, H=size, W=size]
        self.h = torch.zeros((4, size, size), device=device)
        
        # "Stem cell" drop in the center to kickstart the system
        center = size // 2
        self.h[0, center-1:center+2, center-1:center+2] = 0.8 # E
        self.h[2, center-1:center+2, center-1:center+2] = 0.8 # G
        self.h[3, center-1:center+2, center-1:center+2] = 0.8 # L
        
        # 3x3 Mean Convolution Kernel for Neighborhood
        # Shape structure: [out_channels=1, in_channels=1, kH=3, kW=3]
        self.neighbor_kernel = torch.tensor([
            [0.1, 0.15, 0.1],
            [0.15, 0.0, 0.15],
            [0.1, 0.15, 0.1]
        ], device=device).view(1, 1, 3, 3)
        # We set center to 0.0 so we strictly get the *neighbors* mean

    def get_neighbor_mean(self, grid_channel):
        """Calculates neighbor mean using 2D Conv"""
        x = grid_channel.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        # padding=1 ensures boundaries stay the same size
        # padding_mode='replicate' prevents dark edges
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        mean_x = F.conv2d(x_pad, self.neighbor_kernel)
        return mean_x.squeeze(0).squeeze(0)

    def step(self, stim_grid, death_pressure=0.8):
        """
        Executes one timestep of EPGL physics dynamically.
        stim_grid: [size, size] tensor of external energy/stimulus.
        """
        E, P, G, L = self.h[0], self.h[1], self.h[2], self.h[3]
        
        # 1. Parallel neighbor communication
        En = self.get_neighbor_mean(E)
        Pn = self.get_neighbor_mean(P)
        Gn = self.get_neighbor_mean(G)
        Ln = self.get_neighbor_mean(L)
        
        # 2. Parallel state evolution (The user's NotebookLM explicit rules)
        
        # Energy: fed by stimulus, drained by pressure and active growth
        E_new = E + 0.3 * stim_grid - 0.4 * P - 0.2 * G
        
        # Pressure: spiked by stimulus, infectious from neighbors, mitigated by energy
        P_new = P + 0.5 * stim_grid + 0.3 * (Pn - P) - 0.2 * E
        
        # Growth: desires high E and low P, infectious from neighbors, repressed by P
        G_new = G + 0.4 * E * (1.0 - P) + 0.2 * (Gn - G) - 0.3 * P
        
        # Link (Connection Will): desires energetic/growing neighbors, infectious, repressed by P
        good_neighbor = 0.5 * En + 0.5 * Gn
        L_new = L + 0.4 * good_neighbor + 0.3 * (Ln - L) - 0.3 * P
        
        # 3. Apply biological limitations (Clamp to [0.0, 1.0])
        E_new = torch.clamp(E_new, 0.0, 1.0)
        P_new = torch.clamp(P_new, 0.0, 1.0)
        G_new = torch.clamp(G_new, 0.0, 1.0)
        L_new = torch.clamp(L_new, 0.0, 1.0)
        
        # 4. Apoptosis (Programmed Cell Death & Garbage Collection)
        # If Energy depleted AND Pressure exceeds threshold -> Cell dies (All zeros)
        death_mask = (E_new < 0.05) & (P_new > death_pressure)
        
        E_new[death_mask] = 0.0
        P_new[death_mask] = 0.0
        G_new[death_mask] = 0.0
        L_new[death_mask] = 0.0
        
        # Overwrite internal state
        self.h = torch.stack([E_new, P_new, G_new, L_new], dim=0)
        
        return self.h
