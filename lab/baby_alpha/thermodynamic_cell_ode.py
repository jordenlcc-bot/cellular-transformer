import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermodynamicODE(nn.Module):
    """
    Explicit Non-linear ODE System replacing the Black-Box MLP/GRU.
    Follows biological thermodynamics (Le Chatelier-Braun principle) and Dissipative Structures.
    """
    def __init__(self, dt=0.1):
        super().__init__()
        self.dt = dt
        
        # Le Chatelier-Braun constants (Initialized as learnable, but behave physically)
        # alpha: Sensitivity to external stimulus (stress generation rate)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # beta: Efficiency of Energy (E) in neutralizing Pressure/Stress (P)
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # gamma: Energy cost per unit of stress neutralization
        self.gamma = nn.Parameter(torch.tensor(0.2))
        
        # Basal metabolism (rate of gathering negentropy from pure existence)
        self.basal_metabolism = nn.Parameter(torch.tensor(0.05))
        
    def forward(self, E, P, G, h_tilde):
        """
        E: Energy / Negentropy [Batch, Seq, 1]
        P: Pressure / Stress [Batch, Seq, 1]
        G: Growth / Phase-shift [Batch, Seq, 1]
        h_tilde: External message influx [Batch, Seq, DimE]
        """
        # 1. External Stimulus Force (S)
        # S represents the kinetic energy/entropy of the incoming signal norm
        S = torch.norm(h_tilde, dim=-1, keepdim=True) ** 2
        
        # ==========================================
        # 🧬 2. ODE Update: Pressure (Entropy/Stress)
        # ==========================================
        # Pressure increases due to stimulus S, but the cell actively burns
        # Energy (E) to neutralize this pressure (Le Chatelier-Braun negative feedback).
        dP_dt = self.alpha * S - self.beta * (P * E)
        
        # ==========================================
        # 🧬 3. ODE Update: Energy (Negentropy)
        # ==========================================
        # Energy naturally replenishes slowly (basal metabolism), but is consumed 
        # when fighting stress (proportional to P * E).
        dE_dt = self.basal_metabolism - self.gamma * (P * E)
        
        # Integrate using Euler Method
        P_new = P + dP_dt * self.dt
        E_new = E + dE_dt * self.dt
        
        # Constraint: Physical quantities E and P cannot be negative
        P_new = F.relu(P_new)
        E_new = F.relu(E_new)
        
        # ==========================================
        # 🧬 4. Membrane Permeability (m)
        # ==========================================
        # A calm, energetic cell (High E, Low P) opens up to the network.
        # A stressed cell (High P) closes its membrane to avoid more chaos.
        m_new = torch.sigmoid(E_new - P_new)
        
        # ==========================================
        # 🧬 5. Growth & Lifespan / Phase-shift (G & l)
        # ==========================================
        # Cell differentiates/grows when it masters its environment (E > P).
        dG_dt = 0.1 * E_new - 0.5 * P_new
        G_new = G + dG_dt * self.dt
        
        # If Pressure exceeds critical threshold and Energy is depleted, 
        # the cell's lifespan (l_decay) drops precipitously (Trigger Death).
        l_decay = 0.01 + 0.1 * F.relu(P_new - 5.0) 
        
        return E_new, P_new, G_new, m_new, l_decay
