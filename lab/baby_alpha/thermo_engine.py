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
        # 🧬 4. Membrane Permeability (m_target)
        # ==========================================
        # A calm, energetic cell (High E, Low P) opens up to the network.
        # A stressed cell (High P) closes its membrane to avoid more chaos.
        m_target = torch.sigmoid(E_new - P_new)
        
        # ==========================================
        # 🧬 5. Growth & Lifespan / Phase-shift (G & l_decay)
        # ==========================================
        # Cell differentiates/grows when it masters its environment (E > P).
        dG_dt = 0.1 * E_new - 0.5 * P_new
        G_new = G + dG_dt * self.dt
        
        # If Pressure exceeds critical threshold and Energy is depleted, 
        # the cell's lifespan (l_decay) drops precipitously (Trigger Death).
        l_decay = 0.01 + 0.5 * F.relu(P_new - 2.0) 
        
        return E_new, P_new, G_new, m_target, l_decay

class ThermoInnerCell(nn.Module):
    def __init__(self, dt=0.1):
        super().__init__()
        # Entirely replaced the MLPs with the Thermodynamic ODE system!
        self.ode = ThermodynamicODE(dt)

    def forward(self, E, P, G, h_tilde, m, c, l):
        E_new, P_new, G_new, m_target, l_decay = self.ode(E, P, G, h_tilde)
        
        # Smooth physical transition for membrane (m is highly multi-dimensional in original,
        # but here we treat ODE m_target as a global scalar multiplier for the membrane matrix, 
        # or just expand it to match m's dimension).
        m_target = m_target.expand_as(m)
        m_new = 0.9 * m + 0.1 * m_target
        
        # Decrease lifespan based on stress and basal decay
        l_new = l - l_decay * self.ode.dt
        l_new = torch.clamp(l_new, min=0.0, max=1.0)
        
        return E_new, P_new, G_new, m_new, c, l_new

class OuterTrans(nn.Module):
    """
    Macro-coupling. Slime Mold inspired attention routing.
    """
    def __init__(self, dim_e, dim_m, dim_c):
        super().__init__()
        self.q_proj = nn.Linear(dim_e + dim_c, dim_e)
        self.k_proj = nn.Linear(dim_e + dim_c, dim_e)
        self.v_proj = nn.Linear(dim_e + dim_c, dim_e)
        
    def forward(self, e, m, c, l):
        x = torch.cat([e, c], dim=-1)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        # Slime Mold Hook + Lifespan mask
        membrane_mask = torch.matmul(m, m.transpose(-2, -1))
        lifespan_mask = torch.matmul(l, l.transpose(-2, -1))
        
        adaptive_scores = attn_scores * membrane_mask * lifespan_mask
        attn_weights = F.softmax(adaptive_scores, dim=-1)
        
        h_tilde = torch.matmul(attn_weights, v) 
        e_new = e + h_tilde 
        
        return e_new, h_tilde

class ThermoCellularPINN(nn.Module):
    def __init__(self, dim_e, dim_m, dim_c, depth=4, dt=0.1):
        super().__init__()
        self.depth = depth
        self.outer_layers = nn.ModuleList([OuterTrans(dim_e, dim_m, dim_c) for _ in range(depth)])
        self.inner_layers = nn.ModuleList([ThermoInnerCell(dt=dt) for _ in range(depth)])

    def forward(self, e, E, P, G, m, c, l, death_threshold=0.05):
        assert e.size(0) == 1, "Dynamic Pruning requires Batch Size = 1 for streaming"
        
        for i in range(self.depth):
            # Macro-routing
            e_new, h_tilde = self.outer_layers[i](e, m, c, l)
            
            # Micro-metabolism (Thermodynamic ODE)
            E_new, P_new, G_new, m_new, c_new, l_new = self.inner_layers[i](E, P, G, h_tilde, m, c, l)
            
            e, E, P, G, m, c, l = e_new, E_new, P_new, G_new, m_new, c_new, l_new
            
        # ==========================================
        # 💀 DYNAMIC PRUNING (The Executioner)
        # ==========================================
        alive_mask = (l > death_threshold).squeeze(0).squeeze(-1) # [Seq]
        
        if not alive_mask.any():
            return e[:, :0, :], E[:, :0, :], P[:, :0, :], G[:, :0, :], m[:, :0, :], c[:, :0, :], l[:, :0, :], 0
            
        e_alive = e[:, alive_mask, :]
        E_alive = E[:, alive_mask, :]
        P_alive = P[:, alive_mask, :]
        G_alive = G[:, alive_mask, :]
        m_alive = m[:, alive_mask, :]
        c_alive = c[:, alive_mask, :]
        l_alive = l[:, alive_mask, :]
        
        num_died = e.size(1) - e_alive.size(1)
        
        return e_alive, E_alive, P_alive, G_alive, m_alive, c_alive, l_alive, num_died
