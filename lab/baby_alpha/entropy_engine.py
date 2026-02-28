import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerCellUpdate(nn.Module):
    """
    Inner state machine for each CellToken.
    Updates hidden state (h), membrane (m), cell type (c), and lifespan (l).
    """
    def __init__(self, dim_h, dim_e, dim_m, dim_c):
        super().__init__()
        # Metabolism RNN to digest external influx (h_tilde)
        self.metabolism_rnn = nn.GRUCell(input_size=dim_e, hidden_size=dim_h)
        self.membrane_updater = nn.Linear(dim_h, dim_m)

    def forward(self, h, h_tilde, m, c, l, variance_threshold=1e-4, decay_rate=0.1):
        """
        h_tilde: [Batch, Seq, DimE] - External information influx
        h, m, c, l: Current internal states
        """
        B, L, _ = h_tilde.size()
        
        # Flatten for GRUCell
        h_tilde_flat = h_tilde.view(B * L, -1)
        h_flat = h.view(B * L, -1)
        
        # 1. Digest external information, update internal state
        h_new_flat = self.metabolism_rnn(h_tilde_flat, h_flat)
        h_new = h_new_flat.view(B, L, -1)
        
        # 2. Update membrane permeability based on new energy/state
        m_new = torch.sigmoid(self.membrane_updater(h_new))
        
        # 3. [TELEONOMY / ENTROPY-DRIVEN LIFESPAN] 
        # Calculate how much 'h' has changed. If change is small, it has reached its attractor.
        # h_diff indicates the surprise/information gain.
        h_diff = torch.norm(h_new - h, dim=-1) # [Batch, Seq]
        
        # If the state change is below a threshold, the token is 'satisfied' (purpose fulfilled).
        # We accelerate its lifespan decay. Otherwise, standard slow metabolism decay.
        is_satisfied = (h_diff < variance_threshold).float()
        
        # Lifespan update: standard decay + accelerated decay if satisfied
        l_new = l - (decay_rate * (1.0 + 5.0 * is_satisfied))
        l_new = torch.clamp(l_new, min=0.0, max=1.0) # Keep in [0, 1] range
        
        c_new = c # Cell type remains static for now
        
        return h_new, m_new, c_new, l_new

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
        
        # Raw communication intent
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        # Slime Mold Hook: Physical membrane matching (channel thickness)
        membrane_mask = torch.matmul(m, m.transpose(-2, -1))
        
        # Death Hook: Dying tokens cannot transmit or receive efficiently
        lifespan_mask = torch.matmul(l.unsqueeze(-1), l.unsqueeze(-1).transpose(-2, -1))
        
        # Composite physical routing grid (D_ij)
        adaptive_scores = attn_scores * membrane_mask * lifespan_mask
        attn_weights = F.softmax(adaptive_scores, dim=-1)
        
        # Aggregate environment influx
        h_tilde = torch.matmul(attn_weights, v) 
        
        # Update expression
        e_new = e + h_tilde 
        
        return e_new, h_tilde

class EntropyCellularPINN(nn.Module):
    def __init__(self, dim_e, dim_h, dim_m, dim_c, depth=4):
        super().__init__()
        self.depth = depth
        self.dim_e = dim_e
        self.dim_h = dim_h
        self.dim_m = dim_m
        self.dim_c = dim_c
        
        self.outer_layers = nn.ModuleList([OuterTrans(dim_e, dim_m, dim_c) for _ in range(depth)])
        self.inner_layers = nn.ModuleList([InnerCellUpdate(dim_h, dim_e, dim_m, dim_c) for _ in range(depth)])

    def forward(self, e, h, m, c, l, death_threshold=0.05):
        """
        Forward pass with Dynamic Pruning out of Dead Tokens.
        Current inputs: [Batch=1, Seq, Dim]
        """
        # Note: We assume Batch = 1 for this dynamic length simulation. 
        # Dynamic pruning with Batch > 1 requires padding/masking, but for a 
        # continuous stream on a single sequence (like a brain), Batch=1 is perfect.
        assert e.size(0) == 1, "Dynamic Pruning requires Batch Size = 1"
        
        for i in range(self.depth):
            # Macro-routing
            e_new, h_tilde = self.outer_layers[i](e, m, c, l)
            
            # Micro-metabolism
            h_new, m_new, c_new, l_new = self.inner_layers[i](h, h_tilde, m, c, l)
            
            e, h, m, c, l = e_new, h_new, m_new, c_new, l_new
            
        # ==========================================
        # 💀 DYNAMIC PRUNING (The Executioner)
        # ==========================================
        # Find which tokens are still alive
        alive_mask = (l > death_threshold).squeeze(0) # [Seq]
        
        if not alive_mask.any():
            # If all died, return empty tensors of correct shape
            return e[:, :0, :], h[:, :0, :], m[:, :0, :], c[:, :0, :], l[:, :0]
            
        # Slice out only the alive tokens natively in PyTorch
        # This PHYSICALLY frees memory and shrinks the sequence dimension!
        e_alive = e[:, alive_mask, :]
        h_alive = h[:, alive_mask, :]
        m_alive = m[:, alive_mask, :]
        c_alive = c[:, alive_mask, :]
        l_alive = l[:, alive_mask]
        
        num_died = e.size(1) - e_alive.size(1)
        
        return e_alive, h_alive, m_alive, c_alive, l_alive, num_died
