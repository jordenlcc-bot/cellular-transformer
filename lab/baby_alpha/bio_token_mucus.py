import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🧬 1. Genetic Decoding (Bits -> Blocks)
# ==========================================
def decode_bits_to_epgl(bit_tensor):
    """
    Simulates genetic decoding from raw bits to initial EPGL thermodynamic states.
    Input: bit_tensor [Batch=1, Num_Tokens, Num_Blocks, 8] (8-bits per block/byte)
    Output: EPGL state [Batch=1, Num_Tokens, Num_Blocks, 4]
    
    Mapping rule (per 2 bits):
    00 -> 0.0 (Low)
    01 -> 0.33 (Mid-Low)
    10 -> 0.66 (Mid-High)
    11 -> 1.0 (High)
    """
    # Group pairs of bits: (bit1 * 2 + bit2) / 3.0
    val_E = (bit_tensor[..., 0] * 2 + bit_tensor[..., 1]) / 3.0
    val_P = (bit_tensor[..., 2] * 2 + bit_tensor[..., 3]) / 3.0
    val_G = (bit_tensor[..., 4] * 2 + bit_tensor[..., 5]) / 3.0
    val_L = (bit_tensor[..., 6] * 2 + bit_tensor[..., 7]) / 3.0
    
    return torch.stack([val_E, val_P, val_G, val_L], dim=-1)

# ==========================================
# 🦠 2. The Bio-Token (Internal Mucus Ecology)
# ==========================================
class BioTokenMucusSim(nn.Module):
    """
    Executes the pure numerical EPGL updates and the W Mucus Matrix routing
    WITHIN a single token made of multiple blocks.
    No MLP weights employed here!
    """
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        
    def step_internal_ecology(self, h, W, stim):
        """
        h: [Batch, Seq, Blocks, 4] (E, P, G, L)
        W: [Batch, Seq, Blocks, Blocks] (Mucus connection strength)
        stim: [Batch, Seq, Blocks] (External semantic stimulus focused on specific blocks)
        """
        E, P, G, L = h[..., 0], h[..., 1], h[..., 2], h[..., 3]
        
        # ----------------------------------------------------
        # A. Mucus Information Routing (Calculate Neighbor Mean)
        # ----------------------------------------------------
        # We mathematically exclude self-connections for the neighbor calculation
        W_no_self = W * (1.0 - torch.eye(self.num_blocks, device=W.device))
        
        # Inflow = sum_j (W_ji * h_j) -> dot product over blocks
        # h is [B, S, Blk, 4], W is [B, S, Blk(dest), Blk(src)]
        inflow = torch.matmul(W_no_self, h) # [B, S, Blk, 4]
        
        total_w = W_no_self.sum(dim=-1, keepdim=True) + 1e-8 # Prevent div by zero
        En_Pn_Gn_Ln = inflow / total_w
        
        En, Pn, Gn, Ln = En_Pn_Gn_Ln[..., 0], En_Pn_Gn_Ln[..., 1], En_Pn_Gn_Ln[..., 2], En_Pn_Gn_Ln[..., 3]
        
        # ----------------------------------------------------
        # B. EPGL Explicit Thermodynamic Rules (No MLP)
        # ----------------------------------------------------
        E_new = torch.clamp(E + 0.3 * stim - 0.4 * P - 0.2 * G, 0.0, 1.0)
        P_new = torch.clamp(P + 0.5 * stim + 0.3 * (Pn - P) - 0.2 * E, 0.0, 1.0)
        G_new = torch.clamp(G + 0.4 * E * (1.0 - P) + 0.2 * (Gn - G) - 0.3 * P, 0.0, 1.0)
        
        good_neighbor = 0.5 * En + 0.5 * Gn
        L_new = torch.clamp(L + 0.4 * good_neighbor + 0.3 * (Ln - L) - 0.3 * P, 0.0, 1.0)
        
        h_new = torch.stack([E_new, P_new, G_new, L_new], dim=-1)
        
        # ----------------------------------------------------
        # C. dynamic W Mucus Matrix Re-wiring (Slime mode)
        # ----------------------------------------------------
        # Difference in state between block i and block j
        h_diff = torch.cdist(h_new, h_new, p=2.0) # [B, S, Blk, Blk]
        
        # Mutual willingness to link (Average L)
        L_unsqueeze_i = L_new.unsqueeze(3) # [B, S, Blk, 1]
        L_unsqueeze_j = L_new.unsqueeze(2) # [B, S, 1, Blk]
        mutual_L = (L_unsqueeze_i + L_unsqueeze_j) / 2.0
        
        # If they are different but willing to link, branch out!
        alpha = 0.1 * mutual_L
        beta = 0.05 # Natural decay
        
        W_new = W + alpha * h_diff - beta * W
        W_new = torch.clamp(W_new, 0.0, 1.0)
        
        # Reset diagonal to 0
        W_new = W_new * (1.0 - torch.eye(self.num_blocks, device=W.device))
        
        return h_new, W_new

# ==========================================
# 🌌 3. The Triple-Layer Bio-Token Model
# ==========================================
class TripleLayerBioPINN(nn.Module):
    def __init__(self, num_blocks=8, d_model=256):
        super().__init__()
        self.num_blocks = num_blocks
        self.ecology = BioTokenMucusSim(num_blocks)
        
        # The ultimate manifestation: Translating the evolved Mucus W and h
        # back into the standard Macro-space for global Transformer-style attention.
        # W has (num_blocks * num_blocks) features, h has (num_blocks * 4) features
        flattened_dim = (num_blocks * num_blocks) + (num_blocks * 4)
        
        # Outer membrane adapter (The ONLY deep learning weights here, mapping micro physics to macro space)
        self.micro_to_macro = nn.Linear(flattened_dim, d_model)
        
    def forward(self, bit_tensor, macro_stimulus, internal_steps=10):
        """
        bit_tensor: Raw 8-bit sequences [Batch, Seq, Num_Blocks, 8]
        macro_stimulus: Outer contextual stimuli coming from other tokens
        """
        B, S, Blk, _ = bit_tensor.shape
        
        # 1. Genetic Decode: Bits -> Biology
        h = decode_bits_to_epgl(bit_tensor) # [B, S, Blk, 4]
        
        # Initialize Mucus connection randomly
        W = torch.rand((B, S, Blk, Blk), device=bit_tensor.device) * 0.1
        W = W * (1.0 - torch.eye(Blk, device=W.device))
        
        # 2. Run the inner physical ecology for several steps
        for step in range(internal_steps):
            # Pass macro contextual stimulus to the blocks (randomly focusing for now)
            local_stim = macro_stimulus.unsqueeze(-1) * torch.ones((B, S, Blk), device=bit_tensor.device)
            # In real applications, local_stim could be selectively routed to specific blocks!
            
            h, W = self.ecology.step_internal_ecology(h, W, local_stim)
            
        # 3. Micro to Macro Outcropping (The resulting "Meaning")
        # Flatten W and h per token
        h_flat = h.view(B, S, -1)
        W_flat = W.view(B, S, -1)
        
        physical_morphology = torch.cat([h_flat, W_flat], dim=-1)
        
        # The macro expression of this evolved bio-token
        macro_embedding = self.micro_to_macro(physical_morphology)
        
        return macro_embedding, h, W
