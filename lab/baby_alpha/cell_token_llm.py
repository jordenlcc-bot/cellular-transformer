"""
CellToken LLM - Full LLM Architecture
======================================
Architecture: Raw Token IDs → CellEmbedding → (EPGL Inner Steps × n) → TokenCollapse → OuterTransformer × depth → LM Head

The key idea:
- Each token is NOT just a static vector. It contains `num_blocks` CellBlocks.
- Each CellBlock has a 4-dim EPGL state: (Energy, Pressure, Growth, Link).
- Blocks within a token exchange states via a dynamic Mucus W matrix (slime mold routing).
- After inner_steps of EPGL updates, all blocks are collapsed into one macro embedding.
- Macro embeddings form a standard token sequence, then processed by an outer Transformer.
- A final LM Head maps to vocabulary logits for next-token prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧬 Layer 1: Cell Embedding
# Maps raw token IDs to biological initial state (e, h, W)
# ============================================================
class CellEmbedding(nn.Module):
    """
    Converts a token ID into:
      e: outer embedding (standard semantic vector) [B, S, d_e]
      h: initial EPGL block states [B, S, num_blocks, 4]
      W: initial mucus connection matrix [B, S, num_blocks, num_blocks]
    """
    def __init__(self, vocab_size: int, d_e: int, num_blocks: int, max_seq_len: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_e = d_e

        # Standard semantic embedding (this is the outer 'e' vector)
        self.token_embed = nn.Embedding(vocab_size, d_e)
        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_e)

        # Biological initial state projector:
        # Projects embedding → initial EPGL state for ALL blocks at once
        self.h_proj = nn.Linear(d_e, num_blocks * 4)

    def forward(self, token_ids: torch.Tensor):
        """
        token_ids: [B, S]
        Returns:
          e:  [B, S, d_e]
          h:  [B, S, num_blocks, 4]  (EPGL states, initialized from embedding)
          W:  [B, S, num_blocks, num_blocks]  (initially sparse random)
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)

        e = self.token_embed(token_ids) + self.pos_embed(positions)  # [B, S, d_e]

        # Initialize inner block states from the outer embedding
        h_flat = torch.sigmoid(self.h_proj(e))  # [B, S, num_blocks*4] → values in (0,1)
        h = h_flat.view(B, S, self.num_blocks, 4)  # [B, S, num_blocks, 4]

        # Initialize Mucus W as a sparse near-identity matrix
        # (weak connections at start; ecology will wire itself)
        W = torch.eye(self.num_blocks, device=token_ids.device) * 0.1
        W = W.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        # Remove self-connections
        mask = 1.0 - torch.eye(self.num_blocks, device=token_ids.device)
        W = W * mask

        return e, h, W.clone()


# ============================================================
# 🦠 Layer 2: Mucus Inner Cell Update (EPGL + W rewiring)
# The biological heart of the architecture - no MLP weights!
# ============================================================
class MucusInnerCell(nn.Module):
    """
    One step of inner EPGL thermodynamic update + Mucus W self-wiring.
    Pure numerical rules — no trained weights except for learnable coefficients.
    """
    def __init__(self, num_blocks: int):
        super().__init__()
        self.num_blocks = num_blocks

        # Learnable thermodynamic coefficients (they ARE physics-informed,
        # but can be tuned via backprop — they are scalar constants, not matrices)
        self.alpha_E = nn.Parameter(torch.tensor(0.3))  # E: stimulus gain
        self.alpha_P = nn.Parameter(torch.tensor(0.5))  # P: stimulus pressure
        self.alpha_G = nn.Parameter(torch.tensor(0.4))  # G: growth coupling
        self.alpha_L = nn.Parameter(torch.tensor(0.4))  # L: link coupling
        self.beta_P  = nn.Parameter(torch.tensor(0.3))  # P: neighbor sync
        self.beta_G  = nn.Parameter(torch.tensor(0.2))  # G: neighbor sync
        self.beta_L  = nn.Parameter(torch.tensor(0.3))  # L: neighbor sync

        # Mucus rewiring coefficients
        self.w_grow  = nn.Parameter(torch.tensor(0.05)) # W: growth rate
        self.w_decay = nn.Parameter(torch.tensor(0.02)) # W: natural decay

    def forward(self, h: torch.Tensor, W: torch.Tensor, stim: torch.Tensor):
        """
        h:    [B, S, num_blocks, 4]  (E, P, G, L)
        W:    [B, S, num_blocks, num_blocks]
        stim: [B, S]  (macro-level stimulus from outer attention/context)

        Returns updated h_new, W_new
        """
        B, S, Blk, _ = h.shape

        # ---- A. Mucus Information Routing ----
        # Normalize W row-wise so it acts like a weighted mean
        W_norm = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
        # Weighted neighbor state: [B, S, Blk, 4]
        h_nbr = torch.matmul(W_norm, h)

        En, Pn, Gn, Ln = h_nbr[..., 0], h_nbr[..., 1], h_nbr[..., 2], h_nbr[..., 3]
        E,  P,  G,  L  = h[..., 0],    h[..., 1],    h[..., 2],    h[..., 3]

        # Broadcast stim to block dimension: [B, S, 1] → [B, S, Blk]
        s = stim.unsqueeze(-1)  # [B, S, 1]

        # ---- B. EPGL Explicit Thermodynamic Rules ----
        E_new = torch.clamp(E + self.alpha_E * s - 0.4 * P - 0.2 * G, 0.0, 1.0)
        P_new = torch.clamp(P + self.alpha_P * s + self.beta_P * (Pn - P) - 0.2 * E, 0.0, 1.0)
        G_new = torch.clamp(G + self.alpha_G * E * (1.0 - P) + self.beta_G * (Gn - G) - 0.3 * P, 0.0, 1.0)

        good_nbr = 0.5 * En + 0.5 * Gn
        L_new = torch.clamp(L + self.alpha_L * good_nbr + self.beta_L * (Ln - L) - 0.3 * P, 0.0, 1.0)

        h_new = torch.stack([E_new, P_new, G_new, L_new], dim=-1)  # [B, S, Blk, 4]

        # ---- C. Mucus W Rewiring (Slime Mold Dynamics) ----
        # State difference: blocks that are different but both willing to link → strengthen connection
        # h_new: [B, S, Blk, 4]
        h_i = h_new.unsqueeze(4)   # [B, S, Blk, 4, 1]
        h_j = h_new.unsqueeze(3)   # [B, S, 1, 4, Blk]  (broadcast)
        diff = (h_i - h_j.transpose(3,4)).norm(dim=3)  # [B, S, Blk, Blk]

        L_i = L_new.unsqueeze(-1)  # [B, S, Blk, 1]
        L_j = L_new.unsqueeze(-2)  # [B, S, 1, Blk]
        mutual_L = (L_i + L_j) / 2.0  # [B, S, Blk, Blk]

        W_new = torch.clamp(W + self.w_grow * mutual_L * diff - self.w_decay * W, 0.0, 1.0)

        # Zero out self-connections
        eye = torch.eye(Blk, device=W.device).unsqueeze(0).unsqueeze(0)
        W_new = W_new * (1.0 - eye)

        return h_new, W_new


# ============================================================
# 🔬 Layer 3: Token Collapse
# Micro (blocks + W) → Macro (single token vector)
# ============================================================
class TokenCollapse(nn.Module):
    """
    Flattens the evolved CellBlock states and Mucus W matrix into a
    single token-level macro embedding, ready for outer Transformer attention.
    """
    def __init__(self, num_blocks: int, d_e: int):
        super().__init__()
        inner_dim = num_blocks * 4 + num_blocks * num_blocks
        self.proj = nn.Linear(inner_dim, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, e: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        """
        e: [B, S, d_e]  outer embedding (added as residual)
        h: [B, S, num_blocks, 4]
        W: [B, S, num_blocks, num_blocks]
        Returns: macro_embed [B, S, d_e]
        """
        B, S = e.shape[:2]
        h_flat = h.view(B, S, -1)          # [B, S, num_blocks*4]
        W_flat = W.view(B, S, -1)          # [B, S, num_blocks*num_blocks]
        inner_repr = torch.cat([h_flat, W_flat], dim=-1)  # [B, S, inner_dim]

        # Residual: add the micro-physics representation to the macro embedding
        macro_embed = self.norm(e + self.proj(inner_repr))
        return macro_embed


# ============================================================
# 🌐 Layer 4: Outer Transformer Block
# Standard multi-head attention + FFN on the sequence
# ============================================================
class OuterTransformerBlock(nn.Module):
    def __init__(self, d_e: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_e, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_e, d_e * ffn_mult),
            nn.GELU(),
            nn.Linear(d_e * ffn_mult, d_e),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_e)
        self.norm2 = nn.LayerNorm(d_e)

    def forward(self, x: torch.Tensor):
        # Build causal mask explicitly (required by some PyTorch versions with is_causal)
        S = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)

        # Self-attention with pre-norm
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, attn_mask=causal_mask)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 🧠 MASTER: CellToken LLM
# The complete end-to-end model
# ============================================================
class CellTokenLLM(nn.Module):
    """
    CellToken Large Language Model.

    Architecture:
      1. CellEmbedding: token_ids → (e, h, W)
      2. MucusInnerCell × inner_steps: EPGL + Mucus W ecology evolves per-token
      3. TokenCollapse: (h, W) → macro token embedding (residual with e)
      4. OuterTransformerBlock × depth: standard sequence-level attention
      5. LM Head: linear projection to vocab logits

    Stim signal for inner cells: derived from the outer positional embedding norm
    (a simple proxy for "how important is this position").
    """
    def __init__(
        self,
        vocab_size: int,
        d_e: int = 256,
        num_blocks: int = 8,
        inner_steps: int = 5,
        n_heads: int = 8,
        depth: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_e % n_heads == 0, "d_e must be divisible by n_heads"

        self.inner_steps = inner_steps
        self.depth = depth

        # Layers
        self.cell_embed = CellEmbedding(vocab_size, d_e, num_blocks, max_seq_len)
        self.inner_cell = MucusInnerCell(num_blocks)
        self.collapse = TokenCollapse(num_blocks, d_e)
        self.transformer = nn.ModuleList([
            OuterTransformerBlock(d_e, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_e)
        self.lm_head = nn.Linear(d_e, vocab_size, bias=False)

        # Weight tying: share token embedding and LM head weights (standard LLM practice)
        self.lm_head.weight = self.cell_embed.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        token_ids: [B, S]
        targets:   [B, S]  (optional, for loss calculation)
        Returns:
          logits: [B, S, vocab_size]
          loss:   scalar (only if targets provided)
        """
        B, S = token_ids.shape

        # 1. Cell Embedding: ids → (outer embedding, inner blocks, mucus)
        e, h, W = self.cell_embed(token_ids)

        # 2. Stimulus: use the L2 norm of the outer embedding as a proxy
        #    for how much "attention" the outer context is sending to each token.
        #    Shape: [B, S]  — normalized to [0,1]
        stim = e.norm(dim=-1) / (math.sqrt(e.shape[-1]))
        stim = torch.clamp(stim, 0.0, 1.0)

        # 3. Inner EPGL Ecology: run the thermodynamic update for inner_steps
        for _ in range(self.inner_steps):
            h, W = self.inner_cell(h, W, stim)

        # 4. Token Collapse: (e, h, W) → single macro token vector
        x = self.collapse(e, h, W)   # [B, S, d_e]

        # 5. Outer Transformer: sequence-level attention
        #    Build causal mask (upper triangular) for autoregressive LM
        # nn.MultiheadAttention with is_causal=True handles masking internally
        for block in self.transformer:
            x = block(x)

        x = self.final_norm(x)

        # 6. LM Head
        logits = self.lm_head(x)   # [B, S, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
