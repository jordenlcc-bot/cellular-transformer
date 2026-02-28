import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CellularTokenEmbedding(nn.Module):
    """
    Inflates a standard 1D token embedding into a 2D "Cellular" state.
    Instead of [batch, seq, d_model], we get [batch, seq, num_organelles, d_organelle].
    """
    def __init__(self, vocab_size, num_organelles, d_organelle):
        super().__init__()
        self.num_organelles = num_organelles
        self.d_organelle = d_organelle
        self.embedding = nn.Embedding(vocab_size, num_organelles * d_organelle)
        
    def forward(self, x):
        # x: [batch, seq_len]
        # out: [batch, seq_len, num_organelles * d_organelle]
        emb = self.embedding(x)
        batch_size, seq_len, _ = emb.shape
        # reshape to create "cellular organelles"
        # out: [batch, seq_len, num_organelles, d_organelle]
        return emb.view(batch_size, seq_len, self.num_organelles, self.d_organelle)

class CellularAttention(nn.Module):
    """
    Micro-level attention: "Organelles" within ONE cell communicate with each other.
    This simulates the non-linear "Sensing & Reasoning" inside a single Slime Mold cell.
    """
    def __init__(self, d_organelle, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_organelle, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_organelle)
        
    def forward(self, cellular_state):
        # cellular_state: [batch, seq_len, num_organelles, d_organelle]
        batch_size, seq_len, num_org, d_org = cellular_state.shape
        
        # We want attention across organelles, so we merge batch and seq_len
        # x: [batch * seq_len, num_organelles, d_organelle]
        x = cellular_state.view(batch_size * seq_len, num_org, d_org)
        
        # Self-attention within the cell
        attn_out, _ = self.mha(x, x, x)
        
        # Residual + Norm
        x = self.norm(x + attn_out)
        
        # Reshape back to cellular state
        return x.view(batch_size, seq_len, num_org, d_org)
        
class MacroAttention(nn.Module):
    """
    Macro-level attention: Cells communicate with other Cells across the sequence.
    This simulates the multi-cellular tissue/network behavior.
    """
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, sequence_state, mask=None):
        # sequence_state: [batch, seq_len, d_model]
        attn_out, _ = self.mha(sequence_state, sequence_state, sequence_state, attn_mask=mask)
        return self.norm(sequence_state + attn_out)

class SlimeMoldTransformer(nn.Module):
    """
    The Cellular (Nested) Transformer Architecture.
    Reduces the need for many deep macro layers by performing micro-computation inside tokens.
    """
    def __init__(self, vocab_size, num_organelles=8, d_organelle=16, macro_layers=2, macro_heads=8):
        super().__init__()
        self.num_organelles = num_organelles
        self.d_organelle = d_organelle
        self.d_model = num_organelles * d_organelle  # e.g., 8 * 16 = 128
        
        self.embedding = CellularTokenEmbedding(vocab_size, num_organelles, d_organelle)
        self.pos_embedding = nn.Embedding(1024, self.d_model) # Max seq len 1024
        
        # Micro layer (Inside the cell)
        self.cellular_layer = CellularAttention(d_organelle, num_heads=4)
        
        # Macro layers (Between cells)
        self.macro_layers = nn.ModuleList([
            MacroAttention(self.d_model, num_heads=macro_heads) for _ in range(macro_layers)
        ])
        
        self.fc_out = nn.Linear(self.d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 1. Expand to Cellular State: [batch, seq, num_org, d_org]
        cellular_state = self.embedding(x)
        
        # 2. Micro-Computation (Within Token)
        # The token is differentiating based on its internal rules before meeting others
        cellular_state = self.cellular_layer(cellular_state)
        
        # 3. Collapse/Flatten to Sequence State for Macro Communication
        # [batch, seq, num_org * d_org] -> [batch, seq, d_model]
        seq_state = cellular_state.view(batch_size, seq_len, self.d_model)
        
        # Add positional encoding
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        seq_state = seq_state + self.pos_embedding(positions)
        
        # Create causal mask for language modeling
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. Macro-Computation (Between Tokens)
        for layer in self.macro_layers:
            seq_state = layer(seq_state, mask=mask)
            
        # 5. Output Prediction
        logits = self.fc_out(seq_state)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick Test
    model = SlimeMoldTransformer(vocab_size=50, num_organelles=8, d_organelle=16, macro_layers=2)
    print(f"Cellular Transformer Params: {count_parameters(model)}")
    dummy_x = torch.randint(0, 50, (2, 32))  # batch=2, seq=32
    out = model(dummy_x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 32, 50]
