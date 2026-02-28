import torch
import torch.nn as nn
from model import count_parameters

class StandardAttentionBlock(nn.Module):
    """
    Standard Transformer Macro Layer
    """
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

class StandardTransformer(nn.Module):
    """
    A traditional Baseline Transformer.
    No internal cellular state. Needs deeper macro layers to match intelligence.
    """
    def __init__(self, vocab_size, d_model=128, macro_layers=4, macro_heads=8):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model) # Max seq len 1024
        
        # Purely Macro Deep Layers
        self.macro_layers = nn.ModuleList([
            StandardAttentionBlock(d_model, num_heads=macro_heads) for _ in range(macro_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Standard Embedding
        seq_state = self.embedding(x)
        
        # Positional Encoding
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        seq_state = seq_state + self.pos_embedding(positions)
        
        # Mask
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Macro layers ONLY
        for layer in self.macro_layers:
            seq_state = layer(seq_state, mask=mask)
            
        return self.fc_out(seq_state)

if __name__ == "__main__":
    # Quick Test
    model = StandardTransformer(vocab_size=50, d_model=128, macro_layers=4)
    print(f"Standard Transformer Params: {count_parameters(model)}")
    dummy_x = torch.randint(0, 50, (2, 32))  # batch=2, seq=32
    out = model(dummy_x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 32, 50]
