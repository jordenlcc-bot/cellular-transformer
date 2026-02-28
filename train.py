import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CharLanguageModelDataset, generate_synthetic_data
from model import SlimeMoldTransformer, count_parameters
from baseline import StandardTransformer

def train_model(model, dataloader, epochs=5, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropyLoss
            # logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
            # y: [batch, seq_len] -> [batch * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    text = generate_synthetic_data(num_chars=20000)
    seq_len = 32
    dataset = CharLanguageModelDataset(text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    vocab_size = dataset.vocab_size
    
    # 2. Setup Baseline
    # A standard deep transformer with 4 macro layers
    print("\n--- Training Standard Baseline Transformer ---")
    baseline_model = StandardTransformer(
        vocab_size=vocab_size, 
        d_model=128, 
        macro_layers=4, 
        macro_heads=8
    )
    baseline_params = count_parameters(baseline_model)
    print(f"Params: {baseline_params}")
    baseline_losses = train_model(baseline_model, dataloader, epochs=5, lr=2e-3, device=device)
    
    # 3. Setup Cellular Transformer (Slime Mold Attention)
    # A shallow nested transformer with ONLY 2 macro layers, but with internal cellular compute
    print("\n--- Training Cellular Transformer (Slime Mold Attention) ---")
    cellular_model = SlimeMoldTransformer(
        vocab_size=vocab_size, 
        num_organelles=8, 
        d_organelle=16, # Total d_model = 8 * 16 = 128
        macro_layers=2, # Half the macroscopic depth of the baseline!
        macro_heads=8
    )
    cellular_params = count_parameters(cellular_model)
    print(f"Params: {cellular_params}")
    cellular_losses = train_model(cellular_model, dataloader, epochs=5, lr=2e-3, device=device)
    
    # 4. Results
    print("\n=== Experiment Results ===")
    print(f"Baseline Params: {baseline_params} | Final Loss: {baseline_losses[-1]:.4f}")
    print(f"Cellular Params: {cellular_params} | Final Loss: {cellular_losses[-1]:.4f}")
    
    if cellular_losses[-1] <= baseline_losses[-1]:
         print("\nSUCCESS! Cellular Transformer matched/beat Baseline with HALF the depth (macro layers)!")
    else:
         print("\nBaseline won, but Cellular proved it can learn efficiently with shallower macro layers.")

if __name__ == "__main__":
    main()
