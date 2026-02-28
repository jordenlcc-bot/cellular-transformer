import torch
from torch.utils.data import Dataset
import random

class CharLanguageModelDataset(Dataset):
    """
    A simple character-level dataset for sequence prediction.
    Used for quickly testing the modeling capability of Cellular Transformer vs Standard Transformer.
    """
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        # Create a tiny custom vocabulary for simplicity
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = [self.stoi[c] for c in text]

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # input sequence
        x = self.data[idx:idx + self.seq_len]
        # target sequence (shifted by 1)
        y = self.data[idx + 1:idx + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def generate_synthetic_data(num_chars=10000):
    """
    Generates a synthetic sequential string based on simple rules to simulate language structure.
    E.g., "aba cababa daba..."
    """
    patterns = ["apple", "banana", "orange", "grape", "melon", 
                "the slime mold is smart", "octopus has many arms", 
                "intelligence is non-linear", "cellular token acts like life"]
    
    data = ""
    for _ in range(num_chars // 10):
        data += random.choice(patterns) + " "
    return data[:num_chars]

if __name__ == "__main__":
    text = generate_synthetic_data()
    ds = CharLanguageModelDataset(text, seq_len=16)
    print(f"Generated Synthetic Text length: {len(text)}")
    print(f"Vocab size: {ds.vocab_size}")
    print(f"Dataset length: {len(ds)}")
    x, y = ds[0]
    print(f"Sample x: {x}")
    print(f"Sample y: {y}")
