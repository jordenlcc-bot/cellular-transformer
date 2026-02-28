"""
Training Script for CellToken LLM
====================================
next-token prediction on a small character-level text corpus.
Designed to verify:
  1. Forward pass runs correctly
  2. Loss decreases over training
  3. VRAM stays within RTX 3050 6GB limits
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from cell_token_llm import CellTokenLLM

# ============================================================
# 📖 Data: Simple character-level encoding
# ============================================================
def load_tiny_corpus():
    """
    Creates a tiny corpus for quick sanity check.
    In a real run, replace with WikiText-2 / TinyStories.
    """
    text = (
        "The slime mold finds the shortest path through the maze. "
        "Energy flows where resistance is lowest. "
        "Every cell remembers its history and acts with purpose. "
        "The cellular network learns by living, not by being told. "
        "Life is thermodynamics with intention. "
    ) * 300  # Repeat to give enough training signal

    # Build simple character-level vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)
    ch2id = {c: i for i, c in enumerate(chars)}
    id2ch = {i: c for c, i in ch2id.items()}

    data = torch.tensor([ch2id[c] for c in text], dtype=torch.long)
    return data, vocab_size, ch2id, id2ch


def get_batch(data, batch_size, seq_len, device):
    """Sample a random batch of (input, target) pairs."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i: i + seq_len] for i in ix])
    y = torch.stack([data[i + 1: i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# ⚙️ Hyperparameters (RTX 3050 6GB friendly)
# ============================================================
CONFIG = {
    "vocab_size":    None,       # Set after loading corpus
    "d_e":           128,        # Token embedding dimension (small for 3050)
    "num_blocks":    6,          # CellBlocks per token
    "inner_steps":   5,          # EPGL iterations per token per forward pass
    "n_heads":       4,          # Attention heads
    "depth":         3,          # Outer Transformer layers
    "max_seq_len":   128,        # Sequence length
    "dropout":       0.1,
    "batch_size":    16,
    "learning_rate": 3e-4,
    "max_iters":     500,
    "eval_interval": 50,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ============================================================
# 🚀 Training Loop
# ============================================================
def train():
    print("================================================")
    print("🧬 CellToken LLM — Training Run")
    print("================================================")

    device = CONFIG["device"]
    print(f"Device: {device}")

    # Load data
    data, vocab_size, ch2id, id2ch = load_tiny_corpus()
    CONFIG["vocab_size"] = vocab_size
    print(f"Vocabulary size: {vocab_size} characters")

    # Split train / val
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Build model
    model = CellTokenLLM(
        vocab_size   = vocab_size,
        d_e          = CONFIG["d_e"],
        num_blocks   = CONFIG["num_blocks"],
        inner_steps  = CONFIG["inner_steps"],
        n_heads      = CONFIG["n_heads"],
        depth        = CONFIG["depth"],
        max_seq_len  = CONFIG["max_seq_len"],
        dropout      = CONFIG["dropout"],
    ).to(device)

    print(f"Model parameters: {model.num_parameters:,}")
    print(f"Initial VRAM: {get_vram_mb():.2f} MB")

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["max_iters"], eta_min=1e-5)

    print(f"\n{'Iter':<6} | {'Train Loss':<12} | {'VRAM (MB)'}")
    print("-" * 40)

    model.train()
    for step in range(CONFIG["max_iters"]):
        x, y = get_batch(train_data, CONFIG["batch_size"], CONFIG["max_seq_len"], device)

        _, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % CONFIG["eval_interval"] == 0 or step == CONFIG["max_iters"] - 1:
            vram = get_vram_mb()
            print(f"{step:<6} | {loss.item():<12.4f} | {vram:.2f}")

    print("\n✅ Training Complete!")
    print(f"Final VRAM: {get_vram_mb():.2f} MB")

    # Quick generation sample
    print("\n[Sampling from CellToken LLM...]")
    model.eval()
    with torch.no_grad():
        seed = "The slime"
        ids = torch.tensor([[ch2id[c] for c in seed]], dtype=torch.long).to(device)

        for _ in range(80):
            if ids.shape[1] >= CONFIG["max_seq_len"]:
                ids = ids[:, -CONFIG["max_seq_len"]:]
            logits, _ = model(ids)
            next_logit = logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(next_logit, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        generated = "".join([id2ch[i.item()] for i in ids[0]])
        print("Generated:", generated)


if __name__ == "__main__":
    train()
