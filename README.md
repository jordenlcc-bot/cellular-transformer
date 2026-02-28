# CellToken LLM

> **"A token is not a symbol — it is a living colony."**

A biologically-inspired Large Language Model where each token contains a colony of **CellBlocks** connected by a dynamic **Mucus Matrix** ($W$). Internal state evolves via explicit thermodynamic ODEs (Energy, Pressure, Growth, Link) — not black-box MLP weights.

📄 **Read the whitepaper**: [WHITEPAPER.md](./WHITEPAPER.md)

---

## Architecture

```
Token ID → CellEmbedding (e + h=(E,P,G,L) × N blocks + Mucus W)
         → MucusInnerCell × K  (EPGL ODE + Slime Mold W rewiring)
         → TokenCollapse       (micro morphology → macro embedding)
         → OuterTransformer × depth
         → LM Head             (next-token prediction)
```

No MLP in the inner cell. Pure physics.

---

## Results (RTX 3050 6GB)

| Metric | Value |
|---|---|
| Parameters | 626,337 |
| VRAM (constant) | 26.08 MB |
| Loss (0 → 500 steps) | 3.55 → **0.49** |

---

## Repository Structure

```
cell_tokens/
├── WHITEPAPER.md           # Full technical paper
├── lab/
│   ├── baby_alpha/
│   │   ├── cell_token_llm.py      # ⭐ Main model  
│   │   ├── train_cell_llm.py      # Training loop
│   │   ├── epgl_grid.py           # 2D Skin Brain physics engine
│   │   ├── epgl_vis.py            # Animation generator
│   │   ├── thermo_engine.py       # Pure ODE engine
│   │   ├── entropy_engine.py      # Entropy-driven lifespan pruning
│   │   └── bio_token_mucus.py     # Bits→Blocks→Mucus prototype
│   └── slime_mold_attention/      # Slime Mold Attention baseline
└── personal_portfolio/            # Research portfolio website
```

---

## Quick Start

```bash
pip install torch matplotlib
cd lab/baby_alpha

# Train the CellToken LLM
python train_cell_llm.py

# Run the 2D Skin Brain visualization
python epgl_vis.py
# → Outputs: epgl_skin_brain.gif
```

---

## Theoretical Foundations

| Concept | Biological Analog |
|---|---|
| EPGL ODE | Dissipative Structures (Prigogine) |
| Mucus W rewiring | Slime Mold networks (Tero et al. 2010) |
| Apoptosis pruning | Programmed Cell Death |
| Macro attention | Inter-cellular signaling |

---

## License

MIT License — Copyright (c) 2026 Jorden & Antigravity Research Lab

---

*"Life is thermodynamics with intention."*
