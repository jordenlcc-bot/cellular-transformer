# 🦠 Cellular Transformer

## v2.0 — CellToken LLM: Biologically-Inspired LLM with Thermodynamic Inner-Cell Dynamics

> *"A token is not a symbol — it is a living colony."*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com/jordenlcc-bot/cellular-transformer/releases)

**v1 (Slime Mold Attention)** — static nested Transformer with slime mold routing.
**v2 (CellToken LLM)** — each token is a **colony of CellBlocks** with explicit thermodynamic ODE inner dynamics and a self-wiring **Mucus Matrix** ($W$).

A biologically-inspired, nested Transformer proving we can break the "Depth Stack" curse. Tokens are no longer static bricks — they are living cells with internal compute.

📄 **Read the whitepaper**: [WHITEPAPER.md](./WHITEPAPER.md)
📐 **LaTeX (arXiv-ready)**: [paper/main.tex](./paper/main.tex)

---

## Architecture

```text
Token ID → CellEmbedding (e + h=(E,P,G,L) × N blocks + Mucus W)
         → MucusInnerCell × K  (EPGL ODE + Slime Mold W rewiring)
         → TokenCollapse       (micro morphology → macro embedding)
         → OuterTransformer × depth
         → LM Head             (next-token prediction)
```

No MLP in the inner cell. Pure physics.

---

## Results (RTX 3050 6 GB)

| Metric | Value |
|---|---|
| Parameters | 626,337 |
| VRAM (constant) | 26.08 MB |
| Loss (0 → 500 steps) | 3.55 → **0.49** |

---

## Repository Structure

```text
cellular-transformer/
├── WHITEPAPER.md                    # Full technical paper
├── paper/main.tex                   # arXiv-ready LaTeX
├── lab/
│   ├── baby_alpha/
│   │   ├── cell_token_llm.py        # ⭐ Main model (v2)
│   │   ├── train_cell_llm.py        # Training loop
│   │   ├── epgl_grid.py             # 2D Skin Brain physics engine
│   │   ├── epgl_vis.py              # → epgl_skin_brain.gif
│   │   ├── thermo_engine.py         # Pure ODE engine
│   │   ├── entropy_engine.py        # Entropy-driven lifespan pruning
│   │   └── bio_token_mucus.py       # Bits→Blocks→Mucus prototype
│   └── slime_mold_attention/        # v1: Slime Mold Attention baseline
└── personal_portfolio/              # Research portfolio website
```

---

## Quick Start

```bash
pip install torch matplotlib
cd lab/baby_alpha

# Train the CellToken LLM (v2)
python train_cell_llm.py

# Run the 2D Skin Brain visualization
python epgl_vis.py
```

---

## Theoretical Foundations

| Concept | Biological Analog |
|---|---|
| EPGL ODE | Dissipative Structures (Prigogine 1984) |
| Mucus W rewiring | Slime Mold networks (Tero et al. 2010) |
| Apoptosis pruning | Programmed Cell Death |
| Macro attention | Inter-cellular signaling |

---

## License

MIT License — Copyright (c) 2026 Jorden & Antigravity Research Lab

---

*"Life is thermodynamics with intention."*
