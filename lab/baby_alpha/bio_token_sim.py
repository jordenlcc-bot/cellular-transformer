import torch
import numpy as np
from bio_token_mucus import TripleLayerBioPINN, decode_bits_to_epgl

def pprint_matrix(matrix):
    """Pretty prints a 2D numpy matrix"""
    formatted = np.array2string(matrix, formatter={'float_kind':lambda x: "%.2f" % x})
    print(formatted)

def run_bio_token_test():
    print("==================================================")
    print("🧬 Bio-Token Triple Layer: Bits -> Blocks -> Mucus")
    print("==================================================")
    
    # We are simulating a single Token, comprised of 10 Blocks
    NUM_BLOCKS = 10
    
    # 1. 🧮 GENETIC DECODING
    # Let's generate a raw bit stream for 1 Token, 10 Blocks, 8 bits each
    # For Block 0 and 1, we purposefully inject "High Energy / Growth" genetic code (1s)
    raw_bits = torch.zeros((1, 1, NUM_BLOCKS, 8), dtype=torch.float32)
    raw_bits[0, 0, 0, :] = torch.tensor([1,1, 0,0, 1,1, 0,1]) # E=High, P=Low, G=High, L=Med
    raw_bits[0, 0, 1, :] = torch.tensor([1,0, 0,1, 1,1, 1,1]) # E=Med, P=Med, G=High, L=High
    
    # Other blocks get random genetic noise
    raw_bits[0, 0, 2:, :] = torch.randint(0, 2, (1, 1, NUM_BLOCKS-2, 8)).float()
    
    print("\n[Phase 1] Genetic Decoding (Bits to Physics)")
    init_h = decode_bits_to_epgl(raw_bits)
    print("Decoded initialState of Block 0 (E, P, G, L):")
    print(init_h[0, 0, 0].numpy())
    
    # 2. 🦠 MUCUS ECOLOGY EVOLUTION
    model = TripleLayerBioPINN(num_blocks=NUM_BLOCKS, d_model=128)
    
    # Suppose this Token is receiving high attention/stimulus from the Macro-Sequence
    macro_stimulus = torch.tensor([[0.7]]) # [Batch=1, Seq=1]
    
    print("\n[Phase 2] Simulating Internal Ecology (50 Inner Steps)...")
    with torch.no_grad():
        final_macro_embed, final_h, final_W = model(raw_bits, macro_stimulus, internal_steps=50)
    
    print("\n[Phase 3] The Culmination (Outer Macro Meaning)")
    print("After 50 internal physical thermodynamic steps, the Blocks have formed")
    print("a specific morphological structure inside the Token.")
    
    print("\n--- Final W Mucus Matrix (Connections between the 10 Blocks) ---")
    final_W_np = final_W[0, 0].numpy()
    pprint_matrix(final_W_np)
    
    print("\n--- Final Synthesized Macro Embedding (to Transformer) ---")
    print(f"Shape: {final_macro_embed.shape} (Ready for OuterTrans QKV projections)")
    
    print("\nConclusion: The 'meaning' of the token is no longer an arbitrary word2vec lookup.")
    print("It is the literal physical shape of its internal surviving CellBlocks and Mucus connections!")

if __name__ == "__main__":
    run_bio_token_test()
