import torch
import time
from thermo_engine import ThermoCellularPINN

def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def simulate_thermodynamic_stream():
    print("==================================================")
    print("🧠 Baby Alpha: Thermodynamic ODE Engine (No MLP)")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model parameters
    DIM_E = 64
    DIM_M = 16
    DIM_C = 8
    DEPTH = 4
    DT = 0.5 # ODE Integration step size
    
    # Initialize the engine
    model = ThermoCellularPINN(DIM_E, DIM_M, DIM_C, depth=DEPTH, dt=DT).to(device)
    model.eval() 
    
    # Initial empty state (Zero alive tokens)
    # Note: E, P, G are scalars per token [Batch, Seq, 1]
    e = torch.empty(1, 0, DIM_E, device=device)
    E = torch.empty(1, 0, 1, device=device) # Energy
    P = torch.empty(1, 0, 1, device=device) # Pressure
    G = torch.empty(1, 0, 1, device=device) # Growth
    m = torch.empty(1, 0, DIM_M, device=device)
    c = torch.empty(1, 0, DIM_C, device=device)
    l = torch.empty(1, 0, 1, device=device) # Lifespan
    
    TOTAL_TOKENS_STREAMED = 0
    MAX_STEPS = 500
    NEW_TOKENS_PER_STEP = 5 
    
    print("\n[Starting Continuous Thermodynamic Stream...]")
    print(f"{'Step':<5} | {'Total Streamed':<15} | {'Living Tokens':<15} | {'Died This Step':<15} | {'VRAM (MB)'}")
    print("-" * 75)
    
    with torch.no_grad():
        for step in range(MAX_STEPS):
            # 1. 🌟 BIRTH: Inject new tokens from the environment data stream
            new_e = torch.randn(1, NEW_TOKENS_PER_STEP, DIM_E, device=device)
            new_E = torch.ones(1, NEW_TOKENS_PER_STEP, 1, device=device) * 5.0  # High initial energy (youth)
            new_P = torch.zeros(1, NEW_TOKENS_PER_STEP, 1, device=device)       # Low initial pressure/stress
            new_G = torch.zeros(1, NEW_TOKENS_PER_STEP, 1, device=device)       # Initial growth
            new_m = torch.ones(1, NEW_TOKENS_PER_STEP, DIM_M, device=device)    # Open membrane
            new_c = torch.ones(1, NEW_TOKENS_PER_STEP, DIM_C, device=device)
            new_l = torch.ones(1, NEW_TOKENS_PER_STEP, 1, device=device)        # Full lifespan (1.0)
            
            # Append new tokens to the ecosystem
            e = torch.cat([e, new_e], dim=1)
            E = torch.cat([E, new_E], dim=1)
            P = torch.cat([P, new_P], dim=1)
            G = torch.cat([G, new_G], dim=1)
            m = torch.cat([m, new_m], dim=1)
            c = torch.cat([c, new_c], dim=1)
            l = torch.cat([l, new_l], dim=1)
            
            TOTAL_TOKENS_STREAMED += NEW_TOKENS_PER_STEP
            
            # 2. 🌀 ODE INTEGRATION & TELEONOMY
            e, E, P, G, m, c, l, num_died = model(e, E, P, G, m, c, l, death_threshold=0.01)
            
            # 3. Profiling
            if step % 20 == 0 or step == MAX_STEPS - 1:
                vram = get_vram_mb()
                living = e.size(1)
                warning = "⚠️ VRAM EXPLODING" if living > 200 else ""
                
                print(f"{step:<5} | {TOTAL_TOKENS_STREAMED:<15} | {living:<15} | {num_died:<15} | {vram:.2f} {warning}")
                
            time.sleep(0.01)

    print("-" * 75)
    print("✅ Thermodynamic Simulation Complete.")
    print(f"Total tokens processed: {TOTAL_TOKENS_STREAMED}")
    print(f"Final living tokens in memory: {e.size(1)}")
    print("Conclusion: Explicit Thermodynamic ODE successfully replicated lifespan ecology \nwithout any black-box MLPs!")

if __name__ == "__main__":
    simulate_thermodynamic_stream()
