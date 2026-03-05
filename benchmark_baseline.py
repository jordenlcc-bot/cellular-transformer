import torch
import time
from baseline import StandardTransformer

def benchmark(model, device, batch_size=64, seq_len=32, num_iters=1000):
    model.to(device)
    model.eval()
    x = torch.randint(0, 50, (batch_size, seq_len)).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iters
    return avg_time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")

    model = StandardTransformer(vocab_size=50, d_model=128, macro_layers=4)

    # Test with typical seq_len
    avg_time = benchmark(model, device, seq_len=32)
    print(f"Average forward time (seq_len=32): {avg_time*1000:.4f} ms")

    # Test with larger seq_len
    avg_time_long = benchmark(model, device, seq_len=512)
    print(f"Average forward time (seq_len=512): {avg_time_long*1000:.4f} ms")
