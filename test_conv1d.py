import torch
import causal_conv1d
from causal_conv1d import causal_conv1d_fn

def test_installation():
    print("--- Environment Check ---")
    print(f"Causal-Conv1d Version: {causal_conv1d.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is not available. This package requires a GPU.")
        return

    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (Torch): {torch.version.cuda}")

    # Define dimensions
    batch, dim, seqlen = 2, 16, 64
    width = 4
    device = "cuda"

    # Create dummy input tensors
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(dim, device=device, dtype=torch.float16, requires_grad=True)

    print("\n--- Testing Kernel Execution ---")
    try:
        # Run the forward pass
        out = causal_conv1d_fn(x, weight, bias, activation="silu")
        assert out is not None, "causal_conv1d_fn returned None"
        
        print("Forward pass: SUCCESS")
        print(f"Output shape: {out.shape}")

        # Run a simple backward pass to check gradient kernels
        loss = out.sum()
        loss.backward()
        print("Backward pass: SUCCESS")
        
        print("\n[RESULT] Installation is fully functional!")
        
    except Exception as e:
        print(f"\n[FAILURE] Kernel execution failed with error:\n{e}")
        print("\nTip: This often happens if the Torch version or CUDA ABI doesn't match the wheel.")

if __name__ == "__main__":
    test_installation()