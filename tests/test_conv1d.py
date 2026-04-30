import pytest
import torch
import causal_conv1d
from causal_conv1d import causal_conv1d_fn


def test_causal_conv1d_forward_pass():
    """Test that causal_conv1d forward pass executes without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for causal_conv1d kernel testing.")

    batch, dim, seqlen = 2, 16, 64
    width = 4
    device = "cuda"

    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(dim, device=device, dtype=torch.float16, requires_grad=True)

    out = causal_conv1d_fn(x, weight, bias, activation="silu")

    assert out is not None, "causal_conv1d_fn returned None"
    assert out.shape == (batch, dim, seqlen), f"Expected shape {(batch, dim, seqlen)}, got {out.shape}"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"


def test_causal_conv1d_backward_pass():
    """Test that causal_conv1d backward pass executes without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for causal_conv1d kernel testing.")

    batch, dim, seqlen = 2, 16, 64
    width = 4
    device = "cuda"

    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(dim, device=device, dtype=torch.float16, requires_grad=True)

    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients not computed"
    assert weight.grad is not None, "Weight gradients not computed"
    assert bias.grad is not None, "Bias gradients not computed"