import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover - handled at runtime when class is instantiated
    Mamba2 = None


class Mamba2QNetwork(nn.Module):
    """Q-network backed by stacked Mamba2 blocks for sequence modeling."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if Mamba2 is None:
            raise ImportError(
                "mamba-ssm is required for Mamba2QNetwork. "
                "Install with `uv sync` or `uv add mamba-ssm`."
            )

        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                Mamba2(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """
        Supports:
        - [batch, obs_dim] for single-step inference
        - [batch, seq_len, obs_dim] for sequence training
        """
        squeeze_sequence = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_sequence = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape [B, D] or [B, T, D], got {tuple(x.shape)}")

        single_token_sequence = x.size(1) == 1
        if single_token_sequence:
            # Mamba2's CUDA kernel rejects a length-1 sequence on this backend.
            # Pad a second token so single-step inference still works, then trim the output back.
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        q = self.head(h)

        if squeeze_sequence:
            return q[:, 0, :]
        if single_token_sequence:
            return q[:, :1, :]
        return q
