import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover - handled at runtime when class is instantiated
    Mamba2 = None


class Mamba2QuantileQNetwork(nn.Module):
    """Mamba-backed quantile network for QR-DQN.

    Forward returns quantile values with shape [B, T, A, N] for sequence input
    or [B, A, N] for single-step input. Provide `q_values(x)` helper to
    return expected Q-values by averaging over quantiles.
    """

    def __init__(self, obs_dim: int, num_actions: int, num_quantiles: int = 51, hidden_dim: int = 256, num_layers: int = 2, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba2 is None:
            raise ImportError(
                "mamba-ssm is required for Mamba2QuantileQNetwork. "
                "Install with `uv sync` or `uv add mamba-ssm`."
            )

        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_actions * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_sequence = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_sequence = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape [B, D] or [B, T, D], got {tuple(x.shape)}")

        single_token_sequence = x.size(1) == 1
        if single_token_sequence:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        out = self.head(h)  # [B, T, A * N]
        B, T, _ = out.shape
        out = out.view(B, T, self.num_actions, self.num_quantiles)

        if squeeze_sequence:
            return out[:, 0, :, :]
        if single_token_sequence:
            return out[:, :1, :, :]
        return out

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        q = self.forward(x)
        if q.dim() == 3:
            # [B, A, N]
            return q.mean(dim=2)
        # [B, T, A, N] -> [B, T, A]
        return q.mean(dim=3)
