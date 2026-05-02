import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover - handled at runtime when class is instantiated
    Mamba2 = None


class Mamba2DuelingQNetwork(nn.Module):
    """Dueling Q-network using Mamba2 backbone."""

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
                "mamba-ssm is required for Mamba2DuelingQNetwork. "
                "Install with `uv sync` or `uv add mamba-ssm`."
            )

        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                Mamba2(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, num_actions)

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

        value = self.value_head(h)
        adv = self.adv_head(h)
        adv_mean = adv.mean(dim=-1, keepdim=True)
        q = value + (adv - adv_mean)

        if squeeze_sequence:
            return q[:, 0, :]
        if single_token_sequence:
            return q[:, :1, :]
        return q
