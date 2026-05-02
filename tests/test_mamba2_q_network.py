import numpy as np
import pytest
import torch

from app.mamba2_q_network import Mamba2QNetwork
from app.mamba2_dueling_q_network import Mamba2DuelingQNetwork
from app.mamba2_quantile_q_network import Mamba2QuantileQNetwork
from app.open_spiel_2048_env import OpenSpiel2048Env


def _make_model_and_env(device: torch.device):
    env = OpenSpiel2048Env(seed=7)
    model = Mamba2QNetwork(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_dim=256,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=1,
    ).to(device)
    model.eval()
    return env, model


def test_mamba2_q_network_handles_single_2048_observation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env, model = _make_model_and_env(device)

    obs = env.reset(seed=7)
    input_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    assert logits.shape == (1, env.num_actions)
    assert torch.isfinite(logits).all()


def test_mamba2_q_network_handles_2048_sequence_batches():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env, model = _make_model_and_env(device)

    obs = env.reset(seed=11)
    next_obs, _, done, _ = env.step(env.legal_actions()[0])
    if done:
        next_obs = np.zeros_like(obs)

    batch = torch.tensor(
        np.asarray([[obs, next_obs, obs, next_obs]], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (1, 4, env.num_actions)
    assert torch.isfinite(logits).all()


def test_mamba2_dueling_q_network_handles_2048_observation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env = OpenSpiel2048Env(seed=7)
    model = Mamba2DuelingQNetwork(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_dim=256,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=1,
    ).to(device)
    model.eval()

    obs = env.reset(seed=7)
    input_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    assert logits.shape == (1, env.num_actions)
    assert torch.isfinite(logits).all()


def test_mamba2_dueling_q_network_handles_2048_sequence_batches():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env = OpenSpiel2048Env(seed=7)
    model = Mamba2DuelingQNetwork(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_dim=256,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=1,
    ).to(device)
    model.eval()

    obs = env.reset(seed=11)
    next_obs, _, done, _ = env.step(env.legal_actions()[0])
    if done:
        next_obs = np.zeros_like(obs)

    batch = torch.tensor(
        np.asarray([[obs, next_obs, obs, next_obs]], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (1, 4, env.num_actions)
    assert torch.isfinite(logits).all()


def test_mamba2_quantile_q_network_handles_2048_observation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env = OpenSpiel2048Env(seed=7)
    num_quantiles = 51
    model = Mamba2QuantileQNetwork(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        num_quantiles=num_quantiles,
        hidden_dim=256,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=1,
    ).to(device)
    model.eval()

    obs = env.reset(seed=7)
    # For Mamba, always include time dimension: [B, T, obs_dim]
    input_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        quantiles = model(input_tensor)
        q_vals = model.q_values(input_tensor)

    assert quantiles.shape == (1, 1, env.num_actions, num_quantiles)
    assert q_vals.shape == (1, 1, env.num_actions)
    assert torch.isfinite(quantiles).all()
    assert torch.isfinite(q_vals).all()


def test_mamba2_quantile_q_network_handles_2048_sequence_batches():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the causal_conv1d regression path.")

    device = torch.device("cuda")
    env = OpenSpiel2048Env(seed=7)
    num_quantiles = 51
    model = Mamba2QuantileQNetwork(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        num_quantiles=num_quantiles,
        hidden_dim=256,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=1,
    ).to(device)
    model.eval()

    obs = env.reset(seed=11)
    next_obs, _, done, _ = env.step(env.legal_actions()[0])
    if done:
        next_obs = np.zeros_like(obs)

    batch = torch.tensor(
        np.asarray([[obs, next_obs, obs, next_obs]], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        quantiles = model(batch)
        q_vals = model.q_values(batch)

    assert quantiles.shape == (1, 4, env.num_actions, num_quantiles)
    assert q_vals.shape == (1, 4, env.num_actions)
    assert torch.isfinite(quantiles).all()
    assert torch.isfinite(q_vals).all()
