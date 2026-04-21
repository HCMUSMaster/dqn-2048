import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from app.eval import evaluate_multi_seed, greedy_rollout, save_eval_results
from app.helpers import epsilon_by_step, masked_greedy_action
from app.mamba2_q_network import Mamba2QNetwork
from app.open_spiel_2048_env import OpenSpiel2048Env
from app.replay_buffer import make_legal_mask
from app.sequential_replay_buffer import SequentialReplayBuffer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Mamba2-based DQN/Double-DQN on OpenSpiel 2048"
    )
    parser.add_argument("--algorithm", choices=["dqn", "double_dqn"], default="double_dqn")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument("--num_episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--buffer_size", type=int, default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--seq_len", type=int, default=8, help="Temporal sequence length for replay.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--target_sync_every", type=int, default=250, help="Target sync period in env steps.")
    parser.add_argument("--learn_start", type=int, default=1_000, help="Min transitions before updates.")
    parser.add_argument("--learn_every", type=int, default=4, help="Update every N env steps.")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--eps_decay_steps", type=int, default=20_000, help="Linear epsilon decay steps.")
    parser.add_argument("--max_steps_per_episode", type=int, default=5_000, help="Per-episode step cap.")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping norm.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Backbone hidden size.")
    parser.add_argument("--mamba_layers", type=int, default=2, help="Number of Mamba2 blocks.")
    parser.add_argument("--mamba_state_dim", type=int, default=64, help="Mamba2 d_state.")
    parser.add_argument("--mamba_conv_dim", type=int, default=4, help="Mamba2 d_conv.")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba2 expand ratio.")
    parser.add_argument("--num_eval_seeds", type=int, default=100, help="Evaluation seed count.")
    parser.add_argument("--eval_every", type=int, default=20, help="Run eval every N episodes.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for checkpoint and eval files.",
    )
    return parser


def dqn_sequence_update(batch, q_net, target_net, optimizer, gamma: float, grad_clip: float, device):
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(np.asarray(batch.action), dtype=torch.int64, device=device)
    rewards = torch.tensor(np.asarray(batch.reward), dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(np.asarray(batch.done), dtype=torch.float32, device=device)
    next_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=device)

    q_values = q_net(obs)  # [B, T, A]
    q_sa = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]

    with torch.no_grad():
        next_q = target_net(next_obs).masked_fill(~next_legal_mask, -1e9)
        next_max_q = torch.max(next_q, dim=2).values
        next_max_q = torch.where(dones > 0.5, torch.zeros_like(next_max_q), next_max_q)
        target = rewards + gamma * next_max_q

    loss = torch.nn.functional.mse_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def double_dqn_sequence_update(batch, q_net, target_net, optimizer, gamma: float, grad_clip: float, device):
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(np.asarray(batch.action), dtype=torch.int64, device=device)
    rewards = torch.tensor(np.asarray(batch.reward), dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(np.asarray(batch.done), dtype=torch.float32, device=device)
    next_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=device)

    q_values = q_net(obs)  # [B, T, A]
    q_sa = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]

    with torch.no_grad():
        next_online_q = q_net(next_obs).masked_fill(~next_legal_mask, -1e9)
        next_actions = torch.argmax(next_online_q, dim=2, keepdim=True)
        next_target_q = target_net(next_obs).gather(2, next_actions).squeeze(-1)
        next_target_q = torch.where(dones > 0.5, torch.zeros_like(next_target_q), next_target_q)
        target = rewards + gamma * next_target_q

    loss = torch.nn.functional.mse_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace, device: torch.device):
    log_episodes = bool(getattr(args, "log_episodes", True))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env = OpenSpiel2048Env(seed=args.seed)
    obs_dim = train_env.obs_dim
    num_actions = train_env.num_actions

    q_net = Mamba2QNetwork(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        num_layers=args.mamba_layers,
        d_state=args.mamba_state_dim,
        d_conv=args.mamba_conv_dim,
        expand=args.mamba_expand,
    ).to(device)
    target_net = Mamba2QNetwork(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        num_layers=args.mamba_layers,
        d_state=args.mamba_state_dim,
        d_conv=args.mamba_conv_dim,
        expand=args.mamba_expand,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = SequentialReplayBuffer(args.buffer_size, seq_len=args.seq_len)

    global_step = 0
    for episode in tqdm(range(1, args.num_episodes + 1), desc="Training"):
        obs = train_env.reset(seed=args.seed + episode)
        done = False
        ep_return = 0.0
        ep_len = 0
        max_tile = 0
        illegal_attempts = 0
        ep_losses = []

        while not done and ep_len < args.max_steps_per_episode:
            eps = epsilon_by_step(
                global_step,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_steps=args.eps_decay_steps,
            )
            legal = train_env.legal_actions()
            legal_mask = make_legal_mask(num_actions, legal)

            if obs is not None and len(obs) > 0:
                max_tile = max(max_tile, int(np.max(obs)))

            with torch.no_grad():
                q_vals = q_net(torch.tensor(np.asarray([obs]), dtype=torch.float32, device=device))
                q_vals_np = q_vals.cpu().numpy()[0]

            action = masked_greedy_action(
                q_net=q_net,
                obs=obs,
                legal_actions_list=legal,
                num_actions=num_actions,
                epsilon=eps,
                device=device,
            )

            if eps > 0 and len(legal) < num_actions:
                best_raw_action = int(np.argmax(q_vals_np))
                if best_raw_action not in legal:
                    illegal_attempts += 1

            next_obs, reward, done, info = train_env.step(action)
            next_legal = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)
            replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)

            obs = next_obs
            ep_return += reward
            ep_len += 1
            global_step += 1

            if len(replay) >= args.learn_start and global_step % args.learn_every == 0:
                batch = replay.sample(args.batch_size)
                if args.algorithm == "double_dqn":
                    loss = double_dqn_sequence_update(
                        batch=batch,
                        q_net=q_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        gamma=args.gamma,
                        grad_clip=args.grad_clip,
                        device=device,
                    )
                else:
                    loss = dqn_sequence_update(
                        batch=batch,
                        q_net=q_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        gamma=args.gamma,
                        grad_clip=args.grad_clip,
                        device=device,
                    )
                ep_losses.append(loss)

            if global_step % args.target_sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        eval_return = float("nan")
        if episode % args.eval_every == 0:
            eval_return, _, _, _, _ = greedy_rollout(
                q_net=q_net,
                env=OpenSpiel2048Env(seed=1000 + episode),
                num_actions=num_actions,
                max_steps=args.max_steps_per_episode,
                device=device,
            )

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        if log_episodes:
            print(
                f"Episode {episode}/{args.num_episodes} | "
                f"return={ep_return:.1f} | "
                f"length={ep_len} | "
                f"loss={mean_loss:.4f} | "
                f"max_tile={max_tile} | "
                f"illegal_attempts={illegal_attempts} | "
                f"eval_return={eval_return:.1f}"
            )

    return q_net, target_net, obs_dim, num_actions


def evaluate_policy(args: argparse.Namespace, q_net: torch.nn.Module, num_actions: int, device: torch.device):
    print_summary = bool(getattr(args, "print_eval_summary", True))
    eval_data = evaluate_multi_seed(
        q_net=q_net,
        env_class=OpenSpiel2048Env,
        num_eval_seeds=args.num_eval_seeds,
        num_actions=num_actions,
        max_steps_per_episode=args.max_steps_per_episode,
        device=device,
        seed_offset=5000,
    )

    summary = eval_data["summary"]
    if print_summary:
        print(f"Results over {args.num_eval_seeds} seeds:")
        print(f"  Average return:           {summary['avg_return']:.1f} +/- {summary['std_return']:.1f}")
        print(f"  Average episode length:   {summary['avg_length']:.1f}")
        print(f"  Average max tile:         {summary['avg_max_tile']:.1f}")
        print(f"  Average illegal attempts: {summary['avg_illegal_action_attempts']:.1f}")
        print(f"  Total illegal attempts:   {summary['total_illegal_action_attempts']}")

    return eval_data


def save_artifacts(args, q_net, target_net, obs_dim: int, num_actions: int, eval_data):
    log_save = bool(getattr(args, "log_save", True))
    if not args.output_dir:
        if log_save:
            print("Skipping save: pass --output_dir to persist checkpoint and evaluation files.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path, json_path = save_eval_results(
        eval_data=eval_data,
        output_dir=args.output_dir,
        num_actions=num_actions,
        obs_dim=obs_dim,
        source_notebook=f"Mamba2 {args.algorithm}",
        checkpoint_file="mamba2_dqn.pt",
    )
    if log_save:
        print(f"Saved evaluation rollout archive to: {npz_path}")
        print(f"Saved evaluation metadata to: {json_path}")

    checkpoint_path = os.path.join(args.output_dir, "mamba2_dqn.pt")
    torch.save(
        {
            "model_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "obs_dim": obs_dim,
            "num_actions": num_actions,
            "algorithm": args.algorithm,
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "mamba_layers": args.mamba_layers,
            "mamba_state_dim": args.mamba_state_dim,
            "mamba_conv_dim": args.mamba_conv_dim,
            "mamba_expand": args.mamba_expand,
        },
        checkpoint_path,
    )
    if log_save:
        print(f"Saved checkpoint to: {checkpoint_path}")


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Algorithm: {args.algorithm}")
    q_net, target_net, obs_dim, num_actions = train(args, device)
    eval_data = evaluate_policy(args, q_net, num_actions, device)
    save_artifacts(args, q_net, target_net, obs_dim, num_actions, eval_data)


if __name__ == "__main__":
    main()
