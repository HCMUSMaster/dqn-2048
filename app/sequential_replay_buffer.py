import random
from collections import deque, namedtuple

import numpy as np

from app.helpers import Transition

SequenceTransition = namedtuple(
    "SequenceTransition",
    ["obs", "action", "reward", "next_obs", "done", "legal_mask", "next_legal_mask"],
)


class SequentialReplayBuffer:
    """Replay buffer that samples contiguous fixed-length sequences."""

    def __init__(self, capacity: int, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.buffer = deque(maxlen=capacity)
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def _zero_fields_like(self, template_transition):
        zero_obs = np.zeros_like(np.asarray(template_transition.obs, dtype=np.float32))
        zero_next_obs = np.zeros_like(
            np.asarray(template_transition.next_obs, dtype=np.float32)
        )
        zero_legal_mask = np.zeros_like(
            np.asarray(template_transition.legal_mask, dtype=np.float32)
        )
        zero_next_legal_mask = np.zeros_like(
            np.asarray(template_transition.next_legal_mask, dtype=np.float32)
        )
        return {
            "obs": zero_obs,
            "action": 0,
            "reward": 0.0,
            "next_obs": zero_next_obs,
            "done": 0.0,
            "legal_mask": zero_legal_mask,
            "next_legal_mask": zero_next_legal_mask,
        }

    def _build_padded_sequence(self, items, start, zero_fields):
        obs_seq = []
        action_seq = []
        reward_seq = []
        next_obs_seq = []
        done_seq = []
        legal_mask_seq = []
        next_legal_mask_seq = []

        for offset in range(self.seq_len):
            idx = start + offset
            if idx >= len(items):
                obs_seq.append(zero_fields["obs"])
                action_seq.append(zero_fields["action"])
                reward_seq.append(zero_fields["reward"])
                next_obs_seq.append(zero_fields["next_obs"])
                done_seq.append(zero_fields["done"])
                legal_mask_seq.append(zero_fields["legal_mask"])
                next_legal_mask_seq.append(zero_fields["next_legal_mask"])
                continue

            transition = items[idx]
            obs_seq.append(np.asarray(transition.obs, dtype=np.float32))
            action_seq.append(int(transition.action))
            reward_seq.append(float(transition.reward))
            next_obs_seq.append(np.asarray(transition.next_obs, dtype=np.float32))
            done_seq.append(float(transition.done))
            legal_mask_seq.append(np.asarray(transition.legal_mask, dtype=np.float32))
            next_legal_mask_seq.append(
                np.asarray(transition.next_legal_mask, dtype=np.float32)
            )

            if transition.done and offset < self.seq_len - 1:
                # Right-pad with zeros after episode boundary.
                pad = self.seq_len - (offset + 1)
                obs_seq.extend([zero_fields["obs"]] * pad)
                action_seq.extend([zero_fields["action"]] * pad)
                reward_seq.extend([zero_fields["reward"]] * pad)
                next_obs_seq.extend([zero_fields["next_obs"]] * pad)
                done_seq.extend([zero_fields["done"]] * pad)
                legal_mask_seq.extend([zero_fields["legal_mask"]] * pad)
                next_legal_mask_seq.extend([zero_fields["next_legal_mask"]] * pad)
                break

        return (
            np.asarray(obs_seq, dtype=np.float32),
            np.asarray(action_seq, dtype=np.int64),
            np.asarray(reward_seq, dtype=np.float32),
            np.asarray(next_obs_seq, dtype=np.float32),
            np.asarray(done_seq, dtype=np.float32),
            np.asarray(legal_mask_seq, dtype=np.float32),
            np.asarray(next_legal_mask_seq, dtype=np.float32),
        )

    def sample(self, batch_size: int):
        """
        Returns a SequenceTransition with array fields shaped:
        - obs: [B, T, obs_dim]
        - action/reward/done: [B, T]
        - next_obs: [B, T, obs_dim]
        - legal_mask/next_legal_mask: [B, T, num_actions]
        """
        items = list(self.buffer)
        if not items:
            raise ValueError("Cannot sample from an empty replay buffer.")

        zero_fields = self._zero_fields_like(items[0])
        available_starts = list(range(len(items)))

        starts: list[int | None]
        if len(available_starts) >= batch_size:
            starts = random.sample(available_starts, batch_size)
        else:
            starts = random.sample(available_starts, len(available_starts))
            starts.extend([None] * (batch_size - len(available_starts)))

        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        legal_mask_batch = []
        next_legal_mask_batch = []

        for start in starts:
            if start is None:
                obs_seq = np.stack([zero_fields["obs"]] * self.seq_len).astype(
                    np.float32
                )
                action_seq = np.zeros(self.seq_len, dtype=np.int64)
                reward_seq = np.zeros(self.seq_len, dtype=np.float32)
                next_obs_seq = np.stack(
                    [zero_fields["next_obs"]] * self.seq_len
                ).astype(np.float32)
                done_seq = np.zeros(self.seq_len, dtype=np.float32)
                legal_mask_seq = np.stack(
                    [zero_fields["legal_mask"]] * self.seq_len
                ).astype(np.float32)
                next_legal_mask_seq = np.stack(
                    [zero_fields["next_legal_mask"]] * self.seq_len
                ).astype(np.float32)
            else:
                (
                    obs_seq,
                    action_seq,
                    reward_seq,
                    next_obs_seq,
                    done_seq,
                    legal_mask_seq,
                    next_legal_mask_seq,
                ) = self._build_padded_sequence(items, start, zero_fields)

            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            next_obs_batch.append(next_obs_seq)
            done_batch.append(done_seq)
            legal_mask_batch.append(legal_mask_seq)
            next_legal_mask_batch.append(next_legal_mask_seq)

        return SequenceTransition(
            obs=np.stack(obs_batch),
            action=np.stack(action_batch),
            reward=np.stack(reward_batch),
            next_obs=np.stack(next_obs_batch),
            done=np.stack(done_batch),
            legal_mask=np.stack(legal_mask_batch),
            next_legal_mask=np.stack(next_legal_mask_batch),
        )
