"""
A compact, self-contained Decision Transformer
trained offline and evaluated online on CartPole-v1.

- generates offline dataset with a random policy (2 000 episodes).
- trains a GPT-style causal Transformer to predict actions.
- conditions on return-to-go so the single model can target
  different reward levels.
- evaluates learned policy for 20 episodes.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters and config
@dataclass
class Config:
    # environment
    env_id: str = "CartPole-v1"
    max_ep_len: int = 500

    # dataset
    num_episodes: int = 2_000
    pct_traj_for_eval: float = 0.05

    # model architecture
    context_len: int = 30 #K=30 in original DT paper
    embed_dim: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    # optimization
    lr: float = 3e-4
    batch_size: int = 64
    train_steps: int = 50_000
    warmup_steps: int = 2_000
    grad_clip: float = 1.0

    # evaluation
    eval_target_return: float = 200.0
    num_eval_episodes: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: str = "best_dt.pt"

C = Config()

# Offline dataset generation
def collect_trajectories(env: gym.Env, num_eps: int) -> List[dict]:
    """
    Roll random policy to generate offline data.
    Returns a list of dicts: {states, actions, rewards}.
    """
    data = []
    for _ in tqdm(range(num_eps), desc="Generating offline data"):
        s, _ = env.reset()
        traj = {"states": [], "actions": [], "rewards": []}
        for _ in range(C.max_ep_len):
            a = env.action_space.sample()
            s2, r, term, trunc, _ = env.step(a)
            traj["states"].append(s.astype(np.float32))
            traj["actions"].append(a)
            traj["rewards"].append(r)
            s = s2
            if term or trunc:
                break
        data.append(traj)
    return data


def compute_rtgs(rewards: List[float]) -> List[float]:
    """
    Return-to-go for every timestep t: R_t: sum k=t to T-1 of r_k
    """
    rtg = 0.0
    rtgs = []
    for r in reversed(rewards):
        rtg += r
        rtgs.append(rtg)
    return list(reversed(rtgs))


class TrajectoryDataset(Dataset):
    """
    Each sample is a sliding window of length context_len
    containing (return, state, action) triplets.
    Padding is added on the left.
    """
    def __init__(self, trajectories: List[dict], context_len: int):
        self.context_len = context_len
        self.samples: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for traj in trajectories:
            states  = traj["states"]
            actions = traj["actions"]
            rewards = traj["rewards"]
            rtgs    = compute_rtgs(rewards)
            T = len(states)

            for t in range(T):
                end   = t + 1
                start = max(0, end - context_len)
                pad   = context_len - (end - start)

                self.samples.append((
                    np.pad(rtgs[start:end],  (pad, 0), mode="constant"),
                    np.pad(states[start:end], ((pad, 0), (0, 0)),
                           mode="constant"),
                    np.pad(actions[start:end], (pad, 0),
                           mode="constant", constant_values=-1)
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rtg, state, action = self.samples[idx]

        action_mask = (action != -1)
        action[action == -1] = 0

        return (
            torch.tensor(rtg, dtype=torch.float32),
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(action_mask, dtype=torch.bool)
        )

# Decision Transformer definition
class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim: int, act_n: int):
        super().__init__()
        d = C.embed_dim

        # token embeddings
        self.rtg_embed   = nn.Linear(1, d)
        self.state_embed = nn.Linear(obs_dim, d)
        self.act_embed   = nn.Embedding(act_n, d)

        # type and positional embeddings
        self.type_embed = nn.Embedding(3, d) # 0=rtg, 1=state, 2=action
        self.pos_embed  = nn.Parameter(torch.zeros(1, C.context_len * 3, d))

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=C.n_heads,
            dim_feedforward=4*d,
            dropout=C.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, C.n_layers)

        # final head -> action logits
        self.action_head = nn.Linear(d, act_n)

    def forward(self, rtg, states, actions):
        """
	Params:
         - rtg: (B, K) float32
         - states: (B, K, obs_dim) float32
         - actions: (B, K) int64 (prev actions)

        returns logits for actions at timesteps 0 to K-1
        """
        B, K = rtg.shape

        # embed each modality
        rtg_tok   = self.rtg_embed(rtg.unsqueeze(-1))   # (B,K,d)
        state_tok = self.state_embed(states)            # (B,K,d)
        act_tok   = self.act_embed(actions)             # (B,K,d)

        # interleave to [rtg_0, state_0, action_0, etc.] length = 3*K
        tokens = torch.stack((rtg_tok, state_tok, act_tok), dim=2)
        tokens = tokens.view(B, K*3, -1)

        # add type and positional embeddings
        type_ids = torch.arange(3, device=tokens.device).repeat(K)
        tokens += self.type_embed(type_ids)[None, :, :]
        tokens += self.pos_embed[:, :tokens.size(1), :]

        # causal mask ensures autoregressive nature
        seq_len = tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool,
                       device=tokens.device), diagonal=1)

        h = self.backbone(tokens, causal_mask)

	# grab state outputs
        h = h.view(B, K, 3, -1)[:, :, 1, :]
        return self.action_head(h)


# Dataloaders, training, and evaluation
def make_dataloaders(env):
    """
    Generates offline trajectories, split them into train/val sets, 
    and returns batch DataLoaders for each.
    """
    offline_data = collect_trajectories(env, C.num_episodes)
    random.shuffle(offline_data)
    cut = int(len(offline_data) * (1 - C.pct_traj_for_eval))
    train_ds = TrajectoryDataset(offline_data[:cut], C.context_len)
    val_ds   = TrajectoryDataset(offline_data[cut:],  C.context_len)
    train_dl = DataLoader(train_ds, C.batch_size, shuffle=True,  pin_memory=True)
    val_dl   = DataLoader(val_ds,   C.batch_size, shuffle=False, pin_memory=True)
    return train_dl, val_dl


def train(model, train_dl, val_dl):
    """
    Runs AdamW optimization with LR warm-up with periodic validation and
    checkpointing until train_steps updates.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=C.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step+1)/C.warmup_steps, 1.0))

    best_val = float("inf")
    it = 0
    pbar = tqdm(total=C.train_steps, desc="Training")

    while it < C.train_steps:
        for rtg, s, a, mask in train_dl:
            model.train()
            rtg, s, a, mask = (rtg.to(C.device),
                               s.to(C.device),
                               a.to(C.device),
                               mask.to(C.device))
            logits = model(rtg, s, a)

            loss = nn.functional.cross_entropy(
                logits[mask], a[mask])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
            opt.step()
            scheduler.step()

            it += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if it >= C.train_steps:
                break

        # validation pass
        with torch.no_grad():
            model.eval()
            total, n = 0.0, 0
            for rtg, s, a, mask in val_dl:
                rtg, s, a, mask = (rtg.to(C.device),
                                   s.to(C.device),
                                   a.to(C.device),
                                   mask.to(C.device))
                logits = model(rtg, s, a)
                total += nn.functional.cross_entropy(
                    logits[mask], a[mask], reduction='sum').item()
                n += mask.sum().item()
            val_loss = total / n
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), C.ckpt_path)
            pbar.set_postfix(train=f"{loss.item():.3f}",
                             val=f"{val_loss:.3f}")
    pbar.close()
    print(f"finished training — best val CE {best_val:.3f}")


def evaluate(model, env, target_return=C.eval_target_return):
    """
    Runs policy for num_eval_episodes episodes and prints stats.
    """
    model.eval()
    returns = []

    for _ in range(C.num_eval_episodes):
        s, _ = env.reset()
        states, actions, rtgs = [], [], []
        ep_ret = 0.0

        for t in range(C.max_ep_len):
            # maintain sliding context window
            if len(states) >= C.context_len:
                states.pop(0); actions.pop(0); rtgs.pop(0)

            states.append(s.astype(np.float32))
            rtgs.append(target_return - ep_ret)
            actions.append(0 if t == 0 else a)

            # leftpad to context_len
            K = len(states)
            pad = C.context_len - K
            rtg_in   = np.pad(rtgs,  (pad, 0))
            state_in = np.pad(states, ((pad, 0), (0, 0)))
            act_in   = np.pad(actions, (pad, 0), constant_values=0)

            rtg_ten   = torch.tensor(rtg_in,  dtype=torch.float32).unsqueeze(0).to(C.device)
            state_ten = torch.tensor(state_in, dtype=torch.float32).unsqueeze(0).to(C.device)
            act_ten   = torch.tensor(act_in,  dtype=torch.long).unsqueeze(0).to(C.device)

            with torch.no_grad():
                logits = model(rtg_ten, state_ten, act_ten)
                a = torch.argmax(logits[0, -1]).item()

            s, r, term, trunc, _ = env.step(a)
            ep_ret += r
            if term or trunc:
                break

        returns.append(ep_ret)

    mean, std = np.mean(returns), np.std(returns)
    print(f"Evaluation over {C.num_eval_episodes} eps — "
          f"avg return {mean:.1f} ± {std:.1f}")
    return returns


def main():
    env = gym.make(C.env_id)
    obs_dim = env.observation_space.shape[0]
    act_n = env.action_space.n

    train_dl, val_dl = make_dataloaders(env)

    model = DecisionTransformer(obs_dim, act_n).to(C.device)

    train(model, train_dl, val_dl)

    model.load_state_dict(torch.load(C.ckpt_path))
    evaluate(model, env, target_return=C.eval_target_return)


if __name__ == "__main__":
    main()

