"""
Compare episode-return distributions for offline CartPole dataset
against Decision Transformer policy via histogram.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch, gymnasium as gym

from decision_transformer_cartpole import DecisionTransformer, Config, collect_trajectories

# load model
env   = gym.make(Config.env_id)
obs_n = env.observation_space.shape[0]
act_n = env.action_space.n

model = DecisionTransformer(obs_n, act_n).to(Config.device)
model.load_state_dict(torch.load(Config.ckpt_path, map_location=Config.device))
model.eval()

# collect returns
print("Collecting offline dataset …")
offline_data = collect_trajectories(env, Config.num_episodes)
returns_off  = [sum(tr["rewards"]) for tr in offline_data]

# dt policy returns
def run_policy(num_eps=200, target=200):
    rtns = []
    for _ in range(num_eps):
        s, _ = env.reset()
        states, actions, rtgs = [], [], []
        ep_ret = 0
        while True:
            if len(states) >= Config.context_len:
                states.pop(0); actions.pop(0); rtgs.pop(0)
            states.append(s.astype(np.float32))
            rtgs.append(target - ep_ret)
            actions.append(0 if not actions else a)
            pad = Config.context_len - len(states)
            rtg = np.pad(rtgs,(pad,0)); st = np.pad(states,((pad,0),(0,0))); ak = np.pad(actions,(pad,0))
            rtg = torch.tensor(rtg ,dtype=torch.float32)[None].to(Config.device)
            st  = torch.tensor(st  ,dtype=torch.float32)[None].to(Config.device)
            ak  = torch.tensor(ak  ,dtype=torch.long   )[None].to(Config.device)
            with torch.no_grad():
                logits = model(rtg, st, ak)
                a = torch.argmax(logits[0,-1]).item()
            s, r, term, trunc, _ = env.step(a)
            ep_ret += r
            if term or trunc: break
        rtns.append(ep_ret)
    return rtns

print("Running DT policy …")
returns_dt = run_policy(num_eps=200, target=200)

# print bucket counts
bins = np.arange(0, int(Config.eval_target_return) + 60, 10)   # 0-10-20 … 250
hist, edges = np.histogram(returns_off, bins=bins)

print("\nOffline dataset histogram (bin range -> count)")
for left, right, cnt in zip(edges[:-1], edges[1:], hist):
    print(f"[{left:3d}, {right:3d}) : {cnt}")

print("\nDT policy exact return counts")
for ret, cnt in Counter(returns_dt).most_common():
    print(f"Return {ret:6.1f} : {cnt}")

# plot
plt.figure(figsize=(5,4))
plt.hist(returns_off, bins=bins, alpha=0.5, label="offline data")
plt.hist(returns_dt , bins=bins, alpha=0.7, label="DT policy")
plt.xlabel("Episode return")
plt.ylabel("Count")
plt.title("Offline Data vs. Decision-Transformer Performance")
plt.legend()
plt.tight_layout()
plt.savefig("return_histogram.png", dpi=300)
print("\nSaved return_histogram.png")

