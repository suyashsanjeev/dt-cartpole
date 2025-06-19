"""
Plot how closely a Decision Transformer conditioned on RTG=200 hits various CartPole
return-to-go targets ranging from 50-500.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, gymnasium as gym

from decision_transformer_cartpole import DecisionTransformer, Config, evaluate

CKPT = "best_dt.pt"
TARGETS = np.arange(50, 550, 50)
EPISODES_PER_TARGET = 20

def main():
    env   = gym.make(Config.env_id)
    model = DecisionTransformer(env.observation_space.shape[0],
                                env.action_space.n).to(Config.device)
    model.load_state_dict(torch.load(CKPT, map_location=Config.device))

    achieved = [np.mean(evaluate(model, env, t)) for t in TARGETS]

    plt.figure(figsize=(5, 4))
    plt.plot(TARGETS, achieved, marker="o")
    plt.plot([0, 550], [0, 550], ls="--", lw=1)
    plt.xlabel("Target Return")
    plt.ylabel("Achieved Return")
    plt.title("Return-to-Go Target vs. Achieved Return")
    plt.xlim(0, 525); plt.ylim(0, 525)
    plt.tight_layout()
    plt.savefig("rtg_vs_return.png", dpi=300)
    print("Saved rtg_vs_return.png")

if __name__ == "__main__":
    main()

