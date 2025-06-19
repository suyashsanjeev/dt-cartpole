"""
Plots last-layer attention heat-map for the Decision Transformer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch, gymnasium as gym

from decision_transformer_cartpole import DecisionTransformer, Config

# params
TARGET_RTG  = 200
AVG_HEADS   = True
CKPT        = "best_dt.pt" # model weights
OUTFILE     = "attention_heatmap_white.png"


def build_white_to_purple():
    """
    Returns a colormap whose 0-level is white, low end is viridis, high end is purple.
    """
    base = mpl.cm.get_cmap("viridis_r")
    colors = base(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])  # white
    return mpl.colors.ListedColormap(colors, name="white_viridis_r")


@torch.no_grad()
def last_layer_attention(model, rtg, states, actions):
    """
    Forward through all but last encoder layer. Gets last-layer attention.
    """
    K = rtg.size(1)
    d = Config.embed_dim
    rtg_tok   = model.rtg_embed(rtg.unsqueeze(-1))
    state_tok = model.state_embed(states)
    act_tok   = model.act_embed(actions)
    tokens = torch.stack((rtg_tok, state_tok, act_tok), dim=2).reshape(1, K*3, d)
    type_ids = torch.arange(3, device=tokens.device).repeat(K)
    tokens += model.type_embed(type_ids)[None]
    tokens += model.pos_embed[:, :tokens.size(1)]
    L = tokens.size(1)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 1)

    for layer in model.backbone.layers[:-1]:
        tokens = layer(tokens, src_mask=mask)

    last = model.backbone.layers[-1]
    _, attn = last.self_attn(
        tokens, tokens, tokens,
        attn_mask=mask,
        need_weights=True,
        average_attn_weights=False
    )
    return attn.squeeze(0)


def rollout_attention(model, env):
    """
    Collects one rollout and returns averaged attention matrix.
    """
    s, _ = env.reset(seed=0)
    states, actions, rtgs = [], [], []
    ep_ret = 0

    while True:
        if len(states) >= Config.context_len:
            states.pop(0); actions.pop(0); rtgs.pop(0)

        states.append(torch.tensor(s, dtype=torch.float32))
        rtgs.append(torch.tensor(TARGET_RTG - ep_ret, dtype=torch.float32))
        actions.append(torch.tensor(0 if not actions else a, dtype=torch.long))

        pad = Config.context_len - len(states)
        rtg_b  = torch.nn.functional.pad(torch.stack(rtgs),  (pad,0))[None].to(Config.device)
        st_b   = torch.nn.functional.pad(torch.stack(states),(0,0,pad,0))[None].to(Config.device)
        act_b  = torch.nn.functional.pad(torch.stack(actions),(pad,0))[None].to(Config.device)

        attn = last_layer_attention(model, rtg_b, st_b, act_b)
        attn = attn.mean(0) if AVG_HEADS else attn[0]  # (L,L)

        logits = model(rtg_b, st_b, act_b)
        a = torch.argmax(logits[0, -1]).item()
        s, r, term, trunc, _ = env.step(a)
        ep_ret += r
        if term or trunc: return attn.cpu()


def main():
    env = gym.make(Config.env_id)
    model = DecisionTransformer(env.observation_space.shape[0],
                                env.action_space.n).to(Config.device)
    model.load_state_dict(torch.load(CKPT, map_location=Config.device))
    model.eval()

    attn = rollout_attention(model, env)

    cmap = build_white_to_viridis()
    plt.figure(figsize=(6,5))
    plt.imshow(attn, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Attention weight")

    # dashed every 3 for (rtg, state, action) triplets
    for i in range(0, attn.shape[0], 3):
        plt.axhline(i-0.5, ls="--", lw=0.4, color="k", alpha=0.2)
        plt.axvline(i-0.5, ls="--", lw=0.4, color="k", alpha=0.2)
    plt.title("Last-layer Attention Heat-map (Heads Avg.)")
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=300)
    print(f"Saved to: {OUTFILE}")


if __name__ == "__main__":
    main()

