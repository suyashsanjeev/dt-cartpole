"""
Visualizes the final-layer state embeddings of a trained
Decision Transformer on CartPole-v1.

Produces 2-D scatter (blue = left, red = right).
Reducer options: 'tsne', 'umap', or 'pca'.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, gymnasium as gym
from decision_transformer_cartpole import DecisionTransformer, Config, evaluate

REDUCER = "tsne" # options are tsne, pca, or umap
N_POINTS = 3000 # embeddings to collect

def get_reducer(name):
    if name == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(perplexity=30, init="random", random_state=0)
    elif name == "umap":
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0)
    elif name == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0)
    else:
        raise ValueError("Unknown reducer")

def collect_embeddings(model, env, n_points=3000):
    embeds, labels = [], []
    s, _ = env.reset(seed=42)
    states, actions, rtgs = [], [], []
    ep_ret = 0.0
    target = 200

    while len(embeds) < n_points:
        if len(states) >= Config.context_len:
            states.pop(0); actions.pop(0); rtgs.pop(0)

        states.append(s.astype(np.float32))
        rtgs.append(target - ep_ret)
        actions.append(0 if len(actions) == 0 else a)

        pad = Config.context_len - len(states)
        rtg_t  = torch.tensor(np.pad(rtgs,(pad,0)),dtype=torch.float32)[None].to(Config.device)
        st_t   = torch.tensor(np.pad(states,((pad,0),(0,0))),dtype=torch.float32)[None].to(Config.device)
        act_t  = torch.tensor(np.pad(actions,(pad,0)),dtype=torch.long)[None].to(Config.device)

        with torch.no_grad():
            # forward pass to get logits and hidden state for last state token
            logits = model(rtg_t, st_t, act_t)
            a = torch.argmax(logits[0, -1]).item()

            # grab hidden rep of state token at last timestep
            h_tokens = model.backbone(
                model.state_embed(st_t)
            )
            state_emb = h_tokens[0, -1]
            embeds.append(state_emb.cpu().numpy())
            labels.append(a)

        s, r, term, trunc, _ = env.step(a)
        ep_ret += r
        if term or trunc:
            s, _ = env.reset(seed=np.random.randint(1e6))
            states, actions, rtgs, ep_ret = [], [], [], 0.0

    return np.vstack(embeds), np.array(labels)

def main():
    env   = gym.make(Config.env_id)
    model = DecisionTransformer(env.observation_space.shape[0],
                                env.action_space.n).to(Config.device)
    model.load_state_dict(torch.load(Config.ckpt_path, map_location=Config.device))
    model.eval()

    X, y = collect_embeddings(model, env, N_POINTS)
    reducer = get_reducer(REDUCER)
    Z = reducer.fit_transform(X)

    cmap = plt.get_cmap("bwr")
    plt.figure(figsize=(4,4))
    plt.scatter(Z[:,0], Z[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    plt.axis('off')
    plt.title(f"{REDUCER.upper()} of State Embeddings")
    plt.tight_layout()
    plt.savefig("latent_map.png", dpi=300)
    print("Saved latent_map.png")

if __name__ == "__main__":
    main()

