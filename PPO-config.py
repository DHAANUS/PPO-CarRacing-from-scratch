import random 
import torch
import numpy as np

cfg = dict(
    seed=42,
    frame_stack=4,
    total_steps=150_000,     # bump for better results (e.g., 1e6)
    rollout_len=2048,
    minibatch_size=256,
    ppo_epochs=8,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    log_interval=5_000,
)

random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


in_ch = 3 * cfg["frame_stack"]
policy = CNNActorCritic(in_ch, env.action_space).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])