import torch
fs = FrameStacker(cfg["frame_stack"])
obs_raw, _ = reset_env(env, seed=cfg["seed"])
obs_t = preprocess(obs_raw)
state = fs.reset(obs_t)
state = state.to(device)

episode_returns, ep_ret = [], 0.0
global_step = 0
pbar = trange(cfg["total_steps"], desc="Training PPO (CarRacing)", leave=True, miniters=cfg["log_interval"])

while global_step < cfg["total_steps"]:
    buf = RolloutBuffer(cfg["rollout_len"], (in_ch, 84, 84), int(np.prod(env.action_space.shape)))
    for _ in range(cfg["rollout_len"]):
        with torch.no_grad():
            a_env, logp, v = policy.act(state.unsqueeze(0))
        a_np = a_env.squeeze(0).cpu().numpy().astype(np.float32)
        next_obs_raw, r, done, info = step_env(env, a_np)
        ep_ret += r

        # store
        buf.add(state, torch.from_numpy(a_np), logp.squeeze(0), r, done, v.squeeze(0))

        # next state
        next_obs_t = preprocess(next_obs_raw)
        state = fs.step(next_obs_t).to(device)

        if done:
            episode_returns.append(ep_ret); ep_ret = 0.0
            next_obs_raw, _ = reset_env(env)
            next_obs_t = preprocess(next_obs_raw)
            state = fs.reset(next_obs_t).to(device)

        global_step += 1
        if global_step % cfg["log_interval"] == 0:
            last10 = np.mean(episode_returns[-10:]) if episode_returns else 0.0
            pbar.set_postfix(steps=global_step, avg_return=f"{last10:.1f}")
            pbar.update(cfg["log_interval"])

    with torch.no_grad():
        _, last_v = policy.forward(state.unsqueeze(0))
        last_v = last_v.squeeze(0).detach().cpu()
    adv, ret = buf.gae(cfg["gamma"], cfg["lam"], last_v)

    obs_mb = buf.obs[:buf.ptr].to(device)
    act_mb = buf.act[:buf.ptr].to(device)
    old_logp = buf.logp[:buf.ptr].to(device)
    adv_mb = adv.to(device); ret_mb = ret.to(device)

    n = obs_mb.shape[0]
    idxs = np.arange(n)
    for _ in range(cfg["ppo_epochs"]):
        np.random.shuffle(idxs)
        for s in range(0, n, cfg["minibatch_size"]):
            mb = idxs[s:s+cfg["minibatch_size"]]
            new_logp, v_pred = policy.eval_action_logp_v(obs_mb[mb], act_mb[mb])
            ratio = torch.exp(new_logp - old_logp[mb])
            surr1 = ratio * adv_mb[mb]
            surr2 = torch.clamp(ratio, 1-cfg["clip_ratio"], 1+cfg["clip_ratio"]) * adv_mb[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(v_pred, ret_mb[mb])
            dist, _ = policy.forward(obs_mb[mb])
            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + cfg["vf_coef"]*v_loss - cfg["ent_coef"]*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()

print("Episodes:", len(episode_returns))
print("Recent returns:", [round(x,1) for x in episode_returns[-10:]])