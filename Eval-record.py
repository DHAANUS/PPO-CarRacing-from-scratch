def eval_and_record(env, policy, steps=1000, fname="ppo_carracing_demo.mp4", fps=30):
    frames, ep_ret = [], 0.0
    obs_raw, _ = reset_env(env, seed=123)
    fs = FrameStacker(cfg["frame_stack"])
    state = fs.reset(preprocess(obs_raw)).to(device)
    policy.eval()
    with torch.no_grad():
        for t in range(steps):
            a_env, _, _ = policy.act(state.unsqueeze(0))
            obs_raw, r, done, _ = step_env(env, a_env.squeeze(0).cpu().numpy().astype(np.float32))
            ep_ret += r
            frame = env.render()  # rgb_array
            frames.append(frame)
            state = fs.step(preprocess(obs_raw)).to(device)
            if done:
                break
    imageio.mimsave(fname, frames, fps=fps)
    print(f"Eval return: {ep_ret:.1f} | saved {fname}")

eval_and_record(env, policy, steps=800)