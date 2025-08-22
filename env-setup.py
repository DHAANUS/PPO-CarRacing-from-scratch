import torch
import gymnasium as gym
import cv2
import deque

print("Gymnasium:", gym.__version__)
print("Torch:", torch.__version__)

# ---------- Helpers (Gymnasium API) ----------
def reset_env(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs, info

def step_env(env, action):
    obs, r, terminated, truncated, info = env.step(action)
    return obs, r, (terminated or truncated), info

# ---------- Env ----------
# CarRacing-v3: obs = (96,96,3) uint8, action = [steer∈[-1,1], gas∈[0,1], brake∈[0,1]]
env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=True)
print("Obs space:", env.observation_space, "| Act space:", env.action_space)

# ---------- Preprocess & Frame Stack ----------
def preprocess(obs, out=84):
    img = cv2.resize(obs, (out, out), interpolation=cv2.INTER_AREA)
    # keep RGB; normalize to [0,1]
    return torch.from_numpy(img).float().permute(2,0,1) / 255.0  # [3,H,W]

class FrameStacker:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    def reset(self, obs_t):
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs_t.clone())
        return torch.cat(list(self.frames), dim=0)  # [3k,H,W]
    def step(self, obs_t):
        self.frames.append(obs_t)
        return torch.cat(list(self.frames), dim=0)