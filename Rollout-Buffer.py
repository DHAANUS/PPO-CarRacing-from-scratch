import  torch
class RolloutBuffer:
    def __init__(self, n, obs_shape, act_dim):
        self.obs = torch.zeros((n, *obs_shape), dtype=torch.float32)
        self.act = torch.zeros((n, act_dim), dtype=torch.float32)
        self.logp = torch.zeros(n, dtype=torch.float32)
        self.rew  = torch.zeros(n, dtype=torch.float32)
        self.done = torch.zeros(n, dtype=torch.float32)
        self.val  = torch.zeros(n, dtype=torch.float32)
        self.ptr, self.n = 0, n
    def add(self, o, a, lp, r, d, v):
        self.obs[self.ptr].copy_(o.cpu())
        self.act[self.ptr].copy_(a.detach().cpu())
        self.logp[self.ptr] = lp.detach().cpu()
        self.rew[self.ptr]  = r
        self.done[self.ptr] = float(d)
        self.val[self.ptr]  = v.detach().cpu()
        self.ptr += 1
    def gae(self, gamma, lam, last_v):
        adv = torch.zeros_like(self.rew)
        last = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_nonterm, next_v = 1.0 - self.done[t], last_v
            else:
                next_nonterm, next_v = 1.0 - self.done[t+1], self.val[t+1]
            delta = self.rew[t] + gamma * next_v * next_nonterm - self.val[t]
            last = delta + gamma * lam * next_nonterm * last
            adv[t] = last
        ret = adv + self.val[:self.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv[:self.ptr], ret[:self.ptr]