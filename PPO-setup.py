
import torch
import torch.nn as nn
from torch.distributions import Normal
class CNNActorCritic(nn.Module):
    def __init__(self, in_ch, action_space):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),   nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, 84, 84)).view(1,-1).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU(inplace=True))

        self.act_dim = int(np.prod(action_space.shape))  
        self.mu = nn.Linear(256, self.act_dim)
        self.log_std = nn.Parameter(torch.tensor([-0.3, -0.7, -2.0]))  
        self.v = nn.Linear(256, 1)
        nn.init.zeros_(self.mu.weight)
        self.mu.bias.data = torch.tensor([0.0, -0.5, -2.0])

    def encode(self, x):
        z = self.conv(x); z = z.view(z.size(0), -1); return self.fc(z)

    def forward(self, x):
        z = self.encode(x)
        mu = self.mu(z)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        v = self.v(z).squeeze(-1)
        return dist, v

    def _squash(self, a_pre):
        steer_pre, gas_pre, brake_pre = a_pre[...,0:1], a_pre[...,1:2], a_pre[...,2:3]
        steer = torch.tanh(steer_pre)        
        gas   = torch.sigmoid(gas_pre)       
        brake = torch.sigmoid(brake_pre)     
        return torch.cat([steer, gas, brake], dim=-1)

    def act(self, x):
        dist, v = self.forward(x)
        a_pre = dist.rsample()
        a_env = self._squash(a_pre)

        sp, gp, bp = a_pre[...,0], a_pre[...,1], a_pre[...,2]
        steer_corr = torch.log(1 - torch.tanh(sp)**2 + 1e-8)
        s = torch.sigmoid(gp); gas_corr = torch.log(s*(1 - s) + 1e-8)
        b = torch.sigmoid(bp); brake_corr = torch.log(b*(1 - b) + 1e-8)
        logp = dist.log_prob(a_pre).sum(-1) + steer_corr + gas_corr + brake_corr
        return a_env, logp, v

    def eval_action_logp_v(self, x, a_env):
        steer = torch.clamp(a_env[...,0], -0.999, 0.999)
        gas   = torch.clamp(a_env[...,1], 1e-6, 1-1e-6)
        brake = torch.clamp(a_env[...,2], 1e-6, 1-1e-6)

        sp = torch.atanh(steer)
        gp = torch.log(gas) - torch.log(1 - gas)      
        bp = torch.log(brake) - torch.log(1 - brake)  
        a_pre = torch.stack([sp, gp, bp], dim=-1)

        dist, v = self.forward(x)
        base = dist.log_prob(a_pre).sum(-1)
        corr = torch.log(1 - torch.tanh(sp)**2 + 1e-8) \
             + torch.log(gas*(1 - gas) + 1e-8) \
             + torch.log(brake*(1 - brake) + 1e-8)
        logp = base + corr
        return logp, v