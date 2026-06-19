"""
QuantumFlow AI Trading System v2.0 - DreamerV3 Components
World model-based RL for sample-efficient learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

# Symlog / Symexp for stable training
def symlog(x):
    """Symmetric log transformation"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class Encoder(nn.Module):
    """Observation encoder"""
    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, obs):
        return self.net(obs)

class RSSM(nn.Module):
    """Recurrent State-Space Model (World Model core)"""
    def __init__(self, embed_dim, hidden_dim, stoch_dim, num_categories, action_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = num_categories
        self.stoch_size = stoch_dim * num_categories

        # Recurrent model
        self.gru = nn.GRUCell(embed_dim + self.stoch_size + action_dim, hidden_dim)

        # Prior (dynamics predictor)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * num_categories),
        )

        # Posterior (representation)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * num_categories),
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_size, device=device)
        return h, z

    def observe(self, embed, action, h, z):
        # Prior
        prior_logits = self.prior_net(h).reshape(-1, self.stoch_dim, self.num_categories)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)

        # Posterior
        state = self.get_state(h, z)
        posterior_input = torch.cat([state, embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input).reshape(-1, self.stoch_dim, self.num_categories)
        posterior_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)

        z = posterior_dist.sample().reshape(-1, self.stoch_size)

        # Recurrent update
        h = self.gru(torch.cat([embed, z, action], dim=-1), h)

        return h, z, prior_logits, posterior_logits

    def imagine(self, action, h, z):
        state = self.get_state(h, z)
        prior_logits = self.prior_net(h).reshape(-1, self.stoch_dim, self.num_categories)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
        z = prior_dist.sample().reshape(-1, self.stoch_size)
        h = self.gru(torch.cat([torch.zeros_like(z[:, :1]), z, action], dim=-1), h)
        return h, z, prior_logits

    def get_state(self, h, z):
        return torch.cat([h, z], dim=-1)

    def kl_loss(self, prior_logits, posterior_logits, free_nats=1.0, kl_balance=0.8):
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
        posterior_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)

        kl_posterior = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        kl_prior = torch.distributions.kl_divergence(prior_dist, posterior_dist)

        kl = kl_balance * kl_posterior + (1 - kl_balance) * kl_prior
        kl = torch.maximum(kl, torch.tensor(free_nats))
        return kl.mean()

class Decoder(nn.Module):
    """Observation decoder"""
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, obs_dim),
        )

    def forward(self, state):
        return self.net(state)

class RewardPredictor(nn.Module):
    """Reward prediction"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

class Actor(nn.Module):
    """Policy network for Dreamer"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        return mean

    def sample(self, state, deterministic=False):
        mean = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return torch.tanh(action)

class Critic(nn.Module):
    """Value network for Dreamer"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.net(state)
