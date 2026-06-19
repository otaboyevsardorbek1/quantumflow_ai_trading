"""
QuantumFlow AI Trading System v2.0 - DreamerV3 Agent
World model-based RL for sample-efficient trading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional

from agents.dreamer_components import (
    Encoder, RSSM, Decoder, RewardPredictor, Actor, Critic,
    symlog, symexp
)

class DreamerV3Agent:
    """
    DreamerV3 Agent for Trading

    Learns a World Model of the market, then uses it to
    imagine trajectories and improve its policy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        device: str = 'cpu',
        embed_dim: int = 256,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        num_categories: int = 32,
        lr_world_model: float = 3e-4,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        horizon: int = 15,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,
        replay_capacity: int = 100000,
        seq_len: int = 64,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.horizon = horizon

        # Networks
        self.encoder = Encoder(obs_dim, embed_dim).to(device)
        self.rssm = RSSM(embed_dim, hidden_dim, stoch_dim, num_categories, action_dim).to(device)
        self.decoder = Decoder(hidden_dim + stoch_dim * num_categories, obs_dim).to(device)
        self.reward_predictor = RewardPredictor(hidden_dim + stoch_dim * num_categories).to(device)
        self.actor = Actor(hidden_dim + stoch_dim * num_categories, action_dim).to(device)
        self.critic = Critic(hidden_dim + stoch_dim * num_categories).to(device)

        # Optimizers
        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters())
        )
        self.optimizer_world_model = Adam(world_model_params, lr=lr_world_model)
        self.optimizer_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.seq_len = seq_len

        # Hyperparameters
        self.free_nats = free_nats
        self.kl_balance = kl_balance
        self.training_step = 0

        # State tracking
        self.prev_action = None
        self.prev_state = None

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            embed = self.encoder(obs_t)

            if self.prev_state is None:
                h, z = self.rssm.initial_state(1, self.device)
                action = torch.zeros(1, self.action_dim, device=self.device)
            else:
                h, z = self.prev_state
                action = self.prev_action

            h, z, _, _ = self.rssm.observe(embed, action, h, z)
            state = self.rssm.get_state(h, z)
            action = self.actor.sample(state, deterministic=deterministic)

            self.prev_action = action
            self.prev_state = (h, z)

            return action.detach().cpu().numpy()[0]

    def add_experience(self, obs, action, reward, done):
        """Add experience to replay buffer"""
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done
        })

    def train_step(self, batch_size: int = 16) -> Optional[Dict]:
        """Single training step"""
        if len(self.replay_buffer) < self.seq_len + 1:
            return None

        # Sample sequences
        batch = self._sample_sequences(batch_size)
        if batch is None:
            return None

        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        B, T = obs.shape[0], obs.shape[1]

        # Phase 1: Train World Model
        self.optimizer_world_model.zero_grad()

        embed = self.encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)
        h, z = self.rssm.initial_state(B, self.device)

        recon_losses, reward_losses, kl_losses = [], [], []

        for t in range(T):
            h, z, prior_logits, posterior_logits = self.rssm.observe(
                embed[:, t], action[:, t], h, z
            )
            state = self.rssm.get_state(h, z)

            obs_pred = self.decoder(state)
            recon_losses.append(F.mse_loss(obs_pred, obs[:, t]))

            reward_pred = self.reward_predictor(state)
            reward_losses.append(F.mse_loss(reward_pred, symlog(reward[:, t])))

            kl_losses.append(self.rssm.kl_loss(prior_logits, posterior_logits, self.free_nats, self.kl_balance))

        world_model_loss = (torch.stack(recon_losses).mean() + 
                           torch.stack(reward_losses).mean() + 
                           torch.stack(kl_losses).mean())

        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer_world_model.param_groups[0]['params'], 100.0)
        self.optimizer_world_model.step()

        # Phase 2: Imagine & Train Actor-Critic
        with torch.no_grad():
            h_start = h.detach()
            z_start = z.detach()

        # Train critic
        self.optimizer_critic.zero_grad()
        imagined_states, imagined_rewards = self._imagine_trajectory(h_start, z_start, self.horizon)
        value_loss = self._compute_value_loss(imagined_states.detach(), imagined_rewards.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.optimizer_critic.step()

        # Train actor
        imagined_states_actor, imagined_rewards_actor = self._imagine_trajectory(h_start, z_start, self.horizon)
        self.optimizer_actor.zero_grad()
        policy_loss = self._compute_policy_loss(imagined_states_actor, imagined_rewards_actor)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.optimizer_actor.step()

        self.training_step += 1

        return {
            'world_model_loss': world_model_loss.item(),
            'recon_loss': torch.stack(recon_losses).mean().item(),
            'reward_loss': torch.stack(reward_losses).mean().item(),
            'kl_loss': torch.stack(kl_losses).mean().item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    def _sample_sequences(self, batch_size: int) -> Optional[Dict]:
        """Sample sequences from replay buffer"""
        if len(self.replay_buffer) < self.seq_len + 1:
            return None

        sequences = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, len(self.replay_buffer) - self.seq_len)
            seq = self.replay_buffer[start_idx:start_idx + self.seq_len]

            sequences.append({
                'obs': np.stack([s['obs'] for s in seq]),
                'action': np.stack([s['action'] for s in seq]),
                'reward': np.array([s['reward'] for s in seq], dtype=np.float32),
                'done': np.array([s['done'] for s in seq], dtype=np.float32),
            })

        return {
            'obs': torch.FloatTensor(np.stack([s['obs'] for s in sequences])),
            'action': torch.FloatTensor(np.stack([s['action'] for s in sequences])),
            'reward': torch.FloatTensor(np.stack([s['reward'] for s in sequences])),
            'done': torch.FloatTensor(np.stack([s['done'] for s in sequences])),
        }

    def _imagine_trajectory(self, h, z, horizon):
        """Imagine trajectory by rolling out policy in world model"""
        states, rewards = [], []

        for _ in range(horizon):
            state = self.rssm.get_state(h, z)
            states.append(state)

            reward_pred = self.reward_predictor(state)
            rewards.append(symexp(reward_pred))

            action = self.actor.sample(state, deterministic=False)
            h, z, _ = self.rssm.imagine(action, h, z)

        return torch.stack(states, dim=1), torch.stack(rewards, dim=1)

    def _compute_value_loss(self, states, rewards):
        """Train critic using lambda-returns"""
        B, H = states.shape[0], states.shape[1]
        values = self.critic(states.reshape(B * H, -1)).reshape(B, H)

        with torch.no_grad():
            next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=self.device)], dim=1)
            td_targets = rewards + self.gamma * next_values

            returns = torch.zeros_like(rewards)
            returns[:, -1] = td_targets[:, -1]
            for t in reversed(range(H - 1)):
                returns[:, t] = td_targets[:, t] + self.gamma * self.lambda_ * (returns[:, t + 1] - next_values[:, t])

        return F.mse_loss(values, returns)

    def _compute_policy_loss(self, states, rewards):
        """Train actor to maximize expected return"""
        B, H = states.shape[0], states.shape[1]

        with torch.no_grad():
            values = self.critic(states.reshape(B * H, -1)).reshape(B, H)
            next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=self.device)], dim=1)
            td_targets = rewards + self.gamma * next_values
            advantages = td_targets - values

        action_mean = self.actor(states.reshape(B * H, -1)).reshape(B, H, -1)
        dist = torch.distributions.Normal(action_mean, torch.exp(self.actor.log_std))
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions).sum(dim=-1)

        return -(log_probs * advantages).mean()

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'training_step': self.training_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.training_step = checkpoint.get('training_step', 0)
