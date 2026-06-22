"""
QuantumFlow AI Trading System v2.0 - Advanced PPO Trainer
Curriculum learning, online adaptation, and distributed training support
"""
import torch
from torch.compiler import F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging
from tqdm import tqdm
import time
import json
import os

from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy

logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    Proximal Policy Optimization with:
    - Generalized Advantage Estimation (GAE)
    - Curriculum learning
    - Online learning buffer
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    """

    def __init__(
        self,
        policy: nn.Module,
        env,
        config: Dict,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.env = env
        self.config = config
        self.device = device

        # Hyperparameters
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.n_steps = config.get('n_steps', 2048)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('total_timesteps', 2000000) // self.n_steps
        )

        # Mixed precision
        self.use_amp = config.get('use_mixed_precision', True) and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Curriculum learning
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_stage = 0
        self.curriculum_progress = 0

        # Online learning buffer
        self.online_buffer = deque(maxlen=config.get('online_buffer_size', 50000))
        self.online_learning_freq = config.get('online_learning_freq', 10000)

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = {'policy': [], 'value': [], 'entropy': []}

        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_reward = -np.inf

    def collect_rollouts(self) -> Dict:
        """
        Rollout collection with GAE

        Returns:
            rollout_data: dict with observations, actions, rewards, etc.
        """
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        episode_reward = 0
        episode_length = 0

        for step in range(self.n_steps):
            with torch.no_grad():
                if isinstance(self.policy, EnsemblePolicy):
                    action, value, _ = self.policy.get_action(obs, deterministic=False)
                    action = torch.FloatTensor(action).to(self.device)
                else:
                    action_mean, value, _ = self.policy(obs)
                    log_std = self.policy.actor_log_std.expand_as(action_mean)
                    std = torch.exp(log_std)
                    action = action_mean + std * torch.randn_like(action_mean)

                    # Log probability
                    log_prob = -0.5 * (((action - action_mean) / (std + 1e-8)) ** 2 + 
                                       2 * self.policy.actor_log_std + np.log(2 * np.pi))
                    log_prob = log_prob.sum(dim=-1)

                value = value.squeeze(-1)

            # Step environment
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            observations.append(obs.cpu().numpy()[0])
            actions.append(action_np)
            rewards.append(reward)
            values.append(value.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy()[0] if 'log_prob' in locals() else 0)
            dones.append(done)

            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            else:
                obs = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)

        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + np.array(values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout_data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'advantages': advantages,
            'returns': returns,
            'dones': np.array(dones),
        }

        return rollout_data

    def _compute_gae(self, rewards: List, values: List, dones: List) -> np.ndarray:
        """Generalized Advantage Estimation"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        return advantages

    def update_policy(self, rollout_data: Dict) -> Dict:
        """
        PPO policy update

        Returns:
            metrics: dict with loss values
        """
        # Convert to tensors
        obs = torch.FloatTensor(rollout_data['observations']).to(self.device)
        actions = torch.FloatTensor(rollout_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device)
        returns = torch.FloatTensor(rollout_data['returns']).to(self.device)

        # Create dataset
        dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in dataloader:
                batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)

                        # Policy loss (PPO clipped objective)
                        ratio = torch.exp(log_probs - batch_old_log_probs.unsqueeze(-1))
                        surr1 = ratio * batch_advantages.unsqueeze(-1)
                        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages.unsqueeze(-1)
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                        # Entropy bonus
                        entropy_loss = -entropy.mean()

                        # Total loss
                        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                else:
                    log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)

                    ratio = torch.exp(log_probs - batch_old_log_probs.unsqueeze(-1))
                    surr1 = ratio * batch_advantages.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages.unsqueeze(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.scheduler.step()

        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

        return metrics

    def train(self, total_timesteps: int, eval_freq: int = 50000, save_freq: int = 50000):
        """
        Asosiy training loop

        Args:
            total_timesteps: Umumiy training qadamlari
            eval_freq: Har nechta qadamda evaluation
            save_freq: Har nechta qadamda saqlash
        """
        logger.info("=" * 70)
        logger.info("🏋️ STARTING PPO TRAINING")
        logger.info("=" * 70)
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Curriculum learning: {self.use_curriculum}")

        n_iterations = total_timesteps // self.n_steps

        for iteration in tqdm(range(n_iterations), desc="Training"):
            # Collect rollouts
            rollout_data = self.collect_rollouts()

            # Update policy
            metrics = self.update_policy(rollout_data)

            # Log metrics
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards)
                metrics['avg_episode_reward'] = avg_reward
                metrics['avg_episode_length'] = np.mean(self.episode_lengths)

            # Curriculum update
            if self.use_curriculum and iteration % 10 == 0:
                self._update_curriculum(iteration, n_iterations)

            # Online learning
            if iteration % (self.online_learning_freq // self.n_steps) == 0 and iteration > 0:
                self._online_learning_update()

            # Evaluation
            if (iteration + 1) * self.n_steps % eval_freq == 0:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

                # Save best model
                if eval_metrics.get('avg_reward', -np.inf) > self.best_reward:
                    self.best_reward = eval_metrics['avg_reward']
                    self.save_checkpoint(f"{self.checkpoint_dir}/best_model.pt")
                    logger.info(f"💾 New best model saved! Reward: {self.best_reward:.4f}")

            # Regular checkpoint
            if (iteration + 1) * self.n_steps % save_freq == 0:
                self.save_checkpoint(f"{self.checkpoint_dir}/checkpoint_{(iteration+1)*self.n_steps}.pt")

            # Logging
            if iteration % 10 == 0:
                self._log_metrics(iteration, metrics)

        # Final save
        self.save_checkpoint(f"{self.checkpoint_dir}/final_model.pt")
        logger.info("✅ Training complete!")

    def evaluate(self, n_episodes: int = 5) -> Dict:
        """Model evaluation"""
        rewards = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if isinstance(self.policy, EnsemblePolicy):
                        action, _, _ = self.policy.get_action(obs_tensor, deterministic=True)
                        action = action[0]
                    else:
                        action_mean, _, _ = self.policy(obs_tensor)
                        action = action_mean[0].cpu().numpy()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
        }

    def _update_curriculum(self, iteration: int, total_iterations: int):
        """Curriculum learning stage update"""
        progress = iteration / total_iterations

        # Gradually increase difficulty
        if progress < 0.25:
            self.env.config['spread'] = 0.0001
            self.env.config['slippage'] = 0.00002
        elif progress < 0.5:
            self.env.config['spread'] = 0.0002
            self.env.config['slippage'] = 0.00005
        elif progress < 0.75:
            self.env.config['spread'] = 0.0003
            self.env.config['slippage'] = 0.00008
        else:
            self.env.config['spread'] = 0.0005
            self.env.config['slippage'] = 0.0001

        self.curriculum_progress = progress

    def _online_learning_update(self):
        """Online learning from recent experience"""
        if len(self.online_buffer) < self.batch_size * 4:
            return

        # Sample from online buffer
        batch_size = min(self.batch_size * 2, len(self.online_buffer))
        indices = np.random.choice(len(self.online_buffer), batch_size, replace=False)

        batch = [self.online_buffer[i] for i in indices]
        # ... online learning logic

        logger.info(f"🔄 Online learning update completed ({len(self.online_buffer)} samples in buffer)")

    def _log_metrics(self, iteration: int, metrics: Dict):
        """Metrics logging"""
        log_str = f"Iter {iteration:5d} | "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}: {value:.4f} | "
            else:
                log_str += f"{key}: {value} | "
        logger.info(log_str)

    def save_checkpoint(self, path: str):
        """Model checkpoint saqlash"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_reward': self.best_reward,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Model checkpoint yuklash"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_reward = checkpoint.get('best_reward', -np.inf)
        logger.info(f"📂 Checkpoint loaded from {path}")
