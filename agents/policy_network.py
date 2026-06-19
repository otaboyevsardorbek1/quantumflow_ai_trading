"""
QuantumFlow AI Trading System v2.0 - Transformer-based Policy Network
Advanced architecture with multi-head attention for feature importance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FeatureAttention(nn.Module):
    """Feature-level attention for interpretability"""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.feature_query = nn.Linear(d_model, d_model)
        self.feature_key = nn.Linear(d_model, d_model)
        self.feature_value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        Q = self.feature_query(x)
        K = self.feature_key(x)
        V = self.feature_value(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.matmul(attn_weights, V)

        # Feature importance (average across sequence)
        feature_importance = attn_weights.mean(dim=(1, 2))

        return out, feature_importance

class TransformerPolicyNetwork(nn.Module):
    """
    Transformer-based Actor-Critic network for trading

    Architecture:
    - Input: (batch, window, n_features)
    - Feature embedding
    - Positional encoding
    - Transformer encoder (multi-layer)
    - Feature attention (for interpretability)
    - Actor head: mean and log_std for continuous actions
    - Critic head: state value estimation
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        action_dim: int = 4,  # [direction, size, sl, tp]
        use_attention_visualization: bool = True,
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        self.action_dim = action_dim
        self.use_attention_visualization = use_attention_visualization

        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Feature attention for interpretability
        self.feature_attention = FeatureAttention(n_features, d_model)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Actor head (Policy)
        self.actor_hidden = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean and log_std for continuous actions
        self.actor_mean = nn.Linear(dim_feedforward // 2, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head (Value)
        self.critic_hidden = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.critic_value = nn.Linear(dim_feedforward // 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: (batch, window, n_features)

        Returns:
            action_mean: (batch, action_dim)
            value: (batch, 1)
            feature_importance: (batch, n_features) or None
        """
        batch_size = x.size(0)

        # Input embedding: (batch, window, n_features) -> (batch, window, d_model)
        x = self.input_embedding(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Feature attention
        if self.use_attention_visualization:
            x, feature_importance = self.feature_attention(x)
        else:
            feature_importance = None

        # Layer norm
        x = self.layer_norm(x)

        # Global pooling: (batch, window, d_model) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, window)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)

        # Actor
        actor_features = self.actor_hidden(x)
        action_mean = self.actor_mean(actor_features)

        # Action constraints
        # direction: -1 to 1
        action_mean[:, 0] = torch.tanh(action_mean[:, 0])
        # size: 0 to 1
        action_mean[:, 1] = torch.sigmoid(action_mean[:, 1])
        # sl: 0.5 to 3.0 (scaled)
        action_mean[:, 2] = 0.5 + 2.5 * torch.sigmoid(action_mean[:, 2])
        # tp: 1.0 to 5.0 (scaled)
        action_mean[:, 3] = 1.0 + 4.0 * torch.sigmoid(action_mean[:, 3])

        # Critic
        critic_features = self.critic_hidden(x)
        value = self.critic_value(critic_features)

        return action_mean, value, feature_importance

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """Action sampling"""
        with torch.no_grad():
            action_mean, value, _ = self.forward(x)

            if deterministic:
                action = action_mean
            else:
                log_std = self.actor_log_std.expand_as(action_mean)
                std = torch.exp(log_std)
                action = action_mean + std * torch.randn_like(action_mean)

                # Clip actions
                action[:, 0] = torch.clamp(action[:, 0], -1.0, 1.0)
                action[:, 1] = torch.clamp(action[:, 1], 0.0, 1.0)
                action[:, 2] = torch.clamp(action[:, 2], 0.5, 3.0)
                action[:, 3] = torch.clamp(action[:, 3], 1.0, 5.0)

        return action.cpu().numpy(), value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO uchun action evaluation"""
        action_mean, value, _ = self.forward(x)

        log_std = self.actor_log_std.expand_as(action_mean)
        std = torch.exp(log_std)

        # Log probability
        log_prob = -0.5 * (((actions - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Entropy
        entropy = 0.5 * (1.0 + math.log(2 * math.pi) + 2 * log_std).sum(dim=-1, keepdim=True)

        return log_prob, entropy, value

class EnsemblePolicy(nn.Module):
    """
    Multi-Agent Ensemble: 3 ta agent (Trend, Mean-Reversion, Breakout)
    Har biri o'ziga xos strategiyani o'rganadi
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 128,
        ensemble_size: int = 3,
        d_model: int = 256,
        **kwargs
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.agents = nn.ModuleList([
            TransformerPolicyNetwork(n_features, window_size, d_model, **kwargs)
            for _ in range(ensemble_size)
        ])

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)

        # Agent names
        self.agent_names = ['TrendFollower', 'MeanReverter', 'BreakoutTrader']

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Ensemble forward pass

        Returns:
            weighted_action: (batch, action_dim)
            weighted_value: (batch, 1)
            agent_info: dict with individual outputs
        """
        actions = []
        values = []

        for agent in self.agents:
            action_mean, value, _ = agent(x)
            actions.append(action_mean)
            values.append(value)

        actions = torch.stack(actions, dim=0)  # (ensemble_size, batch, action_dim)
        values = torch.stack(values, dim=0)  # (ensemble_size, batch, 1)

        # Softmax weights
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted combination
        weighted_action = (weights.view(-1, 1, 1) * actions).sum(dim=0)
        weighted_value = (weights.view(-1, 1, 1) * values).sum(dim=0)

        agent_info = {
            'individual_actions': actions,
            'individual_values': values,
            'weights': weights,
            'agent_names': self.agent_names,
        }

        return weighted_action, weighted_value, agent_info

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, Dict]:
        """Ensemble action with individual agent outputs"""
        with torch.no_grad():
            weighted_action, weighted_value, agent_info = self.forward(x)

            if not deterministic:
                # Add noise to final action
                log_std = torch.zeros_like(weighted_action)
                std = torch.exp(log_std)
                weighted_action = weighted_action + std * torch.randn_like(weighted_action)

        return weighted_action.cpu().numpy(), weighted_value, agent_info
