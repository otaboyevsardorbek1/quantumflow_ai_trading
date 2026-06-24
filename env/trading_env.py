"""
QuantumFlow AI Trading System v2.0 - Trading Environment
Observation: faqat feature window, pos_info olib tashlangan.
"""
import numpy as np
from typing import Dict, Optional
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)

class QuantumTradingEnv(gym.Env):
    """
    Savdo muhiti:
    - Observation: (window * n_features) o'lchamli vektor
    - Action: [direction, size, sl, tp]
    - Reward: P&L ga asoslangan
    """

    def __init__(self, features: np.ndarray, returns: np.ndarray,
                 timestamps: np.ndarray, window: int = 128,
                 config: Optional[Dict] = None, symbol: str = "XAUUSD"):
        super().__init__()
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.timestamps = timestamps
        self.window = window
        self.config = config or {}
        self.symbol = symbol
        self.T = len(features)
        self.n_features = features.shape[1]

        # Action space: [direction, size, sl, tp]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.5, 1.0]),
            high=np.array([1.0, 1.0, 3.0, 5.0]),
            dtype=np.float32
        )

        # Observation space - faqat window_features (pos_info yo'q)
        obs_dim = window * self.n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window
        self.position = 0.0
        self.entry_price = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.total_pnl = 0.0
        self.trades = 0
        self.done = False
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        direction = np.clip(action[0], -1.0, 1.0)
        size = np.clip(action[1], 0.0, 1.0)
        sl_mult = np.clip(action[2], 0.5, 3.0)
        tp_mult = np.clip(action[3], 1.0, 5.0)

        # Current price (simulated from returns)
        price = 2000.0 * (1 + np.sum(self.returns[:self.idx+1]) * 0.01)

        # Execute trade logic
        reward = 0.0
        trade_pnl = 0.0
        executed = False

        if direction > 0.1 and self.position <= 0:
            # Open long
            self.position = size
            self.entry_price = price
            executed = True
        elif direction < -0.1 and self.position >= 0:
            # Open short
            self.position = -size
            self.entry_price = price
            executed = True
        elif abs(direction) < 0.1 and self.position != 0:
            # Close position
            if self.position > 0:
                trade_pnl = (price - self.entry_price) * self.position * 100
            else:
                trade_pnl = (self.entry_price - price) * abs(self.position) * 100
            self.total_pnl += trade_pnl
            self.equity += trade_pnl
            self.position = 0.0
            self.entry_price = 0.0
            executed = True
            self.trades += 1

        # Stop-loss / Take-profit (simplified)
        if self.position != 0 and self.entry_price != 0:
            if self.position > 0:
                sl_price = self.entry_price * (1 - sl_mult * 0.005)
                tp_price = self.entry_price * (1 + tp_mult * 0.005)
                if price <= sl_price or price >= tp_price:
                    trade_pnl = (price - self.entry_price) * self.position * 100
                    self.total_pnl += trade_pnl
                    self.equity += trade_pnl
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.trades += 1
            else:
                sl_price = self.entry_price * (1 + sl_mult * 0.005)
                tp_price = self.entry_price * (1 - tp_mult * 0.005)
                if price >= sl_price or price <= tp_price:
                    trade_pnl = (self.entry_price - price) * abs(self.position) * 100
                    self.total_pnl += trade_pnl
                    self.equity += trade_pnl
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.trades += 1

        reward = trade_pnl * 0.0001

        self.peak_equity = max(self.peak_equity, self.equity)

        self.idx += 1
        if self.idx >= self.T - 1:
            self.done = True

        info = {
            'equity': self.equity,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'trades': self.trades,
            'trade_executed': executed,
            'trade_pnl': trade_pnl,
            'price': price,
        }
        return self._get_obs(), reward, self.done, False, info

    def _get_obs(self):
        start = max(0, self.idx - self.window)
        window_features = self.features[start:self.idx]
        if len(window_features) < self.window:
            pad = np.zeros((self.window - len(window_features), self.n_features))
            window_features = np.vstack([pad, window_features])
        else:
            window_features = window_features[-self.window:]
        obs = window_features.flatten().astype(np.float32)
        return obs

    def render(self, mode='human'):
        pass