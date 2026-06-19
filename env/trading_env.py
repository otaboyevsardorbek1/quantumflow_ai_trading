"""
QuantumFlow AI Trading System v2.0 - Advanced Trading Environment
Continuous action space with position sizing, trailing stops, and regime-aware rewards
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class QuantumTradingEnv(gym.Env):
    """
    Advanced trading environment with:
    - Continuous action space: [direction, position_size, stop_loss_pct, take_profit_pct]
    - Regime-aware reward shaping
    - Realistic execution costs
    - Trailing stop support
    - Multi-asset support (single asset per env)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        timestamps: np.ndarray,
        config: Optional[Dict] = None,
        window: int = 128,
        symbol: str = "XAUUSD",
        regime_labels: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.config = config or {}
        self.symbol = symbol
        self.window = window

        # Data
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.timestamps = timestamps
        self.T = len(self.r)
        self.regime_labels = regime_labels

        # Cost parameters
        self.commission = self.config.get('commission', 0.0001)
        self.spread = self.config.get('spread', 0.0002)
        self.slippage = self.config.get('slippage', 0.00005)

        # Risk parameters
        self.max_position = self.config.get('max_position', 1.0)
        self.min_position = self.config.get('min_position', 0.0)

        # Action space: [direction (-1 to 1), position_size (0 to 1), 
        #               stop_loss_atr_mult (0.5 to 3.0), take_profit_atr_mult (1.0 to 5.0)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.5, 1.0]),
            high=np.array([1.0, 1.0, 3.0, 5.0]),
            dtype=np.float32
        )

        # Observation space
        obs_dim = self.window * self.X.shape[1] + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

        logger.info(f"🎮 QuantumTradingEnv initialized: {symbol}, Features: {self.X.shape[1]}, Window: {window}")

    def _reset_state(self):
        self.t = self.window
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.cumulative_pnl = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        self.trades = []
        self.current_trade = None
        self.steps_in_trade = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        w = self.X[self.t - self.window : self.t]
        pos_info = np.array([
            self.position,
            self.position_size,
            self.equity / self.peak_equity - 1.0,
            self.cumulative_pnl,
            self.steps_in_trade / 100.0 if self.steps_in_trade > 0 else 0.0,
        ], dtype=np.float32)
        obs = np.concatenate([w.reshape(-1), pos_info])
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        direction = np.clip(action[0], -1.0, 1.0)
        target_size = np.clip(action[1], 0.0, 1.0)
        sl_atr = np.clip(action[2], 0.5, 3.0)
        tp_atr = np.clip(action[3], 1.0, 5.0)

        current_return = self.r[self.t]
        atr = np.std(self.r[max(0, self.t-20):self.t]) if self.t > 20 else 0.001

        # Stop loss / take profit setup
        if self.position != 0 and self.stop_loss is None:
            self.stop_loss = -sl_atr * atr
            self.take_profit = tp_atr * atr
            self.trailing_stop = -sl_atr * atr

        new_position = direction * target_size
        delta_position = new_position - self.position
        trade_cost = abs(delta_position) * (self.commission + self.spread + self.slippage)

        # PnL from holding position
        pnl = self.position * current_return - trade_cost

        # Trailing stop check
        if self.position > 0:
            self.highest_price = max(self.highest_price, current_return)
            if self.trailing_stop is not None:
                trailing_threshold = self.highest_price + self.trailing_stop
                if current_return < trailing_threshold:
                    pnl += self.position * (current_return - self.entry_price) if self.entry_price != 0 else 0
                    new_position = 0.0
                    delta_position = -self.position
        elif self.position < 0:
            self.lowest_price = min(self.lowest_price, current_return)
            if self.trailing_stop is not None:
                trailing_threshold = self.lowest_price - self.trailing_stop
                if current_return > trailing_threshold:
                    new_position = 0.0
                    delta_position = -self.position

        # Update state
        self.equity *= (1.0 + pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.cumulative_pnl += pnl

        # Trade tracking
        if abs(delta_position) > 0.001:
            if self.current_trade is not None:
                self.current_trade['exit_time'] = self.timestamps[self.t]
                self.current_trade['exit_price'] = self.r[self.t]
                self.current_trade['pnl'] = self.cumulative_pnl - self.current_trade.get('entry_pnl', 0)
                self.current_trade['duration'] = self.steps_in_trade
                self.trades.append(self.current_trade)
                if self.current_trade['pnl'] > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0

            if abs(new_position) > 0.001:
                self.current_trade = {
                    'entry_time': self.timestamps[self.t],
                    'entry_price': self.r[self.t],
                    'entry_pnl': self.cumulative_pnl,
                    'position': new_position,
                    'direction': 'long' if new_position > 0 else 'short',
                }
                self.entry_price = self.r[self.t]
                self.steps_in_trade = 0
                self.highest_price = self.r[self.t]
                self.lowest_price = self.r[self.t]
                self.stop_loss = None
                self.take_profit = None
            else:
                self.current_trade = None

        self.position = new_position
        self.position_size = abs(new_position)
        if abs(self.position) > 0.001:
            self.steps_in_trade += 1

        reward = self._compute_reward(pnl, current_return, atr)

        self.t += 1
        terminated = self.t >= self.T - 1
        truncated = False

        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if current_drawdown > self.config.get('max_drawdown', 0.15):
            terminated = True
            reward -= 1.0

        info = {
            'equity': float(self.equity),
            'position': float(self.position),
            'pnl': float(pnl),
            'drawdown': float(current_drawdown),
            'trade_cost': float(trade_cost),
            'cumulative_pnl': float(self.cumulative_pnl),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
        }

        if terminated or truncated:
            info['final_equity'] = float(self.equity)
            info['total_trades'] = len(self.trades)
            info['win_rate'] = sum(1 for t in self.trades if t['pnl'] > 0) / max(len(self.trades), 1)

        return self._get_obs(), float(reward), terminated, truncated, info

    def _compute_reward(self, pnl: float, current_return: float, atr: float) -> float:
        reward = pnl * 100
        if pnl < 0:
            reward *= 1.5
        if abs(self.position) > 0 and self.steps_in_trade > 0:
            stability_bonus = 0.001 * min(self.steps_in_trade, 100) / 100.0
            reward += stability_bonus
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if current_drawdown > 0.05:
            reward -= current_drawdown * 10
        if self.regime_labels is not None and self.t < len(self.regime_labels):
            regime = self.regime_labels[self.t]
            if regime == 2 and abs(self.position) > 0.5:
                reward -= 0.1
            elif regime == 0 and abs(self.position) < 0.3:
                reward -= 0.05
        if self.consecutive_losses >= 3:
            reward -= 0.1 * self.consecutive_losses
        return reward

    def render(self, mode="human"):
        if mode == "human":
            dd = (self.peak_equity - self.equity) / self.peak_equity
            print(f"Step: {self.t} | Equity: {self.equity:.4f} | Pos: {self.position:.2f} | "
                  f"PnL: {self.cumulative_pnl:.4f} | DD: {dd:.2%}")

    def get_trading_stats(self) -> Dict:
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'avg_duration': 0.0, 'profit_factor': 0.0}
        wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] <= 0]
        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades),
            'avg_pnl': np.mean([t['pnl'] for t in self.trades]),
            'avg_duration': np.mean([t.get('duration', 0) for t in self.trades]),
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'total_return': self.equity - 1.0,
            'max_drawdown': (self.peak_equity - min(self.equity, self.peak_equity * 0.85)) / self.peak_equity,
        }
