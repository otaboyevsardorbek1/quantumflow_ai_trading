"""
QuantumFlow AI Trading System v2.0 - Advanced Risk Management
Dual-layer: AI-Powered Risk Manager + Deterministic Safety Layer
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskState:
    """Risk state tracking"""
    equity: float = 1.0
    peak_equity: float = 1.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    current_drawdown: float = 0.0
    volatility_regime: str = "normal"
    last_trade_time: Optional[datetime] = None
    halt_until: Optional[datetime] = None

class AIRiskManager(nn.Module):
    """
    AI-powered risk manager using neural network
    Learns optimal position sizing and risk limits based on market conditions
    """

    def __init__(self, state_dim: int = 32, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def get_risk_adjustment(self, market_state: Dict) -> Dict:
        """Get risk adjustments from AI model"""
        state = self._prepare_state(market_state)
        with torch.no_grad():
            output = self.forward(state)

        return {
            'position_scale': float(output[0, 0]),
            'risk_multiplier': float(output[0, 1]) * 1.5 + 0.5,
            'confidence': float(output[0, 2]),
        }

    def _prepare_state(self, market_state: Dict) -> torch.Tensor:
        state = np.array([
            market_state.get('equity', 1.0),
            market_state.get('drawdown', 0.0),
            market_state.get('volatility', 0.01),
            market_state.get('trend_strength', 0.5),
            market_state.get('rsi', 50.0) / 100.0,
            market_state.get('atr_ratio', 1.0),
            market_state.get('volume_ratio', 1.0),
            market_state.get('correlation_dxy', 0.0),
            market_state.get('vix', 20.0) / 100.0,
            market_state.get('regime', 0) / 3.0,
            market_state.get('time_to_event', 1.0),
            market_state.get('spread', 0.0002) / 0.001,
        ])

        if len(state) < 32:
            state = np.pad(state, (0, 32 - len(state)), mode='constant')

        return torch.FloatTensor(state).unsqueeze(0)

class AdvancedRiskSupervisor:
    """
    Advanced deterministic safety layer
    15+ safety checks with dynamic thresholds
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self.max_daily_loss = self.config.get('max_daily_loss', 0.03)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.max_position = self.config.get('max_position', 0.20)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 20)
        self.min_trade_interval = self.config.get('min_trade_interval', 300)
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 5)
        self.max_spread = self.config.get('max_spread', 0.0005)
        self.volatility_threshold = self.config.get('volatility_threshold', 3.0)

        self.dynamic_mode = self.config.get('dynamic_risk', True)
        self.base_risk = self.config.get('base_risk', 0.02)

        self.risk_state = RiskState()
        self.trade_history = []
        self.rejection_reasons = {}

        self.total_approved = 0
        self.total_rejected = 0

        logger.info("🛡️ Advanced Risk Supervisor initialized")

    def evaluate_trade(
        self,
        action: np.ndarray,
        market_data: Dict,
        ai_risk_adjustment: Optional[Dict] = None
    ) -> Tuple[bool, str, Dict]:

        adjustments = {}

        if ai_risk_adjustment and self.dynamic_mode:
            adjustments['position_scale'] = ai_risk_adjustment.get('position_scale', 1.0)
            adjustments['risk_multiplier'] = ai_risk_adjustment.get('risk_multiplier', 1.0)
        else:
            adjustments['position_scale'] = 1.0
            adjustments['risk_multiplier'] = 1.0

        # 1. Circuit Breaker
        if self.risk_state.daily_pnl < -self.max_daily_loss * adjustments['risk_multiplier']:
            self.risk_state.halt_until = datetime.now() + timedelta(hours=24)
            return False, "CIRCUIT_BREAKER_DAILY_LOSS", adjustments

        # 2. Trading Halt
        if self.risk_state.halt_until and datetime.now() < self.risk_state.halt_until:
            remaining = (self.risk_state.halt_until - datetime.now()).total_seconds() / 3600
            return False, f"TRADING_HALTED_{remaining:.1f}h", adjustments

        # 3. Maximum Drawdown
        current_dd = (self.risk_state.peak_equity - self.risk_state.equity) / self.risk_state.peak_equity
        if current_dd > self.max_drawdown * adjustments['risk_multiplier']:
            return False, f"MAX_DRAWDOWN_{current_dd:.2%}", adjustments

        # 4. Position Size Limit
        position_size = abs(action[1]) if len(action) > 1 else 0.0
        max_pos = self.max_position * adjustments['position_scale']
        if position_size > max_pos:
            adjustments['position_scale'] = max_pos / position_size if position_size > 0 else 0
            return False, f"POSITION_TOO_LARGE_{position_size:.2%}", adjustments

        # 5. Consecutive Losses
        if self.risk_state.consecutive_losses >= self.max_consecutive_losses:
            return False, f"MAX_CONSECUTIVE_LOSSES_{self.risk_state.consecutive_losses}", adjustments

        # 6. Volatility Filter
        volatility = market_data.get('volatility', 0.0)
        if volatility > self.volatility_threshold * adjustments['risk_multiplier']:
            if position_size > 0.1:
                return False, f"HIGH_VOLATILITY_{volatility:.2f}", adjustments

        # 7. Spread Filter
        spread = market_data.get('spread', 0.0)
        if spread > self.max_spread:
            return False, f"SPREAD_TOO_WIDE_{spread:.5f}", adjustments

        # 8. Correlation Guard
        if len(action) > 0 and action[0] > 0:
            dxy_momentum = market_data.get('dxy_momentum', 0.0)
            if dxy_momentum > 0.01:
                adjustments['position_scale'] *= 0.5

        # 9. Event Risk
        if market_data.get('is_high_impact_event', False):
            adjustments['position_scale'] *= 0.3
            if position_size > self.max_position * 0.3:
                return False, "HIGH_IMPACT_EVENT", adjustments

        # 10. Maximum Trades Per Day
        if self.risk_state.daily_trades >= self.max_trades_per_day:
            return False, f"MAX_DAILY_TRADES_{self.risk_state.daily_trades}", adjustments

        # 11. Cooldown Period
        if self.risk_state.last_trade_time:
            elapsed = (datetime.now() - self.risk_state.last_trade_time).total_seconds()
            if elapsed < self.min_trade_interval:
                return False, f"COOLDOWN_{self.min_trade_interval - elapsed:.0f}s", adjustments

        # 12. Market Hours
        if not market_data.get('is_market_open', True):
            return False, "MARKET_CLOSED", adjustments

        # 13. Weekend Check
        if datetime.now().weekday() >= 5:
            return False, "WEEKEND", adjustments

        # 14. Liquidity Check
        if market_data.get('volume', 0) < market_data.get('avg_volume', 1) * 0.5:
            return False, "LOW_LIQUIDITY", adjustments

        # 15. Margin Check
        required_margin = position_size * market_data.get('price', 0) * market_data.get('margin_rate', 0.01)
        available_margin = market_data.get('available_margin', float('inf'))
        if required_margin > available_margin * 0.8:
            return False, "INSUFFICIENT_MARGIN", adjustments

        self.total_approved += 1
        return True, "APPROVED", adjustments

    def update_after_trade(self, pnl: float, trade_info: Dict):
        self.risk_state.equity += pnl
        self.risk_state.daily_pnl += pnl
        self.risk_state.peak_equity = max(self.risk_state.peak_equity, self.risk_state.equity)
        self.risk_state.daily_trades += 1
        self.risk_state.last_trade_time = datetime.now()

        if pnl > 0:
            self.risk_state.consecutive_wins += 1
            self.risk_state.consecutive_losses = 0
        else:
            self.risk_state.consecutive_losses += 1
            self.risk_state.consecutive_wins = 0

        self.trade_history.append({
            'time': datetime.now(),
            'pnl': pnl,
            'equity': self.risk_state.equity,
            'drawdown': (self.risk_state.peak_equity - self.risk_state.equity) / self.risk_state.peak_equity,
        })

    def reset_daily(self):
        logger.info(f"📊 Daily reset - PnL: {self.risk_state.daily_pnl:.4f}, Trades: {self.risk_state.daily_trades}")
        self.risk_state.daily_pnl = 0.0
        self.risk_state.daily_trades = 0
        if self.risk_state.halt_until and datetime.now() >= self.risk_state.halt_until:
            self.risk_state.halt_until = None

    def emergency_shutdown(self):
        self.risk_state.halt_until = datetime.now() + timedelta(days=365)
        logger.critical("🚨 EMERGENCY SHUTDOWN ACTIVATED")
        return "EMERGENCY_SHUTDOWN"

    def get_statistics(self) -> Dict:
        total = self.total_approved + self.total_rejected
        return {
            'total_checks': total,
            'approved': self.total_approved,
            'rejected': self.total_rejected,
            'approval_rate': self.total_approved / total if total > 0 else 0,
            'current_equity': self.risk_state.equity,
            'peak_equity': self.risk_state.peak_equity,
            'current_drawdown': (self.risk_state.peak_equity - self.risk_state.equity) / self.risk_state.peak_equity,
            'daily_pnl': self.risk_state.daily_pnl,
            'daily_trades': self.risk_state.daily_trades,
            'consecutive_losses': self.risk_state.consecutive_losses,
        }

class SafeTradingAgent:
    """AI agent + Risk Supervisor wrapper"""

    def __init__(
        self,
        ai_agent,
        risk_supervisor: AdvancedRiskSupervisor,
        ai_risk_manager: Optional[AIRiskManager] = None,
    ):
        self.ai_agent = ai_agent
        self.risk_supervisor = risk_supervisor
        self.ai_risk_manager = ai_risk_manager

        logger.info("✅ Safe Trading Agent initialized with dual-layer risk management")

    def act(self, obs: np.ndarray, market_data: Dict) -> Tuple[np.ndarray, Dict]:
        if hasattr(self.ai_agent, 'get_action'):
            ai_action, _ = self.ai_agent.get_action(obs)
        else:
            ai_action = self.ai_agent.act(obs)

        ai_adjustment = None
        if self.ai_risk_manager:
            ai_adjustment = self.ai_risk_manager.get_risk_adjustment(market_data)

        approved, reason, adjustments = self.risk_supervisor.evaluate_trade(
            ai_action, market_data, ai_adjustment
        )

        if approved:
            final_action = ai_action * adjustments.get('position_scale', 1.0)
            logger.info(f"✅ Trade approved: {ai_action} -> {final_action} (reason: {reason})")
        else:
            final_action = np.zeros_like(ai_action)
            logger.warning(f"🚫 Trade rejected: {reason}")

        return final_action, {
            'approved': approved,
            'reason': reason,
            'ai_action': ai_action,
            'final_action': final_action,
            'adjustments': adjustments,
        }
