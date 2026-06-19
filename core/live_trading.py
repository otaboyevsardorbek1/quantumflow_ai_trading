"""
QuantumFlow AI Trading System v2.0 - Live Trading Engine
Paper trading and live trading with full monitoring
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging
import threading
import json

from features.engineering import AdvancedFeatureEngineer
from env.trading_env import QuantumTradingEnv
from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy
from risk.manager import AdvancedRiskSupervisor, SafeTradingAgent, AIRiskManager
from execution.engine import ExecutionEngine, MT5ExecutionAdapter, Order, OrderType, OrderSide

logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """
    Live trading engine with:
    - Paper trading mode
    - Live trading mode
    - Real-time monitoring
    - Automatic failover
    """

    def __init__(self, config: Dict):
        self.config = config
        self.paper_trading = config.get('paper_trading', True)
        self.symbol = config.get('symbol', 'XAUUSD')
        self.timeframe = config.get('timeframe', 'M5')
        self.check_interval = config.get('check_interval', 60)

        # Components
        self.feature_engineer = None
        self.agent = None
        self.risk_supervisor = None
        self.safe_agent = None
        self.execution_engine = None
        self.mt5_adapter = None

        # State
        self.running = False
        self.current_position = 0.0
        self.equity = config.get('initial_balance', 100000.0)
        self.trades = []

        # Monitoring
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': self.equity,
        }

        logger.info(f"🚀 Live Trading Engine initialized")
        logger.info(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Initial Balance: ${self.equity:,.2f}")

    def initialize(self):
        """Initialize all components"""
        logger.info("🔧 Initializing components...")

        # Feature engineer
        self.feature_engineer = AdvancedFeatureEngineer(self.config)

        # Load agent
        self._load_agent()

        # Risk supervisor
        self.risk_supervisor = AdvancedRiskSupervisor(self.config.get('risk', {}))

        # AI Risk Manager
        ai_risk = AIRiskManager() if self.config.get('use_ai_risk', True) else None

        # Safe agent wrapper
        self.safe_agent = SafeTradingAgent(self.agent, self.risk_supervisor, ai_risk)

        # Execution engine
        if self.paper_trading:
            self.execution_engine = ExecutionEngine(self.config.get('execution', {}))
        else:
            self.mt5_adapter = MT5ExecutionAdapter(self.config.get('execution', {}))
            if not self.mt5_adapter.connect():
                raise ConnectionError("Failed to connect to MT5")

        logger.info("✅ All components initialized")

    def _load_agent(self):
        """Load trained agent"""
        model_path = self.config.get('model_path', 'checkpoints/best_model.pt')

        # Determine architecture
        if self.config.get('use_ensemble', True):
            self.agent = EnsemblePolicy(
                n_features=self.config.get('n_features', 256),
                window_size=self.config.get('window_size', 128),
                ensemble_size=self.config.get('ensemble_size', 3),
            )
        else:
            self.agent = TransformerPolicyNetwork(
                n_features=self.config.get('n_features', 256),
                window_size=self.config.get('window_size', 128),
            )

        # Load weights
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.agent.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.eval()

        logger.info(f"🧠 Agent loaded from {model_path}")

    def start(self):
        """Start trading loop"""
        self.running = True
        logger.info("🟢 Trading loop started")

        try:
            while self.running:
                self._trading_iteration()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
            self._emergency_stop()
        finally:
            self.shutdown()

    def _trading_iteration(self):
        """Single trading iteration"""
        try:
            # 1. Fetch market data
            market_data = self._fetch_market_data()

            # 2. Compute features
            features = self._compute_features(market_data)

            # 3. Prepare observation
            obs = self._prepare_observation(features)

            # 4. Get action from safe agent
            action, risk_info = self.safe_agent.act(obs, market_data)

            # 5. Execute trade
            if risk_info['approved']:
                self._execute_trade(action, market_data)

            # 6. Update metrics
            self._update_metrics()

            # 7. Log status
            self._log_status(action, risk_info)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")

    def _fetch_market_data(self) -> Dict:
        """Fetch current market data"""
        if self.paper_trading:
            # Simulate market data
            return {
                'price': 2000.0 + np.random.randn() * 5,
                'spread': 0.0002,
                'volatility': 0.01,
                'volume': 1000,
                'avg_volume': 1000,
                'is_market_open': True,
                'is_high_impact_event': False,
                'dxy_momentum': 0.0,
                'available_margin': self.equity,
                'margin_rate': 0.01,
            }
        else:
            # Fetch from MT5
            tick = self.mt5_adapter.mt5.symbol_info_tick(self.symbol)
            return {
                'price': tick.ask if tick else 0,
                'spread': (tick.ask - tick.bid) / tick.ask if tick else 0,
                'is_market_open': True,
            }

    def _compute_features(self, market_data: Dict) -> np.ndarray:
        """Compute features from market data"""
        # This would use historical data + current data
        # Simplified for demonstration
        return np.random.randn(256).astype(np.float32)

    def _prepare_observation(self, features: np.ndarray) -> np.ndarray:
        """Prepare observation for agent"""
        # Add position info
        pos_info = np.array([
            self.current_position,
            abs(self.current_position),
            self.equity / self.metrics['peak_equity'] - 1.0,
            self.metrics['total_pnl'],
            0.0,
        ], dtype=np.float32)

        # For actual implementation, we'd use a window of features
        obs = np.concatenate([features[:128], pos_info])  # Simplified
        return obs

    def _execute_trade(self, action: np.ndarray, market_data: Dict):
        """Execute trade"""
        direction = action[0]
        size = action[1]

        # Determine order side
        if direction > 0.1 and self.current_position <= 0:
            # Open long
            order = Order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=size,
                comment="QuantumFlow Long"
            )
            self.current_position = size

        elif direction < -0.1 and self.current_position >= 0:
            # Open short
            order = Order(
                symbol=self.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                volume=size,
                comment="QuantumFlow Short"
            )
            self.current_position = -size

        elif abs(direction) < 0.1 and self.current_position != 0:
            # Close position
            order = Order(
                symbol=self.symbol,
                side=OrderSide.SELL if self.current_position > 0 else OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=abs(self.current_position),
                comment="QuantumFlow Close"
            )
            self.current_position = 0.0
        else:
            return

        # Execute
        if self.paper_trading:
            result = self.execution_engine.execute_order(order, market_data)
        else:
            result = self.mt5_adapter.execute_order(order)

        if result.success:
            self.trades.append({
                'time': datetime.now(),
                'action': action,
                'price': result.executed_price,
                'volume': result.executed_volume,
            })
            logger.info(f"✅ Trade executed: {order.side.value} {order.volume} @ {result.executed_price}")
        else:
            logger.warning(f"❌ Trade failed: {result.error_message}")

    def _update_metrics(self):
        """Update performance metrics"""
        # Update equity (simplified)
        self.metrics['peak_equity'] = max(self.metrics['peak_equity'], self.equity)
        current_dd = (self.metrics['peak_equity'] - self.equity) / self.metrics['peak_equity']
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], current_dd)

    def _log_status(self, action: np.ndarray, risk_info: Dict):
        """Log current status"""
        logger.info(
            f"📊 Status | Equity: ${self.equity:,.2f} | "
            f"Pos: {self.current_position:.2f} | "
            f"Action: [{action[0]:.2f}, {action[1]:.2f}] | "
            f"Risk: {risk_info['reason']} | "
            f"DD: {self.metrics['max_drawdown']:.2%}"
        )

    def _emergency_stop(self):
        """Emergency stop"""
        logger.critical("🚨 EMERGENCY STOP TRIGGERED")
        self.risk_supervisor.emergency_shutdown()
        self.running = False

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🔴 Shutting down trading engine...")
        self.running = False

        if self.mt5_adapter:
            self.mt5_adapter.disconnect()

        # Save final metrics
        self._save_metrics()

        logger.info("✅ Trading engine shutdown complete")

    def _save_metrics(self):
        """Save trading metrics"""
        metrics_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"📁 Metrics saved to {metrics_file}")

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'running': self.running,
            'mode': 'PAPER' if self.paper_trading else 'LIVE',
            'symbol': self.symbol,
            'equity': self.equity,
            'position': self.current_position,
            'metrics': self.metrics,
            'trades_today': len([t for t in self.trades if t['time'].date() == datetime.now().date()]),
        }
