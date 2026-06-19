"""
QuantumFlow AI Trading System v2.0 - Test Suite
Comprehensive unit and integration tests
"""
import unittest
import numpy as np
import pandas as pd
import torch
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.engineering import AdvancedFeatureEngineer, MarketRegimeDetector
from env.trading_env import QuantumTradingEnv
from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy
from risk.manager import AdvancedRiskSupervisor, AIRiskManager, SafeTradingAgent
from execution.engine import ExecutionEngine, Order, OrderType, OrderSide

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering module"""

    def setUp(self):
        self.config = {'feature_scaling': 'robust', 'use_pca': False}
        self.engineer = AdvancedFeatureEngineer(self.config)

        # Create sample data
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        self.df = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 2000,
            'high': np.random.randn(1000).cumsum() + 2005,
            'low': np.random.randn(1000).cumsum() + 1995,
            'close': np.random.randn(1000).cumsum() + 2000,
            'volume': np.random.randint(1000, 10000, 1000),
        }, index=dates)
        self.df['high'] = np.maximum(self.df['high'], self.df[['open', 'close']].max(axis=1) + 1)
        self.df['low'] = np.minimum(self.df['low'], self.df[['open', 'close']].min(axis=1) - 1)

    def test_feature_computation(self):
        """Test feature computation"""
        features = self.engineer.compute_all_features(self.df)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(features), len(self.df))

    def test_regime_detection(self):
        """Test regime detection"""
        detector = MarketRegimeDetector(n_regimes=4)
        regimes = detector.detect_regime(self.df)

        self.assertIn('regime_volatility', regimes)
        self.assertIn('regime_trending', regimes)
        self.assertEqual(len(regimes['regime_trending']), len(self.df))

class TestTradingEnvironment(unittest.TestCase):
    """Test trading environment"""

    def setUp(self):
        n_samples = 1000
        n_features = 256

        self.features = np.random.randn(n_samples, n_features).astype(np.float32)
        self.returns = np.random.randn(n_samples).astype(np.float32) * 0.001
        self.timestamps = np.arange(n_samples)

        self.env = QuantumTradingEnv(
            features=self.features,
            returns=self.returns,
            timestamps=self.timestamps,
            window=128,
            symbol='XAUUSD'
        )

    def test_environment_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertGreater(self.env.observation_space.shape[0], 0)

    def test_reset(self):
        """Test environment reset"""
        obs, info = self.env.reset()
        self.assertEqual(obs.shape[0], self.env.observation_space.shape[0])
        self.assertEqual(self.env.position, 0.0)

    def test_step(self):
        """Test environment step"""
        obs, _ = self.env.reset()
        action = np.array([0.5, 0.5, 1.5, 2.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIn('equity', info)

    def test_long_short_flat(self):
        """Test all position types"""
        obs, _ = self.env.reset()

        # Long
        obs, _, _, _, info = self.env.step(np.array([1.0, 0.5, 1.5, 2.0]))
        self.assertGreater(self.env.position, 0)

        # Flat
        obs, _, _, _, info = self.env.step(np.array([0.0, 0.0, 1.5, 2.0]))
        self.assertEqual(self.env.position, 0.0)

        # Short
        obs, _, _, _, info = self.env.step(np.array([-1.0, 0.5, 1.5, 2.0]))
        self.assertLess(self.env.position, 0)

class TestPolicyNetwork(unittest.TestCase):
    """Test policy network"""

    def setUp(self):
        self.n_features = 256
        self.window_size = 128
        self.batch_size = 4

        self.policy = TransformerPolicyNetwork(
            n_features=self.n_features,
            window_size=self.window_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
        )

    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, self.window_size, self.n_features)
        action_mean, value, feature_importance = self.policy(x)

        self.assertEqual(action_mean.shape, (self.batch_size, 4))
        self.assertEqual(value.shape, (self.batch_size, 1))

        # Check action constraints
        self.assertTrue((action_mean[:, 0] >= -1.0).all() and (action_mean[:, 0] <= 1.0).all())
        self.assertTrue((action_mean[:, 1] >= 0.0).all() and (action_mean[:, 1] <= 1.0).all())

    def test_get_action(self):
        """Test action sampling"""
        x = torch.randn(1, self.window_size, self.n_features)
        action, value = self.policy.get_action(x, deterministic=True)

        self.assertEqual(action.shape, (1, 4))
        self.assertIsInstance(action, np.ndarray)

class TestEnsemblePolicy(unittest.TestCase):
    """Test ensemble policy"""

    def setUp(self):
        self.ensemble = EnsemblePolicy(
            n_features=256,
            window_size=128,
            ensemble_size=3,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
        )

    def test_ensemble_forward(self):
        """Test ensemble forward pass"""
        x = torch.randn(2, 128, 256)
        action, value, info = self.ensemble(x)

        self.assertEqual(action.shape, (2, 4))
        self.assertEqual(value.shape, (2, 1))
        self.assertIn('individual_actions', info)
        self.assertIn('weights', info)
        self.assertEqual(len(info['agent_names']), 3)

class TestRiskManager(unittest.TestCase):
    """Test risk management"""

    def setUp(self):
        self.risk_config = {
            'max_daily_loss': 0.03,
            'max_drawdown': 0.15,
            'max_position': 0.20,
        }
        self.supervisor = AdvancedRiskSupervisor(self.risk_config)

    def test_approve_normal_trade(self):
        """Test normal trade approval"""
        action = np.array([1.0, 0.1, 1.5, 2.0])
        market_data = {
            'volatility': 1.0,
            'spread': 0.0002,
            'is_market_open': True,
            'is_high_impact_event': False,
            'volume': 1000,
            'avg_volume': 1000,
            'price': 2000.0,
            'available_margin': 100000.0,
            'margin_rate': 0.01,
        }

        approved, reason, adjustments = self.supervisor.evaluate_trade(action, market_data)
        self.assertTrue(approved)
        self.assertEqual(reason, "APPROVED")

    def test_reject_high_volatility(self):
        """Test high volatility rejection"""
        action = np.array([1.0, 0.5, 1.5, 2.0])
        market_data = {
            'volatility': 5.0,
            'spread': 0.0002,
            'is_market_open': True,
            'is_high_impact_event': False,
            'volume': 1000,
            'avg_volume': 1000,
            'price': 2000.0,
            'available_margin': 100000.0,
            'margin_rate': 0.01,
        }

        approved, reason, _ = self.supervisor.evaluate_trade(action, market_data)
        self.assertFalse(approved)
        self.assertIn("HIGH_VOLATILITY", reason)

    def test_reject_wide_spread(self):
        """Test wide spread rejection"""
        action = np.array([1.0, 0.1, 1.5, 2.0])
        market_data = {
            'volatility': 1.0,
            'spread': 0.001,
            'is_market_open': True,
            'is_high_impact_event': False,
            'volume': 1000,
            'avg_volume': 1000,
            'price': 2000.0,
            'available_margin': 100000.0,
            'margin_rate': 0.01,
        }

        approved, reason, _ = self.supervisor.evaluate_trade(action, market_data)
        self.assertFalse(approved)
        self.assertIn("SPREAD", reason)

    def test_circuit_breaker(self):
        """Test circuit breaker"""
        self.supervisor.risk_state.daily_pnl = -0.05

        action = np.array([1.0, 0.1, 1.5, 2.0])
        market_data = {'volatility': 1.0, 'spread': 0.0002, 'is_market_open': True,
                        'is_high_impact_event': False, 'volume': 1000, 'avg_volume': 1000,
                        'price': 2000.0, 'available_margin': 100000.0, 'margin_rate': 0.01}

        approved, reason, _ = self.supervisor.evaluate_trade(action, market_data)
        self.assertFalse(approved)
        self.assertIn("CIRCUIT_BREAKER", reason)

class TestExecutionEngine(unittest.TestCase):
    """Test execution engine"""

    def setUp(self):
        self.config = {'slippage_tolerance': 0.0001, 'commission': 0.0001}
        self.engine = ExecutionEngine(self.config)

    def test_market_order(self):
        """Test market order execution"""
        order = Order(
            symbol='XAUUSD',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=0.1,
        )

        market_data = {
            'price': 2000.0,
            'spread': 0.0002,
            'volatility': 0.01,
            'volume': 1000,
            'avg_volume': 1000,
        }

        result = self.engine.execute_order(order, market_data)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.executed_price)
        self.assertGreater(result.commission, 0)

    def test_limit_order(self):
        """Test limit order execution"""
        order = Order(
            symbol='XAUUSD',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=0.1,
            price=1990.0,
        )

        market_data = {
            'price': 2000.0,
            'spread': 0.0002,
        }

        result = self.engine.execute_order(order, market_data)
        self.assertTrue(result.success)

class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline"""
        # 1. Create environment
        n_samples = 1000
        features = np.random.randn(n_samples, 256).astype(np.float32)
        returns = np.random.randn(n_samples).astype(np.float32) * 0.001
        timestamps = np.arange(n_samples)

        env = QuantumTradingEnv(features, returns, timestamps, window=128)

        # 2. Create agent
        agent = TransformerPolicyNetwork(n_features=256, window_size=128, d_model=64, nhead=4, num_encoder_layers=2)

        # 3. Create risk supervisor
        risk_supervisor = AdvancedRiskSupervisor()

        # 4. Run episode
        obs, _ = env.reset()
        total_reward = 0

        for _ in range(50):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _ = agent.get_action(obs_tensor, deterministic=True)

            # Risk check
            market_data = {'volatility': 1.0, 'spread': 0.0002, 'is_market_open': True,
                          'is_high_impact_event': False, 'volume': 1000, 'avg_volume': 1000,
                          'price': 2000.0, 'available_margin': 100000.0, 'margin_rate': 0.01}
            approved, _, _ = risk_supervisor.evaluate_trade(action[0], market_data)

            if not approved:
                action = np.zeros_like(action[0])

            obs, reward, terminated, truncated, info = env.step(action[0])
            total_reward += reward

            if terminated or truncated:
                break

        self.assertNotEqual(total_reward, 0)
        self.assertIn('equity', info)

if __name__ == '__main__':
    unittest.main()
