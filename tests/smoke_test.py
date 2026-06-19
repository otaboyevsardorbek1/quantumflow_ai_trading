#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Smoke Test
Quick environment verification
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging

from env.trading_env import QuantumTradingEnv
from agents.policy_network import TransformerPolicyNetwork
from features.engineering import AdvancedFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test trading environment"""
    logger.info("🧪 Testing Environment...")

    n_samples = 1000
    features = np.random.randn(n_samples, 256).astype(np.float32)
    returns = np.random.randn(n_samples).astype(np.float32) * 0.001
    timestamps = np.arange(n_samples)

    env = QuantumTradingEnv(features, returns, timestamps, window=128)

    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0], "Observation shape mismatch"
    assert env.position == 0.0, "Initial position should be 0"

    action = np.array([0.5, 0.5, 1.5, 2.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float), "Reward should be float"
    assert 'equity' in info, "Info should contain equity"

    logger.info("✅ Environment test passed!")
    return True

def test_agent():
    """Test agent forward pass"""
    logger.info("🧪 Testing Agent...")

    agent = TransformerPolicyNetwork(n_features=256, window_size=128, d_model=64, nhead=4, num_encoder_layers=2)

    x = torch.randn(2, 128, 256)
    action_mean, value, _ = agent(x)

    assert action_mean.shape == (2, 4), f"Action shape mismatch: {action_mean.shape}"
    assert value.shape == (2, 1), f"Value shape mismatch: {value.shape}"

    # Test action constraints
    assert (action_mean[:, 0] >= -1.0).all() and (action_mean[:, 0] <= 1.0).all(), "Direction out of range"
    assert (action_mean[:, 1] >= 0.0).all() and (action_mean[:, 1] <= 1.0).all(), "Size out of range"

    logger.info("✅ Agent test passed!")
    return True

def test_features():
    """Test feature engineering"""
    logger.info("🧪 Testing Feature Engineering...")

    import pandas as pd
    dates = pd.date_range('2020-01-01', periods=500, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 2000,
        'high': np.random.randn(500).cumsum() + 2005,
        'low': np.random.randn(500).cumsum() + 1995,
        'close': np.random.randn(500).cumsum() + 2000,
        'volume': np.random.randint(1000, 10000, 500),
    }, index=dates)
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1) + 1)
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1) - 1)

    engineer = AdvancedFeatureEngineer(config={})
    features = engineer.compute_all_features(df)

    assert features.shape[0] == len(df), "Feature length mismatch"
    assert np.isfinite(features).all(), "Features contain NaN/Inf"

    logger.info(f"✅ Feature test passed! Generated {features.shape[1]} features")
    return True

def test_integration():
    """Test end-to-end integration"""
    logger.info("🧪 Testing Integration...")

    n_samples = 1000
    features = np.random.randn(n_samples, 256).astype(np.float32)
    returns = np.random.randn(n_samples).astype(np.float32) * 0.001
    timestamps = np.arange(n_samples)

    env = QuantumTradingEnv(features, returns, timestamps, window=128)
    agent = TransformerPolicyNetwork(n_features=256, window_size=128, d_model=64, nhead=4, num_encoder_layers=2)

    obs, _ = env.reset()
    total_reward = 0

    for _ in range(50):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, _ = agent.get_action(obs_tensor, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        total_reward += reward
        if terminated or truncated:
            break

    assert total_reward != 0, "Total reward should not be zero"
    logger.info("✅ Integration test passed!")
    return True

def main():
    logger.info("=" * 60)
    logger.info("🔥 QUANTUMFLOW SMOKE TEST")
    logger.info("=" * 60)

    tests = [
        ("Environment", test_environment),
        ("Agent", test_agent),
        ("Features", test_features),
        ("Integration", test_integration),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"❌ {name} test failed: {e}")
            failed += 1

    logger.info("
" + "=" * 60)
    logger.info(f"📊 Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED - System is ready!")
    else:
        logger.warning("⚠️ Some tests failed - Check errors above")

    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
