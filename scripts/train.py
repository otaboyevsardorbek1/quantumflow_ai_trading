#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Main Training Script
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_default_config
from features.engineering import AdvancedFeatureEngineer
from env.trading_env import QuantumTradingEnv
from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy
from agents.ppo_trainer import PPOTrainer
from evaluation.backtest import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train QuantumFlow AI Trading Agent')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='M5', help='Base timeframe')
    parser.add_argument('--steps', type=int, default=2000000, help='Training steps')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda/mps/cpu')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("🚀 QUANTUMFLOW AI TRADING SYSTEM v2.0")
    logger.info("=" * 80)

    # Load config
    config = get_default_config()

    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    logger.info(f"🖥️ Device: {device}")


    logger.info("📊 Loading data...")
    # This would load actual data
    # For now, create dummy data
    n_samples = 20000
    n_features = 256   # <--- BU QATORNI QO‘SHING
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    returns = np.random.randn(n_samples).astype(np.float32) * 0.001
    timestamps = np.arange(n_samples)
    n_features = features.shape[1]   # bu qolishi mumkin (agar haqiqiy ma'lumotlarda o'zgarib qolsa)
    logger.info(f"✅ Data loaded: {features.shape}")

    # Create environment
    logger.info("🎮 Creating environment...")
    
    env = QuantumTradingEnv(
    features=features,
    returns=returns,
    timestamps=timestamps,
    config=asdict(config['risk']),   # RiskConfig ni dict ga o‘tkazamiz
    window=128,
    symbol=args.symbol,)
    # Create agent
    logger.info("🤖 Creating agent...")
    n_features = env.features.shape[1]   # <-- haqiqiy feature sonini olamiz
    if args.ensemble:
        agent = TransformerPolicyNetwork(
        n_features=n_features,
        window_size=64,          # 128 o‘rniga
        d_model=128,             # 256 o‘rniga
        nhead=4,                 # 8 o‘rniga
        num_encoder_layers=2,    # 4 o‘rniga
        dim_feedforward=512,     # 1024 o‘rniga
        dropout=0.1,
        ensemble_size=3,         # 5 o‘rniga
        )
    else:
        agent = TransformerPolicyNetwork(
        n_features=n_features,
        window_size=64,          # 128 o‘rniga
        d_model=128,             # 256 o‘rniga
        nhead=4,                 # 8 o‘rniga
        num_encoder_layers=2,    # 4 o‘rniga
        dim_feedforward=512,     # 1024 o‘rniga
    )
    logger.info("   Using Transformer Policy")

    # Resume from checkpoint
    if args.resume:
        logger.info(f"📂 Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint['policy_state_dict'])

    # Create trainer
    logger.info("🏋️ Creating trainer...")
    trainer = PPOTrainer(
    policy=agent,
    env=env,
    config=asdict(config['rl']),   # RLConfig ni dict ga o‘tkazamiz
    device=device,)

    if args.eval_only:
        logger.info("📊 Running evaluation...")
        metrics = trainer.evaluate(n_episodes=10)
        logger.info(f"Evaluation results: {metrics}")
        return

    # Train
    logger.info("🎯 Starting training...")
    trainer.train(
        total_timesteps=args.steps,
        eval_freq=50000,
        save_freq=50000,
    )

    # Final evaluation
    logger.info("📊 Final evaluation...")
    final_metrics = trainer.evaluate(n_episodes=5)

    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final metrics: {final_metrics}")

if __name__ == '__main__':
    main()
