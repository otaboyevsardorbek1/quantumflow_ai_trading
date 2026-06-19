#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Hyperparameter Optimization
AutoML with Optuna for optimal hyperparameters
"""
import os
import sys
import optuna
from optuna.samplers import TPESampler
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import get_default_config
from features.engineering import AdvancedFeatureEngineer
from env.trading_env import QuantumTradingEnv
from agents.policy_network import TransformerPolicyNetwork
from agents.ppo_trainer import PPOTrainer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function

    Args:
        trial: Optuna trial

    Returns:
        Sharpe ratio (to maximize)
    """
    # Hyperparameters to optimize
    config = get_default_config()

    # Model architecture
    config['model']['d_model'] = trial.suggest_categorical('d_model', [128, 256, 512])
    config['model']['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
    config['model']['num_encoder_layers'] = trial.suggest_int('num_encoder_layers', 2, 6)
    config['model']['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048])
    config['model']['dropout'] = trial.suggest_float('dropout', 0.0, 0.3)

    # RL hyperparameters
    config['rl']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    config['rl']['gamma'] = trial.suggest_float('gamma', 0.95, 0.999)
    config['rl']['gae_lambda'] = trial.suggest_float('gae_lambda', 0.9, 0.99)
    config['rl']['clip_range'] = trial.suggest_float('clip_range', 0.1, 0.3)
    config['rl']['ent_coef'] = trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
    config['rl']['vf_coef'] = trial.suggest_float('vf_coef', 0.3, 0.7)
    config['rl']['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    config['rl']['n_steps'] = trial.suggest_categorical('n_steps', [1024, 2048, 4096])

    # Environment parameters
    config['risk']['max_position'] = trial.suggest_float('max_position', 0.1, 0.3)
    config['risk']['stop_loss_atr_mult'] = trial.suggest_float('sl_atr', 1.0, 3.0)
    config['risk']['trailing_stop_atr_mult'] = trial.suggest_float('ts_atr', 2.0, 5.0)

    logger.info(f"🧪 Trial {trial.number}: Testing configuration...")

    # Create dummy data for quick evaluation
    n_samples = 50000
    features = np.random.randn(n_samples, 256).astype(np.float32)
    returns = np.random.randn(n_samples).astype(np.float32) * 0.001
    timestamps = np.arange(n_samples)

    env = QuantumTradingEnv(features, returns, timestamps, window=128)

    agent = TransformerPolicyNetwork(
        n_features=256,
        window_size=128,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
    )

    trainer = PPOTrainer(agent, env, config['rl'], device='cpu')

    # Quick training (fewer steps for optimization)
    trainer.train(total_timesteps=100000, eval_freq=50000, save_freq=100000)

    # Evaluate
    metrics = trainer.evaluate(n_episodes=5)
    sharpe = metrics.get('avg_reward', 0)

    logger.info(f"✅ Trial {trial.number}: Sharpe = {sharpe:.4f}")

    return sharpe

def main():
    logger.info("=" * 80)
    logger.info("🔬 HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='quantumflow_optimization'
    )

    # Optimize
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Results
    logger.info("=" * 80)
    logger.info("✅ OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Sharpe: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"   {key}: {value}")

    # Save results
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # Visualization
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optimization_history.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("param_importances.html")

        logger.info("📊 Visualizations saved")
    except ImportError:
        logger.info("Plotly not installed, skipping visualizations")

if __name__ == '__main__':
    main()
