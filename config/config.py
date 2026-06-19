"""
QuantumFlow AI Trading System v2.0 - Global Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class DataConfig:
    """Ma'lumotlar konfiguratsiyasi"""
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD", "EURUSD", "BTCUSD", "SPX500"])
    base_timeframe: str = "M5"
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "H1", "H4", "D1", "W1"])
    data_dir: str = "data"
    lookback_window: int = 128  # Asl 64 dan 128 ga oshirildi (Transformer uchun)
    train_split_date: str = "2022-01-01"
    val_split_date: str = "2023-01-01"

    # Macro indicators
    macro_symbols: List[str] = field(default_factory=lambda: [
        "DXY", "VIX", "US10Y", "WTI", "BTC-USD", "EURUSD=X", 
        "GLD", "SLV", "SPY", "TLT"
    ])

    # Economic calendar
    calendar_file: str = "economic_events.json"
    high_impact_events: List[str] = field(default_factory=lambda: [
        "NFP", "CPI", "FOMC", "GDP", "Retail Sales", 
        "ISM Manufacturing", "PPI", "Unemployment Rate"
    ])

@dataclass
class FeatureConfig:
    """Feature engineering konfiguratsiyasi"""
    # Technical indicators
    use_trend_indicators: bool = True
    use_momentum_indicators: bool = True
    use_volatility_indicators: bool = True
    use_volume_indicators: bool = True
    use_price_action: bool = True

    # Advanced features
    use_market_regime: bool = True
    use_correlation_features: bool = True
    use_microstructure: bool = True
    use_sentiment: bool = True
    use_cross_timeframe: bool = True

    # Feature scaling
    feature_scaling: str = "robust"  # standard, robust, minmax
    use_pca: bool = False
    pca_components: int = 64

    # Total expected features
    expected_features: int = 256  # Asl 150 dan 256 ga oshirildi

@dataclass
class ModelConfig:
    """Model arxitekturasi konfiguratsiyasi"""
    # Architecture
    architecture: str = "transformer"  # transformer, lstm, gru, tcn
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Policy network
    policy_type: str = "continuous"  # continuous (position sizing) vs discrete
    action_dim: int = 3  # Flat, Long, Short
    position_sizing_dim: int = 10  # 0.0, 0.1, 0.2, ..., 1.0

    # Value network
    use_critic: bool = True
    critic_layers: List[int] = field(default_factory=lambda: [512, 256, 128])

    # Memory
    use_memory: bool = True
    memory_type: str = "transformer"  # lstm, transformer, attention
    memory_length: int = 128

    # Ensemble
    use_ensemble: bool = True
    ensemble_size: int = 3  # Trend, Mean-Reversion, Breakout
    ensemble_method: str = "weighted"  # weighted, voting, stacking

@dataclass
class RLConfig:
    """Reinforcement Learning konfiguratsiyasi"""
    algorithm: str = "PPO"  # PPO, SAC, TD3, DreamerV3

    # PPO specific
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 2_000_000  # Asl 1M dan 2M ga oshirildi
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {"volatility": 0.5, "spread": 0.0001, "steps": 500000},
        {"volatility": 1.0, "spread": 0.0002, "steps": 500000},
        {"volatility": 2.0, "spread": 0.0003, "steps": 500000},
        {"volatility": 3.0, "spread": 0.0005, "steps": 500000},
    ])

    # Online learning
    use_online_learning: bool = True
    online_learning_freq: int = 10000  # Har 10k qadamda
    online_buffer_size: int = 50000

@dataclass
class RiskConfig:
    """Risk management konfiguratsiyasi"""
    # Position sizing
    max_position_size: float = 0.20  # 20% of equity per trade
    min_position_size: float = 0.01  # 1% minimum
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.3  # Conservative Kelly (1/3)

    # Stop losses
    use_stop_loss: bool = True
    stop_loss_atr_mult: float = 2.0  # 2x ATR
    use_trailing_stop: bool = True
    trailing_stop_atr_mult: float = 3.0

    # Daily limits
    max_daily_loss: float = 0.03  # 3% daily loss limit
    max_daily_trades: int = 20
    max_consecutive_losses: int = 5

    # Drawdown
    max_drawdown: float = 0.15  # 15% max drawdown
    drawdown_cooldown_hours: int = 24

    # Volatility filter
    volatility_threshold: float = 3.0

    # Spread filter
    max_spread: float = 0.0005  # 5 pips for XAUUSD

    # Correlation guard
    use_correlation_guard: bool = True

    # AI Risk Manager
    use_ai_risk_manager: bool = True
    risk_model_path: Optional[str] = None

@dataclass
class ExecutionConfig:
    """Execution konfiguratsiyasi"""
    # Platform
    platform: str = "mt5"  # mt5, metaapi, oanda, interactive_brokers

    # Order types
    use_limit_orders: bool = True
    use_stop_orders: bool = True
    use_trailing_stops: bool = True

    # Execution
    slippage_tolerance: float = 0.0001
    execution_timeout: int = 30  # seconds
    retry_attempts: int = 3

    # Latency
    max_latency_ms: int = 500
    use_vps: bool = True
    vps_location: str = "london"  # london, new_york, tokyo

    # Paper trading
    paper_trading: bool = True
    paper_initial_balance: float = 100000.0

@dataclass
class MonitoringConfig:
    """Monitoring va logging konfiguratsiyasi"""
    log_level: str = "INFO"
    use_telegram: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Dashboard
    use_dashboard: bool = True
    dashboard_port: int = 8050
    dashboard_refresh_interval: int = 5  # seconds

    # Metrics
    track_sharpe: bool = True
    track_sortino: bool = True
    track_calmar: bool = True
    track_max_drawdown: bool = True
    track_win_rate: bool = True
    track_profit_factor: bool = True

    # Alerts
    alert_on_drawdown: float = 0.10
    alert_on_daily_loss: float = 0.02
    alert_on_consecutive_losses: int = 3

@dataclass
class SystemConfig:
    """Tizim konfiguratsiyasi"""
    device: str = "auto"  # auto, cuda, mps, cpu
    seed: int = 42
    num_workers: int = 4
    use_mixed_precision: bool = True

    # Distributed training
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 50000
    keep_last_n_checkpoints: int = 5

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "quantumflow-trading"
    wandb_entity: Optional[str] = None

def get_default_config():
    """Barcha konfiguratsiyalarni birlashtirish"""
    return {
        "data": DataConfig(),
        "features": FeatureConfig(),
        "model": ModelConfig(),
        "rl": RLConfig(),
        "risk": RiskConfig(),
        "execution": ExecutionConfig(),
        "monitoring": MonitoringConfig(),
        "system": SystemConfig(),
    }

def load_config(config_path: str) -> Dict:
    """Konfiguratsiya faylini yuklash"""
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def save_config(config: Dict, config_path: str):
    """Konfiguratsiyani saqlash"""
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
