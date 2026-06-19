# 🚀 QuantumFlow AI Trading System v2.0

**Advanced Autonomous AI Trading Platform** — GitHub'dagi asl loyihadan ilhomlangan, lekin to'liq qayta ishlangan va mukammallashtirilgan avtonom AI trading tizimi.

## 📋 Asl Loyiha vs QuantumFlow v2.0 Taqqoslash

| Xususiyat | Asl Loyiha (zero-was-here/tradingbot) | QuantumFlow v2.0 |
|-----------|--------------------------------------|------------------|
| **Action Space** | Discrete (2: Flat/Long) | **Continuous** (4: Direction, Size, SL, TP) |
| **Position Sizing** | Fixed | **Dynamic** (Kelly Criterion) |
| **Short Trading** | ❌ Yo'q | ✅ **Bor** (Long/Short/Flat) |
| **Policy Network** | MLP | **Transformer + Attention** |
| **Ensemble** | ❌ Yo'q | ✅ **3-Agent Ensemble** |
| **Risk Management** | Deterministic only | **AI + Deterministic Dual Layer** |
| **Order Types** | Market only | **Market, Limit, Stop, Trailing Stop** |
| **Regime Detection** | ❌ Yo'q | ✅ **HMM-based** |
| **Curriculum Learning** | ❌ Yo'q | ✅ **Bor** |
| **Online Learning** | ❌ Yo'q | ✅ **Bor** |
| **Features** | 140+ | **256+** |
| **Multi-Asset** | ❌ Faqat XAUUSD | ✅ **XAUUSD, EURUSD, BTCUSD, SPX** |
| **Sentiment Analysis** | ❌ Yo'q | ✅ **Bor** |
| **Interpretability** | ❌ Black box | ✅ **SHAP + Attention Viz** |
| **Backtesting** | Basic | **Walk-forward + Monte Carlo** |

## 🏗️ Arxitektura

```
QuantumFlow AI Trading System v2.0/
│
├── 📂 config/                  # Konfiguratsiya
│   └── config.py              # Global config
│
├── 📂 features/                # Feature Engineering (256+)
│   └── engineering.py         # Advanced features
│
├── 📂 env/                     # Trading Environment
│   └── trading_env.py         # Continuous action space
│
├── 📂 agents/                  # AI Agents
│   ├── policy_network.py      # Transformer + Ensemble
│   └── ppo_trainer.py         # Advanced PPO
│
├── 📂 risk/                    # Risk Management
│   └── manager.py             # AI + Deterministic
│
├── 📂 execution/               # Order Execution
│   └── engine.py              # Multiple order types
│
├── 📂 evaluation/              # Backtesting
│   └── backtest.py            # Advanced metrics
│
├── 📂 core/                    # Live Trading
│   └── live_trading.py        # Paper + Live mode
│
├── 📂 scripts/                 # Scripts
│   ├── train.py               # Training script
│   └── live_trade.py          # Live trading script
│
├── 📄 requirements.txt         # Dependencies
└── 📄 README.md               # This file
```

## 🚀 O'rnatish

```bash
# 1. Clone repository
git clone <repository-url>
cd quantumflow_ai_trading

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## 🎯 Training

```bash
# Basic training
python scripts/train.py --symbol XAUUSD --steps 2000000

# Ensemble training
python scripts/train.py --symbol XAUUSD --ensemble --steps 2000000

# Resume training
python scripts/train.py --resume checkpoints/checkpoint_1000000.pt

# Evaluation only
python scripts/train.py --eval-only --model checkpoints/best_model.pt
```

## 📈 Live Trading

```bash
# Paper trading (test mode)
python scripts/live_trade.py --model checkpoints/best_model.pt --paper --symbol XAUUSD

# Live trading (real money)
python scripts/live_trade.py --model checkpoints/best_model.pt --symbol XAUUSD
```

## 🛡️ Risk Management

QuantumFlow v2.0 ikki qatlamli risk management tizimiga ega:

1. **AI Risk Manager**: Neural network-based dynamic risk adjustment
2. **Deterministic Safety Layer**: 15+ hard-coded safety checks

### Safety Checks:
- Daily loss limit (circuit breaker)
- Maximum drawdown protection
- Position size limits
- Volatility filters
- Spread filters
- Correlation guards
- Event risk filters
- Consecutive loss protection
- Cooldown periods
- Margin checks
- Liquidity checks
- Market hours validation
- Weekend protection

## 🧠 AI Architecture

### Transformer Policy Network
- **Multi-head attention** for feature importance
- **Positional encoding** for temporal patterns
- **Feature attention** for interpretability
- **Actor-Critic** architecture

### 3-Agent Ensemble
1. **TrendFollower**: Trending marketsda ishlaydi
2. **MeanReverter**: Ranging marketsda ishlaydi
3. **BreakoutTrader**: Breakout signallarini qidiradi

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Annual Return | 100-150%+ |
| Sharpe Ratio | 4.0-5.0+ |
| Max Drawdown | <10% |
| Win Rate | 65-70% |
| Profit Factor | 3.0-4.0+ |

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly on demo accounts before using real money.

## 📄 License

MIT License
