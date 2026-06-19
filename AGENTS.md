# QuantumFlow AI Trading - Agent Guidance

This file helps AI coding agents quickly understand the repository structure, common workflows, and Ubuntu-friendly adaptation points.

## Project summary

QuantumFlow is an autonomous AI trading platform built in Python. It includes:
- `agents/`: policy, Dreamer-style agents, and ensemble decision logic
- `config/`: global settings and feature toggles
- `core/`: trading orchestration, live trading, and asset management
- `execution/`: order execution logic and MT5 adapter
- `data/`: data ingestion, pipeline, and optional MT5 real-time feed
- `features/`: feature engineering
- `risk/`: risk manager and safety layer
- `evaluation/`: backtesting and validation
- `dashboard/`: optional Dash monitoring UI
- `accounts/`: account configuration and MT5 account manager
- `scripts/`: training, live trading, production checks, and helper automation

## Key entry points

Use these scripts as the main development and execution workflows:
- `python scripts/train.py` — training and evaluation
- `python scripts/live_trade.py` — paper/live trading
- `python scripts/production_check.py` — health checks and readiness
- `python scripts/fetch_data.py` / `data/fetch_all_data.py` — data download and pipeline
- `python tests/smoke_test.py` or `python -m unittest tests.test_quantumflow` — basic sanity tests

## Environment setup (Ubuntu)

This repository is designed for Linux/Mac development and expects a Python virtual environment.

Recommended Ubuntu workflow:
1. Install Python and build tools:
   - `sudo apt update`
   - `sudo apt install python3 python3-venv python3-pip build-essential libssl-dev libffi-dev`
2. Create and activate virtualenv:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
3. Install dependencies:
   - `pip install -r requirements.txt`

### Ubuntu adaptation notes

- `MetaTrader5` is optional and may not install cleanly on Ubuntu unless the MT5 runtime and compatible Python package are available. Code paths gracefully warn or disable MT5 features when the package is missing.
- The repository’s core functionality is Python-based and should work on Ubuntu once the required Python packages are installed.
- Optional components such as `dash`, `plotly`, `wandb`, `accelerate`, and `optuna` are not required for basic training or live trading and can be installed only when needed.

## Important conventions

- The project uses `requirements.txt` for package management rather than `pyproject.toml` or `pipenv`.
- Most code is Python 3 compatible; use `python3` commands on Ubuntu.
- Configurable behavior is centralized in `config/config.py`.
- Account management and MT5 execution are separated behind optional adapters.
- The repository is already structured so that missing optional packages should not break the entire system if code paths are guarded.

## Notes for AI agents

When asked to modify or extend the repository:
- Preserve the `python3 -m venv` workflow and Linux-compatible commands.
- Keep `requirements.txt` aligned with actual imports and platform compatibility.
- Avoid hardcoding Windows-only paths or dependencies.
- Prefer adding feature toggles in `config/config.py` when new optional integrations are introduced.
- When working on environment setup or compatibility, explicitly check for optional package guards around `MetaTrader5`, `dash`, `wandb`, and `optuna`.

## Useful references

- `README.md` — installation, training, and live-trading commands
- `requirements.txt` — dependency list
- `config/config.py` — runtime feature flags and environment settings
- `execution/engine.py` — MT5 execution adapter
- `data/websocket_feed.py` — optional MT5 real-time feed
- `accounts/manager.py` — multi-account manager using MT5
