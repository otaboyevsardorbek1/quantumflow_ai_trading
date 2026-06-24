#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Live Trading Script
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import get_default_config
from core.live_trading import LiveTradingEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run QuantumFlow Live Trading')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("🚀 QUANTUMFLOW LIVE TRADING")
    logger.info("=" * 80)

    config = get_default_config()
    config['symbol'] = args.symbol
    config['model_path'] = args.model
    config['paper_trading'] = args.paper
    config['check_interval'] = args.interval
    config['use_ensemble'] = False   # <-- SHU QATORNI QO‘SHING

    engine = LiveTradingEngine(config)
    engine.initialize()
    engine.start()

if __name__ == '__main__':
    main()
