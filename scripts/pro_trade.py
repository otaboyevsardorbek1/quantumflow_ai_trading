#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Professional Trading Script
Production-ready entry point for real, demo, and paper trading
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.professional_trading import ProfessionalLiveTradingEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='QuantumFlow Professional Trading')

    # Account selection
    parser.add_argument('--accounts', nargs='+', required=True,
                       help='Account IDs to trade (from accounts.json)')
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['real', 'demo', 'paper'],
                       help='Trading mode')

    # Asset selection
    parser.add_argument('--assets', nargs='+', default=['XAUUSD'],
                       help='Assets to trade (XAUUSD, BTCUSD, ETHUSD, etc.)')
    parser.add_argument('--primary', type=str, default='XAUUSD',
                       help='Primary trading asset')

    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble policy')

    # Risk
    parser.add_argument('--risk-per-trade', type=float, default=0.01,
                       help='Risk per trade (default 1%)')
    parser.add_argument('--daily-risk', type=float, default=0.03,
                       help='Daily risk limit (default 3%)')
    parser.add_argument('--max-drawdown', type=float, default=0.10,
                       help='Max drawdown (default 10%)')

    # Execution
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    parser.add_argument('--close-on-shutdown', action='store_true',
                       help='Close all positions on shutdown')

    # Notifications
    parser.add_argument('--telegram-token', type=str, default=None,
                       help='Telegram bot token')
    parser.add_argument('--telegram-chat', type=str, default=None,
                       help='Telegram chat ID')

    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("🚀 QUANTUMFLOW v2.0 - PROFESSIONAL TRADING")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Accounts: {', '.join(args.accounts)}")
    logger.info(f"Assets: {', '.join(args.assets)}")
    logger.info(f"Primary: {args.primary}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Risk/Trade: {args.risk_per_trade:.1%}")
    logger.info(f"Daily Risk: {args.daily_risk:.1%}")
    logger.info(f"Max DD: {args.max_drawdown:.1%}")

    if args.mode == 'real':
        logger.warning("⚠️  REAL MONEY TRADING - Double check all settings!")
        logger.warning("⚠️  Press Ctrl+C within 5 seconds to cancel...")
        import time
        time.sleep(5)

    # Build configuration
    config = {
        'mode': args.mode,
        'accounts_config': 'accounts/accounts.json',
        'active_accounts': args.accounts,
        'primary_asset': args.primary,
        'assets': args.assets,
        'model_path': args.model,
        'use_ensemble': args.ensemble,
        'check_interval': args.interval,
        'close_on_shutdown': args.close_on_shutdown,
        'risk': {
            'max_risk_per_trade': args.risk_per_trade,
            'max_daily_risk': args.daily_risk,
            'max_drawdown': args.max_drawdown,
        },
        'gold_risk': {
            'max_gold_exposure': 0.20,
            'max_gold_positions': 3,
            'gold_daily_loss_limit': args.daily_risk,
        },
        'execution': {
            'slippage_tolerance': 0.0001,
            'commission': 0.00005,
            'max_spread': 0.0005,
        },
        'monitoring': {
            'max_daily_loss': args.daily_risk,
            'max_drawdown': args.max_drawdown,
            'max_latency_ms': 1000,
        },
        'telegram_bot_token': args.telegram_token,
        'telegram_chat_id': args.telegram_chat,
    }

    # Initialize and start
    engine = ProfessionalLiveTradingEngine(config)

    try:
        engine.initialize()
        engine.start()
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        engine.shutdown()

    logger.info("=" * 80)
    logger.info("✅ TRADING SESSION COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
