#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Data Fetching Script
Fetch all required market data
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline import DataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch market data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--symbols', nargs='+', default=['XAUUSD'], help='Symbols to fetch')
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("📊 DATA FETCHING")
    logger.info("=" * 80)

    config = {
        'data_dir': args.output,
        'symbols': args.symbols,
        'macro_symbols': ['DXY', 'VIX', 'US10Y', 'WTI', 'BTC-USD', 'GLD', 'SLV', 'SPY', 'TLT'],
    }

    pipeline = DataPipeline(config)
    data = pipeline.fetch_all_data(start_date=args.start, end_date=args.end)
    pipeline.save_data(data, args.output)

    logger.info("=" * 80)
    logger.info("✅ DATA FETCHING COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
