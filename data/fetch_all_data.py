#!/usr/bin/env python3
"""
QuantumFlow AI - Ma'lumot Yuklash Skripti
Bepul manbalardan ma'lumot yuklash
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FOREX MA'LUMOTLARI (Yahoo Finance - BEPUL)
# ============================================================

FOREX_PAIRS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
    'NZDUSD': 'NZDUSD=X',
    'EURGBP': 'EURGBP=X',
}

# ============================================================
# GOLD & COMMODITIES
# ============================================================

COMMODITIES = {
    'XAUUSD': 'GC=F',        # Gold Futures
    'XAGUSD': 'SI=F',        # Silver Futures
    'WTI': 'CL=F',           # Crude Oil
    'BRENT': 'BZ=F',         # Brent Oil
    'NATGAS': 'NG=F',        # Natural Gas
    'COPPER': 'HG=F',        # Copper
}

# ============================================================
# CRYPTO (Yahoo Finance)
# ============================================================

CRYPTO_YAHOO = {
    'BTCUSD': 'BTC-USD',
    'ETHUSD': 'ETH-USD',
    'LTCUSD': 'LTC-USD',
    'XRPUSD': 'XRP-USD',
    'ADAUSD': 'ADA-USD',
    'SOLUSD': 'SOL-USD',
}

# ============================================================
# INDICES
# ============================================================

INDICES = {
    'SPX500': '^GSPC',       # S&P 500
    'US30': '^DJI',          # Dow Jones
    'NAS100': '^IXIC',       # NASDAQ
    'DAX': '^GDAXI',         # DAX 40
    'FTSE': '^FTSE',         # FTSE 100
    'NIKKEI': '^N225',       # Nikkei 225
}

# ============================================================
# ELECTRONICS / TECH STOCKS
# ============================================================

TECH_STOCKS = {
    'NVDA': 'NVDA',          # NVIDIA
    'AMD': 'AMD',            # AMD
    'INTC': 'INTC',          # Intel
    'TSM': 'TSM',            # TSMC
    'AAPL': 'AAPL',          # Apple
    'MSFT': 'MSFT',          # Microsoft
    'GOOGL': 'GOOGL',        # Google
    'META': 'META',          # Meta
    'AMZN': 'AMZN',          # Amazon
    'TSLA': 'TSLA',          # Tesla
}

def download_data(symbols_dict, interval='1h', start='2020-01-01', end=None, output_dir='data'):
    """
    Ma'lumotlarni yuklash va saqlash

    Args:
        symbols_dict: {name: yahoo_symbol} formatida
        interval: '1m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo'
        start: Boshlanish sanasi (YYYY-MM-DD)
        end: Tugash sanasi (YYYY-MM-DD)
        output_dir: Saqlash papkasi
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, symbol in symbols_dict.items():
        try:
            logger.info(f"📊 {name} ({symbol}) yuklanmoqda...")

            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False
            )

            if df.empty:
                logger.warning(f"⚠️ {name}: Ma'lumot yo'q")
                continue

            # Ustun nomlarini tozalash
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Saqlash
            filename = f"{name}_{interval}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)

            results[name] = df
            logger.info(f"✅ {name}: {len(df)} qator -> {filename}")

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            logger.error(f"❌ {name} xato: {e}")

    return results

def download_all_forex(output_dir='data'):
    """Barcha forex juftliklarini yuklash"""
    logger.info("=" * 60)
    logger.info("📊 FOREX MA'LUMOTLARINI YUKLASH")
    logger.info("=" * 60)
    return download_data(FOREX_PAIRS, interval='1h', output_dir=output_dir)

def download_all_gold(output_dir='data'):
    """Gold va commodities yuklash"""
    logger.info("=" * 60)
    logger.info("🥇 GOLD VA COMMODITIES YUKLASH")
    logger.info("=" * 60)
    return download_data(COMMODITIES, interval='1h', output_dir=output_dir)

def download_all_crypto(output_dir='data'):
    """Crypto yuklash"""
    logger.info("=" * 60)
    logger.info("₿ CRYPTO YUKLASH")
    logger.info("=" * 60)
    return download_data(CRYPTO_YAHOO, interval='1d', output_dir=output_dir)

def download_all_indices(output_dir='data'):
    """Indices yuklash"""
    logger.info("=" * 60)
    logger.info("📈 INDICES YUKLASH")
    logger.info("=" * 60)
    return download_data(INDICES, interval='1d', output_dir=output_dir)

def download_all_tech(output_dir='data'):
    """Tech stocks yuklash"""
    logger.info("=" * 60)
    logger.info("💻 TECH STOCKS YUKLASH")
    logger.info("=" * 60)
    return download_data(TECH_STOCKS, interval='1d', output_dir=output_dir)

def main():
    """Barcha ma'lumotlarni yuklash"""
    logger.info("🚀 QUANTUMFLOW MA'LUMOT YUKLASH")
    logger.info("=" * 60)

    # 1. Forex
    forex_data = download_all_forex()

    # 2. Gold
    gold_data = download_all_gold()

    # 3. Crypto
    crypto_data = download_all_crypto()

    # 4. Indices
    indices_data = download_all_indices()

    # 5. Tech
    tech_data = download_all_tech()

    # Jami
    total = len(forex_data) + len(gold_data) + len(crypto_data) + len(indices_data) + len(tech_data)
    logger.info("=" * 60)
    logger.info(f"✅ JAMI: {total} ta aktiv yuklandi")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
