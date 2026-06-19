#!/usr/bin/env python3
"""
QuantumFlow AI - Ma'lumot Yuklash Skripti
Bepul manbalardan ma'lumot yuklash
"""
import argparse
import json
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

    interval = interval.lower()
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    use_period = False
    period = None

    # Yahoo Finance intraday data often supports only the last 730 days.
    if interval not in ['1d', '1wk', '1mo']:
        now = datetime.now()
        requested_start = datetime.fromisoformat(start)
        requested_end = datetime.fromisoformat(end)
        max_age = timedelta(days=730)

        if requested_end > now:
            requested_end = now
            logger.warning(
                f"⚠️ {interval} interval uchun end hozirgi sanadan katta bo‘lishi mumkin emas. end={requested_end.strftime('%Y-%m-%d')} bo‘ldi."
            )

        if requested_end < now - max_age or requested_end - requested_start > max_age:
            use_period = True
            period = '730d'
            logger.warning(
                "⚠️ 1h ma'lumotlar faqat oxirgi 730 kungacha mavjud. "
                "Siz so‘ragan tarixiy intervalga mos kelmaydi, shuning uchun oxirgi 730 kun olinadi."
            )
            start = (now - max_age).strftime('%Y-%m-%d')
            end = now.strftime('%Y-%m-%d')

    for name, symbol in symbols_dict.items():
        try:
            logger.info(f"📊 {name} ({symbol}) yuklanmoqda...")

            if use_period:
                df = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False
                )
            else:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False
                )

            if df.empty:
                logger.warning(f"⚠️ {name}: Ma'lumot yo'q. 730d period bilan qayta uriniladi...")
                if interval not in ['1d', '1wk', '1mo']:
                    df = yf.download(
                        symbol,
                        period='730d',
                        interval=interval,
                        progress=False
                    )

            if df.empty:
                logger.warning(f"⚠️ {name}: Ma'lumot yo'q")
                continue

            # ========== TUZATISH: Ustun nomlarini ishonchli normalizatsiya ==========
            # 1) MultiIndex bo‘lsa, faqat birinchi darajani olamiz
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 2) Barcha ustun nomlarini kichik harfga o‘tkazamiz va bo‘shliqlarni '_' ga almashtiramiz
            new_cols = []
            for col in df.columns:
                if isinstance(col, str):
                    new_cols.append(col.lower().replace(' ', '_'))
                else:
                    # Agar ustun nomi string bo‘lmasa (masalan, tuple bo‘lsa), uni stringga aylantiramiz
                    new_cols.append(str(col).lower().replace(' ', '_'))
            df.columns = new_cols

            # 3) Kerakli ustunlarni qayta nomlash (agar mavjud bo‘lsa)
            rename_map = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adj_close': 'adj_close',
                'volume': 'volume'
            }
            # Faqat mavjud ustunlarni rename qilamiz
            existing = set(df.columns)
            rename_available = {k: v for k, v in rename_map.items() if k in existing}
            if rename_available:
                df = df.rename(columns=rename_available)

            # 4) Kerakli ustunlar mavjudligini tekshiramiz
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                logger.warning(f"⚠️ {name}: kerakli ustunlar yo'q -> {missing_cols}")
                continue

            # Faqat kerakli ustunlarni saqlaymiz
            df = df[required_columns]

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
    parser = argparse.ArgumentParser(
        description="QuantumFlow ma'lumot yuklaydigan interaktiv skript"
    )
    parser.add_argument(
        '--category',
        choices=['forex', 'gold', 'crypto', 'indices', 'tech', 'all', 'auto'],
        default=None,
        help='Yuklanadigan aktivlar toifasi'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interaktiv rejimda sorovnoma orqali tanlash'
    )
    parser.add_argument(
        '--interval',
        default='1h',
        help='Data intervali (1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)'
    )
    parser.add_argument(
        '--start',
        default='2020-01-01',
        help='Boshlanish sanasi (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        default=None,
        help='Tugash sanasi (YYYY-MM-DD). Default hozirgi sana.'
    )
    parser.add_argument(
        '--output-dir',
        default='data',
        help='CSV fayllar saqlanadigan papka'
    )
    parser.add_argument(
        '--auto-output-dir',
        default=None,
        help='Auto tanlov uchun alohida saqlash papkasi'
    )
    args = parser.parse_args()

    logger.info("🚀 QUANTUMFLOW MA'LUMOT YUKLASH")
    logger.info("=" * 60)

    if args.interactive or args.category is None:
        print("\n--- QuantumFlow interaktiv ma'lumot yuklash ---")
        print("1) forex")
        print("2) gold")
        print("3) crypto")
        print("4) indices")
        print("5) tech")
        print("6) all")
        print("7) auto")
        choice = input("Qaysi toifani yuklamoqchisiz? [1-7] (default 6): ").strip() or '6'
        mapping = {
            '1': 'forex',
            '2': 'gold',
            '3': 'crypto',
            '4': 'indices',
            '5': 'tech',
            '6': 'all',
            '7': 'auto'
        }
        category = mapping.get(choice, 'all')

        interval = input("Intervalni tanlang (1h): ").strip() or '1h'
        start = input("Boshlanish sanasini kiriting (2020-01-01): ").strip() or '2020-01-01'
        end = input("Tugash sanasini kiriting (hozirgi sana): ").strip() or None
        output_dir = input("Saqlash papkasini kiriting (data): ").strip() or 'data'
        auto_output_dir = input("Auto tanlov uchun maxsus papka (bo'sh bo'lsa data): ").strip() or None
    else:
        category = args.category
        interval = args.interval
        start = args.start
        end = args.end
        output_dir = args.output_dir
        auto_output_dir = args.auto_output_dir

    total = 0
    run_info = {
        'category': category,
        'interval': interval,
        'start': start,
        'end': end or datetime.now().strftime('%Y-%m-%d'),
        'output_dir': output_dir,
        'downloaded': []
    }

    if category in ['forex', 'all', 'auto']:
        downloaded = download_data(FOREX_PAIRS, interval=interval, start=start, end=end, output_dir=output_dir)
        total += len(downloaded)
        run_info['downloaded'].extend(downloaded.keys())
    if category in ['gold', 'all', 'auto']:
        downloaded = download_data(COMMODITIES, interval=interval, start=start, end=end, output_dir=output_dir)
        total += len(downloaded)
        run_info['downloaded'].extend(downloaded.keys())
    if category in ['crypto', 'all', 'auto']:
        downloaded = download_data(CRYPTO_YAHOO, interval=interval, start=start, end=end, output_dir=output_dir)
        total += len(downloaded)
        run_info['downloaded'].extend(downloaded.keys())
    if category in ['indices', 'all', 'auto']:
        downloaded = download_data(INDICES, interval=interval, start=start, end=end, output_dir=output_dir)
        total += len(downloaded)
        run_info['downloaded'].extend(downloaded.keys())
    if category in ['tech', 'all', 'auto']:
        downloaded = download_data(TECH_STOCKS, interval=interval, start=start, end=end, output_dir=output_dir)
        total += len(downloaded)
        run_info['downloaded'].extend(downloaded.keys())

    run_info['total_downloaded'] = total
    run_info['timestamp'] = datetime.now().isoformat()

    save_dir = auto_output_dir or output_dir
    os.makedirs(save_dir, exist_ok=True)
    auto_path = os.path.join(save_dir, 'auto_fetch_info.json')
    with open(auto_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ JAMI: {total} ta aktiv yuklandi")
    logger.info(f"ℹ️ Auto info saqlandi: {auto_path}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()