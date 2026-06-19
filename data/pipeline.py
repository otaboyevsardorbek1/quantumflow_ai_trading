"""
QuantumFlow AI Trading System v2.0 - Data Pipeline
Multi-source data ingestion and preprocessing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import logging
from datetime import datetime, timedelta
import json
import requests

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Multi-source data pipeline:
    - Yahoo Finance (macro indicators)
    - MetaTrader 5 (price data)
    - Economic calendar (events)
    - Sentiment data (social media)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.symbols = config.get('symbols', ['XAUUSD'])
        self.macro_symbols = config.get('macro_symbols', [])

    def fetch_all_data(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> Dict:
        """Fetch all required data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"📊 Fetching data from {start_date} to {end_date}")

        data = {}

        # 1. Price data
        for symbol in self.symbols:
            data[f'{symbol}_price'] = self._fetch_price_data(symbol, start_date, end_date)

        # 2. Macro data
        for symbol in self.macro_symbols:
            data[f'{symbol}_macro'] = self._fetch_macro_data(symbol, start_date, end_date)

        # 3. Economic calendar
        data['economic_calendar'] = self._generate_economic_calendar(start_date, end_date)

        logger.info("✅ All data fetched successfully")
        return data

    def _fetch_price_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance or MT5"""
        logger.info(f"   Fetching {symbol} price data...")

        # Map forex symbols to Yahoo Finance format
        yf_symbol = self._map_symbol_to_yf(symbol)

        try:
            df = yf.download(yf_symbol, start=start, end=end, progress=False)
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"   ✅ {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"   ❌ Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_macro_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch macro indicator data"""
        logger.info(f"   Fetching {symbol} macro data...")

        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            logger.info(f"   ✅ {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"   ❌ Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def _generate_economic_calendar(self, start: str, end: str) -> pd.DataFrame:
        """Generate economic calendar events"""
        logger.info("   Generating economic calendar...")

        # Major economic events (simplified)
        events = []

        # NFP (first Friday of each month)
        current = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        while current <= end_dt:
            # Find first Friday
            if current.weekday() <= 4:  # Mon-Fri
                days_to_friday = 4 - current.weekday()
            else:
                days_to_friday = 4 + 7 - current.weekday()

            first_friday = current + timedelta(days=days_to_friday)

            if first_friday <= end_dt:
                events.append({
                    'date': first_friday,
                    'event': 'NFP',
                    'impact': 'high',
                    'currency': 'USD'
                })

            current = (current + pd.DateOffset(months=1)).replace(day=1)

        # FOMC (8 times per year - simplified)
        fomc_dates = pd.date_range(start=start, end=end, freq='6W')
        for date in fomc_dates:
            events.append({
                'date': date,
                'event': 'FOMC',
                'impact': 'high',
                'currency': 'USD'
            })

        df = pd.DataFrame(events)
        logger.info(f"   ✅ Calendar: {len(df)} events")
        return df

    def _map_symbol_to_yf(self, symbol: str) -> str:
        """Map internal symbol to Yahoo Finance format"""
        mapping = {
            'XAUUSD': 'GC=F',  # Gold futures
            'EURUSD': 'EURUSD=X',
            'BTCUSD': 'BTC-USD',
            'SPX500': '^GSPC',
            'DXY': 'DX-Y.NYB',
            'VIX': '^VIX',
            'US10Y': '^TNX',
            'WTI': 'CL=F',
            'GLD': 'GLD',
            'SLV': 'SLV',
            'SPY': 'SPY',
            'TLT': 'TLT',
        }
        return mapping.get(symbol, symbol)

    def save_data(self, data: Dict, output_dir: str):
        """Save fetched data"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for name, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(filepath)
                logger.info(f"   💾 Saved: {filepath}")

class SentimentDataFetcher:
    """Fetch sentiment data from social media and news"""

    def __init__(self, config: Dict):
        self.config = config

    def fetch_sentiment(self, symbol: str = 'XAUUSD', days: int = 30) -> pd.DataFrame:
        """Fetch sentiment data (placeholder for actual implementation)"""
        logger.info(f"📊 Fetching sentiment for {symbol}...")

        # This would integrate with:
        # - Twitter API
        # - Reddit API
        # - News APIs
        # - Google Trends

        # Placeholder: generate synthetic sentiment
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        sentiment = np.random.randn(len(dates)) * 0.3
        volume = np.random.randint(100, 10000, len(dates))

        df = pd.DataFrame({
            'timestamp': dates,
            'score': sentiment,
            'volume': volume,
            'positive': np.maximum(sentiment, 0),
            'negative': np.maximum(-sentiment, 0),
        })

        logger.info(f"✅ Sentiment data: {len(df)} rows")
        return df
