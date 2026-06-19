#!/usr/bin/env python3
"""
QuantumFlow AI Trading System v2.0 - Data & Training Pipeline
Complete pipeline: Download data -> Process -> Train model -> Validate
Supports: Forex, Crypto, Stocks, Commodities
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """
    Multi-source data downloader

    Sources:
    - Yahoo Finance (Forex, Stocks, Crypto, Commodities)
    - Binance (Crypto futures)
    - Dukascopy (Forex tick data)
    - MT5 (Broker data)
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_yahoo(self, symbols: list, start: str, end: str, interval: str = "1h") -> dict:
        """
        Download data from Yahoo Finance

        Args:
            symbols: List of symbols (e.g., ['EURUSD=X', 'GC=F', 'BTC-USD'])
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)

        Returns:
            dict: {symbol: DataFrame}
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("❌ yfinance not installed. Run: pip install yfinance")
            return {}

        results = {}

        for symbol in symbols:
            try:
                logger.info(f"📊 Downloading {symbol} from Yahoo Finance...")

                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False
                )

                if df.empty:
                    logger.warning(f"⚠️ No data for {symbol}")
                    continue

                # Clean column names
                df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
                df = df[['open', 'high', 'low', 'close', 'volume']]

                # Save
                filename = f"{symbol.replace('=', '').replace('-', '_')}_{interval}.csv"
                filepath = os.path.join(self.output_dir, filename)
                df.to_csv(filepath)

                results[symbol] = df
                logger.info(f"✅ {symbol}: {len(df)} rows saved to {filename}")

                time.sleep(0.5)  # Rate limit

            except Exception as e:
                logger.error(f"❌ Error downloading {symbol}: {e}")

        return results

    def download_binance(self, symbols: list, interval: str = "1h", 
                         start_date: str = None, days_back: int = 365) -> dict:
        """
        Download crypto data from Binance

        Args:
            symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD)
            days_back: Days of history (if start_date not provided)
        """
        try:
            import ccxt
        except ImportError:
            logger.error("❌ ccxt not installed. Run: pip install ccxt")
            return {}

        exchange = ccxt.binance({'enableRateLimit': True})
        results = {}

        for symbol in symbols:
            try:
                logger.info(f"📊 Downloading {symbol} from Binance...")

                if start_date:
                    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                else:
                    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

                all_ohlcv = []
                current_ts = since
                end_ts = exchange.milliseconds()

                while current_ts < end_ts:
                    ohlcv = exchange.fetch_ohlcv(symbol, interval, since=current_ts, limit=1000)
                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)

                    if len(ohlcv) < 1000:
                        break

                    current_ts = ohlcv[-1][0] + 1
                    time.sleep(0.1)

                if all_ohlcv:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('date')
                    df = df[['open', 'high', 'low', 'close', 'volume']]

                    # Save
                    filename = f"{symbol}_{interval}.csv"
                    filepath = os.path.join(self.output_dir, filename)
                    df.to_csv(filepath)

                    results[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} rows saved")

            except Exception as e:
                logger.error(f"❌ Error downloading {symbol}: {e}")

        return results

    def download_dukascopy(self, symbol: str, start: str, end: str, 
                          timeframe: str = "M1") -> pd.DataFrame:
        """
        Download forex data from Dukascopy (high quality tick data)

        Note: Requires jforex or manual download from dukascopy.com
        """
        logger.info(f"📊 Dukascopy data for {symbol}")
        logger.info(f"   Visit: https://www.dukascopy.com/swiss/english/marketwatch/historical/")
        logger.info(f"   Download {symbol} {timeframe} from {start} to {end}")
        logger.info(f"   Save to: {self.output_dir}/dukascopy_{symbol}_{timeframe}.csv")

        # Placeholder - actual download requires Dukascopy API or manual download
        return pd.DataFrame()

    def download_mt5(self, symbol: str, timeframe: str = "H1", 
                     start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """
        Download data from MetaTrader 5

        Requires MT5 terminal running
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error("❌ MetaTrader5 not installed")
            return pd.DataFrame()

        if not mt5.initialize():
            logger.error("❌ MT5 initialization failed")
            return pd.DataFrame()

        # Map timeframe string to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }

        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

        if not start:
            start = datetime.now() - timedelta(days=365)
        if not end:
            end = datetime.now()

        rates = mt5.copy_rates_range(symbol, tf, start, end)

        if rates is None or len(rates) == 0:
            logger.warning(f"⚠️ No data from MT5 for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })

        # Save
        filename = f"mt5_{symbol}_{timeframe}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath)

        logger.info(f"✅ MT5 {symbol}: {len(df)} rows")

        mt5.shutdown()
        return df

class DataProcessor:
    """
    Process raw data into training format
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_data(self, symbol: str, source: str = "yahoo") -> pd.DataFrame:
        """Load data from file"""
        # Find file
        files = [f for f in os.listdir(self.data_dir) if symbol in f and f.endswith('.csv')]

        if not files:
            logger.error(f"❌ No data file found for {symbol}")
            return pd.DataFrame()

        filepath = os.path.join(self.data_dir, files[0])
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        logger.info(f"📊 Loaded {symbol}: {len(df)} rows")
        return df

    def prepare_training_data(self, df: pd.DataFrame, window: int = 128) -> tuple:
        """
        Prepare data for model training

        Returns:
            (features, returns, timestamps)
        """
        from features.engineering import AdvancedFeatureEngineer

        # Compute features
        engineer = AdvancedFeatureEngineer(config={})
        features = engineer.compute_all_features(df)

        # Calculate returns
        returns = df['close'].pct_change().fillna(0).values.astype(np.float32)
        timestamps = np.arange(len(df))

        # Remove NaN rows
        valid_idx = ~np.isnan(features).any(axis=1)
        features = features[valid_idx]
        returns = returns[valid_idx]
        timestamps = timestamps[valid_idx]

        logger.info(f"✅ Training data: {len(features)} samples, {features.shape[1]} features")

        return features, returns, timestamps

    def split_data(self, features, returns, timestamps, 
                   train_ratio=0.7, val_ratio=0.15):
        """Split data into train/val/test"""
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = (features[:train_end], returns[:train_end], timestamps[:train_end])
        val_data = (features[train_end:val_end], returns[train_end:val_end], timestamps[train_end:val_end])
        test_data = (features[val_end:], returns[val_end:], timestamps[val_end:])

        logger.info(f"📊 Split: Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")

        return train_data, val_data, test_data

class ModelTrainer:
    """
    Train QuantumFlow model
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cpu')

    def train(self, train_data, val_data, env_config):
        """
        Train model with PPO

        Args:
            train_data: (features, returns, timestamps)
            val_data: (features, returns, timestamps)
            env_config: Environment configuration
        """
        import torch
        from env.trading_env import QuantumTradingEnv
        from agents.policy_network import TransformerPolicyNetwork, EnsemblePolicy
        from agents.ppo_trainer import PPOTrainer

        features, returns, timestamps = train_data

        # Create environment
        env = QuantumTradingEnv(
            features=features,
            returns=returns,
            timestamps=timestamps,
            config=env_config,
            window=self.config.get('window', 128)
        )

        # Create agent
        if self.config.get('use_ensemble', True):
            agent = EnsemblePolicy(
                n_features=features.shape[1],
                window_size=self.config.get('window', 128),
                ensemble_size=self.config.get('ensemble_size', 3),
            )
        else:
            agent = TransformerPolicyNetwork(
                n_features=features.shape[1],
                window_size=self.config.get('window', 128),
            )

        # Create trainer
        trainer = PPOTrainer(
            policy=agent,
            env=env,
            config=self.config.get('rl', {}),
            device=self.device
        )

        # Train
        total_steps = self.config.get('total_steps', 1000000)
        logger.info(f"🏋️ Starting training: {total_steps:,} steps")

        trainer.train(
            total_timesteps=total_steps,
            eval_freq=50000,
            save_freq=50000
        )

        # Save final model
        model_path = self.config.get('model_path', 'checkpoints/final_model.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_checkpoint(model_path)

        logger.info(f"💾 Model saved to {model_path}")

        return agent

def main():
    parser = argparse.ArgumentParser(description='QuantumFlow Data & Training Pipeline')

    # Data download
    parser.add_argument('--download', action='store_true', help='Download data')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD=X'], help='Symbols to download')
    parser.add_argument('--source', type=str, default='yahoo', choices=['yahoo', 'binance', 'mt5'])
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval')

    # Training
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble')
    parser.add_argument('--steps', type=int, default=1000000, help='Training steps')
    parser.add_argument('--window', type=int, default=128, help='Observation window')

    # Device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🚀 QUANTUMFLOW DATA & TRAINING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Download data
    if args.download:
        downloader = DataDownloader(output_dir="data")

        if args.source == 'yahoo':
            data = downloader.download_yahoo(args.symbols, args.start, args.end, args.interval)
        elif args.source == 'binance':
            data = downloader.download_binance(args.symbols, args.interval, args.start)
        elif args.source == 'mt5':
            for symbol in args.symbols:
                downloader.download_mt5(symbol, args.interval)

    # Step 2: Process and train
    if args.train:
        processor = DataProcessor(data_dir="data")

        # Load primary symbol
        symbol = args.symbols[0]
        df = processor.load_data(symbol, args.source)

        if df.empty:
            logger.error("❌ No data loaded. Run with --download first")
            return

        # Prepare training data
        features, returns, timestamps = processor.prepare_training_data(df, window=args.window)

        # Split
        train_data, val_data, test_data = processor.split_data(features, returns, timestamps)

        # Train
        config = {
            'device': args.device,
            'window': args.window,
            'use_ensemble': args.ensemble,
            'model_path': args.model_path,
            'total_steps': args.steps,
            'rl': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'batch_size': 64,
                'n_steps': 2048,
                'n_epochs': 10,
            }
        }

        trainer = ModelTrainer(config)
        model = trainer.train(train_data, val_data, env_config={})

        logger.info("✅ Training complete!")

    logger.info("=" * 80)
    logger.info("🏁 PIPELINE COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
