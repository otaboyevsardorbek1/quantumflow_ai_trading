"""
QuantumFlow AI Trading System v2.0 - Baseline Comparisons
Compare against standard benchmarks
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BaselineComparison:
    """
    Compare trading agent against baselines:
    - Buy & Hold
    - Random Policy
    - MA Crossover
    - RSI Strategy
    - Bollinger Bands
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['returns'] = self.df['close'].pct_change().fillna(0)
        self.results = {}

    def run_all_baselines(self) -> Dict:
        """Run all baseline strategies"""
        logger.info("📊 Running baseline comparisons...")

        self.results['buy_hold'] = self._buy_and_hold()
        self.results['random'] = self._random_policy()
        self.results['ma_crossover'] = self._ma_crossover()
        self.results['rsi_strategy'] = self._rsi_strategy()
        self.results['bollinger_bands'] = self._bollinger_bands()

        return self.results

    def _buy_and_hold(self) -> Dict:
        """Buy and hold benchmark"""
        equity = (1 + self.df['returns']).cumprod()
        return {
            'name': 'Buy & Hold',
            'equity': equity.values,
            'total_return': equity.iloc[-1] - 1,
            'sharpe': self._compute_sharpe(self.df['returns']),
            'max_dd': self._compute_max_dd(equity.values),
            'trades': 1,
        }

    def _random_policy(self) -> Dict:
        """Random trading policy"""
        np.random.seed(42)
        positions = np.random.choice([-1, 0, 1], size=len(self.df))
        strategy_returns = positions * self.df['returns']
        equity = (1 + strategy_returns).cumprod()

        return {
            'name': 'Random Policy',
            'equity': equity.values,
            'total_return': equity.iloc[-1] - 1,
            'sharpe': self._compute_sharpe(strategy_returns),
            'max_dd': self._compute_max_dd(equity.values),
            'trades': sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1]),
        }

    def _ma_crossover(self, fast: int = 20, slow: int = 50) -> Dict:
        """Moving average crossover strategy"""
        self.df[f'ma_fast'] = self.df['close'].rolling(fast).mean()
        self.df[f'ma_slow'] = self.df['close'].rolling(slow).mean()

        positions = np.where(self.df['ma_fast'] > self.df['ma_slow'], 1, -1)
        positions = pd.Series(positions).shift(1).fillna(0).values

        strategy_returns = positions * self.df['returns']
        equity = (1 + strategy_returns).cumprod()

        return {
            'name': f'MA Crossover ({fast}/{slow})',
            'equity': equity.values,
            'total_return': equity.iloc[-1] - 1,
            'sharpe': self._compute_sharpe(strategy_returns),
            'max_dd': self._compute_max_dd(equity.values),
            'trades': sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1]),
        }

    def _rsi_strategy(self, period: int = 14, overbought: int = 70, oversold: int = 30) -> Dict:
        """RSI mean-reversion strategy"""
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        positions = np.zeros(len(self.df))
        positions[rsi < oversold] = 1   # Buy when oversold
        positions[rsi > overbought] = -1 # Sell when overbought
        positions = pd.Series(positions).shift(1).fillna(0).values

        strategy_returns = positions * self.df['returns']
        equity = (1 + strategy_returns).cumprod()

        return {
            'name': f'RSI Strategy ({period})',
            'equity': equity.values,
            'total_return': equity.iloc[-1] - 1,
            'sharpe': self._compute_sharpe(strategy_returns),
            'max_dd': self._compute_max_dd(equity.values),
            'trades': sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1]),
        }

    def _bollinger_bands(self, period: int = 20, std_dev: int = 2) -> Dict:
        """Bollinger Bands mean-reversion strategy"""
        sma = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std

        positions = np.zeros(len(self.df))
        positions[self.df['close'] < lower] = 1   # Buy at lower band
        positions[self.df['close'] > upper] = -1  # Sell at upper band
        positions = pd.Series(positions).shift(1).fillna(0).values

        strategy_returns = positions * self.df['returns']
        equity = (1 + strategy_returns).cumprod()

        return {
            'name': f'Bollinger Bands ({period},{std_dev})',
            'equity': equity.values,
            'total_return': equity.iloc[-1] - 1,
            'sharpe': self._compute_sharpe(strategy_returns),
            'max_dd': self._compute_max_dd(equity.values),
            'trades': sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1]),
        }

    def _compute_sharpe(self, returns: pd.Series) -> float:
        """Compute annualized Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252 * 24)

    def _compute_max_dd(self, equity: np.ndarray) -> float:
        """Compute maximum drawdown"""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return np.max(drawdown)

    def add_agent_result(self, name: str, equity: np.ndarray, trades: int):
        """Add agent result for comparison"""
        returns = np.diff(equity) / equity[:-1]
        self.results[name] = {
            'name': name,
            'equity': equity,
            'total_return': equity[-1] - 1,
            'sharpe': self._compute_sharpe(pd.Series(returns)),
            'max_dd': self._compute_max_dd(equity),
            'trades': trades,
        }

    def plot_comparison(self, save_path: str = None):
        """Plot all equity curves"""
        plt.figure(figsize=(15, 10))

        for name, result in self.results.items():
            plt.plot(result['equity'], label=result['name'], alpha=0.8)

        plt.legend()
        plt.title('Strategy Comparison - Equity Curves')
        plt.xlabel('Time')
        plt.ylabel('Equity (start=1.0)')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print comparison summary"""
        logger.info("" + "=" * 80)
        logger.info("📊 BASELINE COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Trades':>8}")
        logger.info("-" * 80)

        for name, result in sorted(self.results.items(), 
                                  key=lambda x: x[1]['total_return'], reverse=True):
            logger.info(
                f"{result['name']:<30} "
                f"{result['total_return']:>9.2%} "
                f"{result['sharpe']:>7.2f} "
                f"{result['max_dd']:>7.2%} "
                f"{result['trades']:>7d}"
            )
