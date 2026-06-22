"""
QuantumFlow AI Trading System v2.0 - Crisis Period Validation
Test agent on known market crises before live trading
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CrisisValidator:
    """
    Test agent on known crisis periods:
    - 2008 Financial Crisis
    - 2010 Flash Crash
    - 2015 Yuan Devaluation
    - 2020 COVID Crash
    - 2022 Rate Hike Regime
    - 2023 SVB Collapse
    """

    CRISIS_PERIODS = {
        'financial_crisis_2008': {
            'start': '2008-09-01', 'end': '2009-03-01',
            'description': '2008 Financial Crisis',
            'expected_behavior': 'Reduce positions, flight to safety',
            'severity': 'EXTREME', 'typical_gold_move': '+$300',
        },
        'flash_crash_2010': {
            'start': '2010-05-01', 'end': '2010-06-01',
            'description': '2010 Flash Crash',
            'expected_behavior': 'Avoid high volatility',
            'severity': 'HIGH', 'typical_gold_move': '+$50',
        },
        'yuan_deval_2015': {
            'start': '2015-08-01', 'end': '2015-09-01',
            'description': '2015 Yuan Devaluation',
            'expected_behavior': 'Safe haven trade',
            'severity': 'MEDIUM', 'typical_gold_move': '+$40',
        },
        'covid_crash_2020': {
            'start': '2020-02-15', 'end': '2020-04-15',
            'description': 'COVID-19 Market Crash',
            'expected_behavior': 'Reduce positions, avoid falling knives',
            'severity': 'EXTREME', 'typical_gold_move': '+$200',
        },
        'ukraine_invasion_2022': {
            'start': '2022-02-24', 'end': '2022-03-31',
            'description': 'Russia-Ukraine War',
            'expected_behavior': 'Flight to safety, gold rally',
            'severity': 'HIGH', 'typical_gold_move': '+$100',
        },
        'rate_hikes_2022': {
            'start': '2022-01-01', 'end': '2022-12-31',
            'description': 'Fed Aggressive Rate Hikes',
            'expected_behavior': 'Navigate high volatility, strong USD',
            'severity': 'HIGH', 'typical_gold_move': '-$200',
        },
        'svb_collapse_2023': {
            'start': '2023-03-08', 'end': '2023-03-20',
            'description': 'Silicon Valley Bank Collapse',
            'expected_behavior': 'Safe haven trade to Gold',
            'severity': 'MEDIUM', 'typical_gold_move': '+$50',
        },
    }

    def __init__(self, data_path: str = 'data/xauusd_1h.csv'):
        self.data_path = data_path
        self.data = None

        if Path(data_path).exists():
            self.load_data()
        else:
            logger.warning(f"⚠️ Data file not found: {data_path}")

    def load_data(self):
        """Load historical data"""
        logger.info(f"📊 Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        if 'time' in self.data.columns:
            self.data['time'] = pd.to_datetime(self.data['time'])
        logger.info(f"✅ Loaded {len(self.data)} bars")

    def validate_all_crises(self, agent, env, verbose: bool = True) -> Dict:
        """Run agent on all crisis periods"""
        if self.data is None:
            logger.error("❌ No data loaded")
            return {}

        results = {}
        logger.info("🔥 Starting Crisis Period Validation")
        logger.info("=" * 60)

        for crisis_name, period in self.CRISIS_PERIODS.items():
            logger.info(f"📊 Testing: {period['description']}")
            logger.info(f"   Period: {period['start']} to {period['end']}")
            logger.info(f"   Severity: {period['severity']}")

            crisis_data = self.data[
                (self.data['time'] >= period['start']) &
                (self.data['time'] <= period['end'])
            ]

            if len(crisis_data) == 0:
                logger.warning(f"   ⚠️ No data - SKIPPED")
                results[crisis_name] = {'skipped': True}
                continue

            result = self._run_crisis_test(agent, env, crisis_data, period)
            results[crisis_name] = result

            if verbose:
                self._print_crisis_result(crisis_name, result, period)

        if verbose:
            self._print_overall_summary(results)

        return results

    def _run_crisis_test(self, agent, env, crisis_data, period) -> Dict:
        """Run single crisis test"""
        obs, _ = env.reset()
        done = False

        equity_curve = [1.0]
        trades = []
        position = 0

        for idx, row in crisis_data.iterrows():
            if hasattr(agent, 'get_action'):
                action, _ = agent.get_action(obs)
            else:
                action = agent.act(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if 'equity' in info:
                equity_curve.append(info['equity'])

            if done:
                break

        return self._analyze_performance(equity_curve, trades, period)

    def _analyze_performance(self, equity_curve, trades, period) -> Dict:
        """Analyze crisis performance"""
        equity = np.array(equity_curve)
        final_equity = equity[-1] if len(equity) > 0 else 1.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # Returns
        returns = np.diff(equity) / equity[:-1]
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0

        # Passing criteria
        passed_survival = final_equity > 0.7
        passed_drawdown = max_dd < 0.30
        passed_sharpe = sharpe > -1.0
        passed_overtrading = len(trades) < 200

        passed_all = all([passed_survival, passed_drawdown, passed_sharpe, passed_overtrading])

        return {
            'passed': passed_all,
            'final_equity': final_equity,
            'return_pct': (final_equity - 1.0) * 100,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades),
            'survived': passed_survival,
            'drawdown_ok': passed_drawdown,
            'sharpe_ok': passed_sharpe,
            'trading_ok': passed_overtrading,
            'severity': period['severity'],
            'description': period['description'],
        }

    def _print_crisis_result(self, crisis_name, result, period):
        """Print formatted crisis result"""
        if result.get('skipped'):
            logger.warning(f"   ⚠️ SKIPPED")
            return

        passed = result['passed']
        emoji = "✅" if passed else "❌"

        logger.info(f"{emoji} Result: {'PASSED' if passed else 'FAILED'}")
        logger.info(f"      Final Equity: {result['final_equity']:.4f} ({result['return_pct']:+.2f}%)")
        logger.info(f"      Max Drawdown: {result['max_drawdown']:.2%}")
        logger.info(f"      Sharpe Ratio: {result['sharpe_ratio']:.2f}")

        if not passed:
            logger.warning("      ⚠️ Failure reasons:")
            if not result['survived']:
                logger.warning(f"         - Final equity too low")
            if not result['drawdown_ok']:
                logger.warning(f"         - Drawdown too large")

    def _print_overall_summary(self, results):
        """Print overall summary"""
        total = len([r for r in results.values() if not r.get('skipped')])
        passed = len([r for r in results.values() if r.get('passed')])

        if total == 0:
            return

        rate = passed / total

        logger.info("" + "=" * 60)
        logger.info("📊 OVERALL CRISIS VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Run: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Pass Rate: {rate:.1%}")

        if rate == 1.0:
            logger.info("🎉 PERFECT SCORE!")
        elif rate >= 0.75:
            logger.info("✅ GOOD - Agent is reasonably robust")
        elif rate >= 0.50:
            logger.warning("⚠️ MARGINAL - Needs improvement")
        else:
            logger.error("❌ POOR - Not ready for live trading")
