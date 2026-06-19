"""
QuantumFlow AI Trading System v2.0 - Realistic Execution Model
Models slippage, spread widening, market impact, adverse selection
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RealisticExecutionModel:
    """
    Model all real-world execution costs

    The gap between backtest and live trading often comes from:
    1. Slippage (0.1-0.5 pips typically)
    2. Spread widening (1.5-3x during volatility)
    3. Market impact (for larger positions)
    4. Adverse selection
    """

    def __init__(self, config=None):
        if config is None:
            config = self.get_default_config()

        self.base_spread = config.get('base_spread', 0.0003)
        self.base_slippage = config.get('base_slippage', 0.0001)
        self.commission = config.get('commission', 0.00005)
        self.spread_vol_multiplier = config.get('spread_vol_multiplier', 2.0)
        self.slippage_vol_multiplier = config.get('slippage_vol_multiplier', 3.0)
        self.market_impact_coefficient = config.get('market_impact_coef', 0.01)
        self.adverse_selection_cost = config.get('adverse_selection', 0.00005)

        self.total_trades = 0
        self.total_costs = 0.0
        self.cost_breakdown = {'spread': 0.0, 'slippage': 0.0, 'commission': 0.0, 
                               'market_impact': 0.0, 'adverse_selection': 0.0}

        logger.info("⚖️ Realistic Execution Model initialized")
        logger.info(f"  Base Spread: {self.base_spread:.5f} ({self.base_spread * 10000:.1f} pips)")
        logger.info(f"  Base Slippage: {self.base_slippage:.5f}")
        logger.info(f"  Commission: {self.commission:.5f}")

    @staticmethod
    def get_default_config():
        return {
            'base_spread': 0.0003,
            'base_slippage': 0.0001,
            'commission': 0.00005,
            'spread_vol_multiplier': 2.0,
            'slippage_vol_multiplier': 3.0,
            'market_impact_coef': 0.01,
            'adverse_selection': 0.00005,
        }

    def estimate_execution_cost(self, order, market_state):
        """Estimate total cost of executing an order"""
        costs = {}

        # 1. Spread Cost
        current_spread = market_state.get('spread', self.base_spread)
        vol_ratio = self._get_volatility_ratio(market_state)
        if vol_ratio > 1.5:
            current_spread *= min(vol_ratio, self.spread_vol_multiplier)
        if market_state.get('is_event_window', False):
            current_spread *= 2.0
        costs['spread'] = current_spread

        # 2. Slippage
        slippage = self.base_slippage
        if vol_ratio > 1.5:
            slippage *= min(vol_ratio, self.slippage_vol_multiplier)
        if order.get('order_type', 'market') == 'market':
            slippage *= 1.5
        if market_state.get('is_event_window', False):
            slippage *= 3.0
        costs['slippage'] = slippage

        # 3. Commission
        costs['commission'] = self.commission

        # 4. Market Impact
        position_size = order.get('size', 0.05)
        avg_liquidity = market_state.get('liquidity', 1.0)
        if position_size > avg_liquidity:
            costs['market_impact'] = (position_size / avg_liquidity) * self.market_impact_coefficient
        else:
            costs['market_impact'] = 0.0

        # 5. Adverse Selection
        costs['adverse_selection'] = self.adverse_selection_cost

        total_cost = sum(costs.values())

        self.total_trades += 1
        self.total_costs += total_cost
        for key, value in costs.items():
            self.cost_breakdown[key] += value

        return total_cost, costs

    def execute_trade(self, order, market_state, entry_price):
        """Execute trade with realistic costs"""
        total_cost, cost_breakdown = self.estimate_execution_cost(order, market_state)
        side = order.get('side', 'buy')

        if side in ['buy', 'long']:
            fill_price = entry_price * (1 + total_cost)
        else:
            fill_price = entry_price * (1 - total_cost)

        return fill_price, total_cost, cost_breakdown

    def _get_volatility_ratio(self, market_state):
        current_vol = market_state.get('volatility', 1.0)
        normal_vol = market_state.get('normal_volatility', 1.0)
        return current_vol / normal_vol if normal_vol > 0 else 1.0

    def get_statistics(self):
        if self.total_trades == 0:
            return {'total_trades': 0, 'avg_cost_per_trade': 0.0, 'total_costs': 0.0, 
                    'cost_breakdown': self.cost_breakdown}
        avg_cost = self.total_costs / self.total_trades
        return {
            'total_trades': self.total_trades,
            'avg_cost_per_trade': avg_cost,
            'total_costs': self.total_costs,
            'cost_breakdown': {k: v / self.total_trades for k, v in self.cost_breakdown.items()},
            'cost_in_pips': avg_cost * 10000,
        }

class SlippageSimulator:
    """Simple slippage simulator"""

    def __init__(self, avg_slippage=0.0001, volatility_scaling=True):
        self.avg_slippage = avg_slippage
        self.volatility_scaling = volatility_scaling

    def get_slippage(self, market_state):
        slippage = self.avg_slippage
        if self.volatility_scaling:
            vol_ratio = market_state.get('volatility', 1.0)
            slippage *= vol_ratio
        random_component = np.random.normal(0, slippage * 0.5)
        slippage += random_component
        if np.random.random() < 0.8:
            slippage = -abs(slippage)
        else:
            slippage = abs(slippage) * 0.5
        return slippage
