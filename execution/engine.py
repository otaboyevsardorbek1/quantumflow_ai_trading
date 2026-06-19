"""
QuantumFlow AI Trading System v2.0 - Advanced Execution Engine
"""
import time
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_distance: Optional[float] = None
    slippage_tolerance: float = 0.0001
    time_in_force: str = "GTC"
    magic_number: int = 0
    comment: str = ""

@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_volume: Optional[float] = None
    slippage: float = 0.0
    commission: float = 0.0
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

class ExecutionEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.platform = config.get('platform', 'mt5')
        self.default_slippage = config.get('slippage_tolerance', 0.0001)
        self.max_retries = config.get('retry_attempts', 3)
        self.execution_timeout = config.get('execution_timeout', 30)
        self.active_orders = {}
        self.order_history = []
        self.total_executions = 0
        self.successful_executions = 0
        self.avg_slippage = 0.0
        self.avg_execution_time = 0.0
        logger.info(f"🔌 Execution Engine initialized: {self.platform}")

    def execute_order(self, order: Order, market_data: Dict) -> ExecutionResult:
        start_time = time.time()
        if not self._pre_execution_checks(order, market_data):
            return ExecutionResult(success=False, error_message="Pre-execution checks failed")

        if order.order_type == OrderType.MARKET:
            result = self._execute_market_order(order, market_data)
        elif order.order_type == OrderType.LIMIT:
            result = self._execute_limit_order(order, market_data)
        elif order.order_type == OrderType.STOP:
            result = self._execute_stop_order(order, market_data)
        elif order.order_type == OrderType.TRAILING_STOP:
            result = self._execute_trailing_stop(order, market_data)
        else:
            result = ExecutionResult(success=False, error_message="Unsupported order type")

        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time

        if result.success:
            self.successful_executions += 1
            self.avg_slippage = (self.avg_slippage * self.total_executions + result.slippage) / (self.total_executions + 1)
            self.avg_execution_time = (self.avg_execution_time * self.total_executions + execution_time) / (self.total_executions + 1)

        self.total_executions += 1
        self.order_history.append({'order': order, 'result': result, 'market_data': market_data})
        return result

    def _pre_execution_checks(self, order: Order, market_data: Dict) -> bool:
        spread = market_data.get('spread', 0)
        if spread > self.config.get('max_spread', 0.0005):
            logger.warning(f"Spread too wide: {spread}")
            return False
        current_price = market_data.get('price', 0)
        if current_price <= 0:
            return False
        if order.volume <= 0:
            return False
        return True

    def _execute_market_order(self, order: Order, market_data: Dict) -> ExecutionResult:
        current_price = market_data.get('price', 0)
        spread = market_data.get('spread', 0)
        estimated_slippage = self._estimate_slippage(order, market_data)

        if order.side == OrderSide.BUY:
            executed_price = current_price + spread + estimated_slippage
        else:
            executed_price = current_price - estimated_slippage

        if abs(executed_price - current_price) / current_price > order.slippage_tolerance:
            return ExecutionResult(success=False, error_message=f"Slippage exceeded tolerance")

        commission = order.volume * current_price * self.config.get('commission', 0.0001)
        return ExecutionResult(success=True, executed_price=executed_price, executed_volume=order.volume,
                               slippage=estimated_slippage, commission=commission)

    def _execute_limit_order(self, order: Order, market_data: Dict) -> ExecutionResult:
        if order.price is None:
            return ExecutionResult(success=False, error_message="Limit price not specified")
        current_price = market_data.get('price', 0)
        if order.side == OrderSide.BUY and order.price >= current_price:
            return self._execute_market_order(order, market_data)
        elif order.side == OrderSide.SELL and order.price <= current_price:
            return self._execute_market_order(order, market_data)
        return ExecutionResult(success=True, executed_price=order.price, executed_volume=order.volume,
                               slippage=0.0, commission=order.volume * order.price * self.config.get('commission', 0.0001))

    def _execute_stop_order(self, order: Order, market_data: Dict) -> ExecutionResult:
        if order.stop_price is None:
            return ExecutionResult(success=False, error_message="Stop price not specified")
        current_price = market_data.get('price', 0)
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            return self._execute_market_order(order, market_data)
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            return self._execute_market_order(order, market_data)
        return ExecutionResult(success=True, executed_price=None, executed_volume=0.0)

    def _execute_trailing_stop(self, order: Order, market_data: Dict) -> ExecutionResult:
        if order.trailing_distance is None:
            return ExecutionResult(success=False, error_message="Trailing distance not specified")
        return ExecutionResult(success=True, executed_price=None, executed_volume=0.0)

    def _estimate_slippage(self, order: Order, market_data: Dict) -> float:
        base_slippage = self.default_slippage
        volatility = market_data.get('volatility', 0.01)
        vol_adjustment = 1 + volatility * 10
        avg_volume = market_data.get('avg_volume', 1)
        volume_ratio = order.volume / avg_volume if avg_volume > 0 else 1
        volume_adjustment = 1 + volume_ratio * 0.5
        spread = market_data.get('spread', 0)
        spread_adjustment = 1 + spread * 100
        estimated_slippage = base_slippage * vol_adjustment * volume_adjustment * spread_adjustment
        return min(estimated_slippage, 0.01)

    def get_execution_stats(self) -> Dict:
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.successful_executions / max(self.total_executions, 1),
            'avg_slippage': self.avg_slippage,
            'avg_execution_time_ms': self.avg_execution_time,
            'active_orders': len(self.active_orders),
        }

class MT5ExecutionAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.mt5 = None
        self.connected = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            self.connected = True
            logger.info("✅ Connected to MT5")
            return True
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False

    def disconnect(self):
        if self.mt5:
            self.mt5.shutdown()
            self.connected = False

    def get_positions(self, symbol: str) -> List[Dict]:
        if not self.connected:
            return []
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [{'ticket': pos.ticket, 'type': 'buy' if pos.type == self.mt5.ORDER_TYPE_BUY else 'sell',
                 'volume': pos.volume, 'open_price': pos.price_open, 'current_price': pos.price_current,
                 'profit': pos.profit} for pos in positions]
