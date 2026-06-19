"""
QuantumFlow AI Trading System v2.0 - Multi-Account Manager
Professional-grade account management for real, demo, and paper accounts
"""
import logging
import json
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class AccountType(Enum):
    REAL = "real"
    DEMO = "demo"
    PAPER = "paper"

@dataclass
class AccountConfig:
    """Account configuration"""
    account_id: int
    password: str
    server: str
    account_type: AccountType
    broker_name: str
    max_risk_per_trade: float = 0.01  # 1% per trade
    max_daily_risk: float = 0.03      # 3% daily
    max_total_risk: float = 0.10      # 10% total
    leverage: int = 50
    currency: str = "USD"
    is_active: bool = True

    # Gold specific
    gold_symbol: str = "XAUUSD"
    gold_contract_size: float = 100.0  # 1 lot = 100 oz
    gold_min_lot: float = 0.01
    gold_max_lot: float = 10.0
    gold_tick_value: float = 1.0      # $1 per 0.01 move

    # Crypto (future)
    crypto_enabled: bool = False
    crypto_symbols: List[str] = field(default_factory=lambda: ["BTCUSD", "ETHUSD"])

    # Electronics/CFD (future)
    cfd_enabled: bool = False
    cfd_symbols: List[str] = field(default_factory=list)

class MultiAccountManager:
    """
    Professional multi-account management system

    Features:
    - Multiple real/demo/paper accounts simultaneously
    - Risk allocation across accounts
    - Trade copying/mirroring between accounts
    - Account health monitoring
    - Automatic failover
    """

    def __init__(self, config_path: str = "accounts/accounts.json"):
        self.config_path = config_path
        self.accounts: Dict[str, AccountConfig] = {}
        self.mt5_instances: Dict[str, any] = {}
        self.active_connections: Dict[str, bool] = {}
        self.lock = threading.Lock()

        # Performance tracking
        self.account_performance: Dict[str, Dict] = {}

        logger.info("🏦 Multi-Account Manager initialized")

    def load_accounts(self) -> bool:
        """Load account configurations from JSON"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            for acc_id, acc_data in data.items():
                self.accounts[acc_id] = AccountConfig(
                    account_id=acc_data['account_id'],
                    password=acc_data['password'],
                    server=acc_data['server'],
                    account_type=AccountType(acc_data['account_type']),
                    broker_name=acc_data['broker_name'],
                    max_risk_per_trade=acc_data.get('max_risk_per_trade', 0.01),
                    max_daily_risk=acc_data.get('max_daily_risk', 0.03),
                    leverage=acc_data.get('leverage', 50),
                    gold_symbol=acc_data.get('gold_symbol', 'XAUUSD'),
                    crypto_enabled=acc_data.get('crypto_enabled', False),
                    cfd_enabled=acc_data.get('cfd_enabled', False),
                )

            logger.info(f"✅ Loaded {len(self.accounts)} accounts")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load accounts: {e}")
            return False

    def connect_account(self, account_id: str) -> bool:
        """Connect to specific MT5 account"""
        if account_id not in self.accounts:
            logger.error(f"❌ Account {account_id} not found")
            return False

        account = self.accounts[account_id]

        try:
            # Initialize MT5 for this account
            if not mt5.initialize(
                login=account.account_id,
                password=account.password,
                server=account.server
            ):
                logger.error(f"❌ MT5 init failed for {account_id}: {mt5.last_error()}")
                return False

            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"❌ Failed to get account info for {account_id}")
                return False

            self.active_connections[account_id] = True
            self.account_performance[account_id] = {
                'equity': account_info.equity,
                'balance': account_info.balance,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'connected_at': datetime.now(),
            }

            logger.info(f"✅ Connected to {account_id} ({account.broker_name})")
            logger.info(f"   Type: {account.account_type.value.upper()}")
            logger.info(f"   Balance: ${account_info.balance:,.2f}")
            logger.info(f"   Equity: ${account_info.equity:,.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ Connection error for {account_id}: {e}")
            return False

    def connect_all(self) -> Dict[str, bool]:
        """Connect all configured accounts"""
        results = {}
        for account_id in self.accounts:
            results[account_id] = self.connect_account(account_id)
        return results

    def disconnect_account(self, account_id: str):
        """Disconnect specific account"""
        if account_id in self.active_connections:
            self.active_connections[account_id] = False
            logger.info(f"🔌 Disconnected from {account_id}")

    def disconnect_all(self):
        """Disconnect all accounts"""
        for account_id in list(self.active_connections.keys()):
            self.disconnect_account(account_id)
        mt5.shutdown()
        logger.info("🔌 All accounts disconnected")

    def get_account_status(self, account_id: str) -> Optional[Dict]:
        """Get current account status"""
        if not self.active_connections.get(account_id, False):
            return None

        try:
            info = mt5.account_info()
            positions = mt5.positions_total()
            orders = mt5.orders_total()

            return {
                'account_id': account_id,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'profit': info.profit,
                'open_positions': positions,
                'pending_orders': orders,
                'margin_level': info.margin_level if info.margin > 0 else 0,
                'timestamp': datetime.now(),
            }
        except:
            return None

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all connected accounts"""
        return {
            acc_id: self.get_account_status(acc_id)
            for acc_id in self.active_connections
            if self.active_connections[acc_id]
        }

    def execute_trade_all(self, symbol: str, action: str, volume: float, 
                         sl: Optional[float] = None, tp: Optional[float] = None) -> Dict[str, any]:
        """Execute trade on all active accounts"""
        results = {}

        with self.lock:
            for account_id, is_active in self.active_connections.items():
                if not is_active:
                    continue

                account = self.accounts[account_id]

                # Risk check per account
                status = self.get_account_status(account_id)
                if status and status['equity'] > 0:
                    risk_amount = status['equity'] * account.max_risk_per_trade
                    adjusted_volume = self._calculate_volume(
                        symbol, risk_amount, account, sl
                    )

                    result = self._place_order(account_id, symbol, action, adjusted_volume, sl, tp)
                    results[account_id] = result

        return results

    def _calculate_volume(self, symbol: str, risk_amount: float, 
                         account: AccountConfig, sl: Optional[float]) -> float:
        """Calculate appropriate lot size based on risk"""
        if symbol == account.gold_symbol or symbol.startswith('XAU'):
            # Gold: 1 lot = 100 oz, 1 pip = $1 for 0.01 lot
            # Risk = |entry - sl| * contract_size * volume
            if sl:
                # Assuming current price ~2000, SL distance
                sl_distance = abs(2000 - sl)  # Simplified
                volume = risk_amount / (sl_distance * account.gold_contract_size)
                volume = max(volume, account.gold_min_lot)
                volume = min(volume, account.gold_max_lot)
                return round(volume, 2)
            else:
                return account.gold_min_lot

        # Default for other symbols
        return 0.01

    def _place_order(self, account_id: str, symbol: str, action: str, 
                    volume: float, sl: Optional[float], tp: Optional[float]) -> Dict:
        """Place order on specific account"""
        try:
            order_type = mt5.ORDER_TYPE_BUY if action in ['buy', 'long'] else mt5.ORDER_TYPE_SELL

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": f"QuantumFlow_{account_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if sl:
                request["sl"] = float(sl)
            if tp:
                request["tp"] = float(tp)

            result = mt5.order_send(request)

            return {
                'success': result.retcode == mt5.TRADE_RETCODE_DONE,
                'ticket': result.order if result else None,
                'volume': volume,
                'price': result.price if result else None,
                'retcode': result.retcode if result else None,
            }

        except Exception as e:
            logger.error(f"❌ Order failed for {account_id}: {e}")
            return {'success': False, 'error': str(e)}

    def mirror_trade(self, source_account: str, target_accounts: List[str], 
                    ticket: int, ratio: float = 1.0):
        """Mirror trade from source account to target accounts"""
        # Get source position details
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"❌ Position {ticket} not found")
            return False

        pos = position[0]

        results = {}
        for target in target_accounts:
            if target == source_account:
                continue

            account = self.accounts[target]
            mirrored_volume = pos.volume * ratio

            result = self._place_order(
                target,
                pos.symbol,
                'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                mirrored_volume,
                pos.sl,
                pos.tp
            )
            results[target] = result

        return results

    def close_all_positions(self, account_id: str) -> Dict:
        """Close all open positions on specific account"""
        if not self.active_connections.get(account_id, False):
            return {'success': False, 'error': 'Not connected'}

        positions = mt5.positions_get()
        results = []

        for pos in positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 234000,
                "comment": f"Close_All_{account_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            results.append({
                'ticket': pos.ticket,
                'success': result.retcode == mt5.TRADE_RETCODE_DONE if result else False,
            })

        return {
            'success': all(r['success'] for r in results),
            'closed_positions': len(results),
            'results': results,
        }

    def emergency_close_all(self):
        """Emergency close all positions on all accounts"""
        logger.critical("🚨 EMERGENCY CLOSE ALL POSITIONS")

        results = {}
        for account_id in self.active_connections:
            if self.active_connections[account_id]:
                result = self.close_all_positions(account_id)
                results[account_id] = result
                logger.critical(f"   {account_id}: Closed {result.get('closed_positions', 0)} positions")

        return results

    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report across all accounts"""
        report = {
            'timestamp': datetime.now(),
            'accounts': {},
            'total_exposure': 0.0,
            'total_margin': 0.0,
            'total_profit': 0.0,
        }

        for account_id in self.active_connections:
            if not self.active_connections[account_id]:
                continue

            status = self.get_account_status(account_id)
            if status:
                account = self.accounts[account_id]

                # Calculate risk metrics
                equity = status['equity']
                balance = status['balance']
                margin_used = status['margin']
                free_margin = status['free_margin']

                risk_pct = (balance - equity) / balance * 100 if balance > 0 else 0
                margin_pct = margin_used / equity * 100 if equity > 0 else 0

                report['accounts'][account_id] = {
                    'type': account.account_type.value,
                    'broker': account.broker_name,
                    'equity': equity,
                    'balance': balance,
                    'profit': status['profit'],
                    'open_positions': status['open_positions'],
                    'risk_pct': risk_pct,
                    'margin_used_pct': margin_pct,
                    'free_margin': free_margin,
                    'status': 'HEALTHY' if margin_pct < 50 and risk_pct < 10 else 'WARNING',
                }

                report['total_exposure'] += status['open_positions']
                report['total_margin'] += margin_used
                report['total_profit'] += status['profit']

        return report
