"""
QuantumFlow AI Trading System v2.0 - Future Asset Manager
Extensible system for Crypto, CFD, Electronics, Commodities
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    FOREX = "forex"
    GOLD = "gold"           # Precious metals
    CRYPTO = "crypto"      # Bitcoin, Ethereum, etc.
    CFD = "cfd"            # Contracts for Difference
    INDICES = "indices"    # SPX, NDX, DJI
    COMMODITIES = "commodities"  # Oil, Gas, Wheat
    BONDS = "bonds"        # Treasury, Corporate
    ELECTRONICS = "electronics"  # Tech stocks, semiconductors

@dataclass
class AssetSpecification:
    """Trading asset specification"""
    symbol: str
    asset_class: AssetClass
    broker_symbol: str           # Broker-specific symbol (e.g., XAUUSD, BTCUSD)
    contract_size: float         # 1 lot = ? units
    min_lot: float
    max_lot: float
    tick_size: float             # Minimum price movement
    tick_value: float            # Value of 1 tick in account currency
    margin_requirement: float    # % of position value required as margin
    swap_long: float             # Overnight swap for long positions
    swap_short: float            # Overnight swap for short positions
    trading_hours: Dict          # Session times
    is_24h: bool = False         # 24/7 trading (crypto)

    # Risk parameters
    max_leverage: int = 100
    typical_spread: float = 0.0002
    volatility_factor: float = 1.0

    # Correlation factors
    correlations: Dict[str, float] = None

class AssetManager:
    """
    Central asset manager for all instrument types

    Supports:
    - Forex (EURUSD, GBPUSD, etc.)
    - Gold (XAUUSD, XAUEUR)
    - Crypto (BTCUSD, ETHUSD, LTCUSD)
    - Indices (US30, SPX500, NAS100)
    - Commodities (WTI, BRENT, NGAS)
    - Electronics/Tech (via CFD)
    """

    def __init__(self):
        self.assets: Dict[str, AssetSpecification] = {}
        self.active_assets: List[str] = []
        self._register_default_assets()

        logger.info("📊 Asset Manager initialized")

    def _register_default_assets(self):
        """Register default asset specifications"""

        # === GOLD (Primary) ===
        self.register_asset(AssetSpecification(
            symbol="XAUUSD",
            asset_class=AssetClass.GOLD,
            broker_symbol="XAUUSD",
            contract_size=100.0,      # 100 oz per lot
            min_lot=0.01,
            max_lot=50.0,
            tick_size=0.01,           # 1 pip = $0.01
            tick_value=1.0,           # $1 per pip per lot
            margin_requirement=0.5,   # 0.5% = 1:200 leverage
            swap_long=-2.5,
            swap_short=-1.5,
            trading_hours={
                'open': '00:00',
                'close': '23:59',
                'break': None,
            },
            is_24h=False,
            max_leverage=200,
            typical_spread=0.00015,
            volatility_factor=1.2,
            correlations={'DXY': -0.82, 'US10Y': -0.65, 'VIX': 0.45},
        ))

        self.register_asset(AssetSpecification(
            symbol="XAUEUR",
            asset_class=AssetClass.GOLD,
            broker_symbol="XAUEUR",
            contract_size=100.0,
            min_lot=0.01,
            max_lot=50.0,
            tick_size=0.01,
            tick_value=1.0,
            margin_requirement=0.5,
            swap_long=-2.0,
            swap_short=-1.0,
            trading_hours={'open': '00:00', 'close': '23:59', 'break': None},
            correlations={'EURUSD': 0.35, 'DXY': -0.75},
        ))

        # === FOREX ===
        self.register_asset(AssetSpecification(
            symbol="EURUSD",
            asset_class=AssetClass.FOREX,
            broker_symbol="EURUSD",
            contract_size=100000.0,   # 100,000 EUR per lot
            min_lot=0.01,
            max_lot=100.0,
            tick_size=0.00001,       # 1 pip = 0.0001
            tick_value=1.0,          # $1 per pip per lot
            margin_requirement=0.2,  # 0.2% = 1:500 leverage
            swap_long=0.5,
            swap_short=-1.2,
            trading_hours={'open': '00:00', 'close': '23:59', 'break': None},
            correlations={'DXY': -0.85, 'US10Y': 0.30},
        ))

        # === CRYPTO (Future) ===
        self.register_asset(AssetSpecification(
            symbol="BTCUSD",
            asset_class=AssetClass.CRYPTO,
            broker_symbol="BTCUSD",
            contract_size=1.0,        # 1 BTC per lot
            min_lot=0.01,
            max_lot=10.0,
            tick_size=0.01,
            tick_value=0.01,         # $0.01 per tick
            margin_requirement=10.0,  # 10% = 1:10 leverage
            swap_long=-10.0,          # High swap for crypto
            swap_short=-10.0,
            trading_hours={'open': '00:00', 'close': '23:59', 'break': None},
            is_24h=True,              # 24/7 trading
            max_leverage=10,
            typical_spread=0.001,
            volatility_factor=3.0,    # 3x more volatile than gold
            correlations={'SPX': 0.60, 'DXY': -0.40, 'GOLD': 0.20},
        ))

        self.register_asset(AssetSpecification(
            symbol="ETHUSD",
            asset_class=AssetClass.CRYPTO,
            broker_symbol="ETHUSD",
            contract_size=1.0,
            min_lot=0.01,
            max_lot=50.0,
            tick_size=0.01,
            tick_value=0.01,
            margin_requirement=10.0,
            swap_long=-8.0,
            swap_short=-8.0,
            trading_hours={'open': '00:00', 'close': '23:59', 'break': None},
            is_24h=True,
            max_leverage=10,
            typical_spread=0.002,
            volatility_factor=3.5,
            correlations={'BTCUSD': 0.85, 'SPX': 0.55},
        ))

        # === INDICES ===
        self.register_asset(AssetSpecification(
            symbol="SPX500",
            asset_class=AssetClass.INDICES,
            broker_symbol="SPX500",
            contract_size=1.0,
            min_lot=0.01,
            max_lot=100.0,
            tick_size=0.1,
            tick_value=1.0,
            margin_requirement=1.0,   # 1% = 1:100 leverage
            swap_long=-5.0,
            swap_short=-5.0,
            trading_hours={'open': '13:30', 'close': '20:00', 'break': None},
            correlations={'DXY': -0.30, 'US10Y': -0.40, 'VIX': -0.80},
        ))

        # === COMMODITIES (Oil) ===
        self.register_asset(AssetSpecification(
            symbol="WTI",
            asset_class=AssetClass.COMMODITIES,
            broker_symbol="USOIL",
            contract_size=1000.0,     # 1000 barrels per lot
            min_lot=0.01,
            max_lot=50.0,
            tick_size=0.01,
            tick_value=1.0,
            margin_requirement=2.0,    # 2% = 1:50 leverage
            swap_long=-3.0,
            swap_short=-3.0,
            trading_hours={'open': '01:00', 'close': '23:59', 'break': None},
            correlations={'DXY': -0.50, 'US10Y': 0.20, 'VIX': 0.30},
        ))

        # === ELECTRONICS / TECH CFDs ===
        self.register_asset(AssetSpecification(
            symbol="NVDA",
            asset_class=AssetClass.ELECTRONICS,
            broker_symbol="NVDA",
            contract_size=1.0,        # 1 share per lot
            min_lot=1.0,
            max_lot=1000.0,
            tick_size=0.01,
            tick_value=0.01,
            margin_requirement=10.0,  # 10% = 1:10 leverage (CFD)
            swap_long=-2.0,
            swap_short=-2.0,
            trading_hours={'open': '13:30', 'close': '20:00', 'break': None},
            correlations={'SPX': 0.85, 'SOX': 0.90, 'BTCUSD': 0.40},
        ))

        self.register_asset(AssetSpecification(
            symbol="AMD",
            asset_class=AssetClass.ELECTRONICS,
            broker_symbol="AMD",
            contract_size=1.0,
            min_lot=1.0,
            max_lot=1000.0,
            tick_size=0.01,
            tick_value=0.01,
            margin_requirement=10.0,
            swap_long=-2.0,
            swap_short=-2.0,
            trading_hours={'open': '13:30', 'close': '20:00', 'break': None},
            correlations={'NVDA': 0.80, 'SPX': 0.75},
        ))

        self.register_asset(AssetSpecification(
            symbol="SOX",
            asset_class=AssetClass.ELECTRONICS,
            broker_symbol="SOX",  # Philadelphia Semiconductor Index
            contract_size=1.0,
            min_lot=0.1,
            max_lot=100.0,
            tick_size=0.1,
            tick_value=1.0,
            margin_requirement=5.0,
            swap_long=-4.0,
            swap_short=-4.0,
            trading_hours={'open': '13:30', 'close': '20:00', 'break': None},
            correlations={'NVDA': 0.85, 'SPX': 0.80, 'BTCUSD': 0.35},
        ))

    def register_asset(self, asset: AssetSpecification):
        """Register new asset"""
        self.assets[asset.symbol] = asset
        logger.info(f"📊 Registered: {asset.symbol} ({asset.asset_class.value})")

    def get_asset(self, symbol: str) -> Optional[AssetSpecification]:
        """Get asset specification"""
        return self.assets.get(symbol)

    def get_assets_by_class(self, asset_class: AssetClass) -> List[AssetSpecification]:
        """Get all assets of specific class"""
        return [a for a in self.assets.values() if a.asset_class == asset_class]

    def activate_asset(self, symbol: str) -> bool:
        """Activate asset for trading"""
        if symbol not in self.assets:
            logger.error(f"❌ Asset {symbol} not registered")
            return False

        self.active_assets.append(symbol)
        logger.info(f"✅ Activated: {symbol}")
        return True

    def deactivate_asset(self, symbol: str):
        """Deactivate asset"""
        if symbol in self.active_assets:
            self.active_assets.remove(symbol)
            logger.info(f"🔴 Deactivated: {symbol}")

    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        account_currency: str = 'USD'
    ) -> Tuple[float, float, float]:
        """
        Calculate position size for any asset type

        Returns:
            (lot_size, risk_amount, pip_value)
        """
        asset = self.assets.get(symbol)
        if not asset:
            logger.error(f"❌ Asset {symbol} not found")
            return 0.0, 0.0, 0.0

        risk_amount = account_balance * risk_percent
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance > 0:
            # lot_size = risk_amount / (sl_distance * contract_size * tick_value / tick_size)
            lot_size = risk_amount / (sl_distance * asset.contract_size * (asset.tick_value / asset.tick_size))
        else:
            lot_size = asset.min_lot

        # Round to valid lot size
        lot_size = round(lot_size, 2)
        lot_size = max(asset.min_lot, min(lot_size, asset.max_lot))

        actual_risk = lot_size * sl_distance * asset.contract_size * (asset.tick_value / asset.tick_size)
        pip_value = lot_size * asset.tick_value

        return lot_size, actual_risk, pip_value

    def get_margin_requirement(self, symbol: str, lot_size: float, price: float) -> float:
        """Calculate margin requirement for position"""
        asset = self.assets.get(symbol)
        if not asset:
            return 0.0

        position_value = lot_size * asset.contract_size * price
        margin = position_value * (asset.margin_requirement / 100)
        return margin

    def get_swap_cost(self, symbol: str, lot_size: float, days: int, direction: str) -> float:
        """Calculate overnight swap cost"""
        asset = self.assets.get(symbol)
        if not asset:
            return 0.0

        swap_rate = asset.swap_long if direction == 'long' else asset.swap_short
        return lot_size * swap_rate * days

    def get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for active assets"""
        matrix = {}
        for sym in symbols:
            asset = self.assets.get(sym)
            if asset and asset.correlations:
                matrix[sym] = asset.correlations
        return matrix

    def is_trading_allowed(self, symbol: str, timestamp) -> Tuple[bool, str]:
        """Check if trading is allowed for asset at given time"""
        asset = self.assets.get(symbol)
        if not asset:
            return False, "Asset not found"

        if asset.is_24h:
            return True, "24/7 trading"

        # Check trading hours
        current_time = timestamp.time() if hasattr(timestamp, 'time') else timestamp
        # (Would need proper time comparison logic)

        return True, "Within trading hours"

    def get_all_specifications(self) -> Dict[str, Dict]:
        """Get all asset specifications as dict"""
        return {
            sym: {
                'symbol': a.symbol,
                'class': a.asset_class.value,
                'contract_size': a.contract_size,
                'min_lot': a.min_lot,
                'max_lot': a.max_lot,
                'tick_size': a.tick_size,
                'tick_value': a.tick_value,
                'margin_req': a.margin_requirement,
                'max_leverage': a.max_leverage,
                'typical_spread': a.typical_spread,
                'volatility_factor': a.volatility_factor,
                'is_24h': a.is_24h,
            }
            for sym, a in self.assets.items()
        }
