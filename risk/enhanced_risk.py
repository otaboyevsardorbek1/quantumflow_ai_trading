"""
QuantumFlow AI Trading System v2.0 - Enhanced Risk Layer
Balansni himoya qilish uchun qo'shimcha tekshiruvlar
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import numpy  as np
logger = logging.getLogger(__name__)

class EnhancedRiskGuard:
    """
    Qo'shimcha xavfsizlik qatlami:
    - Maksimal kunlik yo'qotish (2%)
    - Maksimal umumiy yo'qotish (10%)
    - Ketma-ket 3 loss -> trading to'xtatiladi
    - High impact news vaqtida to'xtatiladi
    - Spread juda keng bo'lsa to'xtatiladi
    - Majburiy stop-loss
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_daily_loss_pct = self.config.get('max_daily_loss_pct', 0.02)   # 2%
        self.max_total_loss_pct = self.config.get('max_total_loss_pct', 0.10)   # 10%
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.max_spread_pips = self.config.get('max_spread_pips', 5.0)          # 5 pips
        self.require_stop_loss = self.config.get('require_stop_loss', True)

        # Holat
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.initial_equity = None
        self.current_equity = None
        self.consecutive_losses = 0
        self.trades_today = 0
        self.halt_until = None
        self.is_halted = False

        # High impact events (NFP, FOMC va boshqalar)
        self.high_impact_events = [
            'NFP', 'CPI', 'FOMC', 'GDP', 'Retail Sales',
            'ISM Manufacturing', 'PPI', 'Unemployment Rate',
            'Fed Chair Speech', 'Treasury Auction'
        ]

        logger.info("🛡️ Enhanced Risk Guard initialized")
        logger.info(f"   Max daily loss: {self.max_daily_loss_pct:.1%}")
        logger.info(f"   Max total loss: {self.max_total_loss_pct:.1%}")
        logger.info(f"   Max consecutive losses: {self.max_consecutive_losses}")

    def check_trade(self, action: np.ndarray, market_data: Dict, equity: float) -> Tuple[bool, str, Dict]:
        """
        Trade oldidan barcha xavfsizlik tekshiruvlari
        Returns: (approved, reason, adjustments)
        """
        adjustments = {}

        # 1. Initial equity ni saqlash
        if self.initial_equity is None:
            self.initial_equity = equity
        self.current_equity = equity

        # 2. Halt holati
        if self.is_halted:
            return False, "SYSTEM_HALTED", adjustments

        if self.halt_until and datetime.now() < self.halt_until:
            remaining = (self.halt_until - datetime.now()).total_seconds() / 60
            return False, f"HALTED_UNTIL_{remaining:.0f}min", adjustments

        # 3. Maksimal umumiy yo'qotish
        total_loss_pct = (self.initial_equity - self.current_equity) / self.initial_equity
        if total_loss_pct > self.max_total_loss_pct:
            self.is_halted = True
            self.halt_until = datetime.now() + timedelta(days=365)
            logger.critical(f"🚨 TOTAL LOSS LIMIT REACHED: {total_loss_pct:.2%} > {self.max_total_loss_pct:.2%}")
            return False, "MAX_TOTAL_LOSS", adjustments

        # 4. Kunlik yo'qotish
        daily_loss_pct = self.daily_pnl / self.initial_equity
        if daily_loss_pct < -self.max_daily_loss_pct:
            self.halt_until = datetime.now() + timedelta(hours=24)
            logger.critical(f"🚨 DAILY LOSS LIMIT REACHED: {daily_loss_pct:.2%} < -{self.max_daily_loss_pct:.2%}")
            return False, "MAX_DAILY_LOSS", adjustments

        # 5. Ketma-ket yo'qotishlar
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halt_until = datetime.now() + timedelta(hours=6)
            logger.warning(f"⚠️ CONSECUTIVE LOSSES: {self.consecutive_losses}")
            return False, "CONSECUTIVE_LOSSES", adjustments

        # 6. Spread tekshiruvi
        spread = market_data.get('spread', 0.0)
        if spread > self.max_spread_pips * 0.0001:  # pips ga o'tkazish
            logger.warning(f"⚠️ Spread too wide: {spread:.5f} > {self.max_spread_pips*0.0001:.5f}")
            return False, "SPREAD_TOO_WIDE", adjustments

        # 7. High impact event
        if self._is_high_impact_event(market_data):
            logger.warning("⚠️ High impact event detected, trading paused")
            self.halt_until = datetime.now() + timedelta(hours=2)
            return False, "HIGH_IMPACT_EVENT", adjustments

        # 8. Majburiy stop-loss
        if self.require_stop_loss and action[2] <= 0.01:
            logger.warning("⚠️ Stop-loss required but not set")
            adjustments['sl'] = 1.5  # Default SL 1.5% below entry
            # Agar action da SL yo'q bo'lsa, majburlab qo'yamiz

        # 9. Position size cheklovi
        position_size = abs(action[1])
        if position_size > 0.25:  # 25% dan ortiq bo'lmasin
            adjustments['position_scale'] = 0.25 / position_size
            logger.info(f"📉 Position size reduced: {position_size:.2f} -> {0.25:.2f}")

        return True, "APPROVED", adjustments

    def update_after_trade(self, pnl: float, trade_info: Dict):
        """Trade natijasini yangilaydi"""
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.trades_today += 1

        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        logger.debug(f"📊 Trade update: PnL={pnl:.2f}, Daily={self.daily_pnl:.2f}, Consecutive={self.consecutive_losses}")

    def reset_daily(self):
        """Kunlik statistikani tiklaydi"""
        logger.info(f"📊 Daily reset - PnL: {self.daily_pnl:.2f}, Trades: {self.trades_today}")
        self.daily_pnl = 0.0
        self.trades_today = 0
        if self.halt_until and datetime.now() >= self.halt_until:
            self.halt_until = None
            self.is_halted = False

    def _is_high_impact_event(self, market_data: Dict) -> bool:
        """High impact eventni aniqlaydi"""
        # Agar market_data da 'event' kaliti bo'lsa va u ro'yxatdagi eventlardan bo'lsa
        event = market_data.get('event', '')
        if event and event in self.high_impact_events:
            return True

        # Vaqt bo'yicha: NFP (birinchi juma 13:30 NY vaqti)
        now = datetime.now()
        if now.weekday() == 4 and now.hour == 13 and 30 <= now.minute < 35:
            return True

        # FOMC (taxminan har 6 haftada)
        # Soddalashtirilgan: har oyning 2-chorak payshanbasi
        if now.weekday() == 3 and 8 <= now.day <= 14 and now.hour == 14:
            return True

        return False

    def get_status(self) -> Dict:
        """Joriy holat"""
        return {
            'is_halted': self.is_halted,
            'halt_until': self.halt_until.isoformat() if self.halt_until else None,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'total_loss_pct': (self.initial_equity - self.current_equity) / self.initial_equity if self.initial_equity else 0
        }