"""
QuantumFlow AI Trading System v2.0 - Telegram Alerts
Real-time notifications for trading events
"""
import logging
import requests
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class TelegramAlerter:
    """
    Send trading alerts via Telegram Bot

    Features:
    - Trade execution notifications
    - Risk alerts
    - Daily summary
    - Emergency shutdown alerts
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bot_token is not None and chat_id is not None

        if self.enabled:
            logger.info("📱 Telegram alerts enabled")
        else:
            logger.info("📱 Telegram alerts disabled (set bot_token and chat_id)")

    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message via Telegram"""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_trade_alert(self, symbol: str, side: str, volume: float, price: float, pnl: float = 0):
        """Send trade execution alert"""
        emoji = "🟢" if side == 'buy' or side == 'long' else "🔴"
        message = f"""
{emoji} <b>TRADE EXECUTED</b>

Symbol: <code>{symbol}</code>
Side: <b>{side.upper()}</b>
Volume: <code>{volume:.2f}</code>
Price: <code>{price:.5f}</code>
PnL: <code>{pnl:+.2f}</code>
Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_message(message)

    def send_risk_alert(self, alert_type: str, details: str, severity: str = "WARNING"):
        """Send risk management alert"""
        emoji = "🚨" if severity == "CRITICAL" else "⚠️"
        message = f"""
{emoji} <b>RISK ALERT - {severity}</b>

Type: <code>{alert_type}</code>
Details: {details}
Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_message(message)

    def send_daily_summary(self, metrics: Dict):
        """Send daily trading summary"""
        message = f"""
📊 <b>DAILY TRADING SUMMARY</b>

Date: <code>{datetime.now().strftime('%Y-%m-%d')}</code>
Equity: <code>${metrics.get('equity', 0):,.2f}</code>
Daily P&L: <code>{metrics.get('daily_pnl', 0):+.2f}</code>
Total Trades: <code>{metrics.get('total_trades', 0)}</code>
Win Rate: <code>{metrics.get('win_rate', 0):.1%}</code>
Max Drawdown: <code>{metrics.get('max_drawdown', 0):.2%}</code>
Sharpe Ratio: <code>{metrics.get('sharpe_ratio', 0):.2f}</code>
"""
        return self.send_message(message)

    def send_emergency_shutdown(self, reason: str):
        """Send emergency shutdown alert"""
        message = f"""
🚨🚨🚨 <b>EMERGENCY SHUTDOWN</b> 🚨🚨🚨

Reason: <code>{reason}</code>
Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>

All trading has been HALTED.
Manual restart required.
"""
        return self.send_message(message)
