"""
QuantumFlow AI Trading System v2.0 - MetaAPI Integration
Cloud-based trading without VPS
"""
import os
import json
import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetaAPIConfig:
    """MetaAPI configuration"""
    token: str
    account_id: str
    region: str = "eu"

    @classmethod
    def from_env(cls):
        """Load from environment variables"""
        return cls(
            token=os.getenv('METAAPI_TOKEN', ''),
            account_id=os.getenv('METAAPI_ACCOUNT_ID', ''),
            region=os.getenv('METAAPI_REGION', 'eu'),
        )

class MetaAPIClient:
    """
    MetaAPI cloud trading client
    Trade from anywhere without VPS
    """

    def __init__(self, config: MetaAPIConfig):
        self.config = config
        self.connected = False
        self.base_url = f"https://mt-client-api-v1.{config.region}.metaapi.cloud"

    def connect(self) -> bool:
        """Connect to MetaAPI"""
        try:
            import requests
            headers = {'auth-token': self.config.token}
            response = requests.get(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                self.connected = True
                logger.info("✅ Connected to MetaAPI")
                return True
            else:
                logger.error(f"❌ MetaAPI connection failed: {response.status_code}")
                return False

        except ImportError:
            logger.error("❌ requests package not installed")
            return False
        except Exception as e:
            logger.error(f"❌ MetaAPI error: {e}")
            return False

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            return None

        try:
            import requests
            headers = {'auth-token': self.config.token}
            response = requests.get(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}",
                headers=headers,
                timeout=10
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None

    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.connected:
            return []

        try:
            import requests
            headers = {'auth-token': self.config.token}
            response = requests.get(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}/positions",
                headers=headers,
                timeout=10
            )
            return response.json() if response.status_code == 200 else []
        except:
            return []

    def place_order(self, symbol: str, side: str, volume: float, 
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> Optional[Dict]:
        """Place market order"""
        if not self.connected:
            return None

        try:
            import requests
            headers = {
                'auth-token': self.config.token,
                'Content-Type': 'application/json'
            }

            order = {
                'symbol': symbol,
                'actionType': 'ORDER_TYPE_BUY' if side == 'buy' else 'ORDER_TYPE_SELL',
                'volume': volume,
            }

            if stop_loss:
                order['stopLoss'] = stop_loss
            if take_profit:
                order['takeProfit'] = take_profit

            response = requests.post(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}/trade",
                headers=headers,
                json=order,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Order placed: {result.get('orderId')}")
                return result
            else:
                logger.error(f"❌ Order failed: {response.text}")
                return None

        except Exception as e:
            logger.error(f"❌ Order error: {e}")
            return None

    def close_position(self, position_id: str) -> bool:
        """Close position by ID"""
        if not self.connected:
            return False

        try:
            import requests
            headers = {'auth-token': self.config.token}
            response = requests.delete(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}/positions/{position_id}",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price"""
        if not self.connected:
            return None

        try:
            import requests
            headers = {'auth-token': self.config.token}
            response = requests.get(
                f"{self.base_url}/users/current/accounts/{self.config.account_id}/symbols/{symbol}/current-price",
                headers=headers,
                timeout=10
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
