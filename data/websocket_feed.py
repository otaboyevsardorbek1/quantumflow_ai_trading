"""
QuantumFlow AI - Real-time WebSocket Data Connector
Live market data streaming for real trading
"""
import websocket
import json
import threading
import time
from typing import Callable, Dict
import logging

logger = logging.getLogger(__name__)

class WebSocketDataFeed:
    """
    Real-time data feed via WebSocket

    Supports:
    - Binance (crypto)
    - OANDA (forex)
    - Polygon.io (stocks)
    - Custom broker feeds
    """

    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange
        self.ws = None
        self.running = False
        self.callbacks = []
        self.last_price = {}
        self.lock = threading.Lock()

        # Exchange WebSocket URLs
        self.urls = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'binance_futures': 'wss://fstream.binance.com/ws',
            'oanda': 'wss://stream-fxpractice.oanda.com/v3/prices',
        }

    def connect(self, symbols: list):
        """Connect to WebSocket feed"""
        self.symbols = symbols
        self.running = True

        # Build subscription message
        if self.exchange == 'binance':
            streams = '/'.join([f"{s.lower()}@kline_1m" for s in symbols])
            url = f"{self.urls['binance']}/{streams}"
        else:
            url = self.urls.get(self.exchange, self.urls['binance'])

        logger.info(f"🔌 Connecting to {self.exchange} WebSocket...")
        logger.info(f"   URL: {url}")

        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        # Run in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

    def _on_open(self, ws):
        logger.info("✅ WebSocket connected")

        # Subscribe to symbols
        if self.exchange == 'binance':
            # Already subscribed via URL
            pass
        else:
            # Send subscription message
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{s}@kline_1m" for s in self.symbols],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))

    def _on_message(self, ws, message):
        """Handle incoming message"""
        try:
            data = json.loads(message)

            # Parse based on exchange
            if self.exchange == 'binance':
                self._parse_binance(data)

        except Exception as e:
            logger.error(f"Error parsing message: {e}")

    def _parse_binance(self, data: dict):
        """Parse Binance kline data"""
        if 'k' in data:
            kline = data['k']
            symbol = kline['s']

            tick = {
                'symbol': symbol,
                'timestamp': kline['t'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': kline['x'],
            }

            with self.lock:
                self.last_price[symbol] = tick

            # Notify callbacks
            for callback in self.callbacks:
                callback(tick)

    def _on_error(self, ws, error):
        logger.error(f"❌ WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
        self.running = False

    def get_last_price(self, symbol: str) -> dict:
        """Get last received price for symbol"""
        with self.lock:
            return self.last_price.get(symbol, {})

    def add_callback(self, callback: Callable):
        """Add callback for new data"""
        self.callbacks.append(callback)

    def disconnect(self):
        """Disconnect WebSocket"""
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("🔌 WebSocket disconnected")

class MT5RealTimeFeed:
    """
    Real-time feed from MetaTrader 5
    Uses MT5's built-in tick streaming
    """

    def __init__(self):
        self.running = False
        self.symbols = []
        self.callbacks = []
        self.last_tick = {}

        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("⚠️ MetaTrader5 not available")

    def connect(self, symbols: list):
        """Connect to MT5 and start tick stream"""
        if not self.available:
            logger.error("❌ MT5 not available")
            return False

        if not self.mt5.initialize():
            logger.error("❌ MT5 initialization failed")
            return False

        self.symbols = symbols
        self.running = True

        # Subscribe to market depth
        for symbol in symbols:
            self.mt5.market_book_add(symbol)

        # Start tick polling thread
        self.tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self.tick_thread.start()

        logger.info(f"✅ MT5 real-time feed started for {symbols}")
        return True

    def _tick_loop(self):
        """Poll ticks from MT5"""
        while self.running:
            for symbol in self.symbols:
                tick = self.mt5.symbol_info_tick(symbol)
                if tick:
                    tick_data = {
                        'symbol': symbol,
                        'time': tick.time,
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume,
                        'spread': tick.ask - tick.bid,
                    }

                    self.last_tick[symbol] = tick_data

                    for callback in self.callbacks:
                        callback(tick_data)

            time.sleep(0.1)  # 100ms polling

    def get_last_tick(self, symbol: str) -> dict:
        return self.last_tick.get(symbol, {})

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def disconnect(self):
        self.running = False
        for symbol in self.symbols:
            self.mt5.market_book_release(symbol)
        self.mt5.shutdown()
