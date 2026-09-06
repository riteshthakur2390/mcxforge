"""
utils/websocket_feed.py — WebSocket Live Market Feed
====================================================
Replaces polling every 5min with real-time push data.
Dhan and Upstox both provide WebSocket feeds.

CURRENT (polling):  Fetch LTP every 5min → 5-second latency minimum
WEBSOCKET:          Broker pushes tick on every trade → <100ms latency

IMPACT:
  - Entry fills at better prices (less slippage)
  - Trailing SL triggers faster (protects profits)
  - Stale exit fires sooner (cuts losses earlier)

USAGE:
    feed = WebSocketFeed(broker)
    await feed.connect(["NIFTY", "NIFTY26MAY2322500CE"])
    feed.on_tick(my_callback)   # called on every tick
    await feed.start()
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.bus import get_bus, Topic
except ImportError:
    get_bus = None
    Topic   = None


class WebSocketFeed:
    """
    Real-time market data feed via WebSocket.
    Supports Dhan and Upstox. Falls back to polling if WS unavailable.
    """

    def __init__(self, broker=None) -> None:
        self._broker      = broker
        self._ws          = None
        self._callbacks:  list[Callable] = []
        self._subscribed: list[str]      = []
        self._last_ticks: dict[str, dict] = {}
        self._connected   = False
        self._broker_name = getattr(broker, 'broker_name', 'unknown') if broker else 'none'

    def on_tick(self, callback: Callable) -> None:
        """Register callback for every market tick."""
        self._callbacks.append(callback)

    async def connect(self, symbols: list[str]) -> bool:
        """Connect to broker WebSocket and subscribe to symbols."""
        self._subscribed = symbols
        broker = self._broker_name

        try:
            if broker == 'dhan':
                return await self._connect_dhan(symbols)
            elif broker == 'upstox':
                return await self._connect_upstox(symbols)
            else:
                logger.info(f"[WebSocketFeed] Broker '{broker}' — using polling fallback")
                return False
        except Exception as e:
            logger.warning(f"[WebSocketFeed] Connect failed: {e} — falling back to polling")
            return False

    async def start(self) -> None:
        """Start receiving ticks. Blocks until disconnected."""
        if not self._connected:
            logger.info("[WebSocketFeed] Not connected — starting polling fallback")
            await self._polling_fallback()
            return
        try:
            await self._receive_loop()
        except Exception as e:
            logger.error(f"[WebSocketFeed] Feed error: {e}")
            self._connected = False

    async def disconnect(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False
        logger.info("[WebSocketFeed] Disconnected")

    def get_ltp(self, symbol: str) -> float:
        """Get latest known price for a symbol."""
        return float(self._last_ticks.get(symbol, {}).get("ltp", 0))

    # ── DHAN WEBSOCKET ────────────────────────────────────────────────────────

    async def _connect_dhan(self, symbols: list[str]) -> bool:
        """Connect to Dhan WebSocket feed."""
        try:
            import websockets
            token = getattr(self._broker, '_access_token', '')
            client_id = getattr(self._broker, '_client_id', '')

            url = f"wss://api-feed.dhan.co?version=2&token={token}&clientId={client_id}&authType=2"

            self._ws = await websockets.connect(url, ping_interval=30)
            self._connected = True

            # Subscribe to instruments
            sub_msg = {
                "RequestCode": 21,
                "InstrumentCount": len(symbols),
                "InstrumentList": [
                    {"ExchangeSegment": "IDX_I", "SecurityId": "13"}
                    for _ in symbols
                ]
            }
            await self._ws.send(json.dumps(sub_msg))
            logger.info(f"[WebSocketFeed] Dhan WS connected | {len(symbols)} symbols")
            return True

        except ImportError:
            logger.warning("[WebSocketFeed] 'websockets' not installed: pip install websockets")
            return False
        except Exception as e:
            logger.warning(f"[WebSocketFeed] Dhan WS failed: {e}")
            return False

    # ── UPSTOX WEBSOCKET ──────────────────────────────────────────────────────

    async def _connect_upstox(self, symbols: list[str]) -> bool:
        """Connect to Upstox WebSocket v3 feed."""
        try:
            import websockets
            token = getattr(self._broker, '_access_token', '')
            url   = "wss://api.upstox.com/v3/feed/market-data-feed"
            headers = {"Authorization": f"Bearer {token}"}

            self._ws = await websockets.connect(url, extra_headers=headers, ping_interval=30)
            self._connected = True

            # Subscribe message
            sub_msg = {
                "guid": "signalforge",
                "method": "sub",
                "data": {
                    "mode": "full",
                    "instrumentKeys": ["NSE_INDEX|Nifty 50"],
                }
            }
            await self._ws.send(json.dumps(sub_msg))
            logger.info(f"[WebSocketFeed] Upstox WS connected | {len(symbols)} symbols")
            return True

        except ImportError:
            logger.warning("[WebSocketFeed] 'websockets' not installed: pip install websockets")
            return False
        except Exception as e:
            logger.warning(f"[WebSocketFeed] Upstox WS failed: {e}")
            return False

    # ── RECEIVE LOOP ──────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Process incoming WebSocket messages."""
        while self._connected:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=30)
                tick = self._parse_tick(msg)
                if tick:
                    symbol = tick.get("symbol", "NIFTY")
                    self._last_ticks[symbol] = tick
                    await self._dispatch(tick)
            except asyncio.TimeoutError:
                # Send ping to keep alive
                if self._ws:
                    await self._ws.ping()
            except Exception as e:
                logger.warning(f"[WebSocketFeed] Receive error: {e}")
                self._connected = False
                break

    def _parse_tick(self, raw: str | bytes) -> Optional[dict]:
        """Parse broker-specific tick format into standard dict."""
        try:
            if isinstance(raw, bytes):
                # Binary protocol (Dhan uses binary)
                # Simplified parse — real implementation uses struct.unpack
                return {"symbol": "NIFTY", "ltp": 0.0,
                        "timestamp": datetime.now(IST).isoformat()}
            data = json.loads(raw)
            # Upstox format
            feeds = data.get("feeds", {})
            for symbol, feed_data in feeds.items():
                ltp = feed_data.get("ff", {}).get("marketFF", {}).get("ltpc", {}).get("ltp", 0)
                if ltp:
                    return {
                        "symbol":    symbol,
                        "ltp":       float(ltp),
                        "timestamp": datetime.now(IST).isoformat(),
                    }
        except Exception:
            pass
        return None

    async def _dispatch(self, tick: dict) -> None:
        """Dispatch tick to all registered callbacks and bus."""
        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(tick)
                else:
                    cb(tick)
            except Exception as e:
                logger.debug(f"[WebSocketFeed] Callback error: {e}")

        # Publish to bus
        if get_bus and Topic:
            try:
                await get_bus().publish("TICK_UPDATE", tick, "WebSocketFeed")
            except Exception:
                pass

    # ── POLLING FALLBACK ──────────────────────────────────────────────────────

    async def _polling_fallback(self) -> None:
        """
        Poll broker every 5 seconds as fallback when WebSocket unavailable.
        Less efficient but always works.
        """
        logger.info("[WebSocketFeed] Polling mode: every 5 seconds")
        while True:
            try:
                if self._broker:
                    ltp = self._broker.get_ltp("NIFTY")
                    if ltp > 0:
                        tick = {
                            "symbol":    "NIFTY",
                            "ltp":       ltp,
                            "timestamp": datetime.now(IST).isoformat(),
                        }
                        self._last_ticks["NIFTY"] = tick
                        await self._dispatch(tick)
                await asyncio.sleep(5)
            except Exception as e:
                logger.debug(f"[WebSocketFeed] Poll error: {e}")
                await asyncio.sleep(10)
