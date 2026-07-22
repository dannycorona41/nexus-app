"""
NEXUS CEX Execution Module
===========================
Live trade execution on centralised exchanges via CCXT.
Supports: Binance, Coinbase Advanced, Kraken, OKX, Bybit.

PAPER mode  → all "orders" are simulated, no API calls to exchange
LIVE mode   → real orders placed, real money at risk

Usage:
  executor = CCXTExecutor("binance", api_key, api_secret)
  await executor.connect()
  order = await executor.market_buy("BTC/USDT", usdt_amount=500)
  await executor.set_stop_loss("BTC/USDT", quantity, stop_price)
  await executor.market_sell("BTC/USDT", quantity)

Install:
  pip install ccxt
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("nexus.ccxt")

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    log.warning("ccxt not installed — CEX execution disabled. pip install ccxt")
    CCXT_AVAILABLE = False


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class OrderResult:
    order_id:  str
    symbol:    str
    side:      str       # buy or sell
    type:      str       # market / limit / stop
    price:     float
    amount:    float
    cost:      float     # price × amount (approximate)
    fee:       float
    status:    str       # open / closed / canceled
    timestamp: str
    exchange:  str
    simulated: bool = False


# ─────────────────────────────────────────────
# Supported Exchanges
# ─────────────────────────────────────────────

EXCHANGE_MAP = {
    "binance":          "binance",
    "coinbase":         "coinbaseadvanced",
    "coinbasepro":      "coinbaseadvanced",
    "kraken":           "kraken",
    "okx":              "okx",
    "bybit":            "bybit",
}


# ─────────────────────────────────────────────
# CEX Executor
# ─────────────────────────────────────────────

class CCXTExecutor:
    """
    Unified CEX execution across all major exchanges.

    In PAPER mode: logs orders without executing.
    In LIVE mode:  places real orders. Requires valid API keys.
    """

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "",
                 passphrase: str = "", paper_mode: bool = True,
                 testnet: bool = False):
        self.exchange_id = exchange_id.lower()
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.passphrase  = passphrase
        self.paper_mode  = paper_mode
        self.testnet     = testnet
        self._client     = None

    async def connect(self):
        if not CCXT_AVAILABLE:
            log.error("ccxt not available")
            return False
        if self.paper_mode:
            log.info(f"[{self.exchange_id}] Paper mode — no real orders will be placed")
            return True

        ex_class_name = EXCHANGE_MAP.get(self.exchange_id, self.exchange_id)
        ex_class      = getattr(ccxt, ex_class_name, None)
        if not ex_class:
            log.error(f"Exchange '{self.exchange_id}' not supported by CCXT")
            return False

        config = {
            "apiKey":    self.api_key,
            "secret":    self.api_secret,
            "enableRateLimit": True,
            "options":   {"defaultType": "spot"},
        }
        if self.passphrase:
            config["password"] = self.passphrase

        self._client = ex_class(config)

        if self.testnet:
            self._client.set_sandbox_mode(True)

        try:
            await self._client.load_markets()
            log.info(f"[{self.exchange_id}] Connected ({'TESTNET' if self.testnet else 'LIVE'})")
            return True
        except Exception as e:
            log.error(f"[{self.exchange_id}] Connection failed: {e}")
            return False

    async def disconnect(self):
        if self._client:
            await self._client.close()

    # ── Balance ───────────────────────────────

    async def get_balance(self, currency: str = "USDT") -> float:
        """Get free balance for a currency."""
        if self.paper_mode or not self._client:
            return 0.0
        try:
            bal = await self._client.fetch_balance()
            return float(bal.get("free", {}).get(currency, 0))
        except Exception as e:
            log.error(f"Balance error: {e}")
            return 0.0

    async def get_price(self, symbol: str) -> float:
        """Get current ask price for a symbol."""
        if not self._client:
            return 0.0
        try:
            ticker = await self._client.fetch_ticker(symbol)
            return float(ticker.get("ask") or ticker.get("last") or 0)
        except Exception as e:
            log.error(f"Price error for {symbol}: {e}")
            return 0.0

    # ── Order execution ───────────────────────

    async def market_buy(self, symbol: str, usdt_amount: float,
                          max_slippage_pct: float = 0.5) -> Optional[OrderResult]:
        """Buy `usdt_amount` worth of `symbol` at market price."""
        if self.paper_mode:
            return self._simulate_order(symbol, "buy", "market", usdt_amount=usdt_amount)

        try:
            price    = await self.get_price(symbol)
            if price <= 0:
                log.error(f"Cannot buy {symbol}: no price")
                return None
            quantity = usdt_amount / price / (1 + max_slippage_pct / 100)
            quantity = self._client.amount_to_precision(symbol, quantity)

            order = await self._client.create_order(
                symbol=symbol, type="market", side="buy", amount=float(quantity)
            )
            return self._parse_order(order)
        except Exception as e:
            log.error(f"Market buy error {symbol}: {e}")
            return None

    async def market_sell(self, symbol: str, quantity: float) -> Optional[OrderResult]:
        """Sell `quantity` units of `symbol` at market price."""
        if self.paper_mode:
            return self._simulate_order(symbol, "sell", "market", quantity=quantity)

        try:
            qty   = self._client.amount_to_precision(symbol, quantity)
            order = await self._client.create_order(
                symbol=symbol, type="market", side="sell", amount=float(qty)
            )
            return self._parse_order(order)
        except Exception as e:
            log.error(f"Market sell error {symbol}: {e}")
            return None

    async def limit_buy(self, symbol: str, quantity: float,
                         price: float) -> Optional[OrderResult]:
        """Place a limit buy order at `price`."""
        if self.paper_mode:
            return self._simulate_order(symbol, "buy", "limit", quantity=quantity, price=price)

        try:
            qty   = self._client.amount_to_precision(symbol, quantity)
            p     = self._client.price_to_precision(symbol, price)
            order = await self._client.create_order(
                symbol=symbol, type="limit", side="buy", amount=float(qty), price=float(p)
            )
            return self._parse_order(order)
        except Exception as e:
            log.error(f"Limit buy error {symbol}: {e}")
            return None

    async def set_stop_loss(self, symbol: str, quantity: float,
                             stop_price: float) -> Optional[str]:
        """
        Place a stop-loss sell order.
        Uses stop-market on exchanges that support it, otherwise stop-limit.
        """
        if self.paper_mode:
            log.info(f"[PAPER] Stop loss set: {symbol} SL=${stop_price:.4f} qty={quantity:.4f}")
            return "paper_sl_" + symbol

        try:
            if not self._client:
                return None

            # Binance / most exchanges: STOP_MARKET
            try:
                qty   = self._client.amount_to_precision(symbol, quantity)
                sp    = self._client.price_to_precision(symbol, stop_price)
                order = await self._client.create_order(
                    symbol=symbol, type="STOP_MARKET", side="sell",
                    amount=float(qty),
                    params={"stopPrice": float(sp), "closePosition": False},
                )
                log.info(f"Stop-market placed: {symbol} @ ${stop_price:.4f}")
                return order.get("id")
            except Exception:
                # Fallback: stop-limit (stop_price × 0.998 as limit)
                qty   = self._client.amount_to_precision(symbol, quantity)
                sp    = self._client.price_to_precision(symbol, stop_price)
                lp    = self._client.price_to_precision(symbol, stop_price * 0.998)
                order = await self._client.create_order(
                    symbol=symbol, type="STOP_LOSS_LIMIT", side="sell",
                    amount=float(qty), price=float(lp),
                    params={"stopPrice": float(sp)},
                )
                log.info(f"Stop-limit placed: {symbol} SL=${stop_price:.4f}")
                return order.get("id")
        except Exception as e:
            log.error(f"Stop loss error {symbol}: {e}")
            return None

    async def set_take_profit(self, symbol: str, quantity: float,
                               tp_price: float) -> Optional[str]:
        """Place a take-profit sell limit order."""
        if self.paper_mode:
            log.info(f"[PAPER] Take profit set: {symbol} TP=${tp_price:.4f}")
            return "paper_tp_" + symbol

        try:
            qty   = self._client.amount_to_precision(symbol, quantity)
            p     = self._client.price_to_precision(symbol, tp_price)
            order = await self._client.create_order(
                symbol=symbol, type="limit", side="sell",
                amount=float(qty), price=float(p)
            )
            log.info(f"Take profit placed: {symbol} TP=${tp_price:.4f}")
            return order.get("id")
        except Exception as e:
            log.error(f"Take profit error {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.paper_mode or not self._client:
            return True
        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            log.error(f"Cancel order error: {e}")
            return False

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        if self.paper_mode or not self._client:
            return []
        try:
            orders = await self._client.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            log.error(f"Open orders error: {e}")
            return []

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Emergency: cancel all open orders. Returns count cancelled."""
        if self.paper_mode or not self._client:
            return 0
        try:
            orders  = await self.get_open_orders(symbol)
            count   = 0
            for o in orders:
                if await self.cancel_order(o["id"], o["symbol"]):
                    count += 1
            log.warning(f"Cancelled {count} orders (EMERGENCY)")
            return count
        except Exception as e:
            log.error(f"Cancel all error: {e}")
            return 0

    # ── Full position execution ───────────────

    async def execute_nexus_signal(self, signal: dict, usdt_budget: float,
                                    exchange_id: str = "binance") -> dict:
        """
        Execute a complete NEXUS signal: market buy + set SL + set TP.

        signal: dict from nexus_signal_engine score_asset()
          {symbol, entry_zone, stop_loss, take_profit, risk_reward}
        usdt_budget: capital to deploy for this position

        Returns: {ok, buy_order, sl_id, tp_id, error}
        """
        symbol      = signal["symbol"].replace("/USDT", "") + "/USDT"
        stop_loss   = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)

        log.info(f"Executing signal: {symbol} | budget=${usdt_budget:.0f} | "
                 f"SL=${stop_loss:.4f} | TP=${take_profit:.4f}")

        buy_order = await self.market_buy(symbol, usdt_budget)
        if not buy_order:
            return {"ok": False, "error": "Buy order failed"}

        quantity = buy_order.amount
        sl_id = tp_id = None

        if stop_loss > 0:
            sl_id = await self.set_stop_loss(symbol, quantity, stop_loss)
        if take_profit > 0:
            tp_id = await self.set_take_profit(symbol, quantity, take_profit)

        return {
            "ok":        True,
            "buy_order": buy_order.__dict__,
            "sl_id":     sl_id,
            "tp_id":     tp_id,
            "symbol":    symbol,
            "quantity":  quantity,
            "entry":     buy_order.price,
        }

    # ── Internal helpers ──────────────────────

    def _simulate_order(self, symbol: str, side: str, order_type: str,
                         usdt_amount: float = 0, quantity: float = 0,
                         price: float = 0) -> OrderResult:
        log.info(f"[PAPER] {side.upper()} {symbol} type={order_type} "
                 f"{'$'+str(usdt_amount) if usdt_amount else str(quantity)+' units'}")
        return OrderResult(
            order_id  = "paper_" + str(int(datetime.now().timestamp())),
            symbol    = symbol,
            side      = side,
            type      = order_type,
            price     = price or 0,
            amount    = quantity or (usdt_amount / max(price, 1)),
            cost      = usdt_amount or (quantity * price),
            fee       = 0.0,
            status    = "closed",
            timestamp = datetime.now(timezone.utc).isoformat(),
            exchange  = self.exchange_id,
            simulated = True,
        )

    def _parse_order(self, raw: dict) -> OrderResult:
        return OrderResult(
            order_id  = str(raw.get("id", "")),
            symbol    = raw.get("symbol", ""),
            side      = raw.get("side", ""),
            type      = raw.get("type", ""),
            price     = float(raw.get("average") or raw.get("price") or 0),
            amount    = float(raw.get("filled") or raw.get("amount") or 0),
            cost      = float(raw.get("cost") or 0),
            fee       = float((raw.get("fee") or {}).get("cost") or 0),
            status    = raw.get("status", ""),
            timestamp = raw.get("datetime", datetime.now(timezone.utc).isoformat()),
            exchange  = self.exchange_id,
            simulated = False,
        )
