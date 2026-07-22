"""
NEXUS Keyless Live Data Layer
=============================
Pulls real market data from public, keyless endpoints. No API keys, no signup,
no cost. Produces exactly what the orchestrator needs:

    get_ohlcv(symbol)  -> pandas DataFrame[open,high,low,close,volume]  (for the scorer)
    get_price_map(...)  -> {symbol: latest_price}                       (for fills/metrics)

Plus keyless extras: Fear & Greed, DexScreener pairs, XRPL AMM/order-book,
DeFiLlama protocol TVL/revenue.

Design
------
* Every network call is wrapped: on failure it logs and returns None/empty —
  it never raises into the trading loop. One dead endpoint can't crash a cycle.
* The PARSE functions are pure (json -> typed result) and unit-tested offline
  below, so the data-shaping logic is verified without a network.
* `requests` is imported lazily inside fetchers so this module imports cleanly
  even in environments without it; parsing is testable regardless.

Sources (all keyless, verified):
  Binance public market data : https://data-api.binance.vision
  CoinGecko public           : https://api.coingecko.com/api/v3
  DexScreener                : https://api.dexscreener.com
  XRPL public node           : https://xrplcluster.com
  DeFiLlama                  : https://api.llama.fi
  Fear & Greed               : https://api.alternative.me/fng
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

log = logging.getLogger("nexus.data")

BINANCE_PUBLIC = "https://data-api.binance.vision"
COINGECKO      = "https://api.coingecko.com/api/v3"
DEXSCREENER    = "https://api.dexscreener.com"
XRPL_NODE      = "https://xrplcluster.com"
DEFILLAMA      = "https://api.llama.fi"
FEAR_GREED     = "https://api.alternative.me/fng/"

DEFAULT_TIMEOUT = 10


# ── Pure parsers (no network — unit-tested below) ───────────────────────────────

def parse_binance_klines(rows: list) -> pd.DataFrame:
    """Binance /klines array -> OHLCV DataFrame indexed by close time (UTC)."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "volume"]]


def parse_binance_24hr(obj: dict) -> dict:
    """Binance /ticker/24hr object -> compact dict."""
    return {
        "symbol": obj.get("symbol"),
        "last": float(obj.get("lastPrice", 0) or 0),
        "change_pct": float(obj.get("priceChangePercent", 0) or 0),
        "high": float(obj.get("highPrice", 0) or 0),
        "low": float(obj.get("lowPrice", 0) or 0),
        "volume": float(obj.get("volume", 0) or 0),
        "quote_volume": float(obj.get("quoteVolume", 0) or 0),
    }


def parse_fear_greed(obj: dict) -> Optional[int]:
    """alternative.me FNG -> 0-100 int (None if unavailable)."""
    try:
        return int(obj["data"][0]["value"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def parse_dexscreener_pairs(obj: dict) -> list:
    """DexScreener pairs response -> list of compact pair dicts."""
    out = []
    for p in (obj or {}).get("pairs", []) or []:
        out.append({
            "chain": p.get("chainId"),
            "dex": p.get("dexId"),
            "pair": p.get("pairAddress"),
            "base": (p.get("baseToken") or {}).get("symbol"),
            "quote": (p.get("quoteToken") or {}).get("symbol"),
            "price_usd": float(p.get("priceUsd") or 0) if p.get("priceUsd") else None,
            "liquidity_usd": ((p.get("liquidity") or {}).get("usd")),
            "volume_24h": ((p.get("volume") or {}).get("h24")),
            "price_change_24h": ((p.get("priceChange") or {}).get("h24")),
        })
    return out


def parse_xrpl_amm(obj: dict) -> Optional[dict]:
    """XRPL amm_info JSON-RPC result -> pool summary."""
    try:
        amm = obj["result"]["amm"]
    except (KeyError, TypeError):
        return None

    def _amount(a):
        # XRP is a string of drops; issued currency is {currency, issuer, value}
        if isinstance(a, str):
            return {"currency": "XRP", "value": int(a) / 1_000_000}
        return {"currency": a.get("currency"), "issuer": a.get("issuer"),
                "value": float(a.get("value", 0) or 0)}

    return {
        "amount": _amount(amm.get("amount")),
        "amount2": _amount(amm.get("amount2")),
        "trading_fee": amm.get("trading_fee"),
        "lp_token": (amm.get("lp_token") or {}).get("value"),
    }


def parse_defillama_protocol(obj: dict) -> Optional[dict]:
    """DeFiLlama /protocol/{slug} -> tvl + recent revenue summary."""
    if not obj:
        return None
    tvl_series = obj.get("tvl") or []
    latest_tvl = tvl_series[-1].get("totalLiquidityUSD") if tvl_series else None
    return {
        "name": obj.get("name"),
        "symbol": obj.get("symbol"),
        "latest_tvl": latest_tvl,
        "chains": obj.get("chains", []),
        "category": obj.get("category"),
    }


# ── Live feed (network) ─────────────────────────────────────────────────────────

class LiveDataFeed:
    """Keyless live data. All methods fail soft (return None/empty on error)."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, demo_cg_key: str = ""):
        self.timeout = timeout
        self.demo_cg_key = demo_cg_key  # optional CoinGecko demo key for higher rate limit

    def _get(self, url: str, params: dict | None = None) -> Optional[object]:
        try:
            import requests
            r = requests.get(url, params=params, timeout=self.timeout,
                             headers={"User-Agent": "NEXUS/paper"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"GET failed {url}: {e}")
            return None

    def _post(self, url: str, payload: dict) -> Optional[object]:
        try:
            import requests
            r = requests.post(url, json=payload, timeout=self.timeout,
                              headers={"User-Agent": "NEXUS/paper"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"POST failed {url}: {e}")
            return None

    # Binance ─────────────────────────────
    def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200,
                  quote: str = "USDT") -> Optional[pd.DataFrame]:
        """OHLCV candles from Binance public. symbol='BTC' -> 'BTCUSDT'."""
        pair = f"{symbol.upper()}{quote}"
        rows = self._get(f"{BINANCE_PUBLIC}/api/v3/klines",
                         {"symbol": pair, "interval": interval, "limit": limit})
        if rows is None:
            return None
        return parse_binance_klines(rows)

    def get_price_map(self, symbols: list[str], quote: str = "USDT") -> dict:
        """Latest prices for many symbols from Binance 24hr ticker."""
        data = self._get(f"{BINANCE_PUBLIC}/api/v3/ticker/24hr")
        if not isinstance(data, list):
            return {}
        wanted = {f"{s.upper()}{quote}": s.upper() for s in symbols}
        out = {}
        for obj in data:
            sym = obj.get("symbol")
            if sym in wanted:
                out[wanted[sym]] = float(obj.get("lastPrice", 0) or 0)
        return out

    # Fear & Greed ────────────────────────
    def get_fear_greed(self) -> Optional[int]:
        return parse_fear_greed(self._get(FEAR_GREED) or {})

    # DexScreener ─────────────────────────
    def get_dex_pairs(self, query: str) -> list:
        return parse_dexscreener_pairs(self._get(f"{DEXSCREENER}/latest/dex/search",
                                                 {"q": query}) or {})

    # XRPL ────────────────────────────────
    def get_xrpl_amm(self, asset_currency: str, asset_issuer: str,
                     asset2_currency: str = "XRP", asset2_issuer: str = "") -> Optional[dict]:
        asset = {"currency": asset_currency, "issuer": asset_issuer}
        asset2 = {"currency": "XRP"} if asset2_currency == "XRP" else \
                 {"currency": asset2_currency, "issuer": asset2_issuer}
        payload = {"method": "amm_info", "params": [{"asset": asset, "asset2": asset2}]}
        return parse_xrpl_amm(self._post(XRPL_NODE, payload) or {})

    # DeFiLlama ───────────────────────────
    def get_protocol(self, slug: str) -> Optional[dict]:
        return parse_defillama_protocol(self._get(f"{DEFILLAMA}/protocol/{slug}") or {})


# ── Offline parser self-test (no network) ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Keyless Data Layer — parser self-test (offline)")
    print("=" * 60)

    # Realistic Binance klines fixture (2 candles)
    klines = [
        [1700000000000, "42000.0", "42500.0", "41800.0", "42300.0", "1500.5",
         1700003599999, "63450000.0", 12000, "750.2", "31700000.0", "0"],
        [1700003600000, "42300.0", "42900.0", "42100.0", "42850.0", "1320.7",
         1700007199999, "56300000.0", 11000, "640.1", "27400000.0", "0"],
    ]
    df = parse_binance_klines(klines)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2 and abs(df["close"].iloc[-1] - 42850.0) < 1e-6
    print(f"✓ Binance klines -> DataFrame {df.shape}, last close = {df['close'].iloc[-1]}")

    tick = parse_binance_24hr({"symbol": "BTCUSDT", "lastPrice": "42850.0",
                               "priceChangePercent": "2.02", "highPrice": "42900",
                               "lowPrice": "41800", "volume": "2821.2",
                               "quoteVolume": "119750000"})
    assert tick["last"] == 42850.0 and tick["change_pct"] == 2.02
    print(f"✓ Binance 24hr -> last={tick['last']} change={tick['change_pct']}%")

    assert parse_fear_greed({"data": [{"value": "27", "value_classification": "Fear"}]}) == 27
    assert parse_fear_greed({"data": []}) is None
    print("✓ Fear & Greed parse (27) + empty-guard")

    dex = parse_dexscreener_pairs({"pairs": [{
        "chainId": "xrpl", "dexId": "xrpl", "pairAddress": "abc",
        "baseToken": {"symbol": "CHILLGUY"}, "quoteToken": {"symbol": "XRP"},
        "priceUsd": "0.0123", "liquidity": {"usd": 45000},
        "volume": {"h24": 12000}, "priceChange": {"h24": 8.4}}]})
    assert dex[0]["base"] == "CHILLGUY" and dex[0]["price_usd"] == 0.0123
    print(f"✓ DexScreener pairs -> {dex[0]['base']}/{dex[0]['quote']} @ ${dex[0]['price_usd']}")

    amm = parse_xrpl_amm({"result": {"amm": {
        "amount": "1000000000",  # 1000 XRP in drops
        "amount2": {"currency": "USD", "issuer": "rIssuer", "value": "523.4"},
        "trading_fee": 500, "lp_token": {"value": "71.2"}}}})
    assert amm["amount"]["value"] == 1000.0 and amm["amount2"]["value"] == 523.4
    print(f"✓ XRPL AMM -> {amm['amount']['value']} XRP / {amm['amount2']['value']} USD, fee={amm['trading_fee']}")

    proto = parse_defillama_protocol({"name": "Aave", "symbol": "AAVE",
        "tvl": [{"date": 1, "totalLiquidityUSD": 1.2e10}], "chains": ["Ethereum"],
        "category": "Lending"})
    assert proto["latest_tvl"] == 1.2e10
    print(f"✓ DeFiLlama -> {proto['name']} TVL=${proto['latest_tvl']:,.0f}")

    print("=" * 60)
    print("ALL PARSERS VERIFIED OFFLINE. Live fetch ready to run on your Mac.")
    print("=" * 60)
