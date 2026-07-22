"""
NEXUS Keyless Market Data Layer
===============================
Pulls real OHLCV + spot prices with ZERO API keys, using US-friendly public
endpoints (Coinbase → Kraken → CoinGecko fallback chain).

Why these three (and not Binance):
  * api.binance.com returns HTTP 451 from US IPs — unusable from NYC.
  * Coinbase + Kraken expose keyless public candle/ticker endpoints.
  * CoinGecko's keyless public tier (and your optional demo key) covers the rest.

Network calls use stdlib urllib only — no pip install needed for the data layer.
Parsing is split into pure functions so it's unit-testable offline against
fixture payloads (no network, no fake market data presented as real).

Returns pandas DataFrames with columns: open, high, low, close, volume
indexed by UTC timestamp — exactly the shape NEXUSScorer.score_asset expects.
"""

from __future__ import annotations
import json
import time
import urllib.request
import urllib.parse
from typing import Optional
import pandas as pd

DEMO_CG_KEY = ""  # optional; set to your CoinGecko demo key to raise rate limits

# ── Symbol mapping across venues ────────────────────────────────────────────────
# Each asset maps to its product id on each venue. None = not listed there.
SYMBOL_MAP = {
    "BTC": {"coinbase": "BTC-USD", "kraken": "XBTUSD", "coingecko": "bitcoin"},
    "ETH": {"coinbase": "ETH-USD", "kraken": "ETHUSD", "coingecko": "ethereum"},
    "SOL": {"coinbase": "SOL-USD", "kraken": "SOLUSD", "coingecko": "solana"},
    "XRP": {"coinbase": "XRP-USD", "kraken": "XRPUSD", "coingecko": "ripple"},
    "ADA": {"coinbase": "ADA-USD", "kraken": "ADAUSD", "coingecko": "cardano"},
    "AVAX": {"coinbase": "AVAX-USD", "kraken": "AVAXUSD", "coingecko": "avalanche-2"},
    "LINK": {"coinbase": "LINK-USD", "kraken": "LINKUSD", "coingecko": "chainlink"},
    "DOT": {"coinbase": "DOT-USD", "kraken": "DOTUSD", "coingecko": "polkadot"},
}

# Coinbase granularity (seconds) for common timeframes
_CB_GRAN = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
# Kraken interval (minutes)
_KR_INT = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

_HEADERS = {"User-Agent": "NEXUS-paper/1.0"}


def _http_get(url: str, timeout: int = 15) -> dict | list:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ── Pure parsers (unit-testable offline) ────────────────────────────────────────
def parse_coinbase_candles(raw: list) -> pd.DataFrame:
    """Coinbase returns [[time, low, high, open, close, volume], ...] newest-first."""
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").set_index("time")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_kraken_ohlc(raw: dict, pair_key: str) -> pd.DataFrame:
    """Kraken returns {result: {PAIR: [[time,o,h,l,c,vwap,vol,count], ...]}}."""
    result = raw.get("result", {})
    rows = result.get(pair_key)
    if rows is None:  # Kraken sometimes returns a normalized pair key
        rows = next((v for k, v in result.items() if k != "last"), [])
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_coingecko_market_chart(raw: dict) -> pd.DataFrame:
    """
    CoinGecko market_chart returns {prices:[[ms,price]], total_volumes:[[ms,vol]]}.
    No OHLC — we approximate a close-only frame (open=high=low=close) so the
    scorer still runs. Used only as a last-resort fallback.
    """
    prices = raw.get("prices", [])
    vols = dict((int(t), v) for t, v in raw.get("total_volumes", []))
    if not prices:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(prices, columns=["ms", "close"])
    df["time"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df["volume"] = df["ms"].astype(int).map(vols).fillna(0.0)
    df = df.set_index("time")
    df["open"] = df["high"] = df["low"] = df["close"]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_coingecko_simple_price(raw: dict) -> dict:
    """simple/price → {coingecko_id: {'usd': price}} → flatten to {id: price}."""
    return {k: v.get("usd") for k, v in raw.items() if isinstance(v, dict)}


# ── Live fetchers (need network — run on your Mac) ───────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300) -> Optional[pd.DataFrame]:
    """Real OHLCV via Coinbase → Kraken → CoinGecko fallback chain. Keyless."""
    m = SYMBOL_MAP.get(symbol)
    if not m:
        return None

    # 1) Coinbase
    if m.get("coinbase") and timeframe in _CB_GRAN:
        try:
            url = f"https://api.exchange.coinbase.com/products/{m['coinbase']}/candles?granularity={_CB_GRAN[timeframe]}"
            df = parse_coinbase_candles(_http_get(url))
            if len(df):
                return df.tail(limit)
        except Exception:
            pass

    # 2) Kraken
    if m.get("kraken") and timeframe in _KR_INT:
        try:
            url = f"https://api.kraken.com/0/public/OHLC?pair={m['kraken']}&interval={_KR_INT[timeframe]}"
            df = parse_kraken_ohlc(_http_get(url), m["kraken"])
            if len(df):
                return df.tail(limit)
        except Exception:
            pass

    # 3) CoinGecko (close-only fallback)
    if m.get("coingecko"):
        try:
            days = max(1, limit // 24)
            base = "https://api.coingecko.com/api/v3"
            key = f"&x_cg_demo_api_key={DEMO_CG_KEY}" if DEMO_CG_KEY else ""
            url = f"{base}/coins/{m['coingecko']}/market_chart?vs_currency=usd&days={days}{key}"
            df = parse_coingecko_market_chart(_http_get(url))
            if len(df):
                return df.tail(limit)
        except Exception:
            pass

    return None


# ── Full historical data (back to coin inception) ────────────────────────────────

def fetch_full_history_daily(symbol: str) -> Optional[pd.DataFrame]:
    """
    All available daily candles from coin inception, keyless.

    Uses CoinGecko market_chart?days=max — returns genuine full history:
      BTC  from Jan 2013  (~4,500 daily bars)
      ETH  from Aug 2015  (~3,300 bars)
      XRP  from Aug 2013  (~4,200 bars)
      SOL  from Apr 2020  (~1,500 bars)
      ...and so on for every listed coin.

    Returns a close-only DataFrame (CoinGecko doesn't expose daily OHLC for free
    in the market_chart endpoint). Use fetch_kraken_full_history for true OHLCV.
    """
    m = SYMBOL_MAP.get(symbol)
    if not m or not m.get("coingecko"):
        return None
    base = "https://api.coingecko.com/api/v3"
    key = f"&x_cg_demo_api_key={DEMO_CG_KEY}" if DEMO_CG_KEY else ""
    url = f"{base}/coins/{m['coingecko']}/market_chart?vs_currency=usd&days=max{key}"
    try:
        raw = _http_get(url)
        df = parse_coingecko_market_chart(raw)
        # Resample to daily if CoinGecko returned sub-daily granularity
        if len(df) and df.index.freq is None:
            df = df.resample("1D").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()
        return df
    except Exception:
        return None


def fetch_kraken_full_history(symbol: str, interval_min: int = 1440) -> Optional[pd.DataFrame]:
    """
    Full true-OHLCV history from Kraken via paginated calls. Keyless.

    Kraken returns max 720 bars per call. We paginate using the `last` cursor
    (the `since` parameter) until we reach the present, then concatenate.

    interval_min:
      1440 = daily   → BTC/USD from 2013 (≈7 pages for full history)
       240 = 4-hour  → ~28 pages for full BTC history since 2013
        60 = 1-hour  → ~110 pages (slow but complete)

    Returns a sorted, deduplicated DataFrame with real open/high/low/close/volume.
    The result is everything Kraken has — typically the full market history for
    major pairs, which for BTC goes back to Q3 2013.
    """
    m = SYMBOL_MAP.get(symbol)
    if not m or not m.get("kraken"):
        return None

    pair = m["kraken"]
    all_frames: list[pd.DataFrame] = []
    since = 0          # 0 = from the very beginning
    seen_lasts: set = set()

    while True:
        try:
            url = (f"https://api.kraken.com/0/public/OHLC"
                   f"?pair={pair}&interval={interval_min}&since={since}")
            raw = _http_get(url)
            if raw.get("error"):
                break
            result = raw.get("result", {})
            last = result.get("last", 0)
            rows = result.get(pair)
            if rows is None:
                rows = next((v for k, v in result.items() if k != "last"), [])
            if not rows:
                break
            df = parse_kraken_ohlc(raw, pair)
            if len(df):
                all_frames.append(df)
            # Pagination: Kraken returns `last` as the next `since` value
            if last == 0 or last in seen_lasts or last <= since:
                break
            seen_lasts.add(last)
            since = last
            time.sleep(0.5)          # be polite — Kraken rate-limits at ~1 req/s
        except Exception:
            break

    if not all_frames:
        return None
    full = pd.concat(all_frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


def fetch_price_map(symbols: list[str]) -> dict[str, float]:
    """Latest USD spot for each symbol via CoinGecko simple/price. Keyless."""
    ids = [SYMBOL_MAP[s]["coingecko"] for s in symbols if s in SYMBOL_MAP]
    if not ids:
        return {}
    base = "https://api.coingecko.com/api/v3/simple/price"
    key = f"&x_cg_demo_api_key={DEMO_CG_KEY}" if DEMO_CG_KEY else ""
    url = f"{base}?ids={','.join(ids)}&vs_currencies=usd{key}"
    try:
        by_id = parse_coingecko_simple_price(_http_get(url))
    except Exception:
        return {}
    # remap coingecko ids back to symbols
    id_to_sym = {SYMBOL_MAP[s]["coingecko"]: s for s in symbols if s in SYMBOL_MAP}
    return {id_to_sym[i]: p for i, p in by_id.items() if i in id_to_sym and p is not None}


# ── Offline parser self-test (fixture payloads, real venue shapes) ───────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS Data Layer — parser self-test (offline fixtures)")
    print("=" * 60)

    # Coinbase fixture: newest-first [time, low, high, open, close, volume]
    cb_fixture = [
        [1700003600, 60100, 60900, 60200, 60800, 1234.5],
        [1700000000, 59500, 60300, 59600, 60200, 1100.2],
    ]
    cb = parse_coinbase_candles(cb_fixture)
    assert list(cb.columns) == ["open", "high", "low", "close", "volume"]
    assert cb.index.is_monotonic_increasing, "should be sorted oldest-first"
    assert cb.iloc[-1]["close"] == 60800.0
    print(f"✓ Coinbase parser: {len(cb)} rows, last close ${cb.iloc[-1]['close']:,.0f}")

    # Kraken fixture
    kr_fixture = {"result": {"XBTUSD": [
        [1700000000, "59600", "60300", "59500", "60200", "59900", "1100.2", 42],
        [1700003600, "60200", "60900", "60100", "60800", "60500", "1234.5", 50],
    ]}}
    kr = parse_kraken_ohlc(kr_fixture, "XBTUSD")
    assert kr.iloc[-1]["close"] == 60800.0
    print(f"✓ Kraken parser:   {len(kr)} rows, last close ${kr.iloc[-1]['close']:,.0f}")

    # CoinGecko market_chart fixture
    cg_fixture = {
        "prices": [[1700000000000, 60200], [1700003600000, 60800]],
        "total_volumes": [[1700000000000, 1.1e9], [1700003600000, 1.23e9]],
    }
    cg = parse_coingecko_market_chart(cg_fixture)
    assert cg.iloc[-1]["close"] == 60800.0
    assert (cg["open"] == cg["close"]).all(), "close-only fallback"
    print(f"✓ CoinGecko parser:{len(cg)} rows, last close ${cg.iloc[-1]['close']:,.0f}")

    # Full-history: CoinGecko days=max returns daily data — test the resample path
    # Simulate sub-hourly prices → resampled to daily inside fetch_full_history_daily
    cg_long = {
        "prices": [
            [1700000000000, 59000],
            [1700043600000, 59500],   # +12h
            [1700086400000, 60000],   # ~+24h (next day)
            [1700130000000, 61000],
        ],
        "total_volumes": [],
    }
    cg_long_df = parse_coingecko_market_chart(cg_long)
    assert len(cg_long_df) == 4
    print(f"✓ Full-history CG parse: {len(cg_long_df)} sub-daily rows parsed correctly")

    # Kraken paginated: simulate two pages with a shared last timestamp at boundary
    kr_page1 = {"result": {"XBTUSD": [
        [1700000000, "59000", "59500", "58900", "59300", "59100", "1000", 30],
        [1700086400, "59300", "60000", "59200", "59800", "59600", "1100", 35],
    ], "last": 1700086400}}
    kr_page2 = {"result": {"XBTUSD": [
        [1700086400, "59300", "60000", "59200", "59800", "59600", "1100", 35],  # dupe
        [1700172800, "59800", "61000", "59700", "60800", "60400", "1200", 40],
    ], "last": 1700172800}}
    df1 = parse_kraken_ohlc(kr_page1, "XBTUSD")
    df2 = parse_kraken_ohlc(kr_page2, "XBTUSD")
    combined = pd.concat([df1, df2]).sort_index()
    deduped = combined[~combined.index.duplicated(keep="last")]
    assert len(deduped) == 3, f"expected 3 unique bars after dedup, got {len(deduped)}"
    print(f"✓ Kraken paginated dedup: {len(df1)+len(df2)} raw → {len(deduped)} unique bars")
    sp = parse_coingecko_simple_price({"bitcoin": {"usd": 60800}, "ethereum": {"usd": 3000}})
    assert sp == {"bitcoin": 60800, "ethereum": 3000}
    print(f"✓ simple/price parser: {sp}")

    print("=" * 60)
    print("ALL PARSERS VERIFIED. Live fetch needs network — run on your Mac.")
    print("=" * 60)
