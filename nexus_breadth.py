"""
NEXUS Hourly Market Breadth Engine
=====================================
Scans 1H OHLCV across a configurable watchlist and produces a composite
breadth score that acts as a conviction multiplier on all individual signals.

WHAT IS BREADTH?
  A BTC OTE 0.702 hit with 80% of coins above their 20 EMA and surging
  volume breadth is a different trade than the same setup with only 20%
  participation. Breadth tells you whether the market confirms or rejects
  what the individual chart is showing.

FIVE COMPONENTS (hourly scan):
  % above 20 EMA     25%  — Short-term trend participation
  % above 200 EMA    20%  — Medium-term structure health
  RSI Breadth        20%  — % of coins with 1H RSI > 50
  Volume Breadth     20%  — Zweig up-volume / total-volume ratio
  MACD Breadth       15%  — % with positive MACD histogram on 1H

CONVICTION MULTIPLIER applied to all individual signal scores:
  Score ≥ 80   → × 1.25   Boost. Strong broad participation.
  Score 65–80  → × 1.10   Healthy. Most coins confirming.
  Score 45–65  → × 1.00   Neutral. No adjustment.
  Score 30–45  → × 0.85   Caution. Breadth weakening.
  Score < 30   → × 0.70   Circuit breaker. No new longs.

SPECIAL CONDITIONS (fire Telegram alerts):
  Breadth Thrust    — % above 20 EMA jumps <40 → >60 in ≤3 bars  (rare buy)
  Breadth Collapse  — % above 20 EMA drops >70 → <40 in ≤3 bars  (exit longs)
  Extreme Bullish   — >85% above 20 EMA  (overbought — watch for reversal)
  Extreme Bearish   — <15% above 20 EMA  (capitulation — watch for setup)
  NH/NL Thrust      — New hourly highs > 3× new hourly lows (momentum surge)
  Zweig 9:1 Day     — Volume breadth >0.90 or <0.10 (climactic)

Install:
  pip install ccxt pandas pandas-ta aiohttp
"""

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
try:
    import pandas_ta as pta
except Exception:
    pta = None  # optional; breadth falls back if unavailable

log = logging.getLogger("nexus.breadth")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HISTORY_FILE = Path(__file__).parent / "nexus_breadth_history.json"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class AssetBreadth:
    """Per-asset hourly metrics used to compute market breadth."""
    symbol:         str
    close:          float
    ema20:          float
    ema200:         float
    rsi:            float
    macd_hist:      float
    volume_1h:      float
    price_chg_1h:   float     # % change vs prior bar
    above_ema20:    bool
    above_ema200:   bool
    rsi_bullish:    bool      # RSI > 50
    macd_bullish:   bool      # MACD hist > 0
    advancing:      bool      # price_chg > 0
    new_high_20:    bool      # new 20-bar high on 1H
    new_low_20:     bool      # new 20-bar low on 1H


@dataclass
class BreadthSnapshot:
    """Full market breadth state at one point in time."""
    timestamp:            datetime
    total_scanned:        int
    pct_above_ema20:      float      # 0–100
    pct_above_ema200:     float      # 0–100
    pct_rsi_bullish:      float      # 0–100
    volume_breadth:       float      # 0–1 (Zweig up-vol / total-vol)
    pct_macd_bullish:     float      # 0–100
    advance_count:        int
    decline_count:        int
    new_highs:            int
    new_lows:             int
    up_volume:            float
    down_volume:          float
    breadth_score:        float      # 0–100 composite
    breadth_state:        str        # EXTREME_BULLISH / BULLISH / NEUTRAL / BEARISH / COLLAPSE
    conviction_multiplier: float     # 0.70 – 1.25
    alerts:               list[str]  # active special conditions
    leaders:              list[str]  # top advancing symbols
    laggards:             list[str]  # top declining symbols
    assets:               list[AssetBreadth] = field(default_factory=list)

    @property
    def ad_ratio(self) -> float:
        total = self.advance_count + self.decline_count
        return self.advance_count / total if total > 0 else 0.5

    @property
    def nh_nl_ratio(self) -> float:
        total = self.new_highs + self.new_lows
        return self.new_highs / total if total > 0 else 0.5

    def as_telegram(self) -> str:
        state_emoji = {
            "EXTREME_BULLISH": "🟢🟢",
            "BULLISH":         "🟢",
            "NEUTRAL":         "🟡",
            "BEARISH":         "🔴",
            "COLLAPSE":        "🔴🔴",
        }.get(self.breadth_state, "🟡")
        alerts_str = "\n  ⚡ " + "\n  ⚡ ".join(self.alerts) if self.alerts else ""
        return (
            f"📊 NEXUS Breadth Report — {self.timestamp.strftime('%H:%M UTC')}\n"
            f"{state_emoji} {self.breadth_state}  |  Score: {self.breadth_score:.0f}/100\n"
            f"Multiplier: ×{self.conviction_multiplier:.2f}  |  Scanned: {self.total_scanned} assets\n"
            f"──────────────────────\n"
            f"% above 20 EMA : {self.pct_above_ema20:.0f}%\n"
            f"% above 200 EMA: {self.pct_above_ema200:.0f}%\n"
            f"RSI breadth    : {self.pct_rsi_bullish:.0f}% above 50\n"
            f"Volume breadth : {self.volume_breadth * 100:.0f}% up-volume\n"
            f"MACD breadth   : {self.pct_macd_bullish:.0f}% positive\n"
            f"A/D ratio      : {self.advance_count}↑ / {self.decline_count}↓\n"
            f"New H/L        : {self.new_highs} highs / {self.new_lows} lows\n"
            f"Leaders : {', '.join(self.leaders[:5])}\n"
            f"Laggards: {', '.join(self.laggards[:5])}"
            f"{alerts_str}"
        )

    def as_dict_simple(self) -> dict:
        return {
            "timestamp":             self.timestamp.isoformat(),
            "breadth_score":         round(self.breadth_score, 2),
            "breadth_state":         self.breadth_state,
            "conviction_multiplier": self.conviction_multiplier,
            "pct_above_ema20":       round(self.pct_above_ema20, 1),
            "pct_above_ema200":      round(self.pct_above_ema200, 1),
            "pct_rsi_bullish":       round(self.pct_rsi_bullish, 1),
            "volume_breadth":        round(self.volume_breadth, 3),
            "pct_macd_bullish":      round(self.pct_macd_bullish, 1),
            "ad_ratio":              round(self.ad_ratio, 3),
            "nh_nl_ratio":           round(self.nh_nl_ratio, 3),
            "alerts":                self.alerts,
            "leaders":               self.leaders[:5],
            "laggards":              self.laggards[:5],
        }


# ─────────────────────────────────────────────
# Breadth History (rolling 24-hour)
# ─────────────────────────────────────────────

class BreadthHistory:
    """
    Maintains a 24-hour rolling deque of BreadthSnapshot dicts.
    Persisted to JSON so history survives process restarts.
    Used to detect Thrust / Collapse events and trend direction.
    """

    MAX_BARS = 24  # 24 hourly snapshots = 24h lookback

    def __init__(self, filepath: Path = HISTORY_FILE):
        self.filepath = filepath
        self._data: deque[dict] = deque(maxlen=self.MAX_BARS)
        self._load()

    def _load(self):
        try:
            if self.filepath.exists():
                raw = json.loads(self.filepath.read_text())
                for d in raw[-self.MAX_BARS:]:
                    self._data.append(d)
                log.info(f"Breadth history loaded: {len(self._data)} bars")
        except Exception as e:
            log.warning(f"Could not load breadth history: {e}")

    def _save(self):
        try:
            self.filepath.write_text(json.dumps(list(self._data), indent=2))
        except Exception as e:
            log.warning(f"Could not save breadth history: {e}")

    def append(self, snapshot: BreadthSnapshot):
        self._data.append(snapshot.as_dict_simple())
        self._save()

    def pct_above_ema20_series(self) -> list[float]:
        return [d["pct_above_ema20"] for d in self._data]

    def score_series(self) -> list[float]:
        return [d["breadth_score"] for d in self._data]

    def last_n_ema20(self, n: int) -> list[float]:
        series = self.pct_above_ema20_series()
        return series[-n:] if len(series) >= n else series

    def __len__(self):
        return len(self._data)

    def ad_cumulative(self) -> list[float]:
        """Running cumulative A/D line (rising = bullish divergence)."""
        cum, result = 0.0, []
        for d in self._data:
            ratio = d.get("ad_ratio", 0.5)
            cum  += (ratio - 0.5) * 2  # normalize to -1 to +1 contribution
            result.append(round(cum, 3))
        return result


# ─────────────────────────────────────────────
# Alert Manager
# ─────────────────────────────────────────────

class BreadthAlertManager:
    """
    Detects special breadth conditions and generates actionable alerts.
    All conditions described in the module docstring.
    """

    def __init__(self, history: BreadthHistory):
        self.history = history

    def check_thrust(self, current_pct: float) -> Optional[str]:
        """
        Breadth Thrust: % above 20 EMA jumps from below 40 to above 60
        within 3 consecutive hourly bars. Very rare, very bullish.
        """
        series = self.history.last_n_ema20(3)
        if len(series) >= 2:
            oldest = series[0]
            if oldest < 40 and current_pct > 60:
                return f"BREADTH THRUST — jumped from {oldest:.0f}% to {current_pct:.0f}% in {len(series)} bars"
        return None

    def check_collapse(self, current_pct: float) -> Optional[str]:
        """
        Breadth Collapse: % above 20 EMA drops from above 70 to below 40
        within 3 consecutive hourly bars. Exit longs.
        """
        series = self.history.last_n_ema20(3)
        if len(series) >= 2:
            oldest = series[0]
            if oldest > 70 and current_pct < 40:
                return f"BREADTH COLLAPSE — dropped from {oldest:.0f}% to {current_pct:.0f}% — exit longs"
        return None

    def check_extreme(self, current_pct: float) -> Optional[str]:
        if current_pct > 85:
            return f"EXTREME BULLISH — {current_pct:.0f}% above 20 EMA — overbought, watch for reversal"
        if current_pct < 15:
            return f"EXTREME BEARISH — {current_pct:.0f}% above 20 EMA — capitulation zone"
        return None

    def check_nh_nl_thrust(self, new_highs: int, new_lows: int) -> Optional[str]:
        if new_highs > 0 and new_lows > 0 and new_highs >= new_lows * 3:
            return f"NH/NL THRUST — {new_highs} new highs vs {new_lows} new lows"
        return None

    def check_zweig(self, volume_breadth: float) -> Optional[str]:
        if volume_breadth > 0.90:
            return f"ZWEIG 9:1 UP DAY — {volume_breadth * 100:.0f}% up-volume — strong accumulation"
        if volume_breadth < 0.10:
            return f"ZWEIG 9:1 DOWN DAY — {volume_breadth * 100:.0f}% up-volume — heavy distribution"
        return None

    def check_ad_divergence(self) -> Optional[str]:
        """
        Bearish divergence: price (score) making new high, A/D line falling.
        Bullish divergence: score making new low, A/D line rising.
        Uses score_series vs ad_cumulative.
        """
        scores = self.history.score_series()
        ad     = self.history.ad_cumulative()
        if len(scores) < 4 or len(ad) < 4:
            return None

        score_rising = scores[-1] > scores[-4]
        score_falling= scores[-1] < scores[-4]
        ad_rising    = ad[-1] > ad[-4]
        ad_falling   = ad[-1] < ad[-4]

        if score_rising and ad_falling:
            return "BREADTH BEARISH DIVERGENCE — price breadth rising but A/D line falling"
        if score_falling and ad_rising:
            return "BREADTH BULLISH DIVERGENCE — A/D line rising while price breadth falls"
        return None

    def evaluate_all(self, snapshot: BreadthSnapshot) -> list[str]:
        alerts = []
        checks = [
            self.check_thrust(snapshot.pct_above_ema20),
            self.check_collapse(snapshot.pct_above_ema20),
            self.check_extreme(snapshot.pct_above_ema20),
            self.check_nh_nl_thrust(snapshot.new_highs, snapshot.new_lows),
            self.check_zweig(snapshot.volume_breadth),
            self.check_ad_divergence(),
        ]
        alerts = [c for c in checks if c is not None]
        if alerts:
            log.warning(f"Breadth alerts fired: {alerts}")
        return alerts


# ─────────────────────────────────────────────
# Core Breadth Calculator
# ─────────────────────────────────────────────

class BreadthCalculator:
    """
    Main breadth engine. Call scan_all() with a ccxt async exchange instance
    to get a full BreadthSnapshot in one async call.

    Default watchlist: 30 liquid, institutionally-relevant assets across all
    major categories — L1s, L2s, DeFi, Infrastructure, Payments, RWA, AI.
    No speculative or narrative-only assets in the default universe.
    Override via the watchlist constructor parameter, or import
    nexus_asset_universe.Watchlist for pre-built category lists.
    """

    DEFAULT_WATCHLIST = [
        # ── L1 Foundation + Smart Contract ─────────
        "BTC/USDT",  "ETH/USDT",  "SOL/USDT",  "BNB/USDT",  "XRP/USDT",
        "AVAX/USDT", "NEAR/USDT", "APT/USDT",  "SUI/USDT",  "TON/USDT",
        # ── L2 Scaling ──────────────────────────────
        "ARB/USDT",  "OP/USDT",   "MATIC/USDT","IMX/USDT",
        # ── DeFi Blue Chips ─────────────────────────
        "AAVE/USDT", "UNI/USDT",  "MKR/USDT",  "LDO/USDT",  "GMX/USDT",
        "PENDLE/USDT",
        # ── Infrastructure / Oracle ──────────────────
        "LINK/USDT", "GRT/USDT",  "ICP/USDT",
        # ── Cross-Chain ──────────────────────────────
        "DOT/USDT",  "ATOM/USDT", "RUNE/USDT", "TIA/USDT",
        # ── RWA + Payments ───────────────────────────
        "ONDO/USDT", "HBAR/USDT",
        # ── AI / Compute ─────────────────────────────
        "RENDER/USDT","TAO/USDT",
    ]

    # Scoring component weights
    WEIGHTS = {
        "pct_above_ema20":  0.25,
        "pct_above_ema200": 0.20,
        "pct_rsi_bullish":  0.20,
        "volume_breadth":   0.20,
        "pct_macd_bullish": 0.15,
    }

    # Conviction multipliers keyed by minimum breadth score
    MULTIPLIERS = [
        (80, 1.25),
        (65, 1.10),
        (45, 1.00),
        (30, 0.85),
        (0,  0.70),
    ]

    # State labels
    STATES = [
        (80, "EXTREME_BULLISH"),
        (65, "BULLISH"),
        (45, "NEUTRAL"),
        (30, "BEARISH"),
        (0,  "COLLAPSE"),
    ]

    def __init__(self, watchlist: list[str] | None = None,
                 concurrency: int = 8,
                 history: BreadthHistory | None = None):
        self.watchlist   = watchlist or self.DEFAULT_WATCHLIST
        self.concurrency = concurrency
        self.history     = history or BreadthHistory()
        self.alerts      = BreadthAlertManager(self.history)

    # ── Public API ────────────────────────────

    async def scan_all(self, exchange) -> BreadthSnapshot:
        """
        Fetch 1H OHLCV for all watchlist assets concurrently (semaphore-limited),
        compute per-asset metrics, build and return the BreadthSnapshot.
        """
        log.info(f"Scanning breadth across {len(self.watchlist)} assets on 1H...")
        sem   = asyncio.Semaphore(self.concurrency)
        tasks = [self._fetch_asset(sym, exchange, sem) for sym in self.watchlist]
        raw   = await asyncio.gather(*tasks, return_exceptions=True)

        assets = [a for a in raw if isinstance(a, AssetBreadth)]
        if not assets:
            log.error("No assets returned from breadth scan")
            return self._empty_snapshot()

        snapshot = self._build_snapshot(assets)
        snapshot.alerts = self.alerts.evaluate_all(snapshot)
        self.history.append(snapshot)
        log.info(
            f"Breadth scan complete: {snapshot.breadth_score:.1f}/100 "
            f"[{snapshot.breadth_state}] ×{snapshot.conviction_multiplier}"
        )
        return snapshot

    def get_multiplier(self, score: float) -> float:
        for threshold, mult in self.MULTIPLIERS:
            if score >= threshold:
                return mult
        return 0.70

    def get_state(self, score: float) -> str:
        for threshold, state in self.STATES:
            if score >= threshold:
                return state
        return "COLLAPSE"

    # ── Fetch one asset ───────────────────────

    async def _fetch_asset(self, symbol: str, exchange,
                           sem: asyncio.Semaphore) -> AssetBreadth | None:
        # Kraken lists most altcoins against USD, not USDT (e.g. PENDLE trades
        # as PENDLE/USD on Kraken, not PENDLE/USDT) -- the watchlist above is
        # written in USDT, so on a Kraken exchange object every symbol was
        # raising ccxt.BadSymbol and getting swallowed below. Try the given
        # symbol first, then fall back to swapping the quote currency.
        candidates = [symbol]
        if symbol.endswith("/USDT"):
            candidates.append(symbol[: -len("USDT")] + "USD")
        elif symbol.endswith("/USD"):
            candidates.append(symbol[: -len("USD")] + "USDT")

        last_err = None
        async with sem:
            for candidate in candidates:
                try:
                    ohlcv = await exchange.fetch_ohlcv(candidate, timeframe="1h", limit=210)
                    if not ohlcv or len(ohlcv) < 50:
                        continue

                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    return self._compute_asset_metrics(symbol, df)

                except Exception as e:
                    last_err = e
                    continue

            # Logged at warning (not debug) so this is actually visible in
            # Render's default INFO-level logging instead of vanishing silently.
            log.warning(f"Error fetching {symbol} (tried {candidates}): {last_err}")
            return None

    def _compute_asset_metrics(self, symbol: str, df: pd.DataFrame) -> AssetBreadth:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # EMAs
        ema20_s  = pta.ema(close, length=20)
        ema200_s = pta.ema(close, length=200)
        ema20    = float(ema20_s.iloc[-1])  if ema20_s  is not None else float(close.iloc[-1])
        ema200   = float(ema200_s.iloc[-1]) if ema200_s is not None else float(close.iloc[-1])

        # RSI
        rsi_s = pta.rsi(close, length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None else 50.0

        # MACD histogram
        macd_r    = pta.macd(close, fast=12, slow=26, signal=9)
        macd_hist = float(macd_r.iloc[-1, 1]) if macd_r is not None and not macd_r.empty else 0.0

        # Price change % vs prior bar
        prev_close = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
        curr_close = float(close.iloc[-1])
        price_chg  = (curr_close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

        # Volume (current bar)
        curr_vol = float(volume.iloc[-1])

        # New 20-bar high / low
        window20   = 20
        recent_high = float(high.iloc[-window20:-1].max()) if len(high) > window20 else float(high.max())
        recent_low  = float(low.iloc[-window20:-1].min())  if len(low)  > window20 else float(low.min())
        new_high_20 = curr_close >= recent_high
        new_low_20  = curr_close <= recent_low

        return AssetBreadth(
            symbol       = symbol,
            close        = curr_close,
            ema20        = ema20,
            ema200       = ema200,
            rsi          = rsi,
            macd_hist    = macd_hist,
            volume_1h    = curr_vol,
            price_chg_1h = price_chg,
            above_ema20  = curr_close > ema20,
            above_ema200 = curr_close > ema200,
            rsi_bullish  = rsi > 50.0,
            macd_bullish = macd_hist > 0.0,
            advancing    = price_chg > 0,
            new_high_20  = new_high_20,
            new_low_20   = new_low_20,
        )

    # ── Build snapshot ────────────────────────

    def _build_snapshot(self, assets: list[AssetBreadth]) -> BreadthSnapshot:
        n = len(assets)

        above_ema20   = sum(1 for a in assets if a.above_ema20)
        above_ema200  = sum(1 for a in assets if a.above_ema200)
        rsi_bullish   = sum(1 for a in assets if a.rsi_bullish)
        macd_bullish  = sum(1 for a in assets if a.macd_bullish)
        advancing     = sum(1 for a in assets if a.advancing)
        declining     = n - advancing
        new_highs     = sum(1 for a in assets if a.new_high_20)
        new_lows      = sum(1 for a in assets if a.new_low_20)

        # Zweig volume breadth
        up_vol   = sum(a.volume_1h for a in assets if a.advancing)
        down_vol = sum(a.volume_1h for a in assets if not a.advancing)
        total_vol = up_vol + down_vol
        vol_breadth = up_vol / total_vol if total_vol > 0 else 0.5

        pct_ema20  = above_ema20  / n * 100
        pct_ema200 = above_ema200 / n * 100
        pct_rsi    = rsi_bullish  / n * 100
        pct_macd   = macd_bullish / n * 100

        # Weighted composite score (volume_breadth scaled to 0–100)
        score = (
            pct_ema20               * self.WEIGHTS["pct_above_ema20"] +
            pct_ema200              * self.WEIGHTS["pct_above_ema200"] +
            pct_rsi                 * self.WEIGHTS["pct_rsi_bullish"] +
            vol_breadth * 100.0     * self.WEIGHTS["volume_breadth"] +
            pct_macd                * self.WEIGHTS["pct_macd_bullish"]
        )
        score = min(100.0, max(0.0, score))

        # Leaders and laggards
        sorted_assets = sorted(assets, key=lambda a: a.price_chg_1h, reverse=True)
        leaders  = [a.symbol.replace("/USDT", "") for a in sorted_assets[:5]]
        laggards = [a.symbol.replace("/USDT", "") for a in sorted_assets[-5:]]

        return BreadthSnapshot(
            timestamp             = datetime.now(timezone.utc),
            total_scanned         = n,
            pct_above_ema20       = round(pct_ema20,  2),
            pct_above_ema200      = round(pct_ema200, 2),
            pct_rsi_bullish       = round(pct_rsi,    2),
            volume_breadth        = round(vol_breadth, 4),
            pct_macd_bullish      = round(pct_macd,   2),
            advance_count         = advancing,
            decline_count         = declining,
            new_highs             = new_highs,
            new_lows              = new_lows,
            up_volume             = up_vol,
            down_volume           = down_vol,
            breadth_score         = round(score, 2),
            breadth_state         = self.get_state(score),
            conviction_multiplier = self.get_multiplier(score),
            alerts                = [],
            leaders               = leaders,
            laggards              = laggards,
            assets                = assets,
        )

    def _empty_snapshot(self) -> BreadthSnapshot:
        return BreadthSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_scanned=0,
            pct_above_ema20=50.0, pct_above_ema200=50.0,
            pct_rsi_bullish=50.0, volume_breadth=0.5,
            pct_macd_bullish=50.0,
            advance_count=0, decline_count=0,
            new_highs=0, new_lows=0,
            up_volume=0.0, down_volume=0.0,
            breadth_score=50.0, breadth_state="NEUTRAL",
            conviction_multiplier=1.00,
            alerts=["WARNING: breadth scan failed — using neutral defaults"],
            leaders=[], laggards=[],
        )


# ─────────────────────────────────────────────
# XRPL Breadth Extension
# ─────────────────────────────────────────────

class XRPLBreadthCalculator(BreadthCalculator):
    """
    Extended breadth calculator that blends CEX data with XRPL DEX tokens.
    Fetches XRPL token prices from XPMarket and DEXScreener APIs
    and includes them in the breadth universe.
    """

    XRPL_TOKENS = [
        # (currency_hex, issuer, display_name)
        ("XRP",                                        None,                                       "XRP"),
        ("4348494C4C475559000000000000000000000000",   "rKcbDBBjRXCZ7ny3ibBkoKLvfK1iqm9qMa",       "CHILLGUY"),
        ("534F4C4F470000000000000000000000000000000",  "rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh",       "SOLOG"),
    ]

    async def fetch_xrpl_price(self, currency: str, issuer: str | None,
                                name: str) -> AssetBreadth | None:
        """Stub: fetch 1H OHLCV from XPMarket/DEXScreener for XRPL tokens."""
        import aiohttp
        try:
            url = f"https://api.firstledger.net/token-v2/{issuer}/{currency}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    data = await resp.json()
                    close = float(data.get("priceXRP", 0))
                    if close <= 0:
                        return None
                    return AssetBreadth(
                        symbol=f"{name}/XRP",
                        close=close,
                        ema20=close,   # simplified without history
                        ema200=close,
                        rsi=50.0,
                        macd_hist=0.0,
                        volume_1h=float(data.get("volume24h", 0)) / 24,
                        price_chg_1h=float(data.get("change24h", 0)) / 24,
                        above_ema20=True,
                        above_ema200=True,
                        rsi_bullish=True,
                        macd_bullish=True,
                        advancing=float(data.get("change24h", 0)) > 0,
                        new_high_20=False,
                        new_low_20=False,
                    )
        except Exception as e:
            log.debug(f"XRPL breadth fetch error for {name}: {e}")
            return None

    async def scan_all(self, exchange) -> BreadthSnapshot:
        snap = await super().scan_all(exchange)
        xrpl_tasks = [
            self.fetch_xrpl_price(c, i, n) for c, i, n in self.XRPL_TOKENS
        ]
        xrpl_assets = [a for a in await asyncio.gather(*xrpl_tasks) if a]
        if xrpl_assets and snap.assets:
            all_assets = snap.assets + xrpl_assets
            snap = self._build_snapshot(all_assets)
            snap.alerts = self.alerts.evaluate_all(snap)
            log.info(f"XRPL breadth: added {len(xrpl_assets)} XRPL assets")
        return snap


# ─────────────────────────────────────────────
# FastAPI Endpoint (for web dashboard)
# ─────────────────────────────────────────────

def create_breadth_router(calculator: BreadthCalculator):
    """
    Returns a FastAPI APIRouter with breadth endpoints.
    Mount this in your main FastAPI app:

      from nexus_breadth import create_breadth_router, BreadthCalculator
      bc = BreadthCalculator()
      app.include_router(create_breadth_router(bc), prefix="/api/breadth")
    """
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router   = APIRouter()
    _cache: dict = {}        # simple in-memory TTL cache
    _exchange = None

    async def get_exchange():
        nonlocal _exchange
        if _exchange is None:
            import ccxt.async_support as ccxt
            _exchange = ccxt.binance({"enableRateLimit": True})
        return _exchange

    @router.get("/latest")
    async def get_latest_breadth():
        """Return the most recent breadth snapshot (cached 5 min)."""
        now = datetime.now(timezone.utc)
        cached = _cache.get("latest")
        if cached and (now - cached["ts"]).seconds < 300:
            return JSONResponse(cached["data"])
        ex   = await get_exchange()
        snap = await calculator.scan_all(ex)
        data = snap.as_dict_simple()
        _cache["latest"] = {"ts": now, "data": data}
        return JSONResponse(data)

    @router.get("/history")
    async def get_breadth_history():
        """Return 24-hour breadth score series and A/D cumulative line."""
        return JSONResponse({
            "scores":         calculator.history.score_series(),
            "pct_above_ema20": calculator.history.pct_above_ema20_series(),
            "ad_cumulative":  calculator.history.ad_cumulative(),
        })

    @router.get("/components")
    async def get_component_scores():
        """Return detailed component scores for the dashboard bars."""
        cached = _cache.get("latest")
        if not cached:
            return JSONResponse({"error": "No data yet — call /latest first"})
        d = cached["data"]
        return JSONResponse({
            "components": [
                {"name": "% above 20 EMA",  "value": d["pct_above_ema20"],       "weight": 25},
                {"name": "% above 200 EMA", "value": d["pct_above_ema200"],      "weight": 20},
                {"name": "RSI breadth",     "value": d["pct_rsi_bullish"],        "weight": 20},
                {"name": "Volume breadth",  "value": d["volume_breadth"] * 100,  "weight": 20},
                {"name": "MACD breadth",    "value": d["pct_macd_bullish"],       "weight": 15},
            ]
        })

    return router


# ─────────────────────────────────────────────
# Telegram Reporter
# ─────────────────────────────────────────────

class BreadthTelegramReporter:
    """
    Sends breadth snapshots and alerts to a Telegram bot.
    Integrates with the existing NEXUS Telegram bot infrastructure.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id   = chat_id

    async def send(self, text: str):
        import aiohttp
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as sess:
                await sess.post(url, json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "Markdown",
                })
        except Exception as e:
            log.error(f"Telegram send error: {e}")

    async def send_snapshot(self, snapshot: BreadthSnapshot):
        await self.send(snapshot.as_telegram())

    async def send_alerts(self, snapshot: BreadthSnapshot):
        if snapshot.alerts:
            for alert in snapshot.alerts:
                await self.send(f"⚡ *NEXUS Breadth Alert*\n{alert}")

    async def send_hourly_digest(self, snapshot: BreadthSnapshot):
        """Called once per hour from Celery scheduler."""
        if snapshot.breadth_state in ("COLLAPSE", "EXTREME_BULLISH", "EXTREME_BEARISH"):
            await self.send_snapshot(snapshot)
        await self.send_alerts(snapshot)


# ─────────────────────────────────────────────
# Scheduled Runner (Celery-compatible)
# ─────────────────────────────────────────────

class BreadthScheduler:
    """
    Manages the hourly breadth scan cycle.
    Can be run standalone or integrated with Celery beat.

    Standalone:
      scheduler = BreadthScheduler()
      asyncio.run(scheduler.run_forever())

    Celery task (celery_tasks.py):
      @app.task
      def breadth_scan_task():
          asyncio.run(BreadthScheduler().run_once())
    """

    def __init__(self, watchlist: list[str] | None = None,
                 telegram_token: str = "", telegram_chat: str = ""):
        self.calculator = XRPLBreadthCalculator(watchlist=watchlist)
        self.telegram   = BreadthTelegramReporter(telegram_token, telegram_chat) \
                          if telegram_token else None

    async def run_once(self) -> BreadthSnapshot:
        import ccxt.async_support as ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        try:
            snap = await self.calculator.scan_all(exchange)
            if self.telegram:
                await self.telegram.send_hourly_digest(snap)
            return snap
        finally:
            await exchange.close()

    async def run_forever(self, interval_seconds: int = 3600):
        """Run breadth scan every hour indefinitely."""
        log.info(f"Breadth scheduler started (interval: {interval_seconds}s)")
        while True:
            try:
                snap = await self.run_once()
                log.info(f"Next scan in {interval_seconds}s")
            except Exception as e:
                log.error(f"Breadth scan cycle error: {e}")
            await asyncio.sleep(interval_seconds)


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

async def main():
    import os
    import ccxt.async_support as ccxt

    calculator = XRPLBreadthCalculator()
    exchange   = ccxt.binance({"enableRateLimit": True})

    print("Running NEXUS hourly breadth scan...")
    snap = await calculator.scan_all(exchange)
    await exchange.close()

    print("\n" + "=" * 60)
    print(snap.as_telegram())
    print("=" * 60)

    if snap.alerts:
        print("\nACTIVE ALERTS:")
        for alert in snap.alerts:
            print(f"  ⚡ {alert}")

    print(f"\nHistory: {len(calculator.history)} snapshots stored")
    print(f"A/D cumulative line: {calculator.history.ad_cumulative()[-5:]}")


if __name__ == "__main__":
    asyncio.run(main())
