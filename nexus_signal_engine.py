"""
NEXUS Signal Scoring Engine
============================
5-factor conviction scorer (0-100):
  Technical Analysis  25%  - TA patterns + indicators + market structure
  On-Chain Analytics  25%  - Glassnode MVRV, SOPR, exchange flows
  Developer Activity  20%  - GitHub velocity, commit rate, ecosystem rank
  Social Sentiment    15%  - Santiment NVT, social dominance, NLP
  Tokenomics Score    15%  - Supply, unlocks, TVL/mcap, revenue

Conviction tiers:
  0-40   SKIP       No action
  40-60  WATCH      Monitor only
  60-75  RESEARCH   Dig deeper
  75-85  POSITION   Small entry (Kelly sized)
  85+    CONVICTION Full position

Install:
  pip install pandas numpy aiohttp pandas-ta ccxt requests \
              pytrends fredapi newsapi-python

New data clients in this version:
  GoogleTrendsClient  - Search interest (retail FOMO early warning)
  FearGreedClient     - Alternative.me fear/greed index (free)
  CoinGeckoClient     - Market cap, trending, dev stats (free)
  WhaleAlertClient    - Large on-chain transfers (smart money)
  TokenTerminalClient - Protocol P/E, revenue, fundamentals
  FREDClient          - Federal Reserve macro (DXY, CPI, M2, rates)
  NewsAPIClient       - Crypto news NLP sentiment
  LunarCrushClient    - Social galaxy score + AltRank
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import pandas_ta as pta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nexus.scorer")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class PatternResult:
    name: str
    detected: bool
    direction: str          # BULLISH / BEARISH / NEUTRAL
    strength: float         # 0-1
    target_price: float | None = None
    notes: str = ""


@dataclass
class SignalScore:
    symbol: str
    timestamp: datetime
    technical_score: float      # 0-100
    onchain_score: float        # 0-100
    dev_score: float            # 0-100
    sentiment_score: float      # 0-100
    tokenomics_score: float     # 0-100
    conviction_score: float     # 0-100 weighted
    action: str                 # SKIP/WATCH/RESEARCH/POSITION/CONVICTION
    patterns: list[PatternResult] = field(default_factory=list)
    entry_zone: tuple[float, float] | None = None   # (low, high)
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward: float | None = None
    raw: dict = field(default_factory=dict)

    def summary(self) -> str:
        detected = [p.name for p in self.patterns if p.detected]
        return (
            f"[{self.action}] {self.symbol} | Score: {self.conviction_score:.1f} | "
            f"TA:{self.technical_score:.0f} OC:{self.onchain_score:.0f} "
            f"DEV:{self.dev_score:.0f} SENT:{self.sentiment_score:.0f} "
            f"TOK:{self.tokenomics_score:.0f} | Patterns: {detected}"
        )


# ─────────────────────────────────────────────
# TA Calculator
# ─────────────────────────────────────────────

class TACalculator:
    """All technical indicators from the BCB TA Toolkit."""

    # ── Indicators ────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        return pta.rsi(close, length=period)

    @staticmethod
    def stoch_rsi(close: pd.Series, period: int = 14, k: int = 3, d: int = 3):
        """Returns (stochrsi_k, stochrsi_d)."""
        result = pta.stochrsi(close, length=period, rsi_length=period, k=k, d=d)
        if result is None or result.empty:
            return None, None
        cols = result.columns.tolist()
        return result[cols[0]], result[cols[1]]

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Returns (macd_line, signal_line, histogram)."""
        result = pta.macd(close, fast=fast, slow=slow, signal=signal)
        if result is None or result.empty:
            return None, None, None
        cols = result.columns.tolist()
        return result[cols[0]], result[cols[2]], result[cols[1]]

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        return pta.ema(close, length=period)

    @staticmethod
    def sma(close: pd.Series, period: int) -> pd.Series:
        return pta.sma(close, length=period)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        return pta.obv(close, volume)

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series) -> pd.Series:
        return pta.vwap(high, low, close, volume)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        return pta.atr(high, low, close, length=period)

    # ── Fibonacci ──────────────────────────────

    @staticmethod
    def fibonacci_levels(high: float, low: float) -> dict[str, float]:
        """
        BCB TA Toolkit retracement and extension levels.

        KEY LEVEL — 0.702 (Optimal Trade Entry):
          Derived from √0.5 ≈ 0.7071, rounded to 0.702 in trading practice.
          Sits between 0.618 and 0.786, acting as the statistical sweet spot
          within the OTE zone. Used prominently in ICT methodology and by
          institutional smart money for precision entries.

          BULLISH OTE:  Price in an uptrend pulls back to 0.702 → highest-
                        probability long entry within the 0.618–0.786 zone.
                        Stop below the 0.786 or swing low. Target: 1.272–1.618.

          BEARISH OTE:  Price in a downtrend rallies to 0.702 of the last
                        down-leg → optimal short re-entry. Stop above 0.786
                        (of the rally). Target: -0.272 to -0.618 extension.

          EXTENSION 1.702 (= 1 + 0.702):
                        Symmetry target after a clean OTE entry. If entry was
                        taken at the 0.702 retracement of leg A, the 1.702
                        extension of leg A is a high-probability take-profit.
                        Particularly reliable when confluence with 1.618 exists.

          CONFLUENCE RULE: 0.702 hit + RSI recovering from oversold (<40) +
                           BOS confirmed + volume surge = CONVICTION entry.
        """
        diff = high - low
        return {
            # ── Retracements ──────────────────────────────────────────
            "retracement_0.236": high - diff * 0.236,   # Minor pullback
            "retracement_0.382": high - diff * 0.382,   # Shallow retrace entry
            "retracement_0.500": high - diff * 0.500,   # Midpoint — institutional magnet
            "retracement_0.618": high - diff * 0.618,   # Golden ratio — OTE zone start
            "retracement_0.702": high - diff * 0.702,   # ★ OTE sweet spot (√0.5)
            "retracement_0.786": high - diff * 0.786,   # OTE zone end / deep retrace
            # ── Extensions ────────────────────────────────────────────
            "extension_1.272":   high + diff * 0.272,   # First target / partial TP
            "extension_1.618":   high + diff * 0.618,   # Primary golden-ratio target
            "extension_1.702":   high + diff * 0.702,   # ★ OTE mirror target (√0.5 ext)
            "extension_2.000":   high + diff * 1.000,   # Measured move / 200% extension
            "extension_2.618":   high + diff * 1.618,   # Max extension (φ²) — swing target
        }

    @staticmethod
    def ote_proximity(price: float, high: float, low: float,
                      tolerance: float = 0.005) -> dict:
        """
        Check if current price is in or near the OTE zone (0.618–0.786).
        Returns proximity score (0-1) and whether 0.702 sweet spot is hit.

        tolerance = 0.5% cushion around each level (handles wicks and spreads).

        Scoring:
          1.0  = Price within 0.5% of exact 0.702 level  → perfect OTE entry
          0.75 = Price within OTE zone (0.618–0.786)      → valid zone entry
          0.50 = Price near 0.618 or 0.786 boundaries     → borderline
          0.0  = Price outside OTE zone entirely
        """
        diff       = high - low
        ote_618    = high - diff * 0.618
        ote_702    = high - diff * 0.702
        ote_786    = high - diff * 0.786

        in_702     = abs(price - ote_702) / (diff + 1e-12) < tolerance
        in_zone    = ote_786 <= price <= ote_618  # inside full OTE zone
        near_618   = abs(price - ote_618) / (diff + 1e-12) < tolerance * 2
        near_786   = abs(price - ote_786) / (diff + 1e-12) < tolerance * 2

        if in_702:
            proximity = 1.0
        elif in_zone:
            # Linear score: highest at 0.702, lower toward edges
            dist_to_702 = abs(price - ote_702) / (diff * 0.168)  # 0.168 = half zone width
            proximity   = max(0.5, 0.75 * (1 - dist_to_702))
        elif near_618 or near_786:
            proximity = 0.4
        else:
            proximity = 0.0

        return {
            "proximity_score":   round(proximity, 3),
            "in_702_sweet_spot": in_702,
            "in_ote_zone":       in_zone,
            "ote_702_level":     round(ote_702, 8),
            "ote_zone_high":     round(ote_618, 8),   # upper bound (closer to high)
            "ote_zone_low":      round(ote_786, 8),   # lower bound (deeper pullback)
        }

    # ── RSI Divergence ─────────────────────────

    @staticmethod
    def rsi_divergence(close: pd.Series, rsi: pd.Series, window: int = 20) -> dict:
        """Detect bullish/bearish regular and hidden divergence."""
        if len(close) < window * 2:
            return {"bullish": False, "bearish": False, "hidden_bullish": False, "hidden_bearish": False}
        recent_close = close.iloc[-window:]
        recent_rsi   = rsi.iloc[-window:]

        price_higher = recent_close.iloc[-1] > recent_close.max().iloc[0] if hasattr(recent_close.max(), 'iloc') else recent_close.iloc[-1] > recent_close[:-1].max()
        price_lower  = recent_close.iloc[-1] < recent_close[:-1].min()
        rsi_higher   = recent_rsi.iloc[-1] > recent_rsi[:-1].max()
        rsi_lower    = recent_rsi.iloc[-1] < recent_rsi[:-1].min()

        return {
            "bullish":        price_lower and not rsi_lower,
            "bearish":        price_higher and not rsi_higher,
            "hidden_bullish": not price_lower and rsi_lower,
            "hidden_bearish": not price_higher and rsi_higher,
        }

    # ── MACD Divergence ────────────────────────

    @staticmethod
    def macd_divergence(close: pd.Series, hist: pd.Series, window: int = 20) -> dict:
        if len(close) < window:
            return {"bullish": False, "bearish": False}
        close_c = close.dropna().iloc[-window:]
        hist_c  = hist.dropna().iloc[-window:]
        if len(close_c) < 2 or len(hist_c) < 2:
            return {"bullish": False, "bearish": False}
        return {
            "bullish": close_c.iloc[-1] < close_c.iloc[0] and hist_c.iloc[-1] > hist_c.iloc[0],
            "bearish": close_c.iloc[-1] > close_c.iloc[0] and hist_c.iloc[-1] < hist_c.iloc[0],
        }


# ─────────────────────────────────────────────
# Pattern Detector
# ─────────────────────────────────────────────

class PatternDetector:
    """
    Detects all BCB TA Toolkit patterns plus SMC structures.
    Requires OHLCV DataFrame with columns: open, high, low, close, volume.
    """

    def __init__(self):
        self.ta = TACalculator()

    def detect_all(self, df: pd.DataFrame) -> list[PatternResult]:
        patterns = [
            self.head_and_shoulders(df),
            self.bull_flag(df),
            self.bear_flag(df),
            self.cup_and_handle(df),
            self.bcb_crashing_structure(df),
            self.wyckoff_accumulation(df),
            self.wyckoff_distribution(df),
            self.elliott_wave_impulse(df),
            self.break_of_structure(df),
            self.change_of_character(df),
            self.order_blocks(df),
            self.fair_value_gaps(df),
            self.liquidity_zones(df),
            self.golden_cross(df),
            self.rsi_oversold_divergence(df),
        ]
        return patterns

    # ── Head & Shoulders ──────────────────────

    def head_and_shoulders(self, df: pd.DataFrame) -> PatternResult:
        """
        3-peak structure: left shoulder, head (tallest), right shoulder.
        Neckline break = trigger. Volume accelerates at peaks.
        """
        if len(df) < 60:
            return PatternResult("Head & Shoulders", False, "BEARISH", 0.0)

        window = df.tail(60)
        highs  = window["high"].values

        def find_local_max(arr, order=5):
            peaks = []
            for i in range(order, len(arr) - order):
                if arr[i] == max(arr[i - order: i + order + 1]):
                    peaks.append(i)
            return peaks

        peaks = find_local_max(highs, order=5)
        if len(peaks) < 3:
            return PatternResult("Head & Shoulders", False, "BEARISH", 0.0)

        ls, head, rs = peaks[-3], peaks[-2], peaks[-1]
        ls_h, head_h, rs_h = highs[ls], highs[head], highs[rs]

        is_hs = (
            head_h > ls_h and
            head_h > rs_h and
            abs(ls_h - rs_h) / head_h < 0.08  # shoulders within 8%
        )
        if not is_hs:
            return PatternResult("Head & Shoulders", False, "BEARISH", 0.0)

        neckline = min(window["low"].iloc[ls:rs + 1])
        last_close = df["close"].iloc[-1]
        broken = last_close < neckline
        strength = 0.85 if broken else 0.5
        target = neckline - (head_h - neckline) if broken else None
        return PatternResult("Head & Shoulders", is_hs, "BEARISH", strength,
                             target_price=target,
                             notes="Neckline broken" if broken else "Watching neckline")

    # ── Bull Flag ────────────────────────────

    def bull_flag(self, df: pd.DataFrame) -> PatternResult:
        """
        Flagpole (sharp move up) + lower-volume pennant pullback.
        Partial retrace only — trend resumes on breakout.
        """
        if len(df) < 30:
            return PatternResult("Bull Flag", False, "BULLISH", 0.0)

        pole_bars = 10
        flag_bars = 10
        pole  = df.tail(pole_bars + flag_bars).head(pole_bars)
        flag  = df.tail(flag_bars)

        pole_return = (pole["close"].iloc[-1] - pole["close"].iloc[0]) / pole["close"].iloc[0]
        flag_return = (flag["close"].iloc[-1] - flag["close"].iloc[0]) / flag["close"].iloc[0]
        vol_ratio   = flag["volume"].mean() / pole["volume"].mean() if pole["volume"].mean() > 0 else 1

        is_flag = (
            pole_return > 0.08 and          # ≥8% flagpole
            -0.08 < flag_return < 0.01 and  # flag pulls back but not fully
            vol_ratio < 0.75                # lower volume on pullback
        )
        strength = min(1.0, pole_return * 5) if is_flag else 0.0
        target   = pole["high"].iloc[-1] + (pole["close"].iloc[-1] - pole["close"].iloc[0]) if is_flag else None
        return PatternResult("Bull Flag", is_flag, "BULLISH", strength,
                             target_price=target,
                             notes=f"Pole:{pole_return:.1%} Flag:{flag_return:.1%} Vol:{vol_ratio:.2f}")

    # ── Bear Flag ────────────────────────────

    def bear_flag(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 30:
            return PatternResult("Bear Flag", False, "BEARISH", 0.0)

        pole = df.tail(20).head(10)
        flag = df.tail(10)
        pole_return = (pole["close"].iloc[-1] - pole["close"].iloc[0]) / pole["close"].iloc[0]
        flag_return = (flag["close"].iloc[-1] - flag["close"].iloc[0]) / flag["close"].iloc[0]
        vol_ratio   = flag["volume"].mean() / (pole["volume"].mean() + 1e-9)

        is_flag = (
            pole_return < -0.08 and
            -0.01 < flag_return < 0.08 and
            vol_ratio < 0.75
        )
        strength = min(1.0, abs(pole_return) * 5) if is_flag else 0.0
        target   = pole["low"].iloc[-1] + (pole["close"].iloc[-1] - pole["close"].iloc[0]) if is_flag else None
        return PatternResult("Bear Flag", is_flag, "BEARISH", strength,
                             target_price=target,
                             notes=f"Pole:{pole_return:.1%}")

    # ── Cup & Handle ──────────────────────────

    def cup_and_handle(self, df: pd.DataFrame) -> PatternResult:
        """Rounded U-base + smaller handle with downward tilt. Upside breakout."""
        if len(df) < 80:
            return PatternResult("Cup & Handle", False, "BULLISH", 0.0)

        cup_window   = df.tail(80).head(60)
        handle_window = df.tail(20)
        cup_low   = cup_window["low"].min()
        cup_start = cup_window["high"].iloc[0]
        cup_end   = cup_window["high"].iloc[-1]
        handle_low = handle_window["low"].min()

        is_cup_and_handle = (
            cup_low < cup_start * 0.92 and
            abs(cup_start - cup_end) / cup_start < 0.05 and
            handle_low > cup_low and
            handle_low < cup_end and
            handle_window["high"].max() < cup_end * 1.03
        )
        strength = 0.75 if is_cup_and_handle else 0.0
        target   = cup_end + (cup_end - cup_low) if is_cup_and_handle else None
        return PatternResult("Cup & Handle", is_cup_and_handle, "BULLISH", strength,
                             target_price=target)

    # ── BCB Crashing Structure ────────────────

    def bcb_crashing_structure(self, df: pd.DataFrame) -> PatternResult:
        """
        7-wave reversal from BCB toolkit:
        Waves 1-4: two bounces. Wave 5: main drop (internal 5-wave).
        Wave 6: rally to Wave 4. Wave 7: new low. Strong reversal indicator.
        Detection: look for a confirmed Wave 5 drop after structure break.
        """
        if len(df) < 100:
            return PatternResult("BCB Crashing Structure", False, "BEARISH", 0.0)

        window = df.tail(100)
        highs  = window["high"].values
        lows   = window["low"].values
        closes = window["close"].values

        # Simplified: look for progressive lower lows with two bounces (W1-W4)
        # then a sharp W5 drop
        recent_high = highs[:50].max()
        recent_high_idx = np.argmax(highs[:50])
        after_high  = lows[recent_high_idx:]

        if len(after_high) < 20:
            return PatternResult("BCB Crashing Structure", False, "BEARISH", 0.0)

        # Two bounces then continuation drop
        bounce_count = 0
        prev_low = after_high[0]
        for i in range(1, min(len(after_high), 40)):
            if after_high[i] > prev_low * 1.03:
                bounce_count += 1
            prev_low = min(prev_low, after_high[i])

        wave5_drop = (after_high[-1] - recent_high) / recent_high  # negative
        is_bcb = bounce_count >= 2 and wave5_drop < -0.20  # ≥20% drop with 2 bounces

        strength = min(1.0, abs(wave5_drop) * 2) if is_bcb else 0.0
        target   = lows[-1] * 0.85 if is_bcb else None  # project further decline
        return PatternResult("BCB Crashing Structure", is_bcb, "BEARISH", strength,
                             target_price=target,
                             notes=f"Wave5 drop:{wave5_drop:.1%} Bounces:{bounce_count}")

    # ── Wyckoff Accumulation ──────────────────

    def wyckoff_accumulation(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Wyckoff Accumulation: selling climax (SC) → spring → SOS.
        Key: decreasing volume on sells, volume spike at SC, spring (false low break).
        """
        if len(df) < 120:
            return PatternResult("Wyckoff Accumulation", False, "BULLISH", 0.0)

        window = df.tail(120)
        vol    = window["volume"].values
        closes = window["close"].values
        lows   = window["low"].values

        # Selling climax: large red candle with high volume in first 30 bars
        first_third   = window.head(40)
        sc_candidate  = first_third[first_third["volume"] > first_third["volume"].quantile(0.85)]
        has_sc        = len(sc_candidate) > 0

        # Trading range (AR to potential Spring)
        if not has_sc:
            return PatternResult("Wyckoff Accumulation", False, "BULLISH", 0.0)

        range_low  = window["low"].min()
        range_high = window["high"].max()
        range_size = range_high - range_low

        # Spring: price briefly dips below range low then recovers
        last_30 = window.tail(30)
        spring  = (last_30["low"].min() < range_low and
                   last_30["close"].iloc[-1] > range_low)

        # SOS: strong close above midpoint on high volume
        midpoint = range_low + range_size * 0.5
        last_10  = window.tail(10)
        sos      = (last_10["close"].iloc[-1] > midpoint and
                    last_10["volume"].mean() > vol.mean() * 1.2)

        is_accumulation = has_sc and (spring or sos)
        strength = 0.0
        if is_accumulation:
            strength = 0.65 + (0.15 if spring else 0.0) + (0.15 if sos else 0.0)

        target = range_high + range_size * 0.5 if is_accumulation else None
        return PatternResult("Wyckoff Accumulation", is_accumulation, "BULLISH", strength,
                             target_price=target,
                             notes=f"SC:{has_sc} Spring:{spring} SOS:{sos}")

    # ── Wyckoff Distribution ──────────────────

    def wyckoff_distribution(self, df: pd.DataFrame) -> PatternResult:
        """Detect Wyckoff Distribution: buying climax (BC) → upthrust → SOW."""
        if len(df) < 120:
            return PatternResult("Wyckoff Distribution", False, "BEARISH", 0.0)

        window = df.tail(120)
        first_third = window.head(40)
        bc_candidate = first_third[first_third["volume"] > first_third["volume"].quantile(0.85)]
        has_bc       = len(bc_candidate) > 0

        if not has_bc:
            return PatternResult("Wyckoff Distribution", False, "BEARISH", 0.0)

        range_high = window["high"].max()
        range_low  = window["low"].min()
        range_size = range_high - range_low

        last_30 = window.tail(30)
        upthrust = (last_30["high"].max() > range_high and
                    last_30["close"].iloc[-1] < range_high)
        midpoint = range_low + range_size * 0.5
        last_10  = window.tail(10)
        sow      = (last_10["close"].iloc[-1] < midpoint and
                    last_10["volume"].mean() > window["volume"].mean() * 1.2)

        is_dist  = has_bc and (upthrust or sow)
        strength = 0.0
        if is_dist:
            strength = 0.65 + (0.15 if upthrust else 0.0) + (0.15 if sow else 0.0)

        target = range_low - range_size * 0.5 if is_dist else None
        return PatternResult("Wyckoff Distribution", is_dist, "BEARISH", strength,
                             target_price=target,
                             notes=f"BC:{has_bc} UT:{upthrust} SOW:{sow}")

    # ── Elliott Wave Impulse ──────────────────

    def elliott_wave_impulse(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect 5-wave impulse structure using swing highs/lows.
        Rules: W2 < 100% of W1, W3 not shortest, W4 doesn't enter W1.
        """
        if len(df) < 80:
            return PatternResult("Elliott Wave Impulse", False, "BULLISH", 0.0)

        window = df.tail(80)
        closes = window["close"].values

        def find_swings(arr, order=5):
            highs, lows = [], []
            for i in range(order, len(arr) - order):
                seg = arr[i - order: i + order + 1]
                if arr[i] == max(seg):
                    highs.append((i, arr[i]))
                elif arr[i] == min(seg):
                    lows.append((i, arr[i]))
            return highs, lows

        highs, lows = find_swings(closes, order=4)
        if len(highs) < 3 or len(lows) < 2:
            return PatternResult("Elliott Wave Impulse", False, "BULLISH", 0.0)

        # Use last 5 swing points
        all_swings = sorted(highs + lows, key=lambda x: x[0])[-6:]
        if len(all_swings) < 5:
            return PatternResult("Elliott Wave Impulse", False, "BULLISH", 0.0)

        prices = [s[1] for s in all_swings[-5:]]
        w1 = abs(prices[1] - prices[0])
        w2 = abs(prices[2] - prices[1])
        w3 = abs(prices[3] - prices[2])
        w4 = abs(prices[4] - prices[3])

        rule1 = w2 < w1             # W2 retraces < 100% of W1
        rule2 = w3 >= min(w1, w4)   # W3 is not the shortest
        rule3 = prices[4] > prices[1]  # W4 doesn't enter W1 territory (simplified)

        is_impulse = rule1 and rule2 and rule3 and prices[1] < prices[3]
        strength = 0.7 if is_impulse else 0.0
        target = prices[4] + w3 if is_impulse else None
        return PatternResult("Elliott Wave Impulse", is_impulse, "BULLISH", strength,
                             target_price=target,
                             notes=f"Rules: W2<W1:{rule1} W3!=shortest:{rule2} W4:{rule3}")

    # ── Market Structure: BOS / CHoCH ─────────

    def break_of_structure(self, df: pd.DataFrame) -> PatternResult:
        """Bullish BOS: new higher high. Bearish BOS: new lower low."""
        if len(df) < 30:
            return PatternResult("Break of Structure", False, "NEUTRAL", 0.0)

        window    = df.tail(30)
        prev_high = window.iloc[:-5]["high"].max()
        prev_low  = window.iloc[:-5]["low"].min()
        last_high = window.iloc[-5:]["high"].max()
        last_low  = window.iloc[-5:]["low"].min()

        bullish_bos = last_high > prev_high
        bearish_bos = last_low  < prev_low

        direction = "BULLISH" if bullish_bos else ("BEARISH" if bearish_bos else "NEUTRAL")
        detected  = bullish_bos or bearish_bos
        return PatternResult("Break of Structure", detected, direction, 0.7 if detected else 0.0,
                             notes="HH" if bullish_bos else ("LL" if bearish_bos else "None"))

    def change_of_character(self, df: pd.DataFrame) -> PatternResult:
        """First reversal signal — opposite direction structure break."""
        if len(df) < 50:
            return PatternResult("Change of Character", False, "NEUTRAL", 0.0)

        prev_trend = df.tail(50).head(30)
        latest     = df.tail(20)
        prev_dir   = "UP" if prev_trend["close"].iloc[-1] > prev_trend["close"].iloc[0] else "DOWN"

        if prev_dir == "UP":
            choch = latest["low"].min() < prev_trend["low"].min()
            direction = "BEARISH" if choch else "NEUTRAL"
        else:
            choch = latest["high"].max() > prev_trend["high"].max()
            direction = "BULLISH" if choch else "NEUTRAL"

        return PatternResult("Change of Character", choch, direction, 0.8 if choch else 0.0,
                             notes=f"Prior trend: {prev_dir}")

    def order_blocks(self, df: pd.DataFrame) -> PatternResult:
        """
        Bullish OB: last bearish candle before impulsive up move.
        Bearish OB: last bullish candle before impulsive down move.
        """
        if len(df) < 20:
            return PatternResult("Order Block", False, "NEUTRAL", 0.0)

        window = df.tail(20)
        impulse_up   = window["close"].diff().rolling(3).sum()
        impulse_down = impulse_up

        # Find strongest 3-bar move
        max_up_idx  = impulse_up.idxmax()
        max_down_idx = impulse_down.idxmin()

        bullish_ob = max_up_idx is not None and window.index.get_loc(max_up_idx) > 3
        direction  = "BULLISH" if bullish_ob else "NEUTRAL"
        return PatternResult("Order Block", bullish_ob, direction, 0.65 if bullish_ob else 0.0)

    def fair_value_gaps(self, df: pd.DataFrame) -> PatternResult:
        """3-candle imbalance: gap between candle 1 high and candle 3 low (bullish FVG)."""
        if len(df) < 5:
            return PatternResult("Fair Value Gap", False, "NEUTRAL", 0.0)

        detected, direction = False, "NEUTRAL"
        for i in range(len(df) - 3, max(len(df) - 15, 0), -1):
            c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]
            if c3["low"] > c1["high"]:     # Bullish FVG
                detected, direction = True, "BULLISH"
                break
            elif c3["high"] < c1["low"]:   # Bearish FVG
                detected, direction = True, "BEARISH"
                break

        return PatternResult("Fair Value Gap", detected, direction, 0.6 if detected else 0.0)

    def liquidity_zones(self, df: pd.DataFrame) -> PatternResult:
        """
        Equal highs/lows = stop clusters (BSL/SSL).
        Combined with CoinGlass liquidation heatmap externally.
        """
        if len(df) < 40:
            return PatternResult("Liquidity Zone", False, "NEUTRAL", 0.0)

        window = df.tail(40)
        highs  = window["high"].values
        lows   = window["low"].values

        # Equal highs: two highs within 0.3% of each other
        equal_highs = any(
            abs(highs[i] - highs[j]) / highs[i] < 0.003
            for i in range(len(highs) - 1)
            for j in range(i + 1, min(i + 10, len(highs)))
        )
        equal_lows = any(
            abs(lows[i] - lows[j]) / (lows[i] + 1e-9) < 0.003
            for i in range(len(lows) - 1)
            for j in range(i + 1, min(i + 10, len(lows)))
        )

        detected  = equal_highs or equal_lows
        direction = "BEARISH" if equal_highs else ("BULLISH" if equal_lows else "NEUTRAL")
        return PatternResult("Liquidity Zone", detected, direction, 0.6 if detected else 0.0,
                             notes=f"EqualHighs:{equal_highs} EqualLows:{equal_lows}")

    def golden_cross(self, df: pd.DataFrame) -> PatternResult:
        """50 EMA crosses above 200 EMA (Golden Cross) or below (Death Cross)."""
        if len(df) < 210:
            return PatternResult("MA Cross", False, "NEUTRAL", 0.0)

        ema50  = self.ta.ema(df["close"], 50)
        ema200 = self.ta.ema(df["close"], 200)
        if ema50 is None or ema200 is None:
            return PatternResult("MA Cross", False, "NEUTRAL", 0.0)

        prev_above = ema50.iloc[-2] > ema200.iloc[-2]
        curr_above = ema50.iloc[-1] > ema200.iloc[-1]
        if not prev_above and curr_above:
            return PatternResult("Golden Cross", True, "BULLISH", 0.9, notes="50 EMA > 200 EMA")
        if prev_above and not curr_above:
            return PatternResult("Death Cross", True, "BEARISH", 0.9, notes="50 EMA < 200 EMA")
        return PatternResult("MA Cross", False, "NEUTRAL", 0.0)

    def rsi_oversold_divergence(self, df: pd.DataFrame) -> PatternResult:
        """RSI below 30 with bullish divergence — BCB Toolkit key signal."""
        if len(df) < 30:
            return PatternResult("RSI Oversold+Divergence", False, "BULLISH", 0.0)

        rsi  = self.ta.rsi(df["close"], 14)
        if rsi is None:
            return PatternResult("RSI Oversold+Divergence", False, "BULLISH", 0.0)

        oversold = rsi.iloc[-1] < 30
        divergence = self.ta.rsi_divergence(df["close"], rsi)
        detected = oversold and divergence.get("bullish", False)
        return PatternResult("RSI Oversold+Divergence", detected, "BULLISH",
                             0.85 if detected else (0.3 if oversold else 0.0),
                             notes=f"RSI:{rsi.iloc[-1]:.1f} Div:{divergence.get('bullish')}")


# ─────────────────────────────────────────────
# Data Clients (async)
# ─────────────────────────────────────────────

class BaseClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.headers  = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def get(self, path: str, params: dict | None = None) -> dict:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f"{self.base_url}{path}", params=params) as resp:
                resp.raise_for_status()
                return await resp.json()


class CoinGlassClient(BaseClient):
    """CoinGlass derivatives intelligence: liquidation maps, L/S ratios, funding, OI."""

    def __init__(self, api_key: str):
        super().__init__("https://open-api.coinglass.com/public/v2", api_key)
        self.headers["coinglassSecret"] = api_key

    async def get_long_short_ratio(self, symbol: str, period: str = "4h") -> dict:
        return await self.get("/indicator/long_short_account_ratio",
                              {"symbol": symbol, "time_type": period, "limit": 24})

    async def get_funding_rate(self, symbol: str) -> dict:
        return await self.get("/funding/oi_funding_rate", {"symbol": symbol, "limit": 1})

    async def get_open_interest(self, symbol: str) -> dict:
        return await self.get("/indicator/open_interest", {"symbol": symbol, "time_type": "4h", "limit": 24})

    async def compute_derivatives_score(self, symbol: str) -> float:
        """0-100 score from derivatives signals."""
        try:
            ls   = await self.get_long_short_ratio(symbol)
            fund = await self.get_funding_rate(symbol)
            oi   = await self.get_open_interest(symbol)

            score = 50.0
            # Long/short < 1 = majority short = contrarian bullish
            ls_ratio = float(ls.get("data", [{}])[0].get("longShortRatio", 1.0))
            if ls_ratio < 0.8:
                score += 15
            elif ls_ratio > 1.5:
                score -= 15

            # Negative funding = shorts paying longs = bullish squeeze potential
            fr = float(fund.get("data", {}).get("fundingRate", 0))
            if fr < -0.01:
                score += 20
            elif fr > 0.05:
                score -= 10

            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"CoinGlass error for {symbol}: {e}")
            return 50.0


class GlassnodeClient(BaseClient):
    """Glassnode on-chain analytics: MVRV, SOPR, exchange flows."""

    def __init__(self, api_key: str):
        super().__init__("https://api.glassnode.com/v1/metrics", api_key)

    async def get_mvrv(self, asset: str = "BTC") -> float:
        data = await self.get("/market/mvrv", {"a": asset, "f": "JSON", "i": "24h"})
        values = [d.get("v", 1.0) for d in (data if isinstance(data, list) else [])]
        return values[-1] if values else 1.0

    async def get_sopr(self, asset: str = "BTC") -> float:
        data = await self.get("/sopr/sopr", {"a": asset, "f": "JSON", "i": "24h"})
        values = [d.get("v", 1.0) for d in (data if isinstance(data, list) else [])]
        return values[-1] if values else 1.0

    async def get_exchange_net_flow(self, asset: str = "BTC") -> float:
        """Negative flow = net outflow (bullish accumulation)."""
        data = await self.get("/distribution/exchange_net_position_change",
                              {"a": asset, "f": "JSON", "i": "24h"})
        values = [d.get("v", 0.0) for d in (data if isinstance(data, list) else [])]
        return values[-1] if values else 0.0

    async def compute_onchain_score(self, asset: str) -> float:
        try:
            mvrv    = await self.get_mvrv(asset)
            sopr    = await self.get_sopr(asset)
            netflow = await self.get_exchange_net_flow(asset)

            score = 50.0
            # MVRV < 1 = undervalued (bullish), > 3.5 = overheated (bearish)
            if mvrv < 1.0:
                score += 25
            elif 1.0 <= mvrv <= 2.0:
                score += 10
            elif mvrv > 3.5:
                score -= 25

            # SOPR < 1 = selling at loss = capitulation (bullish)
            if sopr < 1.0:
                score += 15
            elif sopr > 1.05:
                score -= 5

            # Net outflow from exchanges = accumulation (bullish)
            if netflow < 0:
                score += 15
            elif netflow > 0:
                score -= 10

            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"Glassnode error for {asset}: {e}")
            return 50.0


class SantimentClient(BaseClient):
    """Santiment behavioral analytics and social intelligence."""

    def __init__(self, api_key: str):
        super().__init__("https://api.santiment.net/graphql", api_key)

    async def query(self, gql: str) -> dict:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(self.base_url, json={"query": gql}) as resp:
                return await resp.json()

    async def get_social_dominance(self, slug: str) -> float:
        from_dt = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_dt   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        gql = f"""{{
          socialDominance(slug: "{slug}", from: "{from_dt}", to: "{to_dt}", interval: "1d") {{
            datetime value
          }}
        }}"""
        result = await self.query(gql)
        data   = result.get("data", {}).get("socialDominance", [])
        return float(data[-1]["value"]) if data else 0.0

    async def compute_sentiment_score(self, slug: str) -> float:
        try:
            dominance = await self.get_social_dominance(slug)
            # Ideal: rising but not parabolic (1-5% = healthy, >8% = fomo risk)
            score = 50.0
            if 1.0 <= dominance <= 5.0:
                score += 20
            elif dominance > 8.0:
                score -= 10
            elif dominance < 0.5:
                score -= 5
            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"Santiment error for {slug}: {e}")
            return 50.0


class DeveloperReportClient(BaseClient):
    """Developer activity via developerreport.com API."""

    def __init__(self, api_key: str):
        super().__init__("https://api.developerreport.com/api/v1", api_key)

    async def get_dev_activity(self, project: str) -> dict:
        return await self.get(f"/projects/{project}/activity")

    async def compute_dev_score(self, project: str) -> float:
        try:
            data    = await self.get_dev_activity(project)
            commits = data.get("commits_30d", 0)
            devs    = data.get("active_developers", 0)
            rank    = data.get("ecosystem_rank", 500)

            score = 20.0
            score += min(40, commits / 5)   # up to 40 pts for 200+ commits/month
            score += min(20, devs * 2)      # up to 20 pts for 10+ active devs
            score += max(0, 20 - rank / 25) # up to 20 pts for top 100 rank
            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"DeveloperReport error for {project}: {e}")
            return 40.0


class DeFiLlamaClient(BaseClient):
    """DeFi TVL and protocol intelligence."""

    def __init__(self):
        super().__init__("https://api.llama.fi")

    async def get_protocol_tvl(self, protocol: str) -> dict:
        return await self.get(f"/protocol/{protocol}")

    async def compute_tokenomics_score(self, protocol: str, mcap: float | None = None) -> float:
        try:
            data = await self.get_protocol_tvl(protocol)
            tvl  = data.get("currentChainTvls", {})
            total_tvl = sum(float(v) for v in tvl.values()) if tvl else 0

            score = 50.0
            if total_tvl > 100_000_000:
                score += 20
            elif total_tvl > 10_000_000:
                score += 10

            if mcap and mcap > 0:
                tvl_mcap = total_tvl / mcap
                if tvl_mcap > 0.5:
                    score += 15
                elif tvl_mcap < 0.1:
                    score -= 10

            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"DeFiLlama error for {protocol}: {e}")
            return 50.0


# ─────────────────────────────────────────────
# NEW DATA CLIENTS
# ─────────────────────────────────────────────

class GoogleTrendsClient:
    """
    Google Trends retail interest via pytrends.
    WHY IT MATTERS: Search interest for crypto terms historically leads price
    by 2–4 weeks. A rising trend before a price move = early retail FOMO signal.
    A parabolic spike usually marks blow-off tops (sell signal).

    Signal logic:
      Rising trend (<70, slope >0) → bullish accumulation (retail not yet in)
      Spike >85 → retail FOMO peak → reduce / avoid new longs
      Near 0–10 → nobody cares → long-term accumulation zone

    Requires: pip install pytrends
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from pytrends.request import TrendReq
            self._client = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        return self._client

    def get_interest(self, keyword: str, timeframe: str = "today 3-m") -> dict:
        """
        Returns recent search interest data for a keyword.
        timeframe: 'today 3-m' | 'today 12-m' | 'today 1-m'
        """
        try:
            pt = self._get_client()
            pt.build_payload([keyword], cat=0, timeframe=timeframe, geo="", gprop="")
            df = pt.interest_over_time()
            if df.empty:
                return {"current": 0, "slope": 0, "peak": 0}
            series  = df[keyword].dropna()
            current = int(series.iloc[-1])
            peak    = int(series.max())
            # Slope over last 4 data points (weekly buckets)
            slope   = float(series.tail(4).diff().mean()) if len(series) >= 4 else 0.0
            return {"current": current, "slope": round(slope, 2), "peak": peak,
                    "series": series.tolist()}
        except Exception as e:
            log.warning(f"Google Trends error for '{keyword}': {e}")
            return {"current": 0, "slope": 0, "peak": 0}

    def get_related_queries(self, keyword: str) -> list[str]:
        """Rising related search queries — useful for narrative discovery."""
        try:
            pt = self._get_client()
            pt.build_payload([keyword], timeframe="today 3-m")
            related = pt.related_queries()
            rising  = related.get(keyword, {}).get("rising")
            if rising is not None and not rising.empty:
                return rising["query"].tolist()[:10]
        except Exception as e:
            log.warning(f"Trends related queries error: {e}")
        return []

    def compute_trends_score(self, keyword: str) -> float:
        """
        0-100 score from Google Trends.
        Ideal: interest rising (slope >0) but not yet at peak (current < 70).
        Danger zone: current > 85 (retail FOMO peak → fading).
        """
        data    = self.get_interest(keyword)
        current = data["current"]
        slope   = data["slope"]
        score   = 50.0

        # Rising but not peaked = best accumulation environment
        if current < 20 and slope >= 0:
            score += 15   # Nobody watching — early mover advantage
        elif 20 <= current < 50 and slope > 0:
            score += 20   # Healthy rising interest — ideal entry window
        elif 50 <= current < 75 and slope > 0:
            score += 10   # Good momentum, still room to grow
        elif current >= 85:
            score -= 25   # Parabolic retail FOMO — top signal
        elif current >= 75:
            score -= 10   # Getting crowded

        # Accelerating slope adds bonus
        if slope > 5:
            score += 10
        elif slope < -5:
            score -= 10

        return min(100.0, max(0.0, score))


class FearGreedClient:
    """
    Alternative.me Crypto Fear & Greed Index (free, no API key).
    WHY IT MATTERS: Extreme fear = others are selling, consider buying.
    Extreme greed = others are buying recklessly, consider reducing.
    Classic contrarian macro context for position sizing.

    Index:
      0–24   Extreme Fear    → aggressive buy zone
      25–49  Fear            → mild buy bias
      50     Neutral
      51–74  Greed           → reduce new longs
      75–100 Extreme Greed   → take profits / avoid chasing
    """

    URL = "https://api.alternative.me/fng/?limit=30&format=json"

    async def get_index(self) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    entries = data.get("data", [])
                    if not entries:
                        return {"value": 50, "label": "Neutral", "trend": "flat"}
                    current = int(entries[0]["value"])
                    label   = entries[0]["value_classification"]
                    # Trend: compare to 7 days ago
                    week_ago = int(entries[min(7, len(entries) - 1)]["value"])
                    trend    = "rising" if current > week_ago else ("falling" if current < week_ago else "flat")
                    return {"value": current, "label": label, "trend": trend,
                            "week_ago": week_ago, "delta": current - week_ago}
        except Exception as e:
            log.warning(f"Fear & Greed API error: {e}")
            return {"value": 50, "label": "Neutral", "trend": "flat"}

    async def compute_feargreed_score(self) -> float:
        """
        Contrarian score: extreme fear → high score (buy), extreme greed → low score.
        """
        data  = await self.get_index()
        value = data["value"]
        trend = data["trend"]
        score = 50.0

        if value <= 15:
            score = 90.0   # Capitulation — peak fear = peak opportunity
        elif value <= 25:
            score = 80.0   # Extreme fear
        elif value <= 40:
            score = 65.0   # Fear zone — decent entry
        elif value <= 55:
            score = 50.0   # Neutral
        elif value <= 70:
            score = 40.0   # Greed building — caution
        elif value <= 85:
            score = 30.0   # High greed — reduce risk
        else:
            score = 15.0   # Extreme greed — avoid new longs

        # Bonus: fear turning to greed = momentum confirmation
        if trend == "rising" and value < 50:
            score += 8
        elif trend == "falling" and value > 50:
            score -= 8

        return min(100.0, max(0.0, score))


class CoinGeckoClient(BaseClient):
    """
    CoinGecko market data — free tier, no API key required.
    WHY IT MATTERS: Trending coins, developer stats, community data,
    and market cap data all in one free source. Identifies emerging
    narratives before they hit mainstream.

    Key signals:
      trending_coins     → what retail/crypto twitter is chasing NOW
      developer_score    → GitHub activity via CoinGecko's own scoring
      community_score    → social community engagement strength
      liquidity_score    → how easy it is to enter/exit positions
    """

    def __init__(self):
        super().__init__("https://api.coingecko.com/api/v3")

    async def get_trending(self) -> list[dict]:
        """Top 7 trending coins in the last 24h by search volume."""
        try:
            data = await self.get("/search/trending")
            coins = data.get("coins", [])
            return [
                {
                    "id":     c["item"]["id"],
                    "name":   c["item"]["name"],
                    "symbol": c["item"]["symbol"],
                    "rank":   c["item"].get("market_cap_rank"),
                    "score":  c["item"].get("score", 0),
                }
                for c in coins
            ]
        except Exception as e:
            log.warning(f"CoinGecko trending error: {e}")
            return []

    async def get_coin_data(self, coin_id: str) -> dict:
        """Full coin data: price, market cap, dev scores, community metrics."""
        try:
            return await self.get(
                f"/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "true",
                    "developer_data": "true",
                }
            )
        except Exception as e:
            log.warning(f"CoinGecko coin data error for {coin_id}: {e}")
            return {}

    async def compute_coingecko_score(self, coin_id: str,
                                       watchlist: list[str] | None = None) -> float:
        """0-100 score combining dev, community, and trending status."""
        try:
            data    = await self.get_coin_data(coin_id)
            dev     = data.get("developer_data", {})
            comm    = data.get("community_data", {})
            score   = 40.0

            # Developer activity
            commits_4w  = dev.get("commit_count_4_weeks", 0)
            stars       = dev.get("stars", 0)
            subscribers = dev.get("subscribers", 0)
            score += min(20, commits_4w / 3)          # up to +20 for 60+ commits/month
            score += min(8, math.log10(max(1, stars))) # log scale for stars

            # Community engagement
            twitter_fol = comm.get("twitter_followers", 0)
            reddit_subs = comm.get("reddit_subscribers", 0)
            score += min(10, math.log10(max(1, twitter_fol)))
            score += min(8,  math.log10(max(1, reddit_subs)))

            # Trending bonus
            trending = await self.get_trending()
            trending_ids = [t["id"] for t in trending]
            if coin_id in trending_ids:
                rank_in_trending = next(
                    (i for i, t in enumerate(trending) if t["id"] == coin_id), 7
                )
                score += max(5, 20 - rank_in_trending * 2)  # #1 trending = +20

            # Watchlist membership bonus
            if watchlist and coin_id in watchlist:
                score += 5

            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"CoinGecko score error for {coin_id}: {e}")
            return 45.0


class WhaleAlertClient(BaseClient):
    """
    Whale Alert API — large on-chain transaction tracker.
    WHY IT MATTERS: Smart money moves before price moves. A sudden spike
    in large wallet transactions from cold storage to exchange = distribution.
    Cold storage inflows (exchange → unknown) = accumulation.

    Key signals:
      Exchange inflow  (whale → exchange)   → bearish (about to sell)
      Exchange outflow (exchange → unknown) → bullish (removing from market)
      Whale accumulation clusters           → pre-pump signal
      Large mint/burn events                → tokenomics change signal

    Requires: pip install (no extra lib, uses aiohttp)
    API: https://whale-alert.io/api
    """

    def __init__(self, api_key: str):
        super().__init__("https://api.whale-alert.io/v1", api_key)
        self.headers["X-WA-API-KEY"] = api_key

    async def get_recent_transactions(self, symbol: str,
                                       min_value_usd: int = 1_000_000,
                                       limit: int = 20) -> list[dict]:
        """Fetch large transactions for a symbol in the last hour."""
        try:
            now  = int(__import__("time").time())
            data = await self.get("/transactions", params={
                "api_key":    self.headers.get("X-WA-API-KEY", ""),
                "min_value":  min_value_usd,
                "start":      now - 3600,
                "limit":      limit,
                "currency":   symbol.lower(),
            })
            return data.get("transactions", [])
        except Exception as e:
            log.warning(f"WhaleAlert error for {symbol}: {e}")
            return []

    def classify_transaction(self, tx: dict) -> str:
        """Classify transaction direction based on from/to wallet type."""
        from_type = tx.get("from", {}).get("owner_type", "unknown")
        to_type   = tx.get("to",   {}).get("owner_type", "unknown")
        if to_type == "exchange" and from_type not in ("exchange",):
            return "EXCHANGE_INFLOW"    # Bearish: moving to sell
        elif from_type == "exchange" and to_type not in ("exchange",):
            return "EXCHANGE_OUTFLOW"   # Bullish: removing from market
        elif from_type == "unknown" and to_type == "unknown":
            return "WALLET_TRANSFER"    # Accumulation or redistribution
        return "OTHER"

    async def compute_whale_score(self, symbol: str) -> float:
        """
        0-100 score from whale transaction flow.
        Net outflow (bullish) vs net inflow (bearish) weighting.
        """
        txs = await self.get_recent_transactions(symbol)
        if not txs:
            return 50.0

        inflow_usd  = 0.0
        outflow_usd = 0.0
        for tx in txs:
            usd = float(tx.get("amount_usd", 0))
            cls = self.classify_transaction(tx)
            if cls == "EXCHANGE_INFLOW":
                inflow_usd  += usd
            elif cls == "EXCHANGE_OUTFLOW":
                outflow_usd += usd

        total = inflow_usd + outflow_usd
        if total == 0:
            return 50.0

        # Outflow ratio: higher = more bullish
        outflow_ratio = outflow_usd / total
        score = 50.0 + (outflow_ratio - 0.5) * 80  # ±40 range
        return min(100.0, max(0.0, score))


class TokenTerminalClient(BaseClient):
    """
    Token Terminal — protocol P/E ratios, revenue, fees, TVL.
    WHY IT MATTERS: Price/Sales and Price/Earnings for crypto protocols.
    A protocol with rising revenue and flat/falling price = undervalued.
    This is the closest thing to fundamental value investing in DeFi.

    Key metrics:
      P/S ratio    < 10 = potentially undervalued
      Revenue 30d  trend rising = growing product-market fit
      Fee/revenue  ratio high = protocol capturing value (not just printing)

    Requires: Token Terminal API key
    """

    def __init__(self, api_key: str):
        super().__init__("https://api.tokenterminal.com/v2", api_key)
        self.headers["Authorization"] = f"Bearer {api_key}"

    async def get_project_metrics(self, project_id: str) -> dict:
        """Fetch fundamentals for a protocol."""
        try:
            return await self.get(f"/projects/{project_id}/metrics/latest")
        except Exception as e:
            log.warning(f"TokenTerminal error for {project_id}: {e}")
            return {}

    async def compute_fundamental_score(self, project_id: str) -> float:
        """0-100 score from on-chain fundamentals."""
        try:
            data    = await self.get_project_metrics(project_id)
            score   = 40.0

            ps      = data.get("price_to_sales", None)
            pe      = data.get("price_to_earnings", None)
            rev_30d = data.get("revenue_30d", 0)
            rev_90d = data.get("revenue_90d", 0)

            # P/S valuation
            if ps is not None:
                if ps < 5:    score += 25
                elif ps < 15: score += 15
                elif ps < 30: score += 5
                elif ps > 100: score -= 15

            # Revenue trend (30d vs 90d daily avg)
            if rev_90d > 0:
                daily_90 = rev_90d / 90
                daily_30 = rev_30d / 30
                rev_growth = (daily_30 - daily_90) / (daily_90 + 1e-9)
                if rev_growth > 0.20:   score += 20  # >20% revenue growth
                elif rev_growth > 0.05: score += 10
                elif rev_growth < -0.20: score -= 15

            return min(100.0, max(0.0, score))
        except Exception as e:
            log.warning(f"TokenTerminal score error for {project_id}: {e}")
            return 45.0


class FREDClient:
    """
    Federal Reserve Economic Data (FRED) — macro intelligence.
    WHY IT MATTERS: Crypto doesn't trade in a vacuum. DXY strength → crypto
    weakness. M2 money supply growth → liquidity expanding → risk assets pump.
    Fed rate decisions and CPI data create crypto volatility events.

    Key series:
      DXY    (US Dollar Index)         → inverse correlation with BTC
      M2SL   (M2 Money Supply)         → liquidity proxy
      FEDFUNDS (Fed Funds Rate)        → risk-off when rising
      CPIAUCSL (CPI Inflation)         → drives Fed hawkishness
      T10Y2Y  (Yield curve inversion)  → recession signal

    Requires: pip install fredapi
    Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    SERIES = {
        "dxy":         "DTWEXBGS",   # Trade-weighted USD index
        "m2":          "M2SL",       # M2 money supply
        "fed_rate":    "FEDFUNDS",   # Federal funds rate
        "cpi":         "CPIAUCSL",   # CPI
        "yield_curve": "T10Y2Y",     # 10Y-2Y spread
        "sp500":       "SP500",      # S&P 500 (risk appetite)
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._fred   = None

    def _get_fred(self):
        if self._fred is None:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def get_series_latest(self, series_id: str, periods: int = 30):
        """Get last N observations for a FRED series."""
        try:
            fred = self._get_fred()
            data = fred.get_series(series_id)
            return data.dropna().tail(periods)
        except Exception as e:
            log.warning(f"FRED error for {series_id}: {e}")
            return None

    def compute_macro_score(self) -> float:
        """
        0-100 macro environment score for crypto.
        High score = accommodative macro (good for risk assets).
        Low score  = restrictive macro (bad for risk assets).
        """
        score = 50.0
        try:
            # DXY: falling dollar = good for crypto
            dxy = self.get_series_latest("DTWEXBGS", 30)
            if dxy is not None and len(dxy) >= 5:
                dxy_slope = (float(dxy.iloc[-1]) - float(dxy.iloc[-5])) / float(dxy.iloc[-5])
                if dxy_slope < -0.01:   score += 15  # Dollar weakening
                elif dxy_slope > 0.01:  score -= 15  # Dollar strengthening

            # M2: rising = liquidity expanding = bullish for risk assets
            m2 = self.get_series_latest("M2SL", 12)
            if m2 is not None and len(m2) >= 3:
                m2_growth = (float(m2.iloc[-1]) - float(m2.iloc[-3])) / float(m2.iloc[-3])
                if m2_growth > 0.005:  score += 12  # Liquidity expanding
                elif m2_growth < -0.005: score -= 8  # Liquidity contracting

            # Fed rate: high and rising = risk-off
            fed = self.get_series_latest("FEDFUNDS", 3)
            if fed is not None and len(fed) >= 2:
                fed_rate = float(fed.iloc[-1])
                fed_trend = float(fed.iloc[-1]) - float(fed.iloc[-2])
                if fed_rate > 5:     score -= 15
                elif fed_rate < 2:   score += 10
                if fed_trend > 0.1:  score -= 8   # Hiking = risk-off
                elif fed_trend < -0.1: score += 8  # Cutting = risk-on

            # Yield curve: inverted (<0) = recession signal = risk-off
            yc = self.get_series_latest("T10Y2Y", 5)
            if yc is not None and len(yc) >= 2:
                spread = float(yc.iloc[-1])
                if spread < -0.5:   score -= 12  # Deep inversion
                elif spread > 0.5:  score += 8   # Healthy curve

        except Exception as e:
            log.warning(f"FRED macro scoring error: {e}")

        return min(100.0, max(0.0, score))


class NewsAPIClient(BaseClient):
    """
    NewsAPI — crypto news sentiment via NLP keyword scoring.
    WHY IT MATTERS: News sentiment drives short-term price action.
    Negative news clusters = fear capitulation = buy signal.
    Positive news spikes + high price = distribution narrative = sell signal.

    NLP approach: keyword scoring on headlines (no heavy model needed).
    Optional: plug in FinBERT from nexus_whitepaper_analyzer for deeper analysis.

    Free tier: 100 requests/day. Paid: 250,000/month.
    Key: https://newsapi.org
    """

    POSITIVE_WORDS = {
        "launch", "partnership", "adoption", "upgrade", "milestone",
        "record", "all-time", "breakthrough", "bullish", "approval",
        "etf", "institutional", "accumulate", "growth", "innovation",
        "integration", "mainnet", "audit", "pass", "compliant",
    }
    NEGATIVE_WORDS = {
        "hack", "exploit", "breach", "rug", "scam", "fraud", "ban",
        "crash", "collapse", "bankruptcy", "lawsuit", "sec", "regulation",
        "fud", "dump", "liquidation", "halt", "suspend", "delist",
        "investigation", "theft", "vulnerable", "exploit", "ponzi",
    }

    def __init__(self, api_key: str):
        super().__init__("https://newsapi.org/v2", api_key)
        self.headers["X-Api-Key"] = api_key

    async def get_articles(self, query: str, days_back: int = 3) -> list[dict]:
        from datetime import timezone
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            data = await self.get("/everything", params={
                "q":        query,
                "from":     since,
                "sortBy":   "relevancy",
                "language": "en",
                "pageSize": 30,
            })
            return data.get("articles", [])
        except Exception as e:
            log.warning(f"NewsAPI error for '{query}': {e}")
            return []

    def score_headline(self, headline: str) -> float:
        """Simple keyword score: +1 per positive word, -1 per negative."""
        lower = headline.lower()
        pos   = sum(1 for w in self.POSITIVE_WORDS if w in lower)
        neg   = sum(1 for w in self.NEGATIVE_WORDS if w in lower)
        return pos - neg

    async def compute_news_sentiment_score(self, keyword: str) -> float:
        """0-100 score from recent news sentiment for a keyword."""
        articles = await self.get_articles(f"{keyword} crypto", days_back=3)
        if not articles:
            return 50.0

        scores = [self.score_headline(a.get("title", "") + " " + a.get("description", ""))
                  for a in articles]
        avg_score = sum(scores) / len(scores)

        # Convert from (-∞, +∞) raw score to 0-100
        # avg_score: -3 → 0/100,  0 → 50/100,  +3 → 100/100
        normalized = 50.0 + avg_score * 15
        return min(100.0, max(0.0, normalized))


class LunarCrushClient(BaseClient):
    """
    LunarCrush — social intelligence for crypto.
    WHY IT MATTERS: Galaxy Score and AltRank are composite metrics that
    combine price, volume, social activity, and sentiment. AltRank < 50
    with rising Galaxy Score is a strong early entry signal for altcoins.

    Key metrics:
      galaxy_score  — overall health score (1-100): combines social + price
      alt_rank      — rank among all coins by social/price momentum (lower = better)
      social_volume — total posts/mentions across all social platforms
      social_score  — quality-weighted social engagement

    Requires: LunarCrush API v3 key (free tier available)
    """

    def __init__(self, api_key: str):
        super().__init__("https://lunarcrush.com/api4/public", api_key)
        self.headers["Authorization"] = f"Bearer {api_key}"

    async def get_coin_data(self, symbol: str) -> dict:
        """Fetch full LunarCrush data for a coin."""
        try:
            data = await self.get(f"/coins/{symbol.upper()}/v1")
            return data.get("data", {})
        except Exception as e:
            log.warning(f"LunarCrush error for {symbol}: {e}")
            return {}

    async def get_trending_altcoins(self, limit: int = 20) -> list[dict]:
        """Get top trending altcoins by social momentum."""
        try:
            data = await self.get("/coins/list/v2", params={"sort": "alt_rank", "limit": limit})
            return data.get("data", [])
        except Exception as e:
            log.warning(f"LunarCrush trending error: {e}")
            return []

    async def compute_lunarcrush_score(self, symbol: str) -> float:
        """
        0-100 score from LunarCrush Galaxy Score and AltRank.

        Galaxy Score 70+      → strong social + price health
        AltRank < 50          → top momentum among all alts
        Rising social volume  → narrative gaining traction
        """
        data = await self.get_coin_data(symbol)
        if not data:
            return 50.0

        galaxy   = float(data.get("galaxy_score",  50))
        alt_rank = float(data.get("alt_rank",      500))
        soc_vol  = float(data.get("social_volume",   0))
        soc_chg  = float(data.get("social_volume_24h_change", 0))
        score    = 30.0

        # Galaxy Score (0-100 directly)
        score += galaxy * 0.5   # up to +50

        # AltRank: lower rank = better momentum (rank 1 = hottest)
        if alt_rank < 10:   score += 20
        elif alt_rank < 50: score += 12
        elif alt_rank < 100: score += 6
        elif alt_rank > 500: score -= 5

        # Rising social volume = narrative gaining
        if soc_chg > 50:   score += 10
        elif soc_chg > 20: score += 5
        elif soc_chg < -30: score -= 8

        return min(100.0, max(0.0, score))

class NEXUSScorer:
    """
    Computes 5-factor conviction score for a given asset.

    Weights:
      TA:          25%   (patterns, indicators, OTE 0.702 confluence)
      On-chain:    25%   (Glassnode + CoinGlass + WhaleAlert)
      Dev:         20%   (DeveloperReport + CoinGecko dev stats)
      Sentiment:   15%   (Santiment + LunarCrush + NewsAPI + Google Trends + Fear/Greed)
      Tokenomics:  15%   (DeFiLlama + TokenTerminal + FRED macro context)
    """

    WEIGHTS = {
        "technical":   0.25,
        "onchain":     0.25,
        "dev":         0.20,
        "sentiment":   0.15,
        "tokenomics":  0.15,
    }

    ACTIONS = [
        (0,  40,  "SKIP"),
        (40, 60,  "WATCH"),
        (60, 75,  "RESEARCH"),
        (75, 85,  "POSITION"),
        (85, 101, "CONVICTION"),
    ]

    def __init__(
        self,
        coinglass_key:      str,
        glassnode_key:      str,
        santiment_key:      str,
        devreport_key:      str,
        # ── New clients (all optional — fall back to neutral 50 if not provided) ──
        whale_alert_key:    str = "",
        token_terminal_key: str = "",
        fred_key:           str = "",
        newsapi_key:        str = "",
        lunarcrush_key:     str = "",
    ):
        self.ta        = TACalculator()
        self.patterns  = PatternDetector()

        # Original clients
        self.coinglass = CoinGlassClient(coinglass_key)
        self.glassnode = GlassnodeClient(glassnode_key)
        self.santiment = SantimentClient(santiment_key)
        self.devreport = DeveloperReportClient(devreport_key)
        self.defillama = DeFiLlamaClient()

        # New data intelligence clients
        self.gtrends   = GoogleTrendsClient()
        self.feargreed = FearGreedClient()
        self.coingecko = CoinGeckoClient()
        self.whale     = WhaleAlertClient(whale_alert_key)   if whale_alert_key    else None
        self.tokterm   = TokenTerminalClient(token_terminal_key) if token_terminal_key else None
        self.fred      = FREDClient(fred_key)                if fred_key           else None
        self.newsapi   = NewsAPIClient(newsapi_key)          if newsapi_key        else None
        self.lunar     = LunarCrushClient(lunarcrush_key)   if lunarcrush_key     else None

    def compute_technical_score(self, df: pd.DataFrame) -> tuple[float, list[PatternResult]]:
        """Score TA signals and pattern confluence. Returns (score, patterns)."""
        score    = 0.0
        detected = self.patterns.detect_all(df)

        # Pattern confluence
        bull_patterns = [p for p in detected if p.detected and p.direction == "BULLISH"]
        bear_patterns = [p for p in detected if p.detected and p.direction == "BEARISH"]
        pattern_score = len(bull_patterns) * 12 - len(bear_patterns) * 12

        # Indicator signals
        close  = df["close"]
        volume = df["volume"]
        rsi    = self.ta.rsi(close, 14)
        macd_l, macd_s, macd_h = self.ta.macd(close)
        ema20  = self.ta.ema(close, 20)
        ema50  = self.ta.ema(close, 50)
        ema200 = self.ta.ema(close, 200)

        if rsi is not None:
            rsi_val = rsi.iloc[-1]
            if 30 <= rsi_val <= 50:  # Recovering from oversold
                score += 15
            elif rsi_val < 30:
                score += 20
            elif rsi_val > 70:
                score -= 15
            div = self.ta.rsi_divergence(close, rsi)
            if div.get("bullish"):
                score += 20

        if macd_l is not None and macd_s is not None:
            if macd_l.iloc[-1] > macd_s.iloc[-1] and macd_l.iloc[-2] <= macd_s.iloc[-2]:
                score += 15  # Bullish crossover
            if macd_h is not None:
                div = self.ta.macd_divergence(close, macd_h)
                if div.get("bullish"):
                    score += 10

        if ema20 is not None and ema50 is not None and ema200 is not None:
            price = close.iloc[-1]
            if price > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
                score += 15  # Full bull alignment
            elif price < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
                score -= 15

        # Volume confirmation
        obv = self.ta.obv(close, volume)
        if obv is not None and obv.iloc[-1] > obv.iloc[-5]:
            score += 5

        # ── OTE / 0.702 Fibonacci proximity bonus ──────────────────
        # Using the last significant swing (50-bar lookback)
        lookback = min(50, len(df))
        swing_high = float(df["high"].tail(lookback).max())
        swing_low  = float(df["low"].tail(lookback).min())
        current    = float(close.iloc[-1])
        ote        = self.ta.ote_proximity(current, swing_high, swing_low)

        if ote["in_702_sweet_spot"]:
            # Exact 0.702 hit: highest-confidence Fibonacci entry
            score += 25
            log.info(f"  ★ OTE 0.702 sweet spot hit at {current:.6f} (level: {ote['ote_702_level']:.6f})")
        elif ote["in_ote_zone"]:
            # Inside the full 0.618–0.786 OTE zone
            ote_bonus = int(ote["proximity_score"] * 18)  # up to +18 pts
            score += ote_bonus
            log.info(f"  OTE zone hit (proximity {ote['proximity_score']:.2f}) +{ote_bonus} pts")

        # Confluence multiplier: OTE + RSI oversold + BOS = CONVICTION add
        rsi_oversold = rsi is not None and float(rsi.iloc[-1]) < 40
        bos_detected = any(p.name == "Break of Structure" and p.detected and
                           p.direction == "BULLISH" for p in detected)
        if ote["in_ote_zone"] and rsi_oversold and bos_detected:
            score += 15
            log.info("  ★★ Triple confluence: OTE + RSI<40 + BOS — +15 bonus")

        total = min(100.0, max(0.0, 50.0 + score + pattern_score))
        return total, detected

    def get_action(self, score: float) -> str:
        for low, high, action in self.ACTIONS:
            if low <= score < high:
                return action
        return "SKIP"

    def compute_entry_exit(self, df: pd.DataFrame, direction: str = "BULLISH"):
        """
        Compute entry zone, stop loss, and take profit using Fibonacci + OTE logic.

        Entry priority (BULLISH):
          1. Price at 0.702 OTE sweet spot  → enter now, stop below 0.786
          2. Price in 0.618–0.786 OTE zone  → limit order at 0.702, stop below 0.786
          3. Price above OTE zone           → wait for pullback; show zone as target
          4. Extension targets: 1.272 (partial), 1.618 (main), 1.702 (OTE mirror)

        Entry priority (BEARISH):
          1. Price at 0.702 OTE of the rally → short entry, stop above 0.786 of rally
          2. Extension targets: -0.272, -0.618, -0.702 (projected below swing low)
        """
        # Use 50-bar swing for major S/R context, 20-bar for immediate entry zone
        lookback_major = min(50, len(df))
        lookback_entry = min(20, len(df))

        major_high = float(df["high"].tail(lookback_major).max())
        major_low  = float(df["low"].tail(lookback_major).min())
        entry_high = float(df["high"].tail(lookback_entry).max())
        entry_low  = float(df["low"].tail(lookback_entry).min())

        fib_major = self.ta.fibonacci_levels(major_high, major_low)
        fib_entry = self.ta.fibonacci_levels(entry_high, entry_low)

        price = float(df["close"].iloc[-1])
        atr   = self.ta.atr(df["high"], df["low"], df["close"], 14)
        atr_v = float(atr.iloc[-1]) if atr is not None else (major_high - major_low) * 0.02

        # OTE proximity check on current price
        ote = self.ta.ote_proximity(price, entry_high, entry_low)

        if direction == "BULLISH":
            # ── Primary entry: OTE zone (0.618–0.786), sweet spot at 0.702 ──
            ote_702    = fib_entry["retracement_0.702"]  # ideal entry
            ote_zone_h = fib_entry["retracement_0.618"]  # top of zone (shallower)
            ote_zone_l = fib_entry["retracement_0.786"]  # bottom of zone (deeper)

            # Stop: below the 0.786 level + half ATR buffer (protects against wicks)
            stop = ote_zone_l - atr_v * 0.5

            # If price is already at OTE, enter at current; otherwise target 0.702
            if ote["in_ote_zone"] or ote["in_702_sweet_spot"]:
                entry_zone_out = (ote_zone_l, ote_702)   # trigger zone
            else:
                entry_zone_out = (ote_zone_l, ote_zone_h) # wider zone if far away

            # Take-profit ladder:
            # TP1 = 1.272 (partial exit, lock in profits)
            # TP2 = 1.618 (primary target — golden ratio extension)
            # TP3 = 1.702 (OTE mirror extension — highest conviction target)
            tp1 = fib_major["extension_1.272"]
            tp2 = fib_major["extension_1.618"]
            tp3 = fib_major["extension_1.702"]

            # Use TP2 (1.618) as primary; TP3 (1.702) for trailing rides
            target = tp2

        else:  # BEARISH
            # ── Bearish OTE: rally to 0.702 of the prior down-leg = short entry ──
            ote_702    = fib_entry["retracement_0.702"]
            ote_zone_h = fib_entry["retracement_0.618"]
            ote_zone_l = fib_entry["retracement_0.786"]

            # Stop: above the 0.786 of the rally
            stop = ote_zone_h + atr_v * 0.5

            if ote["in_ote_zone"] or ote["in_702_sweet_spot"]:
                entry_zone_out = (ote_702, ote_zone_h)
            else:
                entry_zone_out = (ote_zone_l, ote_zone_h)

            tp1    = fib_major["extension_1.272"]
            target = fib_major["extension_1.618"]
            tp3    = fib_major["extension_1.702"]

        rr = abs(target - price) / abs(price - stop) if abs(price - stop) > 1e-9 else 0

        return entry_zone_out, stop, target, rr

    async def score_asset(
        self,
        symbol: str,
        df: pd.DataFrame,
        xrpl_slug: str | None = None,
        defi_protocol: str | None = None,
        coingecko_id:  str | None = None,
        tokterm_id:    str | None = None,
        mcap: float | None = None,
        breadth_snapshot = None,   # BreadthSnapshot from nexus_breadth.py (optional)
    ) -> SignalScore:
        """
        Full 13-source conviction score for an asset.
        df: OHLCV DataFrame (open, high, low, close, volume columns).

        breadth_snapshot: pass the output of BreadthCalculator.scan_all() here.
          The breadth conviction multiplier (0.70–1.25) is applied to the raw
          conviction score before action tier assignment. This means:
            - COLLAPSE breadth (×0.70) can pull a CONVICTION signal down to POSITION
            - EXTREME_BULLISH breadth (×1.25) can lift a POSITION to CONVICTION
          Also: if breadth_state is COLLAPSE, new POSITION/CONVICTION entries are
          blocked (action → RESEARCH) as a circuit breaker.
        """
        log.info(f"Scoring {symbol} across all data sources...")

        # ── Technical (25%) ─────────────────────────────────────
        ta_score, patterns = self.compute_technical_score(df)

        # ── On-chain (25%): Glassnode + CoinGlass + WhaleAlert ──
        base_asset    = symbol.replace("USDT", "").replace("USD", "").replace("/", "")
        onchain_score = await self.glassnode.compute_onchain_score(base_asset)
        cg_score      = await self.coinglass.compute_derivatives_score(symbol)
        whale_score   = await self.whale.compute_whale_score(base_asset) if self.whale else 50.0
        merged_oc     = onchain_score * 0.45 + cg_score * 0.35 + whale_score * 0.20

        # ── Developer Activity (20%): DevReport + CoinGecko dev ─
        dev_score   = await self.devreport.compute_dev_score(xrpl_slug or symbol.lower())
        cg_dev      = await self.coingecko.compute_coingecko_score(coingecko_id or base_asset.lower())
        merged_dev  = dev_score * 0.6 + cg_dev * 0.4

        # ── Sentiment (15%): 5-source composite ─────────────────
        sent_base   = await self.santiment.compute_sentiment_score(xrpl_slug or symbol.lower())
        lunar_score = await self.lunar.compute_lunarcrush_score(base_asset) if self.lunar else 50.0
        news_score  = await self.newsapi.compute_news_sentiment_score(base_asset) if self.newsapi else 50.0
        fg_score    = await self.feargreed.compute_feargreed_score()
        trends_score = self.gtrends.compute_trends_score(base_asset)
        merged_sent = (
            sent_base    * 0.30 +
            lunar_score  * 0.25 +
            news_score   * 0.20 +
            fg_score     * 0.15 +
            trends_score * 0.10
        )

        # ── Tokenomics (15%): DeFiLlama + TokenTerminal + FRED ──
        tok_defi   = await self.defillama.compute_tokenomics_score(defi_protocol or symbol.lower(), mcap)
        tok_term   = await self.tokterm.compute_fundamental_score(tokterm_id or base_asset.lower()) if self.tokterm else 50.0
        macro_score = self.fred.compute_macro_score() if self.fred else 50.0
        merged_tok  = tok_defi * 0.45 + tok_term * 0.35 + macro_score * 0.20

        # ── Conviction with category-aware weights ───────────────
        # Look up asset category and use its specific weight profile.
        # A BTC signal should weight on-chain flows at 40%, not 25%.
        # A DeFi protocol should weight tokenomics/TVL at 25%, not 15%.
        try:
            from nexus_asset_universe import weights_for
            w = weights_for(symbol)
        except ImportError:
            w = self.WEIGHTS  # fallback to balanced defaults

        raw_conviction = (
            ta_score    * w.get("technical",   0.25) +
            merged_oc   * w.get("onchain",     0.25) +
            merged_dev  * w.get("dev",         0.20) +
            merged_sent * w.get("sentiment",   0.15) +
            merged_tok  * w.get("tokenomics",  0.15)
        )

        # ── Apply breadth multiplier ─────────────────────────────
        breadth_score      = 50.0
        breadth_state      = "NEUTRAL"
        breadth_multiplier = 1.00
        breadth_alerts     = []

        if breadth_snapshot is not None:
            breadth_score      = breadth_snapshot.breadth_score
            breadth_state      = breadth_snapshot.breadth_state
            breadth_multiplier = breadth_snapshot.conviction_multiplier
            breadth_alerts     = breadth_snapshot.alerts

            log.info(
                f"  Breadth: {breadth_score:.1f} [{breadth_state}] "
                f"multiplier ×{breadth_multiplier:.2f}"
            )

        # Clamp conviction to 0–100 after multiplier
        conviction = min(100.0, max(0.0, raw_conviction * breadth_multiplier))

        # ── Breadth circuit breaker ──────────────────────────────
        # If breadth is in full collapse, block new POSITION/CONVICTION entries.
        # Downgrade to RESEARCH so the trader sees the signal but doesn't auto-trade.
        action = self.get_action(conviction)
        if breadth_state == "COLLAPSE" and action in ("POSITION", "CONVICTION"):
            action = "RESEARCH"
            log.warning(
                f"  ★ BREADTH CIRCUIT BREAKER: {symbol} action downgraded "
                f"from {self.get_action(conviction)} → RESEARCH "
                f"(breadth {breadth_score:.0f} in COLLAPSE)"
            )

        bull_count  = sum(1 for p in patterns if p.detected and p.direction == "BULLISH")
        direction   = "BULLISH" if bull_count > 0 else "BEARISH"
        entry, sl, tp, rr = self.compute_entry_exit(df, direction)

        signal = SignalScore(
            symbol           = symbol,
            timestamp        = datetime.utcnow(),
            technical_score  = round(ta_score, 2),
            onchain_score    = round(merged_oc, 2),
            dev_score        = round(merged_dev, 2),
            sentiment_score  = round(merged_sent, 2),
            tokenomics_score = round(merged_tok, 2),
            conviction_score = round(conviction, 2),
            action           = action,
            patterns         = patterns,
            entry_zone       = entry,
            stop_loss        = sl,
            take_profit      = tp,
            risk_reward      = round(rr, 2),
            raw = {
                "raw_conviction":    round(raw_conviction, 2),
                "breadth_score":     breadth_score,
                "breadth_state":     breadth_state,
                "breadth_mult":      breadth_multiplier,
                "breadth_alerts":    breadth_alerts,
                "ta_score":          ta_score,
                "onchain":           merged_oc,
                "glassnode":         onchain_score,
                "coinglass":         cg_score,
                "whale_alert":       whale_score,
                "dev_score":         merged_dev,
                "devreport":         dev_score,
                "coingecko":         cg_dev,
                "sentiment":         merged_sent,
                "santiment":         sent_base,
                "lunarcrush":        lunar_score,
                "newsapi":           news_score,
                "fear_greed":        fg_score,
                "google_trends":     trends_score,
                "tokenomics":        merged_tok,
                "defillama":         tok_defi,
                "tokenterminal":     tok_term,
                "fred_macro":        macro_score,
            }
        )
        log.info(signal.summary())
        return signal


# ─────────────────────────────────────────────
# CLI Runner
# ─────────────────────────────────────────────

async def main():
    import os
    import ccxt.async_support as ccxt
    from nexus_breadth import XRPLBreadthCalculator

    scorer = NEXUSScorer(
        coinglass_key      = os.getenv("COINGLASS_API_KEY",      ""),
        glassnode_key      = os.getenv("GLASSNODE_API_KEY",      ""),
        santiment_key      = os.getenv("SANTIMENT_API_KEY",      ""),
        devreport_key      = os.getenv("DEVREPORT_API_KEY",      ""),
        whale_alert_key    = os.getenv("WHALE_ALERT_API_KEY",    ""),
        token_terminal_key = os.getenv("TOKEN_TERMINAL_API_KEY", ""),
        fred_key           = os.getenv("FRED_API_KEY",           ""),
        newsapi_key        = os.getenv("NEWSAPI_KEY",            ""),
        lunarcrush_key     = os.getenv("LUNARCRUSH_API_KEY",     ""),
    )

    breadth_calc = XRPLBreadthCalculator()
    exchange     = ccxt.binance({"enableRateLimit": True})

    # ── Step 1: Scan market breadth across all 1H charts ────────
    print("Step 1: Running 1H breadth scan across 30 assets...")
    breadth = await breadth_calc.scan_all(exchange)
    print(f"\n{breadth.as_telegram()}\n")

    if breadth.alerts:
        print("ACTIVE BREADTH ALERTS:")
        for a in breadth.alerts:
            print(f"  ⚡ {a}")
        print()

    # ── Step 2: Score individual assets with breadth context ────
    watchlist = [
        ("BTC/USDT",  "bitcoin",  "bitcoin",  "bitcoin"),
        ("ETH/USDT",  "ethereum", "ethereum", "ethereum"),
        ("XRP/USDT",  "xrp",      None,       "ripple"),
        ("SOL/USDT",  "solana",   "solana",   "solana"),
    ]

    print("Step 2: Scoring individual assets with breadth multiplier...")
    signals: list[SignalScore] = []
    for pair, cg_id, tt_id, defi_id in watchlist:
        try:
            ohlcv = await exchange.fetch_ohlcv(pair, timeframe="4h", limit=300)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            signal = await scorer.score_asset(
                symbol           = pair,
                df               = df,
                coingecko_id     = cg_id,
                tokterm_id       = tt_id,
                defi_protocol    = defi_id,
                breadth_snapshot = breadth,   # ← breadth multiplier applied here
            )
            signals.append(signal)
        except Exception as e:
            log.error(f"Error scoring {pair}: {e}")

    await exchange.close()

    # ── Step 3: Print ranked signals ────────────────────────────
    signals.sort(key=lambda s: s.conviction_score, reverse=True)
    print("\n" + "=" * 80)
    print(f"  NEXUS SIGNAL REPORT  |  Breadth: {breadth.breadth_score:.0f} "
          f"[{breadth.breadth_state}] ×{breadth.conviction_multiplier}")
    print("=" * 80)
    for s in signals:
        print(s.summary())
        raw_cv = s.raw.get("raw_conviction", s.conviction_score)
        print(f"  Raw conviction: {raw_cv:.1f}  ×{breadth.conviction_multiplier:.2f}  "
              f"→ Final: {s.conviction_score:.1f}  [{s.action}]")
        if s.action in ("POSITION", "CONVICTION"):
            print(f"  Entry zone : {s.entry_zone[0]:.6f} – {s.entry_zone[1]:.6f}")
            print(f"  Stop loss  : {s.stop_loss:.6f}   Take profit: {s.take_profit:.6f}")
            print(f"  R/R        : {s.risk_reward}x")
        print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
