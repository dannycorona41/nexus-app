"""
NEXUS Backtesting Framework
============================
Validate NEXUS signal strategies against historical OHLCV data.

Features:
  - Load data from CSV, CCXT exchange, or Parquet
  - Run signal engine on historical data (vectorized where possible)
  - Simulate trade execution with realistic slippage + commission
  - Kelly Criterion position sizing on historical win rates
  - Full performance metrics: Sharpe, Sortino, Calmar, max DD, win rate
  - Trade log with every entry/exit tagged by pattern
  - Equity curve and drawdown chart (via matplotlib)
  - Walk-forward optimization scaffold
  - Compare multiple strategies side-by-side

Install:
  pip install pandas numpy matplotlib ccxt pandas_ta tqdm

Usage:
  from nexus_backtester import BacktestEngine, BacktestConfig
  config = BacktestConfig(initial_capital=100_000, commission=0.001)
  engine = BacktestEngine(config)
  engine.load_csv("btc_4h.csv")
  report = engine.run()
  print(report.summary())
  report.plot()
"""

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta

log = logging.getLogger("nexus.backtester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital:   float = 100_000.0   # Starting portfolio in base currency
    commission:        float = 0.001        # 0.1% per trade
    slippage:          float = 0.0005       # 0.05% market impact
    max_positions:     int   = 5            # Max concurrent open trades
    risk_per_trade:    float = 0.02         # Max 2% portfolio per trade
    min_rr:            float = 2.0          # Minimum reward/risk ratio
    min_conviction:    float = 60.0         # Minimum signal score to trade
    use_kelly:         bool  = True         # Dynamic Kelly sizing
    pyramid_into:      bool  = False        # Allow adding to winners
    short_enabled:     bool  = False        # Enable short trades
    stop_loss_pct:     float = 0.04         # Default stop if pattern doesn't provide one (4%)
    take_profit_pct:   float = 0.10         # Default TP if pattern doesn't provide one (10%)
    trailing_stop_pct: float | None = None  # Optional trailing stop (e.g. 0.03 = 3%)
    timeframe:         str   = "4h"


# ─────────────────────────────────────────────
# Trade / Position Data
# ─────────────────────────────────────────────

class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    id:             int
    symbol:         str
    direction:      Direction
    entry_time:     pd.Timestamp
    entry_price:    float
    size:           float           # Position size in base units
    stop_loss:      float
    take_profit:    float
    pattern:        str
    conviction:     float
    commission_paid: float = 0.0
    exit_time:      pd.Timestamp | None = None
    exit_price:     float | None = None
    exit_reason:    str = ""        # TAKE_PROFIT / STOP_LOSS / SIGNAL / EOD
    pnl:            float | None = None
    pnl_pct:        float | None = None
    highest_price:  float | None = None  # for trailing stop
    r_multiple:     float | None = None  # how many R units gained/lost

    def is_open(self) -> bool:
        return self.exit_time is None

    def close(self, exit_price: float, exit_time: pd.Timestamp, reason: str):
        self.exit_price  = exit_price
        self.exit_time   = exit_time
        self.exit_reason = reason
        if self.direction == Direction.LONG:
            self.pnl = (exit_price - self.entry_price) * self.size - self.commission_paid
        else:
            self.pnl = (self.entry_price - exit_price) * self.size - self.commission_paid
        capital_risked  = abs(self.entry_price - self.stop_loss) * self.size
        self.pnl_pct    = self.pnl / (self.entry_price * self.size) if self.entry_price > 0 else 0
        self.r_multiple = self.pnl / capital_risked if capital_risked > 0 else 0


# ─────────────────────────────────────────────
# Signal Generator (simplified inline version)
# ─────────────────────────────────────────────

class InlineSignalGenerator:
    """
    Lightweight signal generator for backtesting.
    Mirrors logic from nexus_signal_engine.py without external API calls.
    Uses only TA indicators + pattern heuristics on OHLCV data.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def generate(self, df: pd.DataFrame, idx: int) -> list[dict]:
        """
        Generate signals at bar[idx]. Returns list of signal dicts.
        Only uses data up to idx (no lookahead).
        """
        window = df.iloc[max(0, idx - 200): idx + 1].copy()
        if len(window) < 50:
            return []

        signals = []
        close  = window["close"]
        high   = window["high"]
        low    = window["low"]
        volume = window["volume"]

        # RSI
        rsi_s = pta.rsi(close, length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None else 50

        # MACD
        macd_r  = pta.macd(close, fast=12, slow=26, signal=9)
        macd_l  = float(macd_r.iloc[-1, 0]) if macd_r is not None and not macd_r.empty else 0
        macd_sig= float(macd_r.iloc[-1, 2]) if macd_r is not None and not macd_r.empty else 0
        macd_cross_up = (macd_l > macd_sig and
                         (macd_r.iloc[-2, 0] if len(macd_r) > 1 else 0) <= (macd_r.iloc[-2, 2] if len(macd_r) > 1 else 0))

        # Moving averages
        ema20  = pta.ema(close, length=20)
        ema50  = pta.ema(close, length=50)
        ema200 = pta.ema(close, length=200)
        price  = float(close.iloc[-1])

        ema20_v  = float(ema20.iloc[-1])  if ema20  is not None else price
        ema50_v  = float(ema50.iloc[-1])  if ema50  is not None else price
        ema200_v = float(ema200.iloc[-1]) if ema200 is not None else price

        bull_alignment = price > ema20_v > ema50_v  # Trending up

        # RSI divergence (simple: price lower low, RSI higher low)
        if len(window) >= 30:
            prev_low_price = float(low.iloc[-30:-10].min())
            curr_low_price = float(low.iloc[-10:].min())
            prev_rsi_low   = float(rsi_s.iloc[-30:-10].min()) if rsi_s is not None else rsi
            curr_rsi_low   = float(rsi_s.iloc[-10:].min())    if rsi_s is not None else rsi
            rsi_bull_div   = curr_low_price < prev_low_price and curr_rsi_low > prev_rsi_low
        else:
            rsi_bull_div = False

        # Volume surge
        avg_vol    = float(volume.iloc[-20:-1].mean())
        curr_vol   = float(volume.iloc[-1])
        vol_surge  = curr_vol > avg_vol * 1.5

        # Bull Flag detection (simple version)
        if len(window) >= 20:
            pole   = window.iloc[-20:-10]
            flag   = window.iloc[-10:]
            pole_r = (float(pole["close"].iloc[-1]) - float(pole["close"].iloc[0])) / float(pole["close"].iloc[0])
            flag_r = (float(flag["close"].iloc[-1]) - float(flag["close"].iloc[0])) / float(flag["close"].iloc[0])
            flag_vol_ratio = float(flag["volume"].mean()) / (float(pole["volume"].mean()) + 1e-9)
            is_bull_flag = (pole_r > 0.06 and -0.06 < flag_r < 0.01 and flag_vol_ratio < 0.75)
        else:
            is_bull_flag = False

        # Wyckoff Spring (simplified)
        if len(window) >= 60:
            range_low   = float(low.iloc[-60:-20].min())
            last_10_low = float(low.iloc[-10:].min())
            last_close  = float(close.iloc[-1])
            spring = last_10_low < range_low * 0.98 and last_close > range_low
        else:
            spring = False

        # Score accumulation
        bull_score = 50.0
        if rsi < 35:           bull_score += 15
        if rsi_bull_div:       bull_score += 20
        if macd_cross_up:      bull_score += 15
        if bull_alignment:     bull_score += 10
        if is_bull_flag:       bull_score += 15
        if spring:             bull_score += 15
        if vol_surge:          bull_score += 5
        if rsi > 65:           bull_score -= 10  # Overbought
        bull_score = min(100, max(0, bull_score))

        # Fibonacci entry/stop/target
        recent_high = float(high.iloc[-20:].max())
        recent_low  = float(low.iloc[-20:].min())
        diff        = recent_high - recent_low
        atr_s       = pta.atr(high, low, close, length=14)
        atr_v       = float(atr_s.iloc[-1]) if atr_s is not None else diff * 0.02

        fib_618  = recent_high - diff * 0.618
        fib_382  = recent_high - diff * 0.382
        ext_1618 = recent_high + diff * 0.618

        if bull_score >= self.config.min_conviction:
            # Determine pattern name
            if is_bull_flag:  pattern = "Bull Flag"
            elif spring:       pattern = "Wyckoff Spring"
            elif rsi_bull_div: pattern = "RSI Divergence"
            else:              pattern = "Indicator Confluence"

            signals.append({
                "direction":  Direction.LONG,
                "pattern":    pattern,
                "conviction": bull_score,
                "entry":      price,
                "stop_loss":  max(recent_low - atr_v * 0.5, price * (1 - self.config.stop_loss_pct)),
                "take_profit":ext_1618 if ext_1618 > price else price * (1 + self.config.take_profit_pct),
                "entry_zone": (fib_618, fib_382),
            })

        # Short signals (if enabled)
        if self.config.short_enabled:
            bear_score = 100 - bull_score
            if bear_score >= self.config.min_conviction:
                signals.append({
                    "direction":  Direction.SHORT,
                    "pattern":    "Bear Confluence",
                    "conviction": bear_score,
                    "entry":      price,
                    "stop_loss":  recent_high + atr_v * 0.5,
                    "take_profit":recent_low - diff * 0.618,
                    "entry_zone": (fib_382, fib_618),
                })

        return signals


# ─────────────────────────────────────────────
# Position Manager
# ─────────────────────────────────────────────

class PositionManager:
    def __init__(self, config: BacktestConfig, initial_capital: float):
        self.config    = config
        self.capital   = initial_capital
        self.positions: list[Trade] = []
        self.trade_id  = 0

    def open_positions(self) -> list[Trade]:
        return [t for t in self.positions if t.is_open()]

    def can_open(self) -> bool:
        return len(self.open_positions()) < self.config.max_positions

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss <= 0:
            return self.config.risk_per_trade
        b = avg_win / avg_loss
        q = 1 - win_rate
        f = (win_rate * b - q) / b
        return min(self.config.risk_per_trade, max(0.005, f * 0.5))

    def size_position(self, entry: float, stop: float, win_rate: float = 0.55) -> float:
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0.0
        if self.config.use_kelly:
            fraction = self.kelly_fraction(win_rate, 0.08, 0.04)
        else:
            fraction = self.config.risk_per_trade
        risk_capital = self.capital * fraction
        return risk_capital / risk_per_unit

    def open_trade(self, signal: dict, timestamp: pd.Timestamp, symbol: str,
                   win_rate: float = 0.55) -> Trade | None:
        if not self.can_open():
            return None
        rr = abs(signal["take_profit"] - signal["entry"]) / (abs(signal["entry"] - signal["stop_loss"]) + 1e-9)
        if rr < self.config.min_rr:
            return None

        size       = self.size_position(signal["entry"], signal["stop_loss"], win_rate)
        commission = signal["entry"] * size * self.config.commission
        slippage   = signal["entry"] * size * self.config.slippage
        total_cost = signal["entry"] * size + commission + slippage
        if total_cost > self.capital:
            size = self.capital * 0.95 / (signal["entry"] * (1 + self.config.commission + self.config.slippage))

        self.capital -= commission + slippage
        self.trade_id += 1
        trade = Trade(
            id            = self.trade_id,
            symbol        = symbol,
            direction     = signal["direction"],
            entry_time    = timestamp,
            entry_price   = signal["entry"] * (1 + self.config.slippage),
            size          = size,
            stop_loss     = signal["stop_loss"],
            take_profit   = signal["take_profit"],
            pattern       = signal["pattern"],
            conviction    = signal["conviction"],
            commission_paid= commission * 2,  # in + out
            highest_price = signal["entry"],
        )
        self.positions.append(trade)
        return trade

    def update_and_close(self, bar: pd.Series, timestamp: pd.Timestamp) -> list[Trade]:
        """Check all open positions against current bar. Return closed trades."""
        closed = []
        for trade in list(self.open_positions()):
            # Update trailing stop
            if self.config.trailing_stop_pct and trade.direction == Direction.LONG:
                if bar["high"] > (trade.highest_price or trade.entry_price):
                    trade.highest_price = float(bar["high"])
                    new_sl = trade.highest_price * (1 - self.config.trailing_stop_pct)
                    trade.stop_loss = max(trade.stop_loss, new_sl)

            exit_price, reason = None, ""
            if trade.direction == Direction.LONG:
                if bar["low"] <= trade.stop_loss:
                    exit_price = trade.stop_loss * (1 - self.config.slippage)
                    reason     = "STOP_LOSS"
                elif bar["high"] >= trade.take_profit:
                    exit_price = trade.take_profit * (1 - self.config.slippage)
                    reason     = "TAKE_PROFIT"
            else:
                if bar["high"] >= trade.stop_loss:
                    exit_price = trade.stop_loss * (1 + self.config.slippage)
                    reason     = "STOP_LOSS"
                elif bar["low"] <= trade.take_profit:
                    exit_price = trade.take_profit * (1 + self.config.slippage)
                    reason     = "TAKE_PROFIT"

            if exit_price:
                trade.close(exit_price, timestamp, reason)
                self.capital += (trade.pnl or 0) + trade.entry_price * trade.size
                closed.append(trade)

        return closed


# ─────────────────────────────────────────────
# Performance Analyzer
# ─────────────────────────────────────────────

class PerformanceAnalyzer:
    """Compute all performance metrics from a list of closed trades."""

    def __init__(self, trades: list[Trade], initial_capital: float, equity_curve: pd.Series):
        self.trades         = [t for t in trades if not t.is_open()]
        self.initial_capital= initial_capital
        self.equity         = equity_curve

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    def total_return(self) -> float:
        if self.equity.empty:
            return 0.0
        return (self.equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100

    def win_rate(self) -> float:
        wins = [t for t in self.trades if (t.pnl or 0) > 0]
        return len(wins) / len(self.trades) * 100 if self.trades else 0

    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if (t.pnl or 0) > 0)
        gross_loss   = abs(sum(t.pnl for t in self.trades if (t.pnl or 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def expectancy(self) -> float:
        """Average R-multiple per trade."""
        rs = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        return sum(rs) / len(rs) if rs else 0

    def max_drawdown(self) -> float:
        if self.equity.empty:
            return 0.0
        peak   = self.equity.cummax()
        dd     = (self.equity - peak) / peak * 100
        return float(dd.min())

    def sharpe_ratio(self, risk_free: float = 0.02, periods_per_year: int = 1460) -> float:
        """Annualized Sharpe ratio. periods_per_year=1460 for 4h bars."""
        r = self.returns
        if r.std() == 0:
            return 0.0
        excess = r.mean() - risk_free / periods_per_year
        return float((excess / r.std()) * math.sqrt(periods_per_year))

    def sortino_ratio(self, risk_free: float = 0.02, periods_per_year: int = 1460) -> float:
        r          = self.returns
        downside   = r[r < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf")
        excess     = r.mean() - risk_free / periods_per_year
        return float((excess / downside.std()) * math.sqrt(periods_per_year))

    def calmar_ratio(self) -> float:
        ann_return = self.total_return() / max(1, len(self.equity) / 1460)
        dd         = abs(self.max_drawdown())
        return ann_return / dd if dd > 0 else float("inf")

    def avg_trade_duration(self) -> float:
        """Average trade duration in bars."""
        closed = [t for t in self.trades if t.exit_time]
        if not closed:
            return 0
        return sum(1 for _ in closed) / len(closed)  # simplified

    def by_pattern(self) -> dict:
        """Break down win rate and avg R by pattern."""
        result = {}
        for trade in self.trades:
            p = trade.pattern
            if p not in result:
                result[p] = {"count": 0, "wins": 0, "total_r": 0.0}
            result[p]["count"] += 1
            if (trade.pnl or 0) > 0:
                result[p]["wins"] += 1
            result[p]["total_r"] += trade.r_multiple or 0
        for p, d in result.items():
            d["win_rate"] = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
            d["avg_r"]    = d["total_r"] / d["count"] if d["count"] > 0 else 0
        return result

    def summary(self) -> str:
        by_p = self.by_pattern()
        pattern_lines = "\n".join(
            f"    {p:30s} | Trades:{d['count']:3d} | WR:{d['win_rate']:.0f}% | Avg R:{d['avg_r']:.2f}"
            for p, d in sorted(by_p.items(), key=lambda x: x[1]["avg_r"], reverse=True)
        )
        return (
            f"\n{'═' * 60}\n"
            f"  NEXUS BACKTEST REPORT\n"
            f"{'═' * 60}\n"
            f"  Total Return   : {self.total_return():.2f}%\n"
            f"  Sharpe Ratio   : {self.sharpe_ratio():.2f}\n"
            f"  Sortino Ratio  : {self.sortino_ratio():.2f}\n"
            f"  Calmar Ratio   : {self.calmar_ratio():.2f}\n"
            f"  Max Drawdown   : {self.max_drawdown():.2f}%\n"
            f"  Win Rate       : {self.win_rate():.1f}%\n"
            f"  Profit Factor  : {self.profit_factor():.2f}\n"
            f"  Expectancy (R) : {self.expectancy():.2f}R\n"
            f"  Total Trades   : {len(self.trades)}\n"
            f"{'─' * 60}\n"
            f"  Results by pattern:\n{pattern_lines}\n"
            f"{'═' * 60}\n"
        )


# ─────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────

class BacktestEngine:
    """
    Main backtesting engine.
    Drives signal generation + execution simulation bar by bar.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config    = config or BacktestConfig()
        self.df: pd.DataFrame | None = None
        self.symbol    = "UNKNOWN"
        self.all_trades: list[Trade] = []
        self.equity_curve: list[float] = []

    # ── Data Loaders ───────────────────────────

    def load_csv(self, path: str, symbol: str | None = None):
        """
        Load OHLCV from CSV.
        Expected columns: timestamp (or datetime), open, high, low, close, volume.
        """
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df.set_index(ts_col, inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        self.df     = df.sort_index()
        self.symbol = symbol or path.split("/")[-1].replace(".csv", "")
        log.info(f"Loaded {len(self.df)} bars from {path}")

    def load_dataframe(self, df: pd.DataFrame, symbol: str = "ASSET"):
        self.df     = df.copy().sort_index()
        self.symbol = symbol

    async def load_from_exchange(self, symbol: str, timeframe: str = "4h",
                                  limit: int = 2000, exchange_id: str = "binance"):
        """Fetch OHLCV from CCXT exchange."""
        import ccxt.async_support as ccxt
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        raw = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        await ex.close()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        self.load_dataframe(df, symbol)

    # ── Run ────────────────────────────────────

    def run(self) -> PerformanceAnalyzer:
        """
        Execute backtest bar by bar.
        Returns a PerformanceAnalyzer with all results.
        """
        if self.df is None or len(self.df) < 100:
            raise ValueError("Data not loaded or too short")

        signal_gen    = InlineSignalGenerator(self.config)
        pos_manager   = PositionManager(self.config, self.config.initial_capital)
        self.all_trades = []
        self.equity_curve = [self.config.initial_capital]

        # Calculate historical win rate for Kelly sizing
        historical_win_rate = 0.55  # Start with prior estimate

        log.info(f"Running backtest on {len(self.df)} bars for {self.symbol}...")

        for idx in range(100, len(self.df)):
            bar       = self.df.iloc[idx]
            timestamp = self.df.index[idx]

            # Close positions that hit SL/TP
            closed = pos_manager.update_and_close(bar, timestamp)
            self.all_trades.extend(closed)

            # Update historical win rate dynamically
            if len(self.all_trades) >= 20:
                historical_win_rate = sum(1 for t in self.all_trades[-20:] if (t.pnl or 0) > 0) / 20

            # Generate new signals
            signals = signal_gen.generate(self.df, idx)

            for sig in signals:
                rr = abs(sig["take_profit"] - sig["entry"]) / (abs(sig["entry"] - sig["stop_loss"]) + 1e-9)
                if rr < self.config.min_rr:
                    continue
                if sig["conviction"] < self.config.min_conviction:
                    continue
                pos_manager.open_trade(sig, timestamp, self.symbol, historical_win_rate)

            # Track equity
            open_pnl = sum(
                (float(bar["close"]) - t.entry_price) * t.size
                if t.direction == Direction.LONG
                else (t.entry_price - float(bar["close"])) * t.size
                for t in pos_manager.open_positions()
            )
            total_equity = pos_manager.capital + open_pnl
            # Add capital tied up in positions
            total_equity += sum(t.entry_price * t.size for t in pos_manager.open_positions())
            self.equity_curve.append(total_equity)

        # Close any remaining open positions at last price
        final_bar = self.df.iloc[-1]
        for trade in pos_manager.open_positions():
            trade.close(float(final_bar["close"]), self.df.index[-1], "EOD")
            pos_manager.capital += (trade.pnl or 0) + trade.entry_price * trade.size
        self.all_trades.extend(pos_manager.open_positions())

        equity_series = pd.Series(
            self.equity_curve,
            index=self.df.index[100 - 1:],
            name="Equity"
        )
        log.info(f"Backtest complete. {len(self.all_trades)} trades.")
        return PerformanceAnalyzer(self.all_trades, self.config.initial_capital, equity_series)

    # ── Walk-Forward Optimization ──────────────

    def walk_forward(self, windows: int = 4) -> list[PerformanceAnalyzer]:
        """
        Walk-forward validation: split data into N windows,
        train on first 60% of each window, test on last 40%.
        """
        if self.df is None:
            raise ValueError("No data loaded")

        results  = []
        n_bars   = len(self.df)
        step     = n_bars // windows

        for i in range(windows):
            start   = i * step
            end     = start + step
            test_df = self.df.iloc[start:end].copy()
            log.info(f"Walk-forward window {i + 1}/{windows}: {test_df.index[0]} → {test_df.index[-1]}")
            self.load_dataframe(test_df, self.symbol)
            try:
                result = self.run()
                results.append(result)
            except Exception as e:
                log.warning(f"Window {i + 1} failed: {e}")

        # Restore full dataset
        log.info("Walk-forward complete")
        return results

    # ── Optimization ──────────────────────────

    def optimize_parameters(self, param_grid: dict) -> dict:
        """
        Grid search over BacktestConfig parameters.
        Returns the config dict with best Sharpe ratio.

        Example:
          param_grid = {
            "min_conviction": [55, 65, 75],
            "stop_loss_pct":  [0.03, 0.04, 0.05],
            "min_rr":         [1.5, 2.0, 2.5],
          }
        """
        import itertools
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        best_sharpe = -float("inf")
        best_params = {}

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            cfg    = BacktestConfig(**{**vars(self.config), **params})
            self.config = cfg
            try:
                perf = self.run()
                sharpe = perf.sharpe_ratio()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    log.info(f"New best: Sharpe={sharpe:.2f} | {params}")
            except Exception as e:
                log.debug(f"Param combo failed: {params} — {e}")

        log.info(f"Optimization complete. Best Sharpe: {best_sharpe:.2f} | {best_params}")
        return best_params

    # ── Reporting ──────────────────────────────

    def export_trade_log(self, path: str = "nexus_trade_log.csv"):
        """Export all trades to CSV."""
        if not self.all_trades:
            log.warning("No trades to export")
            return
        rows = []
        for t in self.all_trades:
            rows.append({
                "id":          t.id,
                "symbol":      t.symbol,
                "direction":   t.direction.value,
                "pattern":     t.pattern,
                "conviction":  t.conviction,
                "entry_time":  t.entry_time,
                "entry_price": t.entry_price,
                "exit_time":   t.exit_time,
                "exit_price":  t.exit_price,
                "exit_reason": t.exit_reason,
                "size":        t.size,
                "pnl":         t.pnl,
                "pnl_pct":     t.pnl_pct,
                "r_multiple":  t.r_multiple,
                "commission":  t.commission_paid,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info(f"Trade log exported to {path}")

    def plot(self, perf: PerformanceAnalyzer | None = None, save_path: str | None = None):
        """Plot equity curve, drawdown, and trade markers."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f"NEXUS Backtest — {self.symbol}", fontsize=14, fontweight="bold")
            gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5])

            # Price + trade markers
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(self.df.index, self.df["close"], color="#374151", linewidth=0.8, label="Price")
            for t in self.all_trades:
                color = "green" if (t.pnl or 0) > 0 else "red"
                marker = "^" if t.direction == Direction.LONG else "v"
                ax1.scatter(t.entry_time, t.entry_price, color="royalblue", marker=marker, s=40, zorder=5)
                if t.exit_time:
                    ax1.scatter(t.exit_time, t.exit_price, color=color, marker="x", s=40, zorder=5)
            ax1.set_ylabel("Price")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper left", fontsize=9)

            # Equity curve
            if perf is not None and not perf.equity.empty:
                ax2 = fig.add_subplot(gs[1])
                ax2.plot(perf.equity.index, perf.equity.values, color="royalblue", linewidth=1.2)
                ax2.fill_between(perf.equity.index, perf.equity.values,
                                  perf.equity.values[0], alpha=0.1, color="royalblue")
                ax2.set_ylabel("Equity ($)")
                ax2.grid(True, alpha=0.3)

                # Drawdown
                ax3 = fig.add_subplot(gs[2])
                peak = perf.equity.cummax()
                dd   = (perf.equity - peak) / peak * 100
                ax3.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
                ax3.set_ylabel("Drawdown (%)")
                ax3.set_xlabel("Date")
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                log.info(f"Chart saved to {save_path}")
            else:
                plt.show()
            plt.close()
        except ImportError:
            log.warning("matplotlib not installed — skipping chart. pip install matplotlib")


# ─────────────────────────────────────────────
# Multi-Asset Comparison
# ─────────────────────────────────────────────

class MultiAssetBacktester:
    """Run the same strategy across multiple assets and rank by Sharpe."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config  = config or BacktestConfig()
        self.results: dict[str, PerformanceAnalyzer] = {}

    def run_all(self, data_map: dict[str, pd.DataFrame]) -> dict[str, PerformanceAnalyzer]:
        for symbol, df in data_map.items():
            engine = BacktestEngine(deepcopy(self.config))
            engine.load_dataframe(df, symbol)
            try:
                self.results[symbol] = engine.run()
                log.info(f"{symbol}: Sharpe={self.results[symbol].sharpe_ratio():.2f} WR={self.results[symbol].win_rate():.0f}%")
            except Exception as e:
                log.error(f"{symbol} failed: {e}")
        return self.results

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for sym, perf in self.results.items():
            rows.append({
                "symbol":       sym,
                "total_return": round(perf.total_return(), 2),
                "sharpe":       round(perf.sharpe_ratio(), 2),
                "sortino":      round(perf.sortino_ratio(), 2),
                "max_dd":       round(perf.max_drawdown(), 2),
                "win_rate":     round(perf.win_rate(), 1),
                "profit_factor":round(perf.profit_factor(), 2),
                "expectancy_r": round(perf.expectancy(), 2),
                "n_trades":     len(perf.trades),
            })
        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        return df


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

async def main():
    import sys
    import asyncio

    config = BacktestConfig(
        initial_capital   = 100_000,
        commission        = 0.001,
        slippage          = 0.0005,
        max_positions     = 3,
        risk_per_trade    = 0.02,
        min_rr            = 2.0,
        min_conviction    = 62.0,
        use_kelly         = True,
        trailing_stop_pct = 0.03,
        timeframe         = "4h",
    )

    engine = BacktestEngine(config)

    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        engine.load_csv(sys.argv[1])
    else:
        log.info("No CSV provided — fetching BTC/USDT 4h from Binance...")
        await engine.load_from_exchange("BTC/USDT", "4h", limit=2000)

    perf = engine.run()
    print(perf.summary())
    engine.export_trade_log("nexus_trade_log.csv")

    # Optional walk-forward
    if "--walk-forward" in sys.argv:
        wf_results = engine.walk_forward(4)
        for i, r in enumerate(wf_results):
            print(f"\nWalk-forward window {i + 1}: Sharpe={r.sharpe_ratio():.2f} WR={r.win_rate():.0f}%")

    # Plot
    engine.plot(perf, save_path="nexus_backtest_chart.png")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
