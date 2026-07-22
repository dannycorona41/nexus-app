"""
NEXUS Backtest Runner  (proof-of-track-record gate)
===================================================
Runs the strategy over REAL historical candles and checks the result against
the SAME graduation gate the live paper-trader must clear — so you see whether
the strategy would have earned its way to Pro Trader before it ever trades
forward. This is the "no fake data" proof point.

Usage on your machine (after `pip install -r requirements.txt`):

    # From a CSV you downloaded (columns: timestamp,open,high,low,close,volume)
    python run_backtest.py --csv data/BTC_1d.csv --symbol BTC

    # Or pull real keyless candles (Coinbase/Kraken) for a listed symbol
    python run_backtest.py --symbol BTC --timeframe 1d --bars 400

The backtest uses the engine's InlineSignalGenerator (technical/pattern logic
over history). The full 5-factor live score additionally uses on-chain/dev/
sentiment APIs, which can't be replayed historically without paid archives —
so a passing backtest validates the *technical* core, not the deep-intel layer.
That distinction is stated in the output, not hidden.
"""

from __future__ import annotations
import argparse
import sys

from nexus_backtester import BacktestEngine, BacktestConfig

# Import the gate pieces directly to avoid constructing a full TraderProgression.
from nexus_progression import (
    Metrics, all_gates_met, compute_trust,
    GRAD_SHARPE, GRAD_WINRATE, GRAD_MAX_DD, GRAD_MIN_TRADES, GRAD_MIN_DAYS,
)


def _days_span(engine) -> float:
    """Calendar days covered by the loaded data."""
    try:
        idx = engine.df.index
        return float((idx[-1] - idx[0]).days)
    except Exception:
        return 0.0


def bridge_to_gate(pa, engine) -> Metrics:
    """Map backtest PerformanceAnalyzer → the live graduation Metrics shape."""
    return Metrics(
        sharpe_ratio=pa.sharpe_ratio(),
        win_rate=pa.win_rate(),
        max_drawdown=abs(pa.max_drawdown()),   # analyzer returns negative %; gate wants positive
        n_trades=len(pa.trades),
        days_active=_days_span(engine),
    )


def gate_report(m: Metrics) -> str:
    def row(name, cur, tgt, ok, suffix=""):
        mark = "PASS" if ok else "----"
        return f"  [{mark}] {name:<10} {cur:>8.2f}{suffix}  (need {tgt}{suffix})"
    return "\n".join([
        row("Sharpe",   m.sharpe_ratio, GRAD_SHARPE,     m.sharpe_ratio >= GRAD_SHARPE),
        row("Win rate", m.win_rate,     GRAD_WINRATE,    m.win_rate >= GRAD_WINRATE, "%"),
        row("Max DD",   m.max_drawdown, GRAD_MAX_DD,     m.max_drawdown < GRAD_MAX_DD, "%"),
        row("Trades",   m.n_trades,     GRAD_MIN_TRADES, m.n_trades >= GRAD_MIN_TRADES),
        row("Days",     m.days_active,  GRAD_MIN_DAYS,   m.days_active >= GRAD_MIN_DAYS),
    ])


def main():
    ap = argparse.ArgumentParser(description="NEXUS backtest → graduation-gate verdict")
    ap.add_argument("--csv", help="Path to OHLCV CSV (timestamp,open,high,low,close,volume)")
    ap.add_argument("--symbol", default="BTC", help="Asset symbol (e.g. BTC, ETH, SOL)")
    ap.add_argument("--timeframe", default="1d", help="1h, 4h, 1d (for keyless fetch)")
    ap.add_argument("--bars", type=int, default=400, help="Bars to fetch if no CSV")
    args = ap.parse_args()

    engine = BacktestEngine(BacktestConfig(timeframe=args.timeframe))

    # 1) Load real data
    if args.csv:
        engine.load_csv(args.csv, symbol=args.symbol)
    else:
        from nexus_data import fetch_ohlcv          # keyless; needs network
        df = fetch_ohlcv(args.symbol, timeframe=args.timeframe, limit=args.bars)
        if df is None or len(df) < 100:
            print(f"Could not fetch enough data for {args.symbol} "
                  f"(need >=100 bars). Try --csv, a different symbol, or more --bars.")
            sys.exit(1)
        engine.load_dataframe(df, symbol=args.symbol)

    print("=" * 64)
    print(f"NEXUS BACKTEST  —  {engine.symbol}  ({len(engine.df)} bars, {args.timeframe})")
    print("=" * 64)

    # 2) Run it
    pa = engine.run()

    # 3) Performance summary (real numbers from the backtest)
    print("\nPERFORMANCE")
    print(f"  Total return : {pa.total_return():+.2f}%")
    print(f"  Trades       : {len(pa.trades)}")
    print(f"  Win rate     : {pa.win_rate():.1f}%")
    print(f"  Profit factor: {pa.profit_factor():.2f}")
    print(f"  Expectancy   : {pa.expectancy():+.2f}R")
    print(f"  Max drawdown : {pa.max_drawdown():.1f}%")
    print(f"  Sharpe       : {pa.sharpe_ratio():.2f}")
    print(f"  Sortino      : {pa.sortino_ratio():.2f}")

    # 4) Graduation-gate verdict (same gate the live trader must clear)
    m = bridge_to_gate(pa, engine)
    print("\nGRADUATION GATE (would this track record qualify for Pro Trader?)")
    print(gate_report(m))
    passed = all_gates_met(m)
    print(f"\n  Trust score: {compute_trust(m)}/100")
    print(f"  VERDICT: {'QUALIFIES ✅' if passed else 'does NOT yet qualify'}")

    print("\nNote: backtest validates the TECHNICAL/pattern core (replayable on price).")
    print("The deep-intel layer (on-chain/dev/sentiment) is not replayed here.")
    print("=" * 64)


if __name__ == "__main__":
    main()
