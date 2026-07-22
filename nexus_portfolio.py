"""
NEXUS Portfolio Manager
========================
Manages paper and live positions with SQLite persistence.
All state stored locally in nexus.db — nothing in the cloud.

Modes:
  PAPER — all trades simulated at live prices, no real money
  LIVE  — real execution via CCXTExecutor or XRPLExecutor

Paper → Live graduation requires:
  30 days ⋅ Sharpe ≥ 1.2 ⋅ Win rate ≥ 45% ⋅ Max DD < 20% ⋅ 20+ trades

Install:
  pip install pandas (sqlite3 is built into Python)
"""

import sqlite3
import json
import math
import uuid
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("nexus.portfolio")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PAPER = "PAPER"
LIVE  = "LIVE"

# Graduation thresholds
GRAD_DAYS      = 30
GRAD_SHARPE    = 1.2
GRAD_WINRATE   = 45.0
GRAD_MAX_DD    = 20.0
GRAD_MIN_TRADES = 20

# Default risk parameters (overridden by user config)
DEFAULT_START_CAPITAL = 100_000.0
DEFAULT_RISK_PCT      = 0.02    # 2% per trade
DEFAULT_SL_PCT        = 0.04    # 4% stop loss
DEFAULT_TP_PCT        = 0.10    # 10% take profit
DEFAULT_MAX_POSITIONS = 5


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class Position:
    id:             str
    mode:           str           # PAPER or LIVE
    symbol:         str
    name:           str
    category:       str
    direction:      str           # LONG or SHORT
    entry_price:    float
    entry_time:     str           # ISO
    units:          float
    position_value: float         # entry_price × units
    stop_loss:      float
    take_profit:    float
    conviction:     float
    pattern:        str
    exchange:       str = "paper" # or "binance", "xrpl", etc.
    current_price:  float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pct: float = 0.0

    def update_price(self, price: float):
        self.current_price  = price
        self.unrealized_pnl = (price - self.entry_price) * self.units
        self.unrealized_pct = (price - self.entry_price) / self.entry_price * 100

    def should_stop(self, price: float) -> bool:
        return (self.direction == "LONG"  and price <= self.stop_loss) or \
               (self.direction == "SHORT" and price >= self.stop_loss)

    def should_target(self, price: float) -> bool:
        return (self.direction == "LONG"  and price >= self.take_profit) or \
               (self.direction == "SHORT" and price <= self.take_profit)


@dataclass
class Trade:
    id:             str
    mode:           str
    symbol:         str
    name:           str
    category:       str
    direction:      str
    entry_price:    float
    exit_price:     float
    units:          float
    position_value: float
    pnl:            float
    pnl_pct:        float
    entry_time:     str
    exit_time:      str
    duration_sec:   int
    stop_loss:      float
    take_profit:    float
    conviction:     float
    pattern:        str
    exit_reason:    str   # TAKE_PROFIT / STOP_LOSS / MANUAL / SIGNAL
    exchange:       str


@dataclass
class PortfolioSnapshot:
    timestamp:        str
    mode:             str
    equity:           float
    cash:             float
    open_count:       int
    unrealized_pnl:   float


@dataclass
class PerformanceMetrics:
    mode:            str
    equity:          float
    start_capital:   float
    total_return:    float      # %
    win_rate:        float      # %
    profit_factor:   float
    sharpe_ratio:    float      # annualized
    sortino_ratio:   float
    max_drawdown:    float      # %
    avg_win:         float      # avg P&L on winners
    avg_loss:        float      # avg P&L on losers (positive number)
    avg_hold_h:      float      # average trade duration in hours
    n_trades:        int
    n_open:          int
    days_active:     float
    gross_profit:    float
    gross_loss:      float

    def graduation_status(self) -> dict:
        return {
            "days":      {"met": self.days_active >= GRAD_DAYS,      "current": self.days_active,    "target": GRAD_DAYS},
            "sharpe":    {"met": self.sharpe_ratio >= GRAD_SHARPE,   "current": self.sharpe_ratio,   "target": GRAD_SHARPE},
            "win_rate":  {"met": self.win_rate >= GRAD_WINRATE,      "current": self.win_rate,       "target": GRAD_WINRATE},
            "max_dd":    {"met": self.max_drawdown < GRAD_MAX_DD,    "current": self.max_drawdown,   "target": GRAD_MAX_DD, "invert": True},
            "trades":    {"met": self.n_trades >= GRAD_MIN_TRADES,   "current": self.n_trades,       "target": GRAD_MIN_TRADES},
            "all_met":   (self.days_active    >= GRAD_DAYS      and
                          self.sharpe_ratio   >= GRAD_SHARPE    and
                          self.win_rate       >= GRAD_WINRATE   and
                          self.max_drawdown   <  GRAD_MAX_DD    and
                          self.n_trades       >= GRAD_MIN_TRADES),
        }


# ─────────────────────────────────────────────
# Database Layer
# ─────────────────────────────────────────────

class Database:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id             TEXT PRIMARY KEY,
            mode           TEXT,
            symbol         TEXT,
            name           TEXT,
            category       TEXT,
            direction      TEXT,
            entry_price    REAL,
            entry_time     TEXT,
            units          REAL,
            position_value REAL,
            stop_loss      REAL,
            take_profit    REAL,
            conviction     REAL,
            pattern        TEXT,
            exchange       TEXT DEFAULT 'paper',
            status         TEXT DEFAULT 'OPEN'
        );

        CREATE TABLE IF NOT EXISTS trades (
            id             TEXT PRIMARY KEY,
            mode           TEXT,
            symbol         TEXT,
            name           TEXT,
            category       TEXT,
            direction      TEXT,
            entry_price    REAL,
            exit_price     REAL,
            units          REAL,
            position_value REAL,
            pnl            REAL,
            pnl_pct        REAL,
            entry_time     TEXT,
            exit_time      TEXT,
            duration_sec   INTEGER,
            stop_loss      REAL,
            take_profit    REAL,
            conviction     REAL,
            pattern        TEXT,
            exit_reason    TEXT,
            exchange       TEXT DEFAULT 'paper'
        );

        CREATE TABLE IF NOT EXISTS equity_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            mode            TEXT,
            equity          REAL,
            cash            REAL,
            open_count      INTEGER,
            unrealized_pnl  REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id        TEXT PRIMARY KEY,
            symbol    TEXT,
            name      TEXT,
            category  TEXT,
            score     REAL,
            price     REAL,
            pattern   TEXT,
            timestamp TEXT,
            entered   INTEGER DEFAULT 0,
            mode      TEXT
        );

        CREATE TABLE IF NOT EXISTS breadth_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            score            REAL,
            state            TEXT,
            multiplier       REAL,
            pct_above_ema20  REAL,
            pct_above_ema200 REAL,
            pct_rsi_bullish  REAL,
            volume_breadth   REAL,
            pct_macd_bullish REAL,
            advance_count    INTEGER,
            decline_count    INTEGER,
            new_highs        INTEGER,
            new_lows         INTEGER,
            alerts           TEXT
        );

        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trades_mode   ON trades(mode);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_equity_ts     ON equity_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_ts    ON signals(timestamp);
        """)
        self.conn.commit()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params):
        return self.conn.executemany(sql, params)

    def commit(self):
        self.conn.commit()

    def get_config(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        if row:
            try: return json.loads(row["value"])
            except: return row["value"]
        return default

    def set_config(self, key: str, value):
        self.conn.execute("INSERT OR REPLACE INTO config(key,value) VALUES(?,?)",
                          (key, json.dumps(value)))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────
# Portfolio Manager
# ─────────────────────────────────────────────

class PortfolioManager:
    """
    Central portfolio manager for both paper and live trading.
    Thread-safe via SQLite.
    """

    def __init__(self, db: Database, mode: str = PAPER):
        self.db   = db
        self.mode = mode
        self._ensure_config()

    def _ensure_config(self):
        if not self.db.get_config("start_capital"):
            self.db.set_config("start_capital", DEFAULT_START_CAPITAL)
        if not self.db.get_config("cash"):
            self.db.set_config("cash", DEFAULT_START_CAPITAL)
        if not self.db.get_config("start_date"):
            self.db.set_config("start_date", datetime.now(timezone.utc).isoformat())
        if not self.db.get_config("risk_pct"):
            self.db.set_config("risk_pct",   DEFAULT_RISK_PCT)
        if not self.db.get_config("sl_pct"):
            self.db.set_config("sl_pct",     DEFAULT_SL_PCT)
        if not self.db.get_config("tp_pct"):
            self.db.set_config("tp_pct",     DEFAULT_TP_PCT)
        if not self.db.get_config("max_positions"):
            self.db.set_config("max_positions", DEFAULT_MAX_POSITIONS)

    @property
    def cash(self) -> float:
        return float(self.db.get_config("cash") or DEFAULT_START_CAPITAL)

    @property
    def start_capital(self) -> float:
        return float(self.db.get_config("start_capital") or DEFAULT_START_CAPITAL)

    @property
    def start_date(self) -> datetime:
        raw = self.db.get_config("start_date")
        return datetime.fromisoformat(raw) if raw else datetime.now(timezone.utc)

    @property
    def risk_pct(self)  -> float: return float(self.db.get_config("risk_pct")   or DEFAULT_RISK_PCT)
    @property
    def sl_pct(self)    -> float: return float(self.db.get_config("sl_pct")     or DEFAULT_SL_PCT)
    @property
    def tp_pct(self)    -> float: return float(self.db.get_config("tp_pct")     or DEFAULT_TP_PCT)
    @property
    def max_pos(self)   -> int:   return int(self.db.get_config("max_positions") or DEFAULT_MAX_POSITIONS)

    # ── Positions ──────────────────────────────

    def get_open_positions(self) -> list[Position]:
        rows = self.db.execute(
            "SELECT * FROM positions WHERE status='OPEN' AND mode=?", (self.mode,)
        ).fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_position(self, pos_id: str) -> Optional[Position]:
        row = self.db.execute(
            "SELECT * FROM positions WHERE id=? AND mode=?", (pos_id, self.mode)
        ).fetchone()
        return self._row_to_position(row) if row else None

    def _row_to_position(self, row) -> Position:
        return Position(
            id=row["id"], mode=row["mode"], symbol=row["symbol"],
            name=row["name"], category=row["category"], direction=row["direction"],
            entry_price=row["entry_price"], entry_time=row["entry_time"],
            units=row["units"], position_value=row["position_value"],
            stop_loss=row["stop_loss"], take_profit=row["take_profit"],
            conviction=row["conviction"], pattern=row["pattern"],
            exchange=row["exchange"],
        )

    def open_position(self, symbol: str, name: str, category: str,
                      price: float, conviction: float, pattern: str,
                      exchange: str = "paper") -> Optional[Position]:
        """Create a new position using Kelly-approximated sizing."""
        open_pos = self.get_open_positions()
        if len(open_pos) >= self.max_pos:
            log.warning(f"Max positions ({self.max_pos}) reached")
            return None
        if any(p.symbol == symbol for p in open_pos):
            log.debug(f"Already have open position in {symbol}")
            return None

        # Kelly-approximated sizing
        equity        = self.compute_equity()
        risk_capital  = equity * self.risk_pct
        stop_price    = price * (1 - self.sl_pct)
        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return None
        units         = risk_capital / risk_per_unit
        pos_value     = units * price

        # Concentration cap: a single position may not exceed 45% of cash.
        # Size DOWN to the cap rather than rejecting the trade outright.
        cap = self.cash * 0.45
        if self.cash < price:
            log.warning(f"Insufficient cash for {symbol} position")
            return None
        if pos_value > cap:
            pos_value = cap
            units     = pos_value / price

        pos = Position(
            id             = str(uuid.uuid4())[:12],
            mode           = self.mode,
            symbol         = symbol,
            name           = name,
            category       = category,
            direction      = "LONG",
            entry_price    = price,
            entry_time     = datetime.now(timezone.utc).isoformat(),
            units          = units,
            position_value = pos_value,
            stop_loss      = stop_price,
            take_profit    = price * (1 + self.tp_pct),
            conviction     = conviction,
            pattern        = pattern,
            exchange       = exchange,
        )

        self.db.execute("""
            INSERT INTO positions
            (id,mode,symbol,name,category,direction,entry_price,entry_time,
             units,position_value,stop_loss,take_profit,conviction,pattern,exchange,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')
        """, (pos.id,pos.mode,pos.symbol,pos.name,pos.category,pos.direction,
              pos.entry_price,pos.entry_time,pos.units,pos.position_value,
              pos.stop_loss,pos.take_profit,pos.conviction,pos.pattern,pos.exchange))

        # Deduct from cash
        self.db.set_config("cash", self.cash - pos_value)
        self.db.commit()

        log.info(f"[{self.mode}] Opened {symbol} at ${price:.4f} | "
                 f"units={units:.4f} | SL=${stop_price:.4f} | TP=${pos.take_profit:.4f}")
        return pos

    def close_position(self, pos_id: str, exit_price: float,
                       reason: str = "MANUAL") -> Optional[Trade]:
        """Close a position and record the trade."""
        pos = self.get_position(pos_id)
        if not pos:
            return None

        now       = datetime.now(timezone.utc)
        entry_dt  = datetime.fromisoformat(pos.entry_time)
        duration  = int((now - entry_dt).total_seconds())
        pnl       = (exit_price - pos.entry_price) * pos.units
        pnl_pct   = (exit_price - pos.entry_price) / pos.entry_price * 100

        trade = Trade(
            id=pos.id, mode=pos.mode, symbol=pos.symbol, name=pos.name,
            category=pos.category, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            units=pos.units, position_value=pos.position_value,
            pnl=pnl, pnl_pct=pnl_pct,
            entry_time=pos.entry_time, exit_time=now.isoformat(),
            duration_sec=duration,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            conviction=pos.conviction, pattern=pos.pattern,
            exit_reason=reason, exchange=pos.exchange,
        )

        self.db.execute("""
            INSERT INTO trades
            (id,mode,symbol,name,category,direction,entry_price,exit_price,
             units,position_value,pnl,pnl_pct,entry_time,exit_time,duration_sec,
             stop_loss,take_profit,conviction,pattern,exit_reason,exchange)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (trade.id,trade.mode,trade.symbol,trade.name,trade.category,
              trade.direction,trade.entry_price,trade.exit_price,trade.units,
              trade.position_value,trade.pnl,trade.pnl_pct,trade.entry_time,
              trade.exit_time,trade.duration_sec,trade.stop_loss,trade.take_profit,
              trade.conviction,trade.pattern,trade.exit_reason,trade.exchange))

        self.db.execute(
            "UPDATE positions SET status='CLOSED' WHERE id=?", (pos.id,)
        )

        # Return cash + P&L
        self.db.set_config("cash", self.cash + pos.position_value + pnl)
        self.db.commit()

        log.info(f"[{self.mode}] Closed {pos.symbol} at ${exit_price:.4f} | "
                 f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
        return trade

    def check_positions(self, price_map: dict[str, float]) -> list[Trade]:
        """
        Check all open positions against current prices.
        Closes positions that hit SL or TP.
        Returns list of closed trades.
        """
        closed = []
        for pos in self.get_open_positions():
            price = price_map.get(pos.symbol)
            if not price:
                continue
            if pos.should_stop(price):
                trade = self.close_position(pos.id, pos.stop_loss, "STOP_LOSS")
                if trade: closed.append(trade)
            elif pos.should_target(price):
                trade = self.close_position(pos.id, pos.take_profit, "TAKE_PROFIT")
                if trade: closed.append(trade)
        return closed

    # ── Equity & Metrics ──────────────────────

    def compute_equity(self, price_map: dict[str, float] | None = None) -> float:
        """Current equity = cash + market value of all open positions.

        open_position() deducts the full notional from cash, so equity must add
        back each position's *market value* (units x current price), not just its
        unrealized P&L — otherwise spent cash appears to vanish.
        """
        positions_value = 0.0
        for pos in self.get_open_positions():
            price = (price_map or {}).get(pos.symbol, pos.entry_price)
            positions_value += price * pos.units
        return self.cash + positions_value

    def snapshot_equity(self, price_map: dict[str, float] | None = None):
        """Save current equity to history (called hourly)."""
        open_pos   = self.get_open_positions()
        equity     = self.compute_equity(price_map)
        unrealized = equity - self.cash
        self.db.execute("""
            INSERT INTO equity_history(timestamp,mode,equity,cash,open_count,unrealized_pnl)
            VALUES (?,?,?,?,?,?)
        """, (datetime.now(timezone.utc).isoformat(), self.mode,
              equity, self.cash, len(open_pos), unrealized))
        self.db.commit()

    def get_equity_history(self, days: int = 30) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows  = self.db.execute(
            "SELECT timestamp,equity FROM equity_history WHERE mode=? AND timestamp>=? ORDER BY timestamp",
            (self.mode, since)
        ).fetchall()
        return [{"ts": r["timestamp"], "value": r["equity"]} for r in rows]

    def get_trades(self, limit: int = 50, mode: str | None = None) -> list[Trade]:
        mode = mode or self.mode
        rows = self.db.execute(
            "SELECT * FROM trades WHERE mode=? ORDER BY exit_time DESC LIMIT ?",
            (mode, limit)
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def _row_to_trade(self, row) -> Trade:
        return Trade(**{k: row[k] for k in row.keys()})

    def compute_metrics(self, price_map: dict | None = None) -> PerformanceMetrics:
        """Full performance metrics including Sharpe, Sortino, max drawdown."""
        trades      = self.get_trades(limit=9999)
        equity_hist = self.get_equity_history(days=365)
        equity      = self.compute_equity(price_map)
        start       = self.start_capital
        days        = max(0, (datetime.now(timezone.utc) - self.start_date).total_seconds() / 86400)

        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if wins else 0.0)
        win_rate      = len(wins) / len(trades) * 100 if trades else 0.0
        avg_win       = gross_profit / len(wins) if wins else 0.0
        avg_loss      = gross_loss / len(losses) if losses else 0.0
        avg_hold_h    = sum(t.duration_sec for t in trades) / max(1, len(trades)) / 3600

        sharpe  = self._sharpe(equity_hist)
        sortino = self._sortino(equity_hist)
        max_dd  = self._max_drawdown(equity_hist)

        return PerformanceMetrics(
            mode=self.mode, equity=equity, start_capital=start,
            total_return=(equity - start) / start * 100,
            win_rate=win_rate, profit_factor=profit_factor,
            sharpe_ratio=sharpe, sortino_ratio=sortino,
            max_drawdown=max_dd, avg_win=avg_win, avg_loss=avg_loss,
            avg_hold_h=avg_hold_h, n_trades=len(trades),
            n_open=len(self.get_open_positions()), days_active=days,
            gross_profit=gross_profit, gross_loss=gross_loss,
        )

    @staticmethod
    def _daily_returns(history: list[dict]) -> list[float]:
        if len(history) < 2: return []
        rets = []
        for i in range(1, len(history)):
            prev = history[i-1]["value"]
            curr = history[i]["value"]
            if prev > 0: rets.append((curr - prev) / prev)
        return rets

    def _sharpe(self, history: list[dict], rf_annual: float = 0.05) -> float:
        rets = self._daily_returns(history)
        if len(rets) < 2: return 0.0
        mean  = sum(rets) / len(rets)
        std   = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
        if std == 0: return 0.0
        rf    = rf_annual / 365
        return round(min(9.9, max(-9.9, ((mean - rf) / std) * math.sqrt(365))), 3)

    def _sortino(self, history: list[dict], rf_annual: float = 0.05) -> float:
        rets  = self._daily_returns(history)
        if len(rets) < 2: return 0.0
        mean  = sum(rets) / len(rets)
        neg   = [r for r in rets if r < 0]
        if not neg: return 9.9
        std_d = math.sqrt(sum(r**2 for r in neg) / len(neg))
        if std_d == 0: return 9.9
        rf    = rf_annual / 365
        return round(min(9.9, max(-9.9, ((mean - rf) / std_d) * math.sqrt(365))), 3)

    def _max_drawdown(self, history: list[dict]) -> float:
        if len(history) < 2: return 0.0
        peak  = history[0]["value"]
        max_dd = 0.0
        for h in history:
            if h["value"] > peak: peak = h["value"]
            dd = (peak - h["value"]) / peak * 100
            if dd > max_dd: max_dd = dd
        return round(max_dd, 2)

    # ── Signal logging ─────────────────────────

    def log_signal(self, symbol: str, name: str, category: str,
                   score: float, price: float, pattern: str, entered: bool):
        self.db.execute("""
            INSERT OR IGNORE INTO signals(id,symbol,name,category,score,price,pattern,timestamp,entered,mode)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (str(uuid.uuid4())[:12], symbol, name, category, score, price, pattern,
              datetime.now(timezone.utc).isoformat(), int(entered), self.mode))
        self.db.commit()

    def get_signals(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM signals WHERE mode=? ORDER BY timestamp DESC LIMIT ?",
            (self.mode, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Breadth logging ───────────────────────

    def log_breadth(self, snap) -> None:
        """Log a BreadthSnapshot to the database."""
        self.db.execute("""
            INSERT INTO breadth_log
            (timestamp,score,state,multiplier,pct_above_ema20,pct_above_ema200,
             pct_rsi_bullish,volume_breadth,pct_macd_bullish,advance_count,
             decline_count,new_highs,new_lows,alerts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (snap.timestamp.isoformat(), snap.breadth_score, snap.breadth_state,
              snap.conviction_multiplier, snap.pct_above_ema20, snap.pct_above_ema200,
              snap.pct_rsi_bullish, snap.volume_breadth, snap.pct_macd_bullish,
              snap.advance_count, snap.decline_count, snap.new_highs, snap.new_lows,
              json.dumps(snap.alerts)))
        self.db.commit()

    def get_breadth_history(self, hours: int = 24) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows  = self.db.execute(
            "SELECT * FROM breadth_log WHERE timestamp>=? ORDER BY timestamp",
            (since,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Configuration ─────────────────────────

    def update_config(self, **kwargs):
        allowed = {"start_capital","cash","risk_pct","sl_pct","tp_pct","max_positions"}
        for k, v in kwargs.items():
            if k in allowed:
                self.db.set_config(k, v)

    def set_mode(self, mode: str):
        assert mode in (PAPER, LIVE)
        self.mode = mode
        log.info(f"Portfolio mode switched to: {mode}")

    def reset_paper(self):
        """Wipe all PAPER data and restart from scratch."""
        self.db.execute("DELETE FROM positions   WHERE mode='PAPER'")
        self.db.execute("DELETE FROM trades      WHERE mode='PAPER'")
        self.db.execute("DELETE FROM equity_history WHERE mode='PAPER'")
        self.db.execute("DELETE FROM signals     WHERE mode='PAPER'")
        cap = self.start_capital
        self.db.set_config("cash",       cap)
        self.db.set_config("start_date", datetime.now(timezone.utc).isoformat())
        self.db.commit()
        log.info("Paper trading data reset")
