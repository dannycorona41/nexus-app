"""
NEXUS Backend Server
=====================
FastAPI application that orchestrates the entire NEXUS trading system.
Spawned automatically by the Electron desktop app on port 7432.

REST Endpoints:
  GET  /api/status                 System health + module status
  GET  /api/config                 Current config (no keys)
  POST /api/config                 Update config
  GET  /api/breadth                Latest breadth snapshot
  GET  /api/breadth/history        24h breadth log
  GET  /api/signals                Recent signals
  POST /api/signals/scan           Trigger manual full scan
  GET  /api/score/{symbol}         Full conviction score for one asset
  GET  /api/portfolio              Portfolio summary (equity, mode, metrics)
  GET  /api/portfolio/positions    Open positions
  GET  /api/portfolio/trades       Closed trade history
  POST /api/portfolio/trade        Open a manual position
  DELETE /api/portfolio/trade/{id} Close a position manually
  POST /api/portfolio/mode         Switch PAPER ↔ LIVE
  POST /api/portfolio/reset-paper  Reset paper trading data
  GET  /api/assets                 Full asset universe
  GET  /api/graduation             Paper trading graduation criteria

WebSocket:
  WS /ws    Streams: price_update, signal, position_opened, position_closed,
             breadth_update, alert, status, daily_digest

Install:
  pip install fastapi uvicorn ccxt aiohttp pandas pandas-ta python-telegram-bot
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

try:
    import ccxt.async_support as ccxt_async
except Exception:
    ccxt_async = None  # optional execution/data client
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

# ── NEXUS modules ─────────────────────────────────────────────────────────────
from nexus_portfolio      import PortfolioManager, Database, PAPER, LIVE
from nexus_asset_universe import REGISTRY, Watchlist, get as get_asset
from nexus_breadth        import XRPLBreadthCalculator

log = logging.getLogger("nexus.server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR  = Path(os.getenv("NEXUS_DATA_DIR", BACKEND_DIR / ".." / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORT      = int(os.getenv("PORT", os.getenv("NEXUS_PORT", "7432")))
TG_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Global state ──────────────────────────────────────────────────────────────
db            = Database(DATA_DIR / "nexus.db")
portfolio     = PortfolioManager(db, mode=PAPER)
breadth_calc  = XRPLBreadthCalculator()
exchange      = None
ws_clients: set[WebSocket] = set()

state = {
    "market_data":    {},
    "last_breadth":   None,
    "last_signals":   [],
    "auto_trade":     True,
    "paused":         False,
    "mode":           PAPER,
    "signal_cooldown": {},     # symbol → last_signal_ts
    "backend_started": datetime.now(timezone.utc).isoformat(),
}


# ── WebSocket broadcaster ─────────────────────────────────────────────────────
async def broadcast(msg_type: str, data: Any):
    if not ws_clients:
        return
    payload = json.dumps({"type": msg_type, "data": data,
                           "ts": datetime.now(timezone.utc).isoformat()})
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ── CoinGecko price fetcher ───────────────────────────────────────────────────
async def fetch_prices() -> dict:
    """Fetch live prices for all tracked assets from CoinGecko (free tier)."""
    import aiohttp
    ids = ",".join(set(a.coingecko_id for a in REGISTRY if a.coingecko_id))
    url = (f"https://api.coingecko.com/api/v3/coins/markets"
           f"?vs_currency=usd&ids={ids}&order=market_cap_desc"
           f"&per_page=100&page=1&sparkline=false"
           f"&price_change_percentage=1h,24h,7d")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_CG_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                return {item["id"]: item for item in data}
    except Exception as e:
        log.warning(f"CoinGecko fetch error: {e}")
        return {}


def build_price_map(cg_data: dict) -> dict[str, float]:
    """Map symbol → current price from CoinGecko response."""
    price_map = {}
    for asset in REGISTRY:
        item = cg_data.get(asset.coingecko_id)
        if item:
            price_map[asset.base] = item.get("current_price", 0)
    return price_map


# ── Heuristic NEXUS score (CoinGecko-based, no API keys needed) ───────────────
def quick_score(asset, cg_item: dict | None) -> float:
    """Fast conviction score from CoinGecko data. Use nexus_signal_engine for full score."""
    if not cg_item:
        return 50.0
    score = 50.0
    c24   = cg_item.get("price_change_percentage_24h") or 0
    c7    = cg_item.get("price_change_percentage_7d_in_currency") or 0
    vol   = cg_item.get("total_volume") or 0
    mcap  = cg_item.get("market_cap") or 1
    rank  = cg_item.get("market_cap_rank") or 500

    vm_ratio = vol / mcap
    if vm_ratio > 0.15: score += 8
    elif vm_ratio > 0.08: score += 4

    if c24 > 5:    score += 12
    elif c24 > 2:  score += 8
    elif c24 > 0:  score += 4
    elif c24 < -8: score -= 12
    elif c24 < -4: score -= 8
    elif c24 < 0:  score -= 3

    if c7 > 10:    score += 10
    elif c7 > 3:   score += 6
    elif c7 < -15: score -= 10
    elif c7 < -5:  score -= 5

    if rank <= 10:  score += 8
    elif rank <= 30: score += 5
    elif rank <= 100: score += 2

    return round(min(98, max(10, score)), 1)


# ── Signal engine ─────────────────────────────────────────────────────────────
SIGNAL_THRESHOLD  = 75
SIGNAL_COOLDOWN_H = 8

def should_fire_signal(symbol: str) -> bool:
    last = state["signal_cooldown"].get(symbol, 0)
    return (datetime.now(timezone.utc).timestamp() - last) > SIGNAL_COOLDOWN_H * 3600

def run_signal_scan() -> list[dict]:
    """Scan all assets and fire signals for those scoring above threshold."""
    if state["paused"] or not state["market_data"]:
        return []

    signals_fired = []
    for asset in REGISTRY:
        cg_item = state["market_data"].get(asset.coingecko_id)
        if not cg_item:
            continue

        score = quick_score(asset, cg_item)
        if score < SIGNAL_THRESHOLD:
            continue
        if not should_fire_signal(asset.base):
            continue

        price   = cg_item.get("current_price", 0)
        c24     = cg_item.get("price_change_percentage_24h") or 0
        c7      = cg_item.get("price_change_percentage_7d_in_currency") or 0
        pattern = ("Momentum breakout"       if score >= 85 and c24 > 3
                   else "7D trend confluence" if score >= 80 and c7 > 8
                   else "OTE zone alignment"  if score >= 78
                   else "Multi-factor signal")

        state["signal_cooldown"][asset.base] = datetime.now(timezone.utc).timestamp()

        entered = False
        if state["auto_trade"]:
            pos = portfolio.open_position(
                symbol=asset.base, name=asset.name, category=asset.category.value,
                price=price, conviction=score, pattern=pattern,
            )
            if pos:
                entered = True

        sig = {
            "symbol": asset.base, "name": asset.name,
            "category": asset.category.value, "score": score,
            "price": price, "pattern": pattern,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entered": entered,
        }
        portfolio.log_signal(
            asset.base, asset.name, asset.category.value,
            score, price, pattern, entered
        )
        state["last_signals"] = [sig] + state["last_signals"][:49]
        signals_fired.append(sig)

    return signals_fired


# ── Background loops ──────────────────────────────────────────────────────────
async def price_loop():
    """Fetch prices every 60 seconds, check positions, run signal scan."""
    global exchange
    try:
        exchange = ccxt_async.kraken({"enableRateLimit": True})
    except Exception:
        log.warning("CCXT init failed — using CoinGecko only")

    while True:
        try:
            data = await fetch_prices()
            if data:
                state["market_data"] = data
                price_map = build_price_map(data)

                # Check positions for SL/TP
                closed = portfolio.check_positions(price_map)
                for trade in closed:
                    await broadcast("position_closed", {
                        "symbol": trade.symbol, "pnl": trade.pnl,
                        "pnl_pct": trade.pnl_pct, "reason": trade.exit_reason,
                        "exit_price": trade.exit_price,
                    })
                    await telegram_alert(
                        f"{'🎯' if trade.exit_reason == 'TAKE_PROFIT' else '🛡'} "
                        f"{trade.symbol} closed | "
                        f"{'TP' if trade.exit_reason == 'TAKE_PROFIT' else 'SL'} hit\n"
                        f"P&L: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)"
                    )

                # Run signal scan
                signals = run_signal_scan()
                for sig in signals:
                    await broadcast("signal", sig)
                    if sig["score"] >= 85:
                        await telegram_alert(
                            f"⭐ CONVICTION signal: {sig['symbol']} [{sig['score']}]\n"
                            f"{sig['pattern']} · ${sig['price']:.4f}\n"
                            f"{'→ Position opened' if sig['entered'] else '→ Auto-trade off'}"
                        )

                # Broadcast price update
                await broadcast("price_update", {
                    a.base: {
                        "price": data[a.coingecko_id].get("current_price"),
                        "change_24h": data[a.coingecko_id].get("price_change_percentage_24h"),
                        "change_7d":  data[a.coingecko_id].get("price_change_percentage_7d_in_currency"),
                        "volume":     data[a.coingecko_id].get("total_volume"),
                        "image":      data[a.coingecko_id].get("image"),
                    }
                    for a in REGISTRY
                    if a.coingecko_id in data
                })

        except Exception as e:
            log.error(f"Price loop error: {e}")
        await asyncio.sleep(60)


async def breadth_loop():
    """Run full 30-asset breadth scan every hour."""
    await asyncio.sleep(10)  # let price loop run first
    while True:
        try:
            ex = ccxt_async.kraken({"enableRateLimit": True})
            snap = await breadth_calc.scan_all(ex)
            await ex.close()
            state["last_breadth"] = snap
            portfolio.log_breadth(snap)

            # Broadcast
            await broadcast("breadth_update", snap.as_dict_simple())

            # Alert on special conditions
            for alert in snap.alerts:
                await telegram_alert(f"⚡ Breadth Alert\n{alert}")
                await broadcast("alert", {"type": "BREADTH", "message": alert})

            # Update portfolio conviction multiplier
            log.info(f"Breadth: {snap.breadth_score:.1f} [{snap.breadth_state}] ×{snap.conviction_multiplier}")

        except Exception as e:
            log.error(f"Breadth loop error: {e}")
        await asyncio.sleep(3600)  # hourly


async def equity_snap_loop():
    """Save equity snapshots every 4 hours for the equity curve."""
    await asyncio.sleep(300)
    while True:
        try:
            price_map = build_price_map(state["market_data"])
            portfolio.snapshot_equity(price_map)
        except Exception as e:
            log.error(f"Equity snap error: {e}")
        await asyncio.sleep(14400)  # 4 hours


async def daily_digest_loop():
    """Send daily Telegram digest at configured time."""
    while True:
        now = datetime.now(timezone.utc)
        # Send at 08:00 UTC
        target = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now.hour >= 8:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        try:
            await send_daily_digest()
        except Exception as e:
            log.error(f"Daily digest error: {e}")


async def send_daily_digest():
    price_map = build_price_map(state["market_data"])
    m = portfolio.compute_metrics(price_map)
    g = m.graduation_status()
    open_pos = portfolio.get_open_positions()
    snap = state.get("last_breadth")
    breadth_str = f"Breadth: {snap.breadth_score:.0f}/100 [{snap.breadth_state}]\n" if snap else ""

    criteria_met = sum(1 for v in g.values() if isinstance(v, dict) and v.get("met"))
    grad_line = "✅ Graduation: all criteria met" if g.get("all_met") else f"🎓 Graduation: {criteria_met}/5 criteria"

    msg = (
        f"📊 NEXUS Daily Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"Mode: {'🔴 LIVE' if portfolio.mode == LIVE else '🧪 PAPER'} | "
        f"Auto: {'ON' if state['auto_trade'] else 'OFF'}\n\n"
        f"💰 Portfolio: ${m.equity:,.0f} ({m.total_return:+.2f}%)\n"
        f"{breadth_str}"
        f"Open positions: {m.n_open} | Closed trades: {m.n_trades}\n"
        f"Win rate: {m.win_rate:.1f}% | Sharpe: {m.sharpe_ratio:.2f} | Max DD: {m.max_drawdown:.1f}%\n\n"
        f"{grad_line}"
    )
    await telegram_alert(msg)


# ── Telegram alerting ─────────────────────────────────────────────────────────
async def telegram_alert(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={
                "chat_id": TG_CHAT,
                "text": msg,
                "parse_mode": "Markdown",
            }, timeout=aiohttp.ClientTimeout(total=8))
    except Exception as e:
        log.debug(f"Telegram error: {e}")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"NEXUS backend starting on port {PORT}")
    log.info(f"Data directory: {DATA_DIR}")
    log.info(f"Portfolio mode: {portfolio.mode}")

    asyncio.create_task(price_loop())
    asyncio.create_task(breadth_loop())
    asyncio.create_task(equity_snap_loop())
    asyncio.create_task(daily_digest_loop())

    yield

    if exchange:
        await exchange.close()
    db.close()
    log.info("NEXUS backend shutdown")


app = FastAPI(title="NEXUS API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Serve the dashboard (single-unit deploy) ──────────────────────────────────
DASHBOARD_FILE = BACKEND_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
async def dashboard_root():
    if DASHBOARD_FILE.exists():
        return FileResponse(str(DASHBOARD_FILE))
    return HTMLResponse("<h1>NEXUS</h1><p>Dashboard file missing.</p>", status_code=200)

@app.get("/health")
async def health():
    return {"ok": True}


# ── API Models ────────────────────────────────────────────────────────────────
class TradeRequest(BaseModel):
    symbol:    str
    price:     float | None = None
    conviction: float = 75.0
    pattern:   str = "Manual"

class ConfigUpdate(BaseModel):
    auto_trade:     bool | None   = None
    paused:         bool | None   = None
    risk_pct:       float | None  = None
    sl_pct:         float | None  = None
    tp_pct:         float | None  = None
    max_positions:  int | None    = None
    signal_threshold: int | None  = None

class ModeSwitch(BaseModel):
    mode:    str  # PAPER or LIVE
    confirm: str  # Must be "CONFIRM" for LIVE mode


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {
        "status":    "online",
        "mode":      portfolio.mode,
        "paused":    state["paused"],
        "auto_trade":state["auto_trade"],
        "open_positions": len(portfolio.get_open_positions()),
        "last_breadth_score": state["last_breadth"].breadth_score if state["last_breadth"] else None,
        "market_data_age_s": None,
        "started":   state["backend_started"],
        "telegram":  bool(TG_TOKEN and TG_CHAT),
        "data_dir":  str(DATA_DIR),
    }


@app.get("/api/config")
async def get_config():
    return {
        "mode":              portfolio.mode,
        "auto_trade":        state["auto_trade"],
        "paused":            state["paused"],
        "start_capital":     portfolio.start_capital,
        "risk_pct":          portfolio.risk_pct * 100,
        "sl_pct":            portfolio.sl_pct * 100,
        "tp_pct":            portfolio.tp_pct * 100,
        "max_positions":     portfolio.max_pos,
        "signal_threshold":  SIGNAL_THRESHOLD,
    }


@app.post("/api/config")
async def update_config(cfg: ConfigUpdate):
    global SIGNAL_THRESHOLD
    if cfg.auto_trade  is not None: state["auto_trade"] = cfg.auto_trade
    if cfg.paused      is not None: state["paused"]     = cfg.paused
    if cfg.risk_pct    is not None: portfolio.update_config(risk_pct=cfg.risk_pct / 100)
    if cfg.sl_pct      is not None: portfolio.update_config(sl_pct=cfg.sl_pct / 100)
    if cfg.tp_pct      is not None: portfolio.update_config(tp_pct=cfg.tp_pct / 100)
    if cfg.max_positions is not None: portfolio.update_config(max_positions=cfg.max_positions)
    if cfg.signal_threshold is not None: SIGNAL_THRESHOLD = cfg.signal_threshold
    await broadcast("config_update", await get_config())
    return {"ok": True}


@app.get("/api/breadth")
async def get_breadth():
    snap = state.get("last_breadth")
    if not snap:
        return {"error": "No breadth data yet — waiting for first hourly scan"}
    return snap.as_dict_simple()


@app.get("/api/breadth/history")
async def get_breadth_history():
    return portfolio.get_breadth_history(hours=24)


@app.get("/api/signals")
async def get_signals(limit: int = 50):
    return portfolio.get_signals(limit=limit)


@app.post("/api/signals/scan")
async def trigger_scan():
    signals = run_signal_scan()
    for sig in signals:
        await broadcast("signal", sig)
    return {"fired": len(signals), "signals": signals}


@app.get("/api/score/{symbol}")
async def score_asset(symbol: str):
    """Quick score from CoinGecko data. Full score requires nexus_signal_engine."""
    symbol = symbol.upper()
    asset  = get_asset(symbol)
    if not asset:
        raise HTTPException(404, f"Asset {symbol} not found in universe")
    cg_item = state["market_data"].get(asset.coingecko_id)
    score   = quick_score(asset, cg_item)
    price   = cg_item.get("current_price") if cg_item else None
    multiplier = state["last_breadth"].conviction_multiplier if state["last_breadth"] else 1.0
    return {
        "symbol":      symbol,
        "score":       score,
        "adjusted":    round(min(100, score * multiplier), 1),
        "multiplier":  multiplier,
        "price":       price,
        "category":    asset.category.value,
        "weights":     asset.weights,
        "action":      ("CONVICTION" if score >= 85 else "POSITION" if score >= 75
                        else "RESEARCH" if score >= 60 else "WATCH" if score >= 45 else "SKIP"),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    price_map = build_price_map(state["market_data"])
    m = portfolio.compute_metrics(price_map)
    g = m.graduation_status()
    return {
        "mode":          m.mode,
        "equity":        round(m.equity, 2),
        "cash":          round(portfolio.cash, 2),
        "start_capital": m.start_capital,
        "total_return":  round(m.total_return, 2),
        "metrics": {
            "win_rate":     round(m.win_rate, 1),
            "sharpe":       m.sharpe_ratio,
            "sortino":      m.sortino_ratio,
            "max_drawdown": m.max_drawdown,
            "profit_factor":round(m.profit_factor, 2),
            "n_trades":     m.n_trades,
            "n_open":       m.n_open,
            "days_active":  round(m.days_active, 1),
            "avg_hold_h":   round(m.avg_hold_h, 1),
            "gross_profit": round(m.gross_profit, 2),
            "gross_loss":   round(m.gross_loss, 2),
        },
        "graduation":    g,
    }


@app.get("/api/portfolio/positions")
async def get_positions():
    pos_list = portfolio.get_open_positions()
    result   = []
    for pos in pos_list:
        price = (build_price_map(state["market_data"]).get(pos.symbol) or pos.entry_price)
        pos.update_price(price)
        result.append({
            "id":             pos.id,
            "symbol":         pos.symbol,
            "name":           pos.name,
            "category":       pos.category,
            "entry_price":    pos.entry_price,
            "current_price":  pos.current_price,
            "units":          pos.units,
            "position_value": pos.position_value,
            "stop_loss":      pos.stop_loss,
            "take_profit":    pos.take_profit,
            "unrealized_pnl": round(pos.unrealized_pnl, 2),
            "unrealized_pct": round(pos.unrealized_pct, 2),
            "conviction":     pos.conviction,
            "pattern":        pos.pattern,
            "entry_time":     pos.entry_time,
            "exchange":       pos.exchange,
        })
    return result


@app.get("/api/portfolio/trades")
async def get_trades(limit: int = 50):
    trades = portfolio.get_trades(limit=limit)
    return [t.__dict__ for t in trades]


@app.get("/api/portfolio/equity-history")
async def get_equity_history(days: int = 30):
    return portfolio.get_equity_history(days=days)


@app.post("/api/portfolio/trade")
async def open_trade(req: TradeRequest):
    asset = get_asset(req.symbol.upper())
    if not asset:
        raise HTTPException(404, f"Asset {req.symbol} not found")

    cg_item = state["market_data"].get(asset.coingecko_id)
    price   = req.price or (cg_item.get("current_price") if cg_item else None)
    if not price:
        raise HTTPException(400, "No price available — provide price or wait for market data")

    pos = portfolio.open_position(
        symbol=asset.base, name=asset.name, category=asset.category.value,
        price=price, conviction=req.conviction, pattern=req.pattern,
    )
    if not pos:
        raise HTTPException(409, "Could not open position — check max positions, cash, or existing position")

    await broadcast("position_opened", pos.__dict__)
    return {"ok": True, "position": pos.__dict__}


@app.delete("/api/portfolio/trade/{pos_id}")
async def close_trade(pos_id: str):
    price_map = build_price_map(state["market_data"])
    pos       = portfolio.get_position(pos_id)
    if not pos:
        raise HTTPException(404, "Position not found")
    price = price_map.get(pos.symbol, pos.entry_price)
    trade = portfolio.close_position(pos_id, price, "MANUAL")
    if not trade:
        raise HTTPException(500, "Close failed")
    await broadcast("position_closed", trade.__dict__)
    return {"ok": True, "trade": trade.__dict__}


@app.post("/api/portfolio/mode")
async def switch_mode(req: ModeSwitch):
    if req.mode == LIVE and req.confirm != "CONFIRM":
        raise HTTPException(400, "Live trading requires confirm='CONFIRM'")
    if req.mode == LIVE:
        m = portfolio.compute_metrics()
        g = m.graduation_status()
        if not g["all_met"]:
            raise HTTPException(403,
                "Cannot switch to LIVE — graduation criteria not met. "
                "Complete 30-day paper trading period first.")
    portfolio.set_mode(req.mode)
    state["mode"] = req.mode
    await broadcast("mode_change", {"mode": req.mode})
    await telegram_alert(f"⚡ NEXUS mode switched to: *{req.mode}*")
    return {"ok": True, "mode": req.mode}


@app.post("/api/portfolio/reset-paper")
async def reset_paper():
    portfolio.reset_paper()
    state["signal_cooldown"] = {}
    state["last_signals"]    = []
    await broadcast("paper_reset", {})
    return {"ok": True}


@app.get("/api/assets")
async def get_assets():
    return [
        {
            "symbol":      a.base,
            "name":        a.name,
            "category":    a.category.value,
            "coingecko_id":a.coingecko_id,
            "description": a.description,
            "tags":        a.tags,
            "weights":     a.weights,
        }
        for a in REGISTRY
    ]


@app.get("/api/graduation")
async def get_graduation():
    m = portfolio.compute_metrics()
    return m.graduation_status()


# ── CoinGecko chart proxy (fixes browser rate-limiting) ───────────────────────
# The chart modal and gem-finder were calling api.coingecko.com directly from
# the browser, which uses the visitor's public IP. CoinGecko's free tier
# throttles hard per-IP; on a shared/office network you hit the ceiling
# quickly and every subsequent chart request returns "Chart unavailable".
# Proxying through the server + caching per-symbol/per-range fixes both the
# rate-limit issue AND cuts calls to CoinGecko dramatically.
_chart_cache: dict[str, tuple[float, dict]] = {}
_CHART_TTL_SEC = 300  # 5 min — charts of 30/90/365 days don't need finer resolution

# CoinGecko's free tier is stricter with default aiohttp/requests User-Agents.
# Passing a proper UA reduces 403s meaningfully; still the free tier, still
# rate-limited, but at least we're not getting blocked at the door.
_CG_HEADERS = {
    "User-Agent": "NEXUS/1.0 (+https://nexus-app-t2vt.onrender.com)",
    "Accept": "application/json",
}

@app.get("/api/chart/{coingecko_id}")
async def get_chart(coingecko_id: str, days: int = 30):
    import aiohttp, time
    if days not in (7, 14, 30, 90, 180, 365):
        days = 30
    key = f"{coingecko_id}:{days}"
    now = time.time()
    hit = _chart_cache.get(key)
    if hit and (now - hit[0]) < _CHART_TTL_SEC:
        return hit[1]
    url = (f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_CG_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    # Return the last cached value if we have one, else structured error
                    if hit:
                        return hit[1]
                    return {"error": f"CoinGecko returned {resp.status}", "prices": [], "total_volumes": []}
                data = await resp.json()
                _chart_cache[key] = (now, data)
                return data
    except Exception as e:
        log.warning(f"Chart fetch error for {coingecko_id}: {e}")
        if hit:
            return hit[1]
        return {"error": str(e), "prices": [], "total_volumes": []}


@app.get("/api/chart-ohlc/{coingecko_id}")
async def get_chart_ohlc(coingecko_id: str, days: int = 90):
    """Real OHLC candles (open/high/low/close) for the Patterns tab.
       Same caching approach as /api/chart above."""
    import aiohttp, time
    if days not in (1, 7, 14, 30, 90, 180, 365):
        days = 90
    key = f"ohlc:{coingecko_id}:{days}"
    now = time.time()
    hit = _chart_cache.get(key)
    if hit and (now - hit[0]) < _CHART_TTL_SEC:
        return hit[1]
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/ohlc?vs_currency=usd&days={days}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_CG_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    if hit:
                        return hit[1]
                    return {"error": f"CoinGecko returned {resp.status}", "candles": []}
                data = await resp.json()
                # CoinGecko returns [[timestamp, open, high, low, close], ...]
                result = {"candles": [
                    {"t": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4]}
                    for row in data
                ]}
                _chart_cache[key] = (now, result)
                return result
    except Exception as e:
        log.warning(f"OHLC fetch error for {coingecko_id}: {e}")
        if hit:
            return hit[1]
        return {"error": str(e), "candles": []}


@app.get("/api/gems/scan")
async def scan_gems():
    """Server-side gem scan: pulls mid-cap CoinGecko markets pages 3-6, scores
       them by vol/mcap ratio + momentum. Cached 5 min. Surfaces HTTP status
       code from CoinGecko so if the scan comes back empty you actually know
       why (rate limit vs. block vs. genuine no-match)."""
    import aiohttp, time
    key = "gems:scan"
    now = time.time()
    hit = _chart_cache.get(key)
    if hit and (now - hit[0]) < _CHART_TTL_SEC:
        return hit[1]
    all_coins = []
    status_counts = {}  # e.g. {200: 3, 429: 1}
    try:
        async with aiohttp.ClientSession() as session:
            for page in (3, 4, 5, 6):
                url = (f"https://api.coingecko.com/api/v3/coins/markets"
                       f"?vs_currency=usd&order=volume_desc&per_page=50&page={page}&sparkline=false")
                async with session.get(url, headers=_CG_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    status_counts[resp.status] = status_counts.get(resp.status, 0) + 1
                    if resp.status != 200:
                        continue
                    all_coins.extend(await resp.json())
        # If CoinGecko blocked us on every page, surface that clearly
        if not all_coins:
            blocked = 429 in status_counts or 403 in status_counts
            reason = ("CoinGecko is rate-limiting this server right now — try again in ~1 min."
                      if blocked else f"CoinGecko returned no data (status counts: {status_counts})")
            result = {"gems": [], "scanned": 0, "error": reason, "cg_status": status_counts}
            _chart_cache[key] = (now, result)
            return result
        scored = []
        for c in all_coins:
            mc = c.get("market_cap") or 0
            vol = c.get("total_volume") or 0
            mom = c.get("price_change_percentage_24h") or 0
            if not (5_000_000 < mc < 500_000_000) or vol == 0:
                continue
            ratio = vol / mc
            mc_score = 25 if mc < 20_000_000 else (20 if mc < 100_000_000 else 15)
            ratio_score = 30 if ratio > 0.5 else (20 if ratio > 0.3 else (10 if ratio > 0.15 else 0))
            mom_score = 25 if mom > 10 else (15 if mom > 5 else (8 if mom > 0 else 0))
            scored.append({
                "s":     c["symbol"].upper(),
                "n":     c["name"],
                "mc":    f"${mc/1e6:.0f}M",
                "ratio": round(ratio, 2),
                "mom":   round(mom, 1),
                "score": min(98, mc_score + ratio_score + mom_score + 40),
                "why":   f"Vol/MC {ratio:.2f} · {mom:.1f}% 24h",
                "coingecko_id": c.get("id"),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        result = {"gems": scored[:8], "scanned": len(all_coins), "cg_status": status_counts}
        _chart_cache[key] = (now, result)
        return result
    except Exception as e:
        log.warning(f"Gem scan error: {e}")
        if hit:
            return hit[1]
        return {"error": str(e), "gems": [], "scanned": 0}


@app.get("/api/trending")
async def get_trending():
    """Real CoinGecko trending data (their public /search/trending endpoint).
       Cached 5 minutes. Replaces the TRENDING_SEED hardcoded fake data that
       shipped in v5 by mistake."""
    import aiohttp, time
    key = "trending"
    now = time.time()
    hit = _chart_cache.get(key)
    if hit and (now - hit[0]) < _CHART_TTL_SEC:
        return hit[1]
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_CG_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    if hit:
                        return hit[1]
                    return {"error": f"CoinGecko returned {resp.status}", "coins": []}
                data = await resp.json()
                # CoinGecko returns { "coins": [{ "item": {...} }, ...] }
                # Normalize to what the frontend expects: [{s, n, r, ch, coingecko_id}]
                coins = []
                for i, entry in enumerate((data.get("coins") or [])[:7]):
                    item = entry.get("item", {})
                    coins.append({
                        "s":     (item.get("symbol") or "").upper(),
                        "n":     item.get("name") or "",
                        "r":     i + 1,
                        # /search/trending doesn't include price change; look at the data field if present
                        "ch":    (item.get("data") or {}).get("price_change_percentage_24h", {}).get("usd", 0) or 0,
                        "coingecko_id": item.get("id"),
                        "market_cap_rank": item.get("market_cap_rank"),
                    })
                result = {"coins": coins}
                _chart_cache[key] = (now, result)
                return result
    except Exception as e:
        log.warning(f"Trending fetch error: {e}")
        if hit:
            return hit[1]
        return {"error": str(e), "coins": []}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    log.info(f"WS client connected ({len(ws_clients)} total)")
    try:
        # Send initial state on connect
        price_map = build_price_map(state["market_data"])
        m = portfolio.compute_metrics(price_map)
        await ws.send_text(json.dumps({
            "type": "init",
            "data": {
                "mode":      portfolio.mode,
                "auto_trade":state["auto_trade"],
                "paused":    state["paused"],
                "equity":    round(m.equity, 2),
                "breadth":   state["last_breadth"].as_dict_simple() if state["last_breadth"] else None,
            }
        }))

        # Keep alive — echo pings
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                else:
                    cmd = json.loads(data)
                    await handle_ws_command(ws, cmd)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "heartbeat",
                                                "ts": datetime.now(timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)
        log.info(f"WS client disconnected ({len(ws_clients)} total)")


async def handle_ws_command(ws: WebSocket, cmd: dict):
    """Handle commands sent over WebSocket from the renderer."""
    action = cmd.get("action")
    if action == "pause":
        state["paused"] = True
        await broadcast("status", {"paused": True})
    elif action == "resume":
        state["paused"] = False
        await broadcast("status", {"paused": False})
    elif action == "toggle_auto":
        state["auto_trade"] = not state["auto_trade"]
        await broadcast("status", {"auto_trade": state["auto_trade"]})
    elif action == "scan":
        signals = run_signal_scan()
        for sig in signals:
            await broadcast("signal", sig)


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    log.info(f"Starting NEXUS backend on {args.host}:{args.port}")
    log.info("Data directory: " + str(DATA_DIR))

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
        access_log=False,
    )
