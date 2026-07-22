"""
NOT THE DEPLOYED ENTRYPOINT.
============================
Procfile and render.yaml both start "server:app" -- this file (app.py) is an
earlier/parallel FastAPI build that Render never runs. It was the actual
source of the "static-looking dashboard" bug: the old index.html was fetching
this file's single endpoint (GET /api/paper/summary), which does not exist on
server.py, so every request 404'd silently and the page just showed its
placeholder markup.

index.html has been rewritten to call server.py's real endpoints instead, so
this file is no longer wired to anything. Left in place in case you want to
recover ideas from it (LiveDataFeed usage, TraderProgression wiring), but it
is dead code as far as the live site is concerned. See NEXUS_STATUS.md.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# Import your existing modules
from nexus_portfolio import Database, PortfolioManager, PAPER
from nexus_keyless_live_data import LiveDataFeed
from nexus_progression import TraderProgression

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nexus.backend")

# Shared state
DB_PATH = Path("data/nexus_paper.db")
DB_PATH.parent.mkdir(exist_ok=True)

db = Database(DB_PATH)
portfolio = PortfolioManager(db, mode=PAPER)
feed = LiveDataFeed()
progression = TraderProgression()

WATCHLIST = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT"]
live_prices = {}
is_running = True
background_task = None

async def live_trading_loop():
    global live_prices, is_running
    log.info("Starting live Paper Trading background engine...")
    while is_running:
        try:
            prices = feed.get_price_map(WATCHLIST)
            if prices:
                live_prices.update(prices)
            price_map_usdt = {f"{k}/USDT": v for k, v in live_prices.items()}
            if price_map_usdt:
                portfolio.check_positions(price_map_usdt)
                portfolio.snapshot_equity(price_map_usdt)
        except Exception as e:
            log.error(f"Error in background live loop: {e}")
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task
    background_task = asyncio.create_task(live_trading_loop())
    yield
    global is_running
    is_running = False
    background_task.cancel()
    db.close()

app = FastAPI(title="NEXUS Neural Exchange API", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/paper/summary")
async def get_paper_summary():
    price_map_usdt = {f"{k}/USDT": v for k, v in live_prices.items()}
    metrics = portfolio.compute_metrics(price_map_usdt)
    open_positions = portfolio.get_open_positions()
    closed_trades = portfolio.get_trades(limit=50)
    
    prog_state = progression.update(
        type('Metrics', (object,), {
            "sharpe_ratio": metrics.sharpe_ratio,
            "win_rate": metrics.win_rate,
            "max_drawdown": metrics.max_drawdown,
            "n_trades": metrics.n_trades,
            "days_active": metrics.days_active,
        })()
    )
    
    return {
        "portfolio_value": metrics.equity,
        "phase": prog_state.phase.value,
        "days_active": metrics.days_active,
        "trust_score": prog_state.trust,
        "closed_trades": [{"symbol": t.symbol, "pnl": t.pnl} for t in closed_trades]
    }