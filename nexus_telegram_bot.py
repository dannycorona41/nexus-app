"""
NEXUS Telegram Bot
===================
Mobile command interface and real-time alerts.

Commands:
  /start    — Welcome message and quick-start guide
  /status   — System health: mode, breadth, auto-trade, paused
  /score BTC — Full conviction score for any asset
  /p or /positions — Open positions with live P&L
  /t or /trades  — Last 10 closed trades
  /m or /metrics — Full performance metrics
  /breadth  — Live breadth snapshot with all components
  /signals  — Last 10 fired signals
  /graduate — Check graduation criteria progress
  /pause    — Pause auto-trading (keep positions)
  /resume   — Resume auto-trading
  /auto on|off — Toggle auto-trade
  /mode     — Show current mode (PAPER/LIVE)
  /equity   — Current equity and return %
  /help     — Full command list

Auto-alerts (sent without prompting):
  ⭐ CONVICTION signal fires (score 85+)
  🎯 Take profit hit
  🛡 Stop loss hit
  ⚡ Breadth thrust or collapse
  📊 Daily digest at 08:00 UTC
  🎓 All graduation criteria met

Install:
  pip install python-telegram-bot

Usage:
  bot = TelegramBot(token, chat_id, portfolio_manager, state)
  await bot.start()
"""

import asyncio
import logging
from datetime import datetime, timezone

log = logging.getLogger("nexus.telegram")

try:
    from telegram import Update, Bot
    from telegram.ext import (Application, CommandHandler, MessageHandler,
                               ContextTypes, filters)
    TG_AVAILABLE = True
except ImportError:
    log.warning("python-telegram-bot not installed. pip install python-telegram-bot")
    TG_AVAILABLE = False


def fmt_price(p):
    if p is None: return "—"
    if p >= 1000: return f"${p:,.2f}"
    if p >= 1:    return f"${p:.4f}"
    return f"${p:.6f}"

def fmt_pct(v):
    if v is None: return "—"
    return f"{v:+.2f}%"

def fmt_money(v):
    if v is None: return "—"
    return f"${v:+,.2f}"


class TelegramBot:
    """
    Full NEXUS Telegram bot.
    Connects to the PortfolioManager and state dict from server.py.
    """

    def __init__(self, token: str, chat_id: str,
                 portfolio=None, state: dict | None = None):
        self.token     = token
        self.chat_id   = str(chat_id)
        self.portfolio = portfolio
        self.state     = state or {}
        self._app      = None
        self._bot      = None

    async def start(self):
        if not TG_AVAILABLE or not self.token:
            log.warning("Telegram bot disabled (missing token or library)")
            return
        self._app = Application.builder().token(self.token).build()
        self._bot = self._app.bot

        # Register commands
        handlers = [
            ("start",    self._cmd_start),
            ("help",     self._cmd_help),
            ("status",   self._cmd_status),
            ("score",    self._cmd_score),
            ("p",        self._cmd_positions),
            ("positions",self._cmd_positions),
            ("t",        self._cmd_trades),
            ("trades",   self._cmd_trades),
            ("m",        self._cmd_metrics),
            ("metrics",  self._cmd_metrics),
            ("breadth",  self._cmd_breadth),
            ("signals",  self._cmd_signals),
            ("graduate", self._cmd_graduate),
            ("pause",    self._cmd_pause),
            ("resume",   self._cmd_resume),
            ("auto",     self._cmd_auto),
            ("mode",     self._cmd_mode),
            ("equity",   self._cmd_equity),
        ]
        for cmd, handler in handlers:
            self._app.add_handler(CommandHandler(cmd, handler))

        log.info("Telegram bot starting...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        log.info("Telegram bot running")

    async def stop(self):
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ── Auth check ────────────────────────────
    def _authorized(self, update: "Update") -> bool:
        return str(update.effective_chat.id) == self.chat_id

    async def _deny(self, update: "Update"):
        await update.message.reply_text("⛔ Unauthorized.")

    # ── Commands ──────────────────────────────

    async def _cmd_start(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        await update.message.reply_text(
            "⬡ *NEXUS — Neural Exchange Unified System*\n\n"
            "Your private AI trading intelligence bot is connected.\n\n"
            "Quick commands:\n"
            "  /status — system health\n"
            "  /positions — open positions\n"
            "  /metrics — performance\n"
            "  /breadth — market breadth\n"
            "  /graduate — graduation progress\n"
            "  /help — full command list\n\n"
            f"Mode: *{self.portfolio.mode if self.portfolio else 'PAPER'}*",
            parse_mode="Markdown"
        )

    async def _cmd_help(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        await update.message.reply_text(
            "📋 *NEXUS Commands*\n\n"
            "*Information*\n"
            "  /status — system + breadth + mode\n"
            "  /equity — portfolio value + return\n"
            "  /breadth — full breadth snapshot\n"
            "  /signals — last 10 signals\n"
            "  /score BTC — score any asset\n\n"
            "*Portfolio*\n"
            "  /p or /positions — open positions\n"
            "  /t or /trades — closed trade history\n"
            "  /m or /metrics — Sharpe, WR, DD, etc.\n"
            "  /graduate — 30-day graduation status\n\n"
            "*Controls*\n"
            "  /pause — pause auto-trading\n"
            "  /resume — resume auto-trading\n"
            "  /auto on|off — toggle auto-trade\n"
            "  /mode — show current trading mode",
            parse_mode="Markdown"
        )

    async def _cmd_status(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        mode  = self.portfolio.mode if self.portfolio else "PAPER"
        auto  = self.state.get("auto_trade", True)
        paused = self.state.get("paused", False)
        snap  = self.state.get("last_breadth")
        nopen = len(self.portfolio.get_open_positions()) if self.portfolio else 0
        breadth_str = (f"\n📊 Breadth: {snap.breadth_score:.0f}/100 [{snap.breadth_state}] "
                       f"×{snap.conviction_multiplier}" if snap else "")
        await update.message.reply_text(
            f"*NEXUS Status*\n\n"
            f"Mode: {'🔴 LIVE' if mode == 'LIVE' else '🧪 PAPER'}\n"
            f"Auto-trade: {'✅ ON' if auto else '❌ OFF'}\n"
            f"{'⏸ PAUSED' if paused else '▶️ Running'}\n"
            f"Open positions: {nopen}"
            f"{breadth_str}",
            parse_mode="Markdown"
        )

    async def _cmd_score(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        args = ctx.args
        if not args:
            return await update.message.reply_text("Usage: /score BTC")
        symbol = args[0].upper()
        # Quick local score
        from nexus_asset_universe import get as get_asset
        asset = get_asset(symbol)
        if not asset:
            return await update.message.reply_text(f"Asset '{symbol}' not in NEXUS universe")
        md    = self.state.get("market_data", {})
        cg    = md.get(asset.coingecko_id, {})
        price = cg.get("current_price")
        c24   = cg.get("price_change_percentage_24h") or 0
        # Import quick score from server if available
        try:
            from server import quick_score
            score = quick_score(asset, cg)
        except ImportError:
            score = 50.0
        action = ("⭐ CONVICTION" if score >= 85 else "🟢 POSITION" if score >= 75
                  else "🟡 RESEARCH" if score >= 60 else "🔵 WATCH" if score >= 45 else "⚪ SKIP")
        mult  = self.state.get("last_breadth").conviction_multiplier if self.state.get("last_breadth") else 1.0
        adj   = min(100, score * mult)
        await update.message.reply_text(
            f"*{symbol}* — {asset.name}\n"
            f"Category: {asset.category.value}\n\n"
            f"Score: *{score:.1f}* × {mult:.2f} breadth = *{adj:.1f}*\n"
            f"Action: {action}\n"
            f"Price: {fmt_price(price)}\n"
            f"24h: {fmt_pct(c24)}",
            parse_mode="Markdown"
        )

    async def _cmd_positions(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        positions = self.portfolio.get_open_positions()
        if not positions:
            return await update.message.reply_text("No open positions")

        price_map = {}
        md = self.state.get("market_data", {})
        from nexus_asset_universe import get as get_asset
        for pos in positions:
            asset = get_asset(pos.symbol)
            if asset:
                price_map[pos.symbol] = md.get(asset.coingecko_id, {}).get("current_price", pos.entry_price)

        lines = [f"*Open Positions ({len(positions)})*\n"]
        for pos in positions:
            cur = price_map.get(pos.symbol, pos.entry_price)
            pos.update_price(cur)
            pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
            lines.append(
                f"{pnl_emoji} *{pos.symbol}* | Score: {pos.conviction:.0f}\n"
                f"   Entry: {fmt_price(pos.entry_price)} → Now: {fmt_price(cur)}\n"
                f"   P&L: {fmt_money(pos.unrealized_pnl)} ({fmt_pct(pos.unrealized_pct)})\n"
                f"   SL: {fmt_price(pos.stop_loss)} | TP: {fmt_price(pos.take_profit)}\n"
                f"   {pos.pattern}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_trades(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        trades = self.portfolio.get_trades(limit=10)
        if not trades:
            return await update.message.reply_text("No closed trades yet")
        lines = [f"*Last {len(trades)} Trades*\n"]
        for t in trades:
            emoji = "🎯" if t.exit_reason == "TAKE_PROFIT" else ("🛡" if t.exit_reason == "STOP_LOSS" else "🔄")
            pnl_e = "🟢" if t.pnl >= 0 else "🔴"
            lines.append(
                f"{emoji} {pnl_e} *{t.symbol}* {fmt_pct(t.pnl_pct)}\n"
                f"   {fmt_money(t.pnl)} | {t.exit_reason}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_metrics(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        m = self.portfolio.compute_metrics()
        ret_e = "🟢" if m.total_return >= 0 else "🔴"
        await update.message.reply_text(
            f"*Performance Metrics — {m.mode}*\n\n"
            f"{ret_e} Return: *{fmt_pct(m.total_return)}*\n"
            f"💰 Equity: *${m.equity:,.0f}*\n"
            f"📈 Sharpe: *{m.sharpe_ratio:.2f}* {'✅' if m.sharpe_ratio >= 1.2 else '⏳'}\n"
            f"🎯 Win rate: *{m.win_rate:.1f}%* {'✅' if m.win_rate >= 45 else '⏳'}\n"
            f"🛡 Max DD: *{m.max_drawdown:.1f}%* {'✅' if m.max_drawdown < 20 else '❌'}\n"
            f"💹 Profit factor: *{m.profit_factor:.2f}*\n"
            f"📊 Trades: {m.n_trades} closed, {m.n_open} open\n"
            f"⏱ Avg hold: {m.avg_hold_h:.1f}h\n"
            f"📅 Days active: {m.days_active:.1f}",
            parse_mode="Markdown"
        )

    async def _cmd_breadth(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        snap = self.state.get("last_breadth")
        if not snap:
            return await update.message.reply_text("No breadth data yet — waiting for first hourly scan")
        await update.message.reply_text(snap.as_telegram(), parse_mode="Markdown")

    async def _cmd_signals(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        sigs = self.portfolio.get_signals(limit=10)
        if not sigs:
            return await update.message.reply_text("No signals yet")
        lines = [f"*Last {len(sigs)} Signals*\n"]
        for s in sigs:
            e = "⭐" if s["score"] >= 85 else "🟢" if s["score"] >= 75 else "🔵"
            entered = "→ Entered" if s.get("entered") else "→ Skipped"
            lines.append(f"{e} *{s['symbol']}* [{s['score']:.0f}] — {s['pattern']}\n   {entered}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_graduate(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        m = self.portfolio.compute_metrics()
        g = m.graduation_status()
        criteria = [
            ("Days", g["days"], "30 days", g["days"]["current"], 30, "d"),
            ("Sharpe ≥ 1.2", g["sharpe"], "≥ 1.2", g["sharpe"]["current"], 1.2, ""),
            ("Win rate ≥ 45%", g["win_rate"], "≥ 45%", g["win_rate"]["current"], 45, "%"),
            ("Max DD < 20%", g["max_dd"], "< 20%", g["max_dd"]["current"], 20, "%"),
            ("20+ trades", g["trades"], "20 trades", g["trades"]["current"], 20, ""),
        ]
        lines = [f"*Graduation Progress ({m.mode})*\n"]
        for label, crit, req, cur, tgt, unit in criteria:
            check = "✅" if crit["met"] else "⏳"
            val = f"{cur:.1f}{unit}" if isinstance(cur, float) else f"{cur}{unit}"
            lines.append(f"  {check} {label}: {val} (need {req})")
        lines.append("")
        if g["all_met"]:
            lines.append("🎓 *All criteria met! Ready to graduate to live trading.*")
        else:
            met = sum(1 for k, v in g.items() if isinstance(v, dict) and v.get("met"))
            lines.append(f"Progress: {met}/5 criteria met")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_pause(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        self.state["paused"] = True
        await update.message.reply_text("⏸ Auto-trading paused. Existing positions remain open.")

    async def _cmd_resume(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        self.state["paused"] = False
        await update.message.reply_text("▶️ Auto-trading resumed.")

    async def _cmd_auto(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        args = ctx.args
        if args and args[0].lower() == "on":
            self.state["auto_trade"] = True
            await update.message.reply_text("✅ Auto-trade ON — signals will open positions automatically")
        elif args and args[0].lower() == "off":
            self.state["auto_trade"] = False
            await update.message.reply_text("❌ Auto-trade OFF — signals shown but no positions opened")
        else:
            status = "ON" if self.state.get("auto_trade") else "OFF"
            await update.message.reply_text(f"Auto-trade is currently: *{status}*\nUse /auto on or /auto off", parse_mode="Markdown")

    async def _cmd_mode(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        mode = self.portfolio.mode if self.portfolio else "PAPER"
        await update.message.reply_text(
            f"Trading mode: *{'🔴 LIVE' if mode == 'LIVE' else '🧪 PAPER'}*\n\n"
            f"Switch modes via the desktop dashboard Settings panel.\n"
            f"LIVE mode requires passing all 5 graduation criteria first.",
            parse_mode="Markdown"
        )

    async def _cmd_equity(self, update: "Update", ctx: "ContextTypes.DEFAULT_TYPE"):
        if not self._authorized(update): return await self._deny(update)
        if not self.portfolio:
            return await update.message.reply_text("Portfolio not connected")
        m = self.portfolio.compute_metrics()
        e = "🟢" if m.total_return >= 0 else "🔴"
        await update.message.reply_text(
            f"*Portfolio Equity — {m.mode}*\n\n"
            f"{e} Value: *${m.equity:,.2f}*\n"
            f"Started: ${m.start_capital:,.0f}\n"
            f"Return: *{fmt_pct(m.total_return)}*\n"
            f"Cash available: ${self.portfolio.cash:,.2f}\n"
            f"Open positions: {m.n_open}",
            parse_mode="Markdown"
        )

    # ── Send alert (called from server.py) ────

    async def send(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the configured chat ID."""
        if not TG_AVAILABLE or not self.token or not self.chat_id:
            return
        try:
            if self._bot:
                await self._bot.send_message(
                    chat_id=self.chat_id, text=text, parse_mode=parse_mode
                )
            else:
                import aiohttp
                url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                async with aiohttp.ClientSession() as s:
                    await s.post(url, json={"chat_id": self.chat_id, "text": text,
                                             "parse_mode": parse_mode},
                                  timeout=aiohttp.ClientTimeout(total=8))
        except Exception as e:
            log.debug(f"Telegram send error: {e}")
