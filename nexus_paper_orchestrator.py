"""
NEXUS Paper Orchestrator  (paper-only — no live execution)
==========================================================
Wires the three real subsystems into one loop:

    Signal Engine (scores)  →  Portfolio (paper trades)  →  Progression (trust/phase)

Design notes
------------
* PAPER-ONLY. There is no import of any execution/CCXT/XRPL module here. The
  orchestrator literally cannot place a real order — it only calls
  PortfolioManager.open_position(..., exchange="paper").

* The scorer is INJECTED (scorer_fn). In production you pass
  NEXUSScorer.score_asset (needs aiohttp/pandas-ta on your Mac). In tests you
  pass a deterministic stub. This keeps the wiring verifiable offline without
  pretending any market data is real.

* Deep-intelligence APIs (Glassnode, Santiment, CoinGlass, etc.) live INSIDE the
  scorer and are optional — every keyed client falls back to a neutral 50 when
  its key is absent, so the system runs keyless and lights up as you add keys.

* When the trader reaches PRO_TRADER, its POSITION/CONVICTION calls are flagged
  `mirror_worthy=True` and carry full reasoning so you can choose to follow them
  manually in your own wallet, outside the app.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Any
import logging

from nexus_portfolio import PortfolioManager, Database, PAPER
from nexus_progression import TraderProgression, Metrics, Phase

log = logging.getLogger("nexus.orchestrator")

# Actions from the signal engine that justify opening a paper position
ENTRY_ACTIONS = {"POSITION", "CONVICTION"}


@dataclass
class SurfacedCall:
    """A signal surfaced to the human. When the trader is PRO, mirror_worthy=True."""
    symbol: str
    action: str
    conviction: float
    entry_zone: Optional[tuple]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    patterns: list
    mirror_worthy: bool
    reasoning: str


@dataclass
class CycleResult:
    timestamp: str
    phase: str
    trust: int
    equity: float
    opened: list = field(default_factory=list)   # symbols opened this cycle
    closed: list = field(default_factory=list)    # closed trades this cycle
    surfaced: list = field(default_factory=list)  # SurfacedCall list
    metrics: dict = field(default_factory=dict)


class PaperOrchestrator:
    def __init__(
        self,
        db_path: str = "nexus_paper.db",
        scorer_fn: Optional[Callable] = None,
        progression: Optional[TraderProgression] = None,
    ):
        self.db = Database(Path(db_path))
        self.portfolio = PortfolioManager(self.db, mode=PAPER)   # hard-locked to paper
        self.progression = progression or TraderProgression()
        self.scorer_fn = scorer_fn   # async callable(symbol, df, **kw) -> SignalScore

    # ── Metric bridge: PerformanceMetrics → progression.Metrics ─────────────────
    @staticmethod
    def _to_progression_metrics(pm) -> Metrics:
        return Metrics(
            sharpe_ratio=pm.sharpe_ratio,
            win_rate=pm.win_rate,
            max_drawdown=pm.max_drawdown,
            n_trades=pm.n_trades,
            days_active=pm.days_active,
        )

    def _reasoning(self, sig) -> str:
        detected = [p.name for p in getattr(sig, "patterns", []) if getattr(p, "detected", False)]
        bits = [f"conviction {sig.conviction_score:.0f}", f"action {sig.action}"]
        if sig.entry_zone:
            bits.append(f"entry {sig.entry_zone[0]:.4g}-{sig.entry_zone[1]:.4g}")
        if sig.stop_loss:
            bits.append(f"stop {sig.stop_loss:.4g}")
        if sig.take_profit:
            bits.append(f"target {sig.take_profit:.4g}")
        if sig.risk_reward:
            bits.append(f"R/R {sig.risk_reward:.2f}")
        if detected:
            bits.append("patterns: " + ", ".join(detected))
        return " | ".join(bits)

    async def run_cycle(
        self,
        price_data: dict,          # symbol -> OHLCV DataFrame (for the scorer)
        price_map: dict,           # symbol -> latest price (for fills/metrics)
        meta: Optional[dict] = None,  # symbol -> {name, category, ...}
        **scorer_kwargs,
    ) -> CycleResult:
        if self.scorer_fn is None:
            raise RuntimeError("No scorer_fn injected. Pass NEXUSScorer.score_asset on your Mac.")
        meta = meta or {}

        is_pro = self.progression.state.phase == Phase.PRO_TRADER
        opened, surfaced = [], []

        # 1) Score every asset and act on entries
        for symbol, df in price_data.items():
            sig = await self.scorer_fn(symbol, df, **scorer_kwargs)
            price = price_map.get(symbol)
            entered = False

            if sig.action in ENTRY_ACTIONS and price:
                pos = self.portfolio.open_position(
                    symbol=symbol,
                    name=meta.get(symbol, {}).get("name", symbol),
                    category=meta.get(symbol, {}).get("category", "Unknown"),
                    price=price,
                    conviction=sig.conviction_score,
                    pattern=(sig.patterns[0].name if getattr(sig, "patterns", []) else "signal"),
                    exchange="paper",
                )
                if pos:
                    opened.append(symbol)
                    entered = True
                # Surface the call (mirror-worthy only once the trader is PRO)
                surfaced.append(SurfacedCall(
                    symbol=symbol, action=sig.action, conviction=sig.conviction_score,
                    entry_zone=getattr(sig, "entry_zone", None),
                    stop_loss=getattr(sig, "stop_loss", None),
                    take_profit=getattr(sig, "take_profit", None),
                    risk_reward=getattr(sig, "risk_reward", None),
                    patterns=[p.name for p in getattr(sig, "patterns", []) if getattr(p, "detected", False)],
                    mirror_worthy=is_pro,
                    reasoning=self._reasoning(sig),
                ))

            # Log every signal we evaluated (entered flag now known)
            self.portfolio.log_signal(
                symbol,
                meta.get(symbol, {}).get("name", symbol),
                meta.get(symbol, {}).get("category", "Unknown"),
                sig.conviction_score,
                price if price else 0.0,
                (sig.patterns[0].name if getattr(sig, "patterns", []) else "signal"),
                entered,
            )

        # 2) Close any positions that hit SL/TP on current prices
        closed_trades = self.portfolio.check_positions(price_map)
        self.portfolio.snapshot_equity(price_map)

        # 3) Recompute metrics → drive the progression state machine
        pm = self.portfolio.compute_metrics(price_map)
        prog_state = self.progression.update(self._to_progression_metrics(pm))

        return CycleResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=prog_state.phase.value,
            trust=prog_state.trust,
            equity=pm.equity,
            opened=opened,
            closed=[{"symbol": t.symbol, "pnl": t.pnl, "reason": t.exit_reason}
                    for t in closed_trades] if closed_trades else [],
            surfaced=[s.__dict__ for s in surfaced],
            metrics={
                "sharpe": pm.sharpe_ratio, "win_rate": pm.win_rate,
                "max_dd": pm.max_drawdown, "n_trades": pm.n_trades,
                "n_open": pm.n_open, "days": pm.days_active,
                "total_return": pm.total_return,
            },
        )

    def status(self) -> dict:
        pm = self.portfolio.compute_metrics()
        st = self.progression.state
        return {
            "phase": st.phase.value,
            "trust": st.trust,
            "peak_trust": st.peak_trust,
            "demotions": st.demotions,
            "gate_report": self.progression.gate_report(self._to_progression_metrics(pm)),
            "transitions": st.transitions,
        }


# ── Integration self-test (offline, no network) ─────────────────────────────────
# Uses the REAL portfolio + REAL progression engine, driven by a deterministic
# stub scorer and a deterministic price path. This verifies the WIRING — signal
# → paper trade → SL/TP close → metrics → phase — not market performance.
if __name__ == "__main__":
    import asyncio, tempfile, os
    from dataclasses import dataclass as _dc

    logging.basicConfig(level=logging.WARNING)

    @_dc
    class _StubPattern:
        name: str
        detected: bool = True

    @_dc
    class _StubSignal:
        symbol: str
        conviction_score: float
        action: str
        entry_zone: tuple = None
        stop_loss: float = None
        take_profit: float = None
        risk_reward: float = None
        patterns: list = None

    # Deterministic scorer: BTC always a CONVICTION buy, ETH a WATCH (no entry)
    async def stub_scorer(symbol, df, **kw):
        if symbol == "BTC":
            return _StubSignal("BTC", 88.0, "CONVICTION",
                               entry_zone=(60000, 61000), stop_loss=57600,
                               take_profit=66000, risk_reward=2.5,
                               patterns=[_StubPattern("OTE 0.702")])
        return _StubSignal("ETH", 55.0, "WATCH", patterns=[])

    async def main():
        tmp = tempfile.mktemp(suffix=".db")
        orch = PaperOrchestrator(db_path=tmp, scorer_fn=stub_scorer)
        orch.portfolio.reset_paper()

        meta = {"BTC": {"name": "Bitcoin", "category": "L1 Foundation"},
                "ETH": {"name": "Ethereum", "category": "L1 Smart Contract"}}

        print("=" * 60)
        print("PAPER ORCHESTRATOR — integration self-test")
        print("=" * 60)

        # Cycle 1: BTC at 60500 → should open a paper position
        r1 = await orch.run_cycle(
            price_data={"BTC": None, "ETH": None},
            price_map={"BTC": 60500, "ETH": 3000},
            meta=meta,
        )
        print(f"\nCycle 1 @ BTC 60500")
        print(f"  phase={r1.phase} trust={r1.trust} equity=${r1.equity:,.0f}")
        print(f"  opened={r1.opened}  surfaced={[s['symbol']+'/'+('MIRROR' if s['mirror_worthy'] else 'unproven') for s in r1.surfaced]}")

        # Cycle 2: BTC rips to 67000 → above internal TP (entry*1.10 = 66550) → must close in profit
        r2 = await orch.run_cycle(
            price_data={"BTC": None, "ETH": None},
            price_map={"BTC": 67000, "ETH": 3000},
            meta=meta,
        )
        print(f"\nCycle 2 @ BTC 67000 (above internal TP 66550)")
        print(f"  closed={r2.closed}")
        print(f"  metrics: trades={r2.metrics['n_trades']} win_rate={r2.metrics['win_rate']:.0f}% return={r2.metrics['total_return']:+.2f}%")
        assert len(r2.closed) == 1, "TP close path was not exercised"
        assert r2.metrics['n_trades'] == 1, "closed trade not recorded in metrics"

        print(f"\nFinal status:")
        s = orch.status()
        print(f"  phase={s['phase']} trust={s['trust']}")
        print("  (Apprentice expected — only 1 trade, <30 days; correct.)")
        print("=" * 60)
        print("WIRING VERIFIED: signal → paper open → TP close → metrics → progression")
        print("=" * 60)
        os.remove(tmp)

    asyncio.run(main())
