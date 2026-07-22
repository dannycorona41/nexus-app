"""
NEXUS Trader Progression Engine  (paper-only)
=============================================
Turns raw paper-trading performance into a trust-based career progression:

    APPRENTICE  →  GRADUATION REVIEW  →  PRO TRADER
                         ↑___________________|   (demotion if trust decays)

The AI never touches real money. As it proves itself on paper it climbs phases.
Once it reaches PRO TRADER, its calls are surfaced as high-conviction signals
the human can choose to mirror manually — outside the app.

Pure stdlib. No network. Deterministic. Unit-testable offline.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


# ── Graduation gate (must hold to become and stay PRO) ──────────────────────────
GRAD_SHARPE       = 1.2     # risk-adjusted return
GRAD_WINRATE      = 45.0    # %
GRAD_MAX_DD       = 20.0    # % (lower is better — hard ceiling)
GRAD_MIN_TRADES   = 20      # closed trades
GRAD_MIN_DAYS     = 30      # calendar days of track record

# ── Trust hysteresis (prevents flapping between phases) ─────────────────────────
PROMOTE_TRUST     = 70      # need >= this AND all gates met to reach PRO
DEMOTE_TRUST      = 60      # drop below this (or break a hard gate) → demoted

# ── Trust score weights (performance-weighted, sums to 1.0) ─────────────────────
W_SHARPE   = 0.30
W_WINRATE  = 0.25
W_DRAWDOWN = 0.20
W_TRADES   = 0.15
W_DAYS     = 0.10


class Phase(str, Enum):
    APPRENTICE        = "APPRENTICE"
    GRADUATION_REVIEW = "GRADUATION_REVIEW"
    PRO_TRADER        = "PRO_TRADER"


@dataclass
class Metrics:
    """The performance inputs — produced by the real portfolio module."""
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0          # percent 0-100
    max_drawdown: float = 0.0      # percent 0-100
    n_trades: int = 0
    days_active: float = 0.0


@dataclass
class ProgressionState:
    phase: Phase = Phase.APPRENTICE
    trust: int = 0
    peak_trust: int = 0
    promoted_at_day: Optional[float] = None
    demotions: int = 0
    transitions: list = field(default_factory=list)  # human-readable log

    def to_json(self) -> str:
        d = asdict(self)
        d["phase"] = self.phase.value
        return json.dumps(d, indent=2)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_trust(m: Metrics) -> int:
    """
    0-100 trust score from real performance. Each gate contributes its
    progress ratio (capped at 1.0). Drawdown is inverted — less is better,
    and breaching the ceiling zeroes that component.
    """
    sharpe_p   = _clamp01(m.sharpe_ratio / GRAD_SHARPE) if GRAD_SHARPE else 0.0
    winrate_p  = _clamp01(m.win_rate / GRAD_WINRATE) if GRAD_WINRATE else 0.0
    drawdown_p = _clamp01((GRAD_MAX_DD - m.max_drawdown) / GRAD_MAX_DD) if m.max_drawdown < GRAD_MAX_DD else 0.0
    trades_p   = _clamp01(m.n_trades / GRAD_MIN_TRADES) if GRAD_MIN_TRADES else 0.0
    days_p     = _clamp01(m.days_active / GRAD_MIN_DAYS) if GRAD_MIN_DAYS else 0.0

    score = (
        sharpe_p   * W_SHARPE +
        winrate_p  * W_WINRATE +
        drawdown_p * W_DRAWDOWN +
        trades_p   * W_TRADES +
        days_p     * W_DAYS
    )
    return round(score * 100)


def all_gates_met(m: Metrics) -> bool:
    return (
        m.sharpe_ratio >= GRAD_SHARPE and
        m.win_rate     >= GRAD_WINRATE and
        m.max_drawdown <  GRAD_MAX_DD and
        m.n_trades     >= GRAD_MIN_TRADES and
        m.days_active  >= GRAD_MIN_DAYS
    )


def is_eligible(m: Metrics) -> bool:
    """Minimum track record before graduation can even be reviewed."""
    return m.days_active >= GRAD_MIN_DAYS and m.n_trades >= GRAD_MIN_TRADES


class TraderProgression:
    """
    Drives the phase state machine. Feed it fresh Metrics each cycle via
    update(); it returns the current state. Persist state.to_json() to disk
    so progression survives restarts.
    """

    def __init__(self, state: Optional[ProgressionState] = None):
        self.state = state or ProgressionState()

    def _log(self, day: float, msg: str):
        self.state.transitions.append({"day": round(day, 1), "event": msg})

    def update(self, m: Metrics) -> ProgressionState:
        s = self.state
        s.trust = compute_trust(m)
        s.peak_trust = max(s.peak_trust, s.trust)
        gates = all_gates_met(m)

        if s.phase == Phase.APPRENTICE:
            if is_eligible(m):
                s.phase = Phase.GRADUATION_REVIEW
                self._log(m.days_active, "Reached eligibility — entering Graduation Review")

        if s.phase == Phase.GRADUATION_REVIEW:
            if gates and s.trust >= PROMOTE_TRUST:
                s.phase = Phase.PRO_TRADER
                s.promoted_at_day = m.days_active
                self._log(m.days_active, f"PROMOTED to Pro Trader (trust {s.trust})")
            elif not is_eligible(m):
                # lost the minimum track record (e.g. data reset)
                s.phase = Phase.APPRENTICE
                self._log(m.days_active, "Dropped below eligibility — back to Apprentice")

        elif s.phase == Phase.PRO_TRADER:
            if not gates or s.trust < DEMOTE_TRUST:
                s.phase = Phase.GRADUATION_REVIEW
                s.demotions += 1
                reason = "broke a hard gate" if not gates else f"trust fell to {s.trust}"
                self._log(m.days_active, f"DEMOTED to Graduation Review ({reason})")

        return s

    # ── Presentation helpers ────────────────────────────────────────────────────
    def status_line(self, m: Metrics) -> str:
        s = self.state
        bar_filled = "█" * (s.trust // 10)
        bar_empty  = "░" * (10 - s.trust // 10)
        label = {
            Phase.APPRENTICE:        "🌱 Apprentice — building track record",
            Phase.GRADUATION_REVIEW: "📊 Graduation Review — proving consistency",
            Phase.PRO_TRADER:        "🎓 PRO TRADER — calls are high-conviction",
        }[s.phase]
        return f"{label}\nTrust [{bar_filled}{bar_empty}] {s.trust}/100"

    def gate_report(self, m: Metrics) -> str:
        def row(name, cur, tgt, ok, suffix=""):
            mark = "✅" if ok else "⏳"
            return f"  {mark} {name:<12} {cur:>7.2f}{suffix} / {tgt}{suffix}"
        lines = [
            row("Sharpe",   m.sharpe_ratio, GRAD_SHARPE,     m.sharpe_ratio >= GRAD_SHARPE),
            row("Win rate", m.win_rate,     GRAD_WINRATE,    m.win_rate >= GRAD_WINRATE, "%"),
            row("Max DD",   m.max_drawdown, GRAD_MAX_DD,     m.max_drawdown < GRAD_MAX_DD, "%"),
            row("Trades",   m.n_trades,     GRAD_MIN_TRADES, m.n_trades >= GRAD_MIN_TRADES),
            row("Days",     m.days_active,  GRAD_MIN_DAYS,   m.days_active >= GRAD_MIN_DAYS),
        ]
        return "\n".join(lines)


# ── Self-test: drives the state machine with a deterministic fixture ────────────
# This is a UNIT TEST of the progression LOGIC. The numbers below are a synthetic
# career arc (NOT market data) used to verify transitions fire correctly:
#   build up → graduate → slip → get demoted → recover → re-promote.
if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS Progression Engine — self-test")
    print("=" * 60)

    # (day, sharpe, win_rate, max_dd, n_trades)
    career = [
        (5,   0.4, 40, 6,   3),    # early apprentice
        (15,  0.8, 43, 9,  11),    # improving, still apprentice (under 30d/20tr)
        (32,  1.0, 46, 12, 21),    # eligible now → review (trust likely < 70)
        (40,  1.3, 52, 11, 26),    # strong → should PROMOTE to PRO
        (55,  1.4, 55, 10, 34),    # holding pro
        (70,  0.9, 41, 22, 41),    # slumps + DD breaches ceiling → DEMOTE
        (88,  1.5, 58, 9,  53),    # recovers → re-PROMOTE
    ]

    prog = TraderProgression()
    for day, sharpe, wr, dd, n in career:
        m = Metrics(sharpe_ratio=sharpe, win_rate=wr, max_drawdown=dd,
                    n_trades=n, days_active=day)
        st = prog.update(m)
        print(f"\nDay {day:>3} | {st.phase.value:<18} | trust {st.trust:>3}")
        print(prog.gate_report(m))

    print("\n" + "=" * 60)
    print("Transition log:")
    for t in prog.state.transitions:
        print(f"  Day {t['day']:>5} — {t['event']}")
    print("=" * 60)
    print(f"Demotions: {prog.state.demotions} | Peak trust: {prog.state.peak_trust}")
