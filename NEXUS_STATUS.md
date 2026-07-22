# NEXUS — Project Status & Architecture Notes

Last audited: 2026-07-16. Read this before asking Claude (or yourself, in six
months) "wait, which file actually does X?"

## ⚠️ ACTION REQUIRED BEFORE THIS DEPLOYS: change a Render dashboard setting

This build will fail again unless you do this manually -- it's not something
fixable by editing repo files:

**Render dashboard → `nexus-app-t2vt` service → Environment tab → change
`PYTHON_VERSION` to `3.12.4` (or delete the variable entirely).**

Your `runtime.txt` and `render.yaml` already say `3.12.4`, but your service's
actual build log showed `Using Python version 3.11.8 via environment
variable PYTHON_VERSION` -- a manually-set dashboard variable is overriding
both files. This is the same underlying issue as the `/opt/render/project/data`
ephemeral-disk problem noted further down: this service was not created from
`render.yaml` as a Blueprint, so dashboard settings win over repo files.
`pandas_ta` (needed for the breadth scan to work at all) only has builds for
Python >=3.12, so this step is required, not optional.

## The bug you asked about

Your Render deploy was serving a static-looking dashboard because
**`index.html` and the live backend were built to two different APIs.**

- `Procfile` and `render.yaml` both start **`server.py`** (`uvicorn server:app`).
  That's the only backend Render ever runs.
- The old `index.html` fetched `http://localhost:8000/api/paper/summary` —
  an endpoint that only exists on **`app.py`**, a separate, never-deployed
  FastAPI build. On Render, that fetch either 404'd or failed outright
  (hardcoded `localhost` from a browser hitting your Render URL), the
  `try/catch` swallowed the error, `data` stayed `null` or the component
  never got real values, and you were left looking at the hardcoded
  placeholder markup ("Static Chart View Loaded").

**Fix applied:** `index.html` was rewritten from scratch to call `server.py`'s
real endpoints using relative paths (same-origin `fetch("/api/...")`), plus a
live WebSocket connection to `/ws` for push updates, with polling as a
fallback. I ran `server.py` locally end-to-end (started uvicorn, curled every
endpoint the dashboard touches, diffed the served HTML against the file) —
every endpoint returns the shape the frontend expects. CoinGecko calls
themselves are blocked in my sandbox's network policy (not on Render), but
the failure path is caught and logged, not fatal, by design.

`app.py` is left in the folder but headed with a docstring marking it dead —
I didn't delete it in case you want to salvage ideas from it (it used
`LiveDataFeed` and `TraderProgression`, which `server.py` doesn't).

## Architecture — what's actually wired into the live site

```
Render (Procfile: uvicorn server:app)
  └─ server.py                     ← the ONLY deployed entrypoint
       ├─ nexus_portfolio.py       (paper trading, positions, metrics, DB)
       ├─ nexus_asset_universe.py  (44-asset registry)
       ├─ nexus_breadth.py         (hourly market-breadth scan via ccxt/kraken)
       ├─ its own quick_score()    (CoinGecko-heuristic scorer, NOT the real engine)
       └─ its own telegram_alert() (inline, doesn't use nexus_telegram_bot.py)
  └─ index.html                    ← served at "/" by server.py, now fixed
```

## Modules that exist in your project but are NOT wired into the live app

These aren't broken — they're just dead code from the deployed site's
perspective. Nothing imports them:

| File | What it's for | Imported by |
|---|---|---|
| `nexus_signal_engine.py` (89KB) | The *real* multi-factor conviction scorer (patterns, on-chain, sentiment) | Nobody. `server.py` uses a much cruder `quick_score()` heuristic instead. |
| `nexus_ccxt_execution.py` | Real order execution via ccxt | Nobody |
| `nexus_paper_orchestrator.py` | Was meant to wire signal engine → portfolio → progression | Nobody |
| `nexus_progression.py` | Trust/phase (Apprentice → Pro Trader) state machine | Only `app.py` (dead) and its own self-test |
| `nexus_telegram_bot.py` | Dedicated Telegram bot module | Nobody — `server.py` has its own inline alert function |
| `nexus_whitepaper_analyzer.py` | Whitepaper analysis | Nobody |
| `nexus_data.py` / `nexus_keyless_live_data.py` | Keyless OHLCV fetchers (Coinbase/Kraken/CoinGecko/Binance) | Only `run_backtest.py` and `app.py` (dead) — `server.py` fetches CoinGecko inline itself |
| `app.py` | Earlier/parallel backend build | Nobody (not the deploy entrypoint) |

**The practical implication:** what's live right now is a simpler system than
what most of your files represent. The 5-factor deep-intelligence scoring,
the trust/phase progression ladder, real ccxt execution, and the paper
orchestrator's wiring all exist as *code* but none of it runs on your actual
site. If you want the real signal engine driving trades instead of
`quick_score()`, that's a real integration task, not a bug fix — worth a
separate conversation with a clear scope (e.g. "wire `nexus_signal_engine`
into `server.py`'s `run_signal_scan()`").

## requirements.txt / deploy notes -- history of getting this right (read this if the build ever fails again)

This took three iterations to get right, and I want you to see the actual
chain of reasoning rather than just the final answer, since dependency
resolution is exactly the kind of thing that silently breaks again later:

1. **First claim (wrong):** I said `pandas_ta` was safe to omit from
   `requirements.txt` since `server.py` still imports without it. True, but I'd
   only checked import, not runtime behavior.
2. **Second attempt (wrong, caused a failed build):** added
   `pandas_ta>=0.3.14b`. Two problems: that exact version string is invalid
   PEP440 syntax, and -- separately -- `pandas-ta==0.3.14b` **was deleted from
   PyPI in September 2025** and no longer exists at all. Pip had nothing to
   install, hence your `ERROR: No matching distribution found`.
3. **Third attempt (correct, actually verified):** the only current `pandas-ta`
   release is `0.4.71b0`, which requires **Python >=3.12**. Your `runtime.txt`
   and `render.yaml` already specify `3.12.4` -- but your live Render service's
   build log showed `Using Python version 3.11.8 via environment variable
   PYTHON_VERSION`, meaning the Render **dashboard** has a manually-set env var
   overriding both files (same root cause as the `/opt/render/project/data`
   disk issue from earlier -- this service isn't reading `render.yaml`).
   **You must change `PYTHON_VERSION` to `3.12.4` in the Render dashboard's
   Environment tab yourself** -- I cannot do this from here.
4. Pinning `pandas_ta==0.4.71b0` alone then failed dependency resolution: it
   requires `numpy>=2.2.6` and `pandas>=2.3.2`, which conflicted with the
   existing `numpy<2.0` / `pandas<2.2` caps. Bumping those unlocked `pandas
   3.0.3` -- a major version jump that could plausibly break other pandas
   usage in this codebase.

**Before shipping this**, I built a completely clean virtualenv, ran the exact
`pip install -r requirements.txt` Render would run, confirmed it resolves,
then actually started `server.py` under that environment and hit
`/health`, `/api/status`, `/api/portfolio`, and `HEAD /` -- all 200. I also
re-ran the breadth-computation unit test (the Kraken USD/USDT fallback +
`pandas_ta` indicators) under this exact stack, not the version installed in
my general sandbox. `nexus_portfolio.py` doesn't use pandas at all, so the
pandas 3.0 jump doesn't touch it. I did not check `nexus_backtester.py` /
`nexus_signal_engine.py` (both use `pandas_ta` too) against pandas 3.0 --
they aren't imported by the live `server.py`, so they're out of scope for
this deploy, but flag it if you ever wire them in.

Final `requirements.txt` versions: `numpy>=2.2.6`, `pandas>=2.3.2`,
`pandas_ta==0.4.71b0`.

## Three real bugs found in your actual Render deploy log (all fixed)

1. **`broadcast()` crashed every single call** (`server.py`). The line
   `ws_clients -= dead` rebinds the name, which makes Python treat
   `ws_clients` as a local variable for the *entire* function -- so the
   earlier read (`if not ws_clients: return`) raised
   `UnboundLocalError: cannot access local variable 'ws_clients'` every time,
   exactly as seen in your log (`Price loop error: ...`, `Breadth loop
   error: ...`). This silently killed every WebSocket push (`price_update`,
   `signal`, `position_opened/closed`, `breadth_update`) -- the dashboard's
   polling fallback covered for it, but nothing was ever pushed live.
   **Fixed:** changed to `ws_clients.difference_update(dead)` (in-place
   mutation, no rebinding). Directly unit-tested the exact failing path (a
   dead socket triggering removal) -- confirmed no crash.
2. **Breadth scan returned zero assets, every scan** -- see the `pandas_ta`
   note above. Also hardened `_fetch_asset()` to try both `/USDT` and `/USD`
   quote variants, since Kraken lists most of these alts against USD, not
   USDT (verified externally: PENDLE trades as `PENDLE/USD` on Kraken, not
   `PENDLE/USDT`) -- your watchlist was written entirely in `/USDT`. The
   swallowed exception is now logged at `warning`, not `debug`, so if a
   symbol still fails you'll actually see why in the Render logs instead of
   silence.
3. **`HEAD /` returned 405** (visible in your log:
   `"HEAD / HTTP/1.1" 405 Method Not Allowed`). Some uptime monitors and load
   balancers probe with HEAD, not GET. Added `@app.head("/")` alongside the
   existing `@app.get("/")`. Low severity -- your deploy went live anyway --
   but free to fix.

## Not fixed, worth your attention

- **Ephemeral data directory.** Your deploy log shows
  `Data directory: /opt/render/project/data`, not the `/var/data` your
  `render.yaml` specifies. That means this Render service almost certainly
  was **not** created from `render.yaml` as a Blueprint -- it's a manually
  configured Web Service that never picked up the `NEXUS_DATA_DIR` env var or
  the persistent disk. Practical effect: your SQLite DB (paper trades,
  equity history, progression data) lives on ephemeral storage and **will be
  wiped on every redeploy.** If you want trading history to survive deploys,
  either recreate the service as a Blueprint from `render.yaml`, or manually
  add a persistent disk + the `NEXUS_DATA_DIR` env var in the Render
  dashboard for this existing service. I didn't touch this myself since it's
  a Render dashboard setting, not a code fix.

## Verified working (tested, not assumed)

- All 15 `.py` files compile clean (`py_compile`).
- `server.py` imports cleanly against a fresh install of `requirements.txt`.
- Started the real server locally and curled every endpoint the dashboard
  calls (`/`, `/health`, `/api/status`, `/api/portfolio`,
  `/api/portfolio/positions`, `/api/portfolio/equity-history`,
  `/api/signals`, `/api/breadth`, `/api/assets`) — all return the shapes
  the frontend expects, including sane empty-state responses (`[]`,
  `{"error": "..."}`) that the new `index.html` explicitly handles.
- Confirmed `/ws` completes a real WebSocket handshake (HTTP 101).
- Diffed the HTML `server.py` serves at `/` against the file in this repo —
  identical, so what you deploy is what you get.

## Not verified (needs your Render environment or real network)

- Live CoinGecko/Kraken data — my sandbox's egress policy blocks those
  domains; Render's won't. The failure path is caught and logged either way.
- Telegram alerts — needs real `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` env vars.
- The hourly breadth scan and 4-hourly equity snapshot loops, since those
  need real elapsed time to fire.
