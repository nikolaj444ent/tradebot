"""The paper-trading daemon CLI.

    # live quotes (public endpoints, no keys), runs until stopped or halted:
    python -m tradebot.run.paper --strategy momentum

    # no network: drive the daemon over synthetic data, end to end:
    python -m tradebot.run.paper --replay-synthetic trending --strategy momentum

    # replay a pinned snapshot through the daemon (plumbing rehearsal):
    python -m tradebot.run.paper --replay-snapshot a1b2c3 --strategy momentum

A halted run refuses to start until a human clears it:

    python -m tradebot.run.paper --strategy momentum --clear-halt

Exit codes: 0 clean stop, 2 halted by the risk gate, 3 state reconciliation
failed. Docker restarts the process, but a halted or unreconciled run exits
again immediately — restart policy cannot quietly resume it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..config import Config
from ..data import SnapshotStore, synthetic_panel
from ..paper import LiveCcxtFeed, PaperHalted, PaperState, PaperTrader, ReconciliationError, ReplayFeed
from ._common import STRATEGIES, load_cfg


def heartbeat_check(cfg: Config, max_age: int | None) -> int:
    """Dead-man's switch for `docker compose` healthchecks.

    Exit 0 while every paper run under the state dir has a fresh heartbeat
    (halted runs excepted: they are loudly stopped, not silently dead).
    """
    max_age = max_age or cfg.paper.heartbeat_max_age_seconds
    base = Path(cfg.paper.state_dir) / "paper"
    if not base.exists():
        print(f"no paper runs under {base}")
        return 1
    now = time.time()
    bad = 0
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        hb = run_dir / "heartbeat"
        if (run_dir / "HALTED").exists():
            print(f"{run_dir.name}: HALTED (deliberately stopped, not stale)")
            continue
        if not hb.exists():
            print(f"{run_dir.name}: no heartbeat file")
            bad += 1
            continue
        age = now - hb.stat().st_mtime
        if age > max_age:
            print(f"{run_dir.name}: heartbeat is {age:.0f}s old (limit {max_age}s)")
            bad += 1
        else:
            print(f"{run_dir.name}: ok ({age:.0f}s)")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="momentum")
    p.add_argument("--run-name", default=None, help="state is kept per run name")
    replay = p.add_mutually_exclusive_group()
    replay.add_argument(
        "--replay-synthetic",
        choices=["random_walk", "trending", "mean_reverting"],
        help="drive the daemon over generated data: proves plumbing, no network",
    )
    replay.add_argument("--replay-snapshot", help="drive the daemon over a pinned snapshot")
    p.add_argument("--synthetic-bars", type=int, default=2000)
    p.add_argument("--synthetic-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N bars (smoke tests)")
    p.add_argument("--clear-halt", action="store_true", help="human decision: resume a halted run")
    p.add_argument("--heartbeat-check", action="store_true", help="exit 0 if heartbeats are fresh")
    p.add_argument("--max-age", type=int, default=None, help="heartbeat staleness limit, seconds")
    args = p.parse_args()

    cfg = load_cfg(args)
    if args.heartbeat_check:
        sys.exit(heartbeat_check(cfg, args.max_age))

    strategy = STRATEGIES[args.strategy]()
    if args.replay_synthetic:
        run_name = args.run_name or f"replay-{args.replay_synthetic}-{args.strategy}"
        frames = synthetic_panel(
            cfg.data.symbols,
            args.synthetic_bars,
            cfg.data.timeframe,
            seed=args.synthetic_seed,
            kind=args.replay_synthetic,
        )
        feed = ReplayFeed(frames, timeframe=cfg.data.timeframe)
    elif args.replay_snapshot:
        run_name = args.run_name or f"replay-{args.replay_snapshot}-{args.strategy}"
        snap = SnapshotStore(cfg.data.snapshot_dir).load(args.replay_snapshot)
        feed = ReplayFeed(snap.frames, timeframe=cfg.data.timeframe)
    else:
        run_name = args.run_name or args.strategy
        feed = LiveCcxtFeed(
            cfg.data.exchange,
            cfg.data.symbols,
            cfg.data.timeframe,
            grace_seconds=cfg.paper.grace_seconds,
        )

    state = PaperState.at(cfg.paper.state_dir, run_name)
    if args.clear_halt:
        if state.clear_halt():
            print(f"halt cleared for run '{run_name}'. The gate's memory (high-water "
                  f"mark, day-open equity) is kept: if the book is still past a "
                  f"limit, it will halt again on the next bar.")
        else:
            print(f"run '{run_name}' was not halted; nothing to clear.")

    trader = PaperTrader(
        feed=feed,
        strategy=strategy,
        costs=cfg.costs,
        limits=cfg.risk,
        state=state,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
    )

    mode = "replay" if isinstance(feed, ReplayFeed) else "live"
    print(f"paper run '{run_name}': {strategy.describe()}  mode={mode}")
    print(f"state: {state.root}")
    try:
        trader.run_forever(poll_seconds=cfg.paper.poll_seconds, max_steps=args.max_steps)
    except PaperHalted as exc:
        print(f"\n*** HALTED: {exc} ***", file=sys.stderr)
        sys.exit(2)
    except ReconciliationError as exc:
        print(f"\n*** RECONCILIATION FAILED: {exc} ***", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\nstopped by hand; state is persisted and the run can resume.")

    eq = trader.equity_series
    final = float(eq.iloc[-1]) if len(eq) else trader.pf.cash
    print(f"\nbars processed: {len(eq)}   fills: {len(trader.fills)}   "
          f"fees: {trader.pf.fees_paid:.2f}   final equity: {final:.2f}")
    if isinstance(feed, ReplayFeed):
        print("(replay results prove plumbing, not profitability — see DESIGN.md)")


if __name__ == "__main__":
    main()
