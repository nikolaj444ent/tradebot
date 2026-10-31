"""Run, inspect or supervise the paper trader.

    python -m tradebot.run.paper --strategy momentum
    python -m tradebot.run.paper --replay --synthetic trending --max-bars 500
    python -m tradebot.run.paper --status
    python -m tradebot.run.paper --watchdog
    python -m tradebot.run.paper --clear-halt

Paper mode needs no API keys: it reads public market data and simulates fills
locally with the same cost model and the same risk gate as a backtest. There is
no live order path in this repository (DESIGN.md section 7).

Exit codes matter here, because something other than a human is reading them:

    0  stopped cleanly (source exhausted, or --max-bars reached)
    1  refused to start, or crashed
    2  HALTED by the risk gate — a position-flattening event a human must see
    3  watchdog found a dead or halted run

`--replay` is a full dry run: real journal, real reconciliation, real risk
gate, data from a pinned snapshot instead of the network, at the speed of the
CPU rather than one bar an hour. Run it before trusting a deployment.
"""

from __future__ import annotations

import argparse
import sys

from ..data.quotes import CcxtQuoteSource, ReplayQuoteSource, TIMEFRAME_SECONDS
from ..engine.paper import (
    PaperStartupRefused,
    result_from_journal,
    run_paper_loop,
    start_paper_trader,
)
from ..state import Journal
from ..validation.metrics import summarize
from ._common import STRATEGIES, add_data_args, load_cfg, resolve_data

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_HALTED = 2
EXIT_WATCHDOG = 3


def _print_status(journal: Journal, cfg, strategy) -> int:
    result = result_from_journal(
        journal, cfg.backtest.initial_equity, strategy=strategy
    )
    print(f"\n{'=' * 64}")
    print(f"  paper run: {journal.run_name}")
    print(f"  {result.meta['strategy']}")
    print(f"  journal: {journal.path}")
    print(f"{'=' * 64}")

    if not result.meta["strategy_matches_journal"]:
        print(
            f"\n  NOTE: this journal was written by {result.meta['strategy']}, "
            f"not by\n        {result.meta['strategy_arg']} as --strategy says. "
            f"The numbers below are\n        that run's. Pass a matching "
            f"--strategy to report its parameters too."
        )

    if not len(result.equity):
        print("\n  No marks recorded yet: the run has not completed a bar.")
        book = journal.read_book()
        if book:
            print(f"  book checkpoint: cash {book['cash']:.2f}, "
                  f"positions {book.get('positions')}")
        return EXIT_OK

    if result.halted_at is not None:
        print(f"\n  *** HALTED at {result.halted_at} ***")
        print(f"      {result.halt_reason}")
        print("      The run is stopped and will refuse to restart until a")
        print("      human clears it (--clear-halt). Everything below describes")
        print("      the sample up to the halt.")
        print()

    summary = summarize(result, cfg.bars_per_year)
    print(summary.table())

    age = journal.heartbeat_age()
    print(f"\n  {'-- liveness ' + '-' * 50}")
    if age is None:
        print(f"  {'heartbeat':<28} {'never written':>12}")
    else:
        print(f"  {'heartbeat age (s)':<28} {age:>12.0f}")

    notes = []
    if summary["sample_years"] < 1.0:
        notes.append(
            f"only {summary['sample_years']:.2f} years of paper data: cagr is an "
            f"extrapolation, not a result."
        )
    if summary["sharpe_t_stat"] < 2.0:
        notes.append(
            f"Sharpe {summary['sharpe_annual']:.2f} +/- "
            f"{summary['sharpe_std_error']:.2f} (t={summary['sharpe_t_stat']:.1f}) "
            f"is not distinguishable from zero on this sample."
        )
    if summary["clamp_rate"] > 0.1:
        notes.append(
            f"risk gate clamped {summary['clamp_rate']:.0%} of decisions: this "
            f"run measures the gate, not the strategy."
        )
    notes.append(
        "paper fills are modelled, not executed: the impact term is a prior, "
        "and divergence from the backtest is the signal you are paying for "
        "here (DESIGN.md section 11)."
    )
    print(f"\n  {'-- read this ' + '-' * 49}")
    for note in notes:
        print(f"  ! {note}")
    print()
    return EXIT_HALTED if result.halted_at is not None else EXIT_OK


def _watchdog(journal: Journal, cfg, max_age_s: float | None) -> int:
    """Answer one question, with an exit code: is this run alive and trading?

    A process cannot detect its own death, which is why this runs as a separate
    command — the docker healthcheck, a cron job, whatever will actually wake a
    human. DESIGN.md section 6: the failure that costs money is the process
    dying silently while holding a position, and nothing inside that process
    can report it.
    """
    halted = journal.halted()
    if halted is not None:
        print(f"HALTED: {halted.get('reason')} (at {halted.get('ts')})")
        positions = halted.get("positions_at_halt") or {}
        print(f"  positions when it halted: {positions or 'flat'}")
        return EXIT_WATCHDOG

    age = journal.heartbeat_age()
    if age is None:
        print(f"NO HEARTBEAT: {journal.heartbeat_path} does not exist or is unreadable.")
        return EXIT_WATCHDOG

    if max_age_s is None:
        # Two bars of grace: one for a slow venue, one for a slow poll.
        max_age_s = 2.5 * TIMEFRAME_SECONDS[cfg.data.timeframe]

    book = journal.read_book() or {}
    positions = book.get("positions") or {}
    if age > max_age_s:
        print(f"STALE: last heartbeat {age:.0f}s ago, limit {max_age_s:.0f}s.")
        print(f"  positions it was holding: {positions or 'flat'}")
        if positions:
            print("  THE PROCESS IS NOT RESPONDING AND HOLDS A POSITION.")
        return EXIT_WATCHDOG

    print(f"alive: heartbeat {age:.0f}s ago, equity {book.get('equity', float('nan')):.2f}, "
          f"positions {positions or 'flat'}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="momentum")
    p.add_argument(
        "--run-name",
        default="paper",
        help="names the state directory under state/, so two strategies can "
        "paper-trade side by side without sharing a journal",
    )
    p.add_argument("--state-dir", default="state")
    p.add_argument(
        "--replay",
        action="store_true",
        help="drive the loop from pinned or synthetic data instead of the "
        "network: a full dry run of the daemon at CPU speed",
    )
    p.add_argument(
        "--max-bars", type=int, help="stop after this many bars (for dry runs)"
    )
    p.add_argument(
        "--status", action="store_true", help="print the run's report and exit"
    )
    p.add_argument(
        "--watchdog",
        action="store_true",
        help="check liveness and exit non-zero if the run is dead or halted",
    )
    p.add_argument(
        "--watchdog-max-age",
        type=float,
        help="seconds of heartbeat silence to tolerate (default: 2.5 bars)",
    )
    p.add_argument(
        "--clear-halt",
        action="store_true",
        help="clear a halt so the run may restart. A deliberate human decision: "
        "read the journal first.",
    )
    args = p.parse_args(argv)

    cfg = load_cfg(args)
    journal = Journal(args.state_dir, args.run_name)
    strategy = STRATEGIES[args.strategy]()

    if args.watchdog:
        return _watchdog(journal, cfg, args.watchdog_max_age)

    if args.clear_halt:
        halted = journal.halted()
        if journal.clear_halt():
            print(f"Cleared halt on '{args.run_name}': {halted.get('reason')}")
            print("The next start will resume from the journal. Positions held at")
            print("the halt were flattened, so it resumes flat.")
            return EXIT_OK
        print(f"'{args.run_name}' is not halted; nothing to clear.")
        return EXIT_OK

    if args.status:
        return _print_status(journal, cfg, strategy)

    # ----------------------------------------------------------------- running

    try:
        trader = start_paper_trader(cfg, strategy, journal, run_id=None)
    except PaperStartupRefused as exc:
        print(f"\nRefusing to start:\n\n{exc}\n", file=sys.stderr)
        return EXIT_REFUSED

    if args.replay:
        frames, source_id = resolve_data(args, cfg)
        warmup = max(int(strategy.warmup), 1)
        if len(frames["close"]) < warmup + 2:
            print(
                f"\n{len(frames['close'])} bars is too few for {strategy.name} "
                f"(warmup {warmup}).\n",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        source = ReplayQuoteSource(
            frames,
            timeframe=cfg.data.timeframe,
            start_index=warmup - 1,
            volume_mode="actual_next",
        )
        print(f"\nREPLAY (no network): {source_id}")
        print("  This is a dry run. Fills are modelled against historical bars,")
        print("  so the result is a backtest wearing the daemon's clothes — it")
        print("  exercises the journal, reconciliation and risk gate, and says")
        print("  nothing new about the strategy.")
    else:
        source = CcxtQuoteSource(
            cfg.data.symbols,
            exchange=cfg.data.exchange,
            timeframe=cfg.data.timeframe,
            history_bars=max(int(strategy.warmup) + 50, 200),
        )
        print(f"\nPAPER TRADING (simulated fills, live {cfg.data.exchange} quotes)")
        print(f"  symbols   {list(cfg.data.symbols)} @ {cfg.data.timeframe}")

    print(f"  strategy  {strategy.describe()}")
    print(f"  run       {args.run_name}  (journal: {journal.path})")
    print(f"  book      {trader.portfolio.cash:.2f} cash, "
          f"{len(trader.portfolio.positions)} position(s) "
          f"{dict(trader.portfolio.positions) or ''}")
    print(f"  stops     daily {cfg.risk.max_daily_loss_pct:.0%}, "
          f"drawdown {cfg.risk.max_drawdown_pct:.0%}, halt_on_breach="
          f"{cfg.risk.halt_on_breach}")
    if trader.last_bar_ts is not None:
        print(f"  resuming  after bar {trader.last_bar_ts}")
    print()

    def report(outcome) -> None:
        if outcome.status == "traded" and outcome.n_fills:
            print(
                f"  {outcome.bar_ts}  equity {outcome.equity:>12.2f}  "
                f"{outcome.n_fills} fill(s)  targets "
                f"{ {k: round(v, 3) for k, v in outcome.targets.items()} }"
            )

    outcome = run_paper_loop(
        trader,
        source,
        heartbeat_interval_s=TIMEFRAME_SECONDS[cfg.data.timeframe],
        max_bars=args.max_bars,
        on_step=report,
    )

    if outcome is not None and outcome.halted:
        print(f"\n*** HALTED at {outcome.bar_ts}: {outcome.halt_reason}")
        print("    The book was flattened and the run is stopped. It will refuse")
        print("    to restart until a human clears it:")
        print(f"      python -m tradebot.run.paper --status --run-name {args.run_name}")
        print(f"      python -m tradebot.run.paper --clear-halt --run-name {args.run_name}")
        print()
        return EXIT_HALTED

    print("\nStopped cleanly.")
    _print_status(journal, cfg, strategy)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
