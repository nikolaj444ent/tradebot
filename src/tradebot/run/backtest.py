"""Run one backtest and print an honest report.

    python -m tradebot.run.backtest --synthetic trending --strategy momentum
    python -m tradebot.run.backtest --snapshot a1b2c3 --strategy momentum --n-trials 12

`--n-trials` is how many variants you have tried in total while developing
this idea. It feeds the deflated Sharpe. Understating it is self-deception,
not optimisation.
"""

from __future__ import annotations

import argparse

from ..costs import CostModel
from ..engine.backtest import run_backtest
from ..risk import RiskLimits
from ..strategy import BuyAndHold
from ..validation.metrics import summarize
from ._common import STRATEGIES, add_data_args, load_cfg, resolve_data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="momentum")
    p.add_argument(
        "--n-trials", type=int, default=1,
        help="total variants tried for this idea; see RESEARCH_LOG.md",
    )
    p.add_argument(
        "--cost-attribution", action="store_true",
        help="also run frictionlessly, to separate strategy from fees",
    )
    args = p.parse_args()

    cfg = load_cfg(args)
    frames, source = resolve_data(args, cfg)
    strategy = STRATEGIES[args.strategy]()

    result = run_backtest(
        frames, strategy,
        costs=cfg.costs, limits=cfg.risk,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
        snapshot_id=source,
    )
    # The benchmark runs UNCONSTRAINED, and that is deliberate. Under the
    # configured limits it is not a benchmark at all: a 0.30 per-symbol cap
    # forces continuous selling of whichever asset is winning, which turned
    # buy-and-hold into 6,754 trades and a capped rebalancing strategy. A
    # reference has to be a reference.
    #
    # The useful consequence: the gap between a constrained strategy and the
    # unconstrained hold now includes the cost of the risk policy itself,
    # which is a real cost and should be visible rather than netted away.
    bench_limits = RiskLimits(
        max_weight_per_symbol=1.0,
        max_gross_exposure=1.0,
        halt_on_breach=False,
        symbol_allowlist=frozenset(cfg.data.symbols),
    )
    bench = run_backtest(
        frames, BuyAndHold(),
        costs=cfg.costs, limits=bench_limits,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
        snapshot_id=source,
    )

    print(f"\n{'=' * 64}")
    print(f"  {strategy.describe()}")
    print(f"  data: {source}  ({result.meta['n_bars']} bars)")
    print(f"  {result.meta['first_bar'][:19]} .. {result.meta['last_bar'][:19]}")
    print(f"{'=' * 64}")

    summary = summarize(result, cfg.bars_per_year, n_trials=args.n_trials)

    # The halt banner goes FIRST. Every metric below is computed on the
    # truncated sample, so burying this under a spectacular-looking table is
    # exactly the kind of dishonest report this harness exists to prevent.
    if result.halted_at is not None:
        print(f"\n  *** RUN HALTED at {result.halted_at} ***")
        print(f"      {result.halt_reason}")
        print(f"      Traded {summary['effective_bars']} of {result.meta['n_bars']} "
              f"bars ({summary['sample_years']:.2f} years). Everything below "
              f"describes that stub, not the full sample.")
        print()

    print(summary.table())
    if summary["sample_years"] < 1.0:
        print(f"\n  NOTE: cagr is extrapolated from {summary['sample_years']:.2f} "
              f"years and is not a meaningful annual figure.")

    print(f"\n  {'-- benchmark: buy & hold, unconstrained ' + '-' * 23}")
    bs = summarize(bench, cfg.bars_per_year)
    for key in ("final_equity", "cagr", "sharpe_annual", "max_drawdown"):
        print(f"  {key:<28} {bs[key]:>12.4f}")
    print(f"  {'n_trades':<28} {bs['n_trades']:>12}")
    excess = summary["cagr"] - bs["cagr"]
    print(f"  {'excess cagr vs hold':<28} {excess:>12.4f}")

    if args.cost_attribution:
        free = CostModel(
            taker_fee_bps=0.0, half_spread_bps=0.0,
            impact_bps_at_full_bar=0.0, min_notional=0.0,
        )
        frictionless = run_backtest(
            frames, STRATEGIES[args.strategy](),
            costs=free, limits=cfg.risk,
            initial_equity=cfg.backtest.initial_equity,
            cash_buffer=cfg.backtest.cash_buffer,
        )
        gap = frictionless.meta["final_equity"] - result.meta["final_equity"]
        print(f"\n  {'-- cost attribution ' + '-' * 42}")
        print(f"  {'frictionless final equity':<28} {frictionless.meta['final_equity']:>12.2f}")
        print(f"  {'cost of trading':<28} {gap:>12.2f}")
        print("  (the frictionless run also drops min_notional, so it takes a")
        print("   slightly different trade path: a diagnostic, not a decomposition)")

    # Warnings last, where they are read.
    notes = []
    if summary["sharpe_t_stat"] < 2.0:
        notes.append(
            f"Sharpe {summary['sharpe_annual']:.2f} +/- {summary['sharpe_std_error']:.2f} "
            f"(t={summary['sharpe_t_stat']:.1f}) is NOT distinguishable from zero. "
            f"{summary['sample_years']:.2f} years of data is too short to tell."
        )
    if summary["clamp_rate"] > 0.1:
        notes.append(
            f"risk gate clamped {summary['clamp_rate']:.0%} of decisions "
            f"({summary['violations_per_decision']:.2f} violations per decision): "
            f"this result measures the gate, not the strategy. Either widen the "
            f"limits or size the strategy to fit them."
        )
    if summary["sample_years"] < 1.0:
        notes.append(
            f"only {summary['sample_years']:.2f} years of data: cagr is an "
            f"extrapolation, not a result."
        )
    if result.halted_at is not None:
        notes.append(f"HALTED at {result.halted_at}: {result.halt_reason}")
    if args.n_trials == 1:
        notes.append(
            "n_trials=1 assumed. If you have tried more variants, pass --n-trials "
            "so the deflated Sharpe is honest."
        )
    if excess <= 0:
        notes.append("does not beat buy-and-hold after costs.")
    if notes:
        print(f"\n  {'-- read this ' + '-' * 49}")
        for n in notes:
            print(f"  ! {n}")
    print()


if __name__ == "__main__":
    main()
