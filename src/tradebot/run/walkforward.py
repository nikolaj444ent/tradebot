"""Walk-forward validation across a parameter grid.

    python -m tradebot.run.walkforward --synthetic trending --strategy momentum

Reports stitched out-of-sample performance, per-fold parameter churn, the
deflated Sharpe against the real trial count, and PBO across the candidate
family.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..validation.metrics import sharpe, sharpe_standard_error, max_drawdown
from ..validation.metrics import deflated_sharpe_ratio
from ..validation.pbo import probability_of_backtest_overfitting
from ..validation.walkforward import walk_forward
from ._common import PARAM_GRIDS, STRATEGIES, add_data_args, load_cfg, resolve_data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="momentum")
    p.add_argument("--pbo-blocks", type=int, default=10)
    args = p.parse_args()

    cfg = load_cfg(args)
    frames, source = resolve_data(args, cfg)
    grid = PARAM_GRIDS[args.strategy]
    if not grid:
        raise SystemExit(
            f"{args.strategy} has no tunable parameters, so there is nothing to "
            f"walk forward. Use `python -m tradebot.run.backtest` instead."
        )

    print(f"\nWalk-forward: {args.strategy} on {source}")
    print(f"  grid: { {k: list(v) for k, v in grid.items()} }")
    print(f"  train={cfg.walkforward.train_bars} test={cfg.walkforward.test_bars} "
          f"step={cfg.walkforward.step_bars} bars")

    wf = walk_forward(frames, STRATEGIES[args.strategy], grid, cfg)

    print(f"\n{'=' * 64}\n  PER FOLD\n{'=' * 64}")
    print(wf.per_fold.to_string(index=False))

    oos_sr = sharpe(wf.oos_returns, cfg.bars_per_year)
    se = sharpe_standard_error(len(wf.oos_returns), cfg.bars_per_year)
    per_bar = 0.0
    r = wf.oos_returns
    if len(r) > 1 and float(r.std(ddof=1)) > 0:
        per_bar = float(r.mean() / r.std(ddof=1))
    dsr = deflated_sharpe_ratio(
        per_bar, len(r), float(r.skew()), float(r.kurt()) + 3.0,
        wf.n_trials, wf.trial_sr_variance,
    )

    print(f"\n{'=' * 64}\n  STITCHED OUT-OF-SAMPLE\n{'=' * 64}")
    print(f"  {'oos bars':<28} {len(r):>12}")
    print(f"  {'oos years':<28} {len(r) / cfg.bars_per_year:>12.2f}")
    print(f"  {'oos sharpe (annual)':<28} {oos_sr:>12.2f}")
    print(f"  {'standard error':<28} {se:>12.2f}")
    print(f"  {'t-stat':<28} {abs(oos_sr) / se if se else 0:>12.2f}")
    print(f"  {'oos max drawdown':<28} {max_drawdown(wf.oos_equity):>12.2%}")
    print(f"  {'trials run':<28} {wf.n_trials:>12}")
    print(f"  {'deflated sharpe':<28} {dsr:>12.4f}")

    print(f"\n  {'-- parameter stability (CV across folds) ' + '-' * 21}")
    stability = wf.param_stability
    if stability:
        for k, v in stability.items():
            flag = "  <-- unstable" if v > 0.5 else ""
            print(f"  {k:<28} {v:>12.3f}{flag}")
    else:
        print("  (single fold or non-numeric parameters)")

    if wf.all_trial_returns.shape[1] >= 2:
        trials = wf.all_trial_returns.dropna(how="all")
        try:
            pbo = probability_of_backtest_overfitting(trials, n_blocks=args.pbo_blocks)
            print(f"\n  {'-- probability of backtest overfitting ' + '-' * 23}")
            print(f"  {'pbo':<28} {pbo.pbo:>12.3f}")
            print(f"  {'candidates':<28} {pbo.n_candidates:>12}")
            print(f"  {'combinations':<28} {pbo.n_combinations:>12}")
            print(f"  verdict: {pbo.verdict()}")
        except ValueError as exc:
            print(f"\n  pbo unavailable: {exc}")

    notes = []
    if abs(oos_sr) / se < 2.0 if se else True:
        notes.append("out-of-sample Sharpe is not distinguishable from zero.")
    if dsr < 0.5:
        notes.append(
            f"deflated sharpe {dsr:.2f}: after accounting for {wf.n_trials} trials, "
            f"this is more likely selection luck than edge."
        )
    if any(v > 0.5 for v in stability.values()):
        notes.append("parameters jump between folds: the search is fitting noise.")
    if notes:
        print(f"\n  {'-- read this ' + '-' * 49}")
        for n in notes:
            print(f"  ! {n}")
    print()


if __name__ == "__main__":
    main()
