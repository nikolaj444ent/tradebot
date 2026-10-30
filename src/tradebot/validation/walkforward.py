"""Walk-forward validation.

One train/test split tells you almost nothing: you get a single out-of-sample
number, and if it disappoints you will be tempted to "fix" the strategy and
look again — at which point the test set is contaminated and you have no
out-of-sample left.

Walk-forward instead rolls the window: fit on bars [t, t+train), evaluate on
[t+train, t+train+test), step forward, repeat. The stitched test returns form
one continuous out-of-sample curve from parameters that were always chosen
using only prior data. Per-fold parameter churn is as informative as the
returns: parameters that jump between folds are fitting noise, even when the
stitched curve looks acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..engine.backtest import run_backtest
from ..strategy.base import Strategy
from ..types import OHLCV_FIELDS
from .metrics import sharpe


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    def describe(self, idx: pd.DatetimeIndex) -> str:
        return (
            f"fold {self.index}: train {idx[self.train_start].date()}"
            f"..{idx[self.train_end - 1].date()}  "
            f"test {idx[self.test_start].date()}..{idx[self.test_end - 1].date()}"
        )


def make_folds(
    n_bars: int, train_bars: int, test_bars: int, step_bars: int, min_folds: int = 1
) -> list[Fold]:
    folds: list[Fold] = []
    start = 0
    i = 0
    while start + train_bars + test_bars <= n_bars:
        folds.append(
            Fold(
                index=i,
                train_start=start,
                train_end=start + train_bars,
                test_start=start + train_bars,
                test_end=min(start + train_bars + test_bars, n_bars),
            )
        )
        start += step_bars
        i += 1
    if len(folds) < min_folds:
        raise ValueError(
            f"only {len(folds)} walk-forward fold(s) fit in {n_bars} bars with "
            f"train={train_bars}, test={test_bars}, step={step_bars}; need "
            f"{min_folds}. Fetch more history or shorten the windows — a "
            f"walk-forward with one fold is just a train/test split."
        )
    return folds


@dataclass(slots=True)
class WalkForwardResult:
    oos_returns: pd.Series
    oos_equity: pd.Series
    per_fold: pd.DataFrame
    chosen_params: list[dict]
    n_trials: int
    trial_sr_variance: float
    all_trial_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def param_stability(self) -> dict[str, float]:
        """Coefficient of variation per parameter across folds.

        High values mean the chosen parameters are unstable, which is evidence
        the search is fitting noise regardless of how the curve looks.
        """
        if not self.chosen_params:
            return {}
        out: dict[str, float] = {}
        keys = set().union(*(p.keys() for p in self.chosen_params))
        for k in sorted(keys):
            vals = [p.get(k) for p in self.chosen_params]
            nums = [float(v) for v in vals if isinstance(v, (int, float))]
            if len(nums) > 1:
                mean = float(np.mean(nums))
                out[k] = float(np.std(nums) / abs(mean)) if mean else float("nan")
        return out


def _slice(frames: dict[str, pd.DataFrame], start: int, end: int) -> dict:
    return {f: frames[f].iloc[start:end] for f in OHLCV_FIELDS}


def walk_forward(
    frames: dict[str, pd.DataFrame],
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, Sequence],
    cfg: Config,
    *,
    select: Callable[[pd.Series, int], float] | None = None,
) -> WalkForwardResult:
    """Roll `param_grid` forward, selecting on train and reporting on test."""
    select = select or (lambda r, bpy: sharpe(r, bpy))
    close = frames["close"]
    n = len(close)
    wf = cfg.walkforward
    folds = make_folds(n, wf.train_bars, wf.test_bars, wf.step_bars, wf.min_folds)

    keys = list(param_grid)
    combos = [dict(zip(keys, vals)) for vals in product(*(param_grid[k] for k in keys))]
    if not combos:
        raise ValueError("param_grid produced no combinations")

    oos_parts: list[pd.Series] = []
    rows: list[dict] = []
    chosen: list[dict] = []
    trial_srs: list[float] = []
    trial_cols: dict[str, pd.Series] = {}

    for fold in folds:
        train = _slice(frames, fold.train_start, fold.train_end)

        scored: list[tuple[float, dict]] = []
        for combo in combos:
            strat = strategy_factory(**combo)
            try:
                res = run_backtest(
                    train, strat,
                    costs=cfg.costs,
                    initial_equity=cfg.backtest.initial_equity,
                    cash_buffer=cfg.backtest.cash_buffer,
                )
            except ValueError:
                continue  # too few bars for this warmup
            score = select(res.returns, cfg.bars_per_year)
            scored.append((score, combo))
            trial_srs.append(score)
            label = f"f{fold.index}|" + ",".join(f"{k}={v}" for k, v in combo.items())
            trial_cols[label] = res.returns.reset_index(drop=True)

        if not scored:
            continue
        best_score, best_combo = max(scored, key=lambda t: t[0])
        chosen.append(dict(best_combo))

        # The test run needs warmup history, so extend the slice backwards and
        # then keep only the genuinely out-of-sample part. Without this the
        # strategy would spend the first part of every fold unable to act.
        strat = strategy_factory(**best_combo)
        warm = max(int(strat.warmup), 1) + 1
        ext_start = max(fold.test_start - warm, 0)
        test_ext = _slice(frames, ext_start, fold.test_end)
        try:
            res = run_backtest(
                test_ext, strat,
                costs=cfg.costs,
                initial_equity=cfg.backtest.initial_equity,
                cash_buffer=cfg.backtest.cash_buffer,
            )
        except ValueError:
            continue

        boundary = close.index[fold.test_start]
        oos = res.returns[res.returns.index >= boundary]
        oos_parts.append(oos)

        rows.append(
            {
                "fold": fold.index,
                "train_sharpe": best_score,
                "test_sharpe": sharpe(oos, cfg.bars_per_year),
                "test_bars": len(oos),
                "n_trades": res.n_trades,
                "halted": res.halted_at is not None,
                **{f"param_{k}": v for k, v in best_combo.items()},
            }
        )

    if not oos_parts:
        raise RuntimeError("no fold produced out-of-sample returns")

    oos_returns = pd.concat(oos_parts).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = (1.0 + oos_returns).cumprod() * cfg.backtest.initial_equity

    trials = pd.DataFrame(trial_cols) if trial_cols else pd.DataFrame()
    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        per_fold=pd.DataFrame(rows),
        chosen_params=chosen,
        n_trials=len(trial_srs),
        trial_sr_variance=float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0,
        all_trial_returns=trials,
    )
