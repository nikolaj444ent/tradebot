"""Baseline strategies.

These exist to be *tested against*, not to be traded. Two of them are the
classic retail ideas (momentum, mean reversion) and one is buy-and-hold, which
is the benchmark that matters: in a market that rose as much as crypto did
over most sample windows, a strategy that does not beat buy-and-hold after
costs has produced nothing except turnover. Report it alongside every result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MarketView
from .base import Strategy


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Indices of `values` in descending order, NaNs dropped.

    Written to reproduce `Series.dropna().sort_values(ascending=False)` index
    for index: pandas sorts the non-NaN values ascending with quicksort and
    then reverses the whole indexer, so that is what this does. The point is
    not elegance — it is that switching these strategies to numpy must not
    change which asset gets picked when two returns tie, because
    `tests/test_engine_golden.py` pins the output bit for bit.
    """
    keep = np.flatnonzero(~np.isnan(values))
    return keep[np.argsort(values[keep], kind="quicksort")][::-1]


def _rank_asc(values: np.ndarray) -> np.ndarray:
    """Indices in ascending order, NaNs dropped: `dropna().sort_values()`."""
    keep = np.flatnonzero(~np.isnan(values))
    return keep[np.argsort(values[keep], kind="quicksort")]


def _equal_weights(symbols, picked: np.ndarray) -> dict[str, float]:
    if picked.size == 0:
        return {}
    w = 1.0 / picked.size
    return {symbols[j]: w for j in picked}


@dataclass(frozen=True, slots=True)
class BuyAndHold(Strategy):
    """Buy equal notional on the first bar, then never trade again.

    The honest benchmark, and the one most strategies quietly fail to beat.
    Zero parameters, so there is nothing here to overfit.

    Subtlety worth understanding: holding is *not* the same as asking for equal
    weights every bar. Once prices move, a static holding drifts away from
    equal weight, so a strategy that re-requests 1/n each bar is really a
    continuously rebalanced portfolio that pays fees forever (see
    `EqualWeightRebalanced`). To express true holding through a weight-based
    interface, this returns the weights the original purchase has *drifted*
    into — which is a pure function of the price path:

        w_i(t) = (P_i(t) / P_i(t0)) / sum_j (P_j(t) / P_j(t0))

    The engine then finds nothing to trade, because the target already equals
    what is held.
    """

    warmup: int = 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.values("close")
        symbols = view.symbols
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = close[-1] / close[0]
        # `.replace([inf, -inf], nan).dropna()` then `> 0`, in one mask.
        keep = np.flatnonzero(np.isfinite(growth) & (growth > 0.0))
        if keep.size == 0:
            return {}
        total = float(growth[keep].sum())
        return {symbols[j]: float(growth[j]) / total for j in keep}


@dataclass(frozen=True, slots=True)
class EqualWeightRebalanced(Strategy):
    """Hold 1/n of every asset, rebalancing back to equal weight.

    A legitimate benchmark in its own right — rebalancing harvests volatility
    between uncorrelated assets — but a different one from buy-and-hold, and
    more expensive. Comparing the two tells you whether a strategy's edge is
    really just a rebalancing premium you could have had for free.
    """

    warmup: int = 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        syms = view.symbols
        w = 1.0 / len(syms)
        return {s: w for s in syms}


@dataclass(frozen=True, slots=True)
class CrossSectionalMomentum(Strategy):
    """Hold the strongest `n_long` assets by trailing return.

    Cross-sectional rather than time-series: it is always fully invested in
    *something*, which removes the market-timing question and isolates whether
    relative strength persists.
    """

    lookback: int = 168  # 7 days of 1h bars
    n_long: int = 2
    rebalance_every: int = 24  # trade once a day, not every bar

    @property
    def warmup(self) -> int:  # type: ignore[override]
        return self.lookback + 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.values("close")
        n = len(close)
        if n < self.lookback + 1:
            return {}

        # Only act on rebalance bars. Between them, return the same targets so
        # the engine naturally holds (it compares target to current and finds
        # nothing to do).
        if self.rebalance_every > 1 and (n - 1) % self.rebalance_every != 0:
            return self._held(view)

        return self._winners(close, n - 1, view.symbols)

    def _winners(self, close: np.ndarray, at: int, symbols) -> dict[str, float]:
        """Equal weight across the top `n_long` by trailing return at bar `at`."""
        with np.errstate(divide="ignore", invalid="ignore"):
            trailing = close[at] / close[at - self.lookback] - 1.0
        ranked = _rank_desc(trailing)[: self.n_long]
        # Positive momentum only: if nothing is rising, sit in cash rather than
        # holding the least-bad asset.
        return _equal_weights(symbols, ranked[trailing[ranked] > 0.0])

    def _held(self, view: MarketView) -> dict[str, float]:
        """Targets from the most recent rebalance bar, recomputed from history.

        Recomputed rather than remembered: see `Strategy` rule 1. This is
        slightly wasteful and entirely worth it — and since it runs on every
        non-rebalance bar, it was also the single most expensive line in a
        backtest until it stopped going through pandas. Profiling a 6,000-bar
        run put 57% of total time inside this method.
        """
        close = view.values("close")
        n = len(close)
        last_rebal = ((n - 1) // self.rebalance_every) * self.rebalance_every
        if last_rebal < self.lookback:
            return {}
        return self._winners(close, last_rebal, view.symbols)


@dataclass(frozen=True, slots=True)
class ZScoreMeanReversion(Strategy):
    """Buy what has fallen relative to its own recent mean.

    Long-only (spot), so this expresses "oversold" only. Weight is spread
    equally across whatever currently qualifies.
    """

    lookback: int = 72
    entry_z: float = -1.5
    max_positions: int = 2

    @property
    def warmup(self) -> int:  # type: ignore[override]
        return self.lookback + 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.values("close")
        if len(close) < self.lookback + 1:
            return {}
        window = close[-self.lookback :]
        # nanmean/nanstd rather than mean/std, because pandas skips NaN by
        # default and this has to keep doing exactly what it did before.
        mean = np.nanmean(window, axis=0)
        sd = np.nanstd(window, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # A zero standard deviation means a flat price, not a huge z-score.
            z = (close[-1] - mean) / np.where(sd == 0.0, np.nan, sd)
        # Sorting ascending and then filtering is the same set as filtering and
        # then sorting, because the filter is a threshold on the sort key.
        ranked = _rank_asc(z)
        return _equal_weights(
            view.symbols, ranked[z[ranked] <= self.entry_z][: self.max_positions]
        )
