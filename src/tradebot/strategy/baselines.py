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
        first = view.close.iloc[0]
        last = view.close.iloc[-1]
        growth = (last / first).replace([np.inf, -np.inf], np.nan).dropna()
        growth = growth[growth > 0]
        if growth.empty:
            return {}
        total = float(growth.sum())
        return {s: float(g) / total for s, g in growth.items()}


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
        close = view.close
        if len(close) < self.lookback + 1:
            return {}

        # Only act on rebalance bars. Between them, return the same targets so
        # the engine naturally holds (it compares target to current and finds
        # nothing to do).
        if self.rebalance_every > 1 and (len(close) - 1) % self.rebalance_every != 0:
            return self._held(view)

        trailing = close.iloc[-1] / close.iloc[-1 - self.lookback] - 1.0
        trailing = trailing.dropna()
        if trailing.empty:
            return {}
        winners = trailing.sort_values(ascending=False).head(self.n_long)
        # Positive momentum only: if nothing is rising, sit in cash rather than
        # holding the least-bad asset.
        winners = winners[winners > 0]
        if winners.empty:
            return {}
        w = 1.0 / len(winners)
        return {s: w for s in winners.index}

    def _held(self, view: MarketView) -> dict[str, float]:
        """Targets from the most recent rebalance bar, recomputed from history.

        Recomputed rather than remembered: see `Strategy` rule 1. This is
        slightly wasteful and entirely worth it.
        """
        close = view.close
        n = len(close)
        last_rebal = ((n - 1) // self.rebalance_every) * self.rebalance_every
        if last_rebal < self.lookback:
            return {}
        trailing = close.iloc[last_rebal] / close.iloc[last_rebal - self.lookback] - 1.0
        trailing = trailing.dropna()
        winners = trailing.sort_values(ascending=False).head(self.n_long)
        winners = winners[winners > 0]
        if winners.empty:
            return {}
        w = 1.0 / len(winners)
        return {s: w for s in winners.index}


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
        close = view.close
        if len(close) < self.lookback + 1:
            return {}
        window = close.iloc[-self.lookback :]
        mean = window.mean()
        sd = window.std(ddof=1)
        last = close.iloc[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (last - mean) / sd.replace(0.0, np.nan)
        z = z.dropna()
        picks = z[z <= self.entry_z].sort_values().head(self.max_positions)
        if picks.empty:
            return {}
        w = 1.0 / len(picks)
        return {s: w for s in picks.index}
