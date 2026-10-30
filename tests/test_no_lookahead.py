"""The most important tests in the repo.

A backtest that can see the future is not optimistic, it is fiction. These
tests attack that from three directions:

1. Structurally: every fill must land on a strictly later bar than the
   decision that produced it.
2. By demonstration: a strategy that *does* get the future makes absurd
   money, while the same logic restricted to `MarketView` does not. This
   proves the view is what blocks the leak, and that these tests would catch
   a regression that reintroduced it.
3. Statistically: on a driftless random walk, no causal strategy should earn
   a meaningful risk-adjusted return after costs.
"""

from __future__ import annotations

import math

import pandas as pd

from tradebot.data import synthetic_panel
from tradebot.engine import run_backtest
from tradebot.strategy import CrossSectionalMomentum, ZScoreMeanReversion
from tradebot.strategy.base import Strategy
from tradebot.types import MarketView
from tradebot.validation.metrics import sharpe, sharpe_standard_error

BARS_PER_YEAR = 365 * 24


class SpyStrategy(Strategy):
    """Records the last bar it was shown, so the delay can be checked."""

    warmup = 2

    def __init__(self) -> None:
        self.seen: list[pd.Timestamp] = []
        self.view_lengths: list[int] = []

    def target_weights(self, view: MarketView) -> dict[str, float]:
        self.seen.append(view.now)
        self.view_lengths.append(len(view))
        syms = view.symbols
        return {syms[0]: 0.5}


class FutureOracle(Strategy):
    """Deliberately cheats: holds the full panel and reads the NEXT bar.

    This is what a lookahead bug looks like. It exists so the tests can prove
    the difference between "cannot see the future" and "chose not to look".
    """

    warmup = 2

    def __init__(self, full_close: pd.DataFrame) -> None:
        self._full = full_close

    def target_weights(self, view: MarketView) -> dict[str, float]:
        loc = self._full.index.get_loc(view.now)
        if loc + 1 >= len(self._full):
            return {}
        growth = self._full.iloc[loc + 1] / self._full.iloc[loc]
        if growth.max() <= 1.0:
            return {}
        return {str(growth.idxmax()): 1.0}


def test_every_fill_is_after_the_decision_that_caused_it():
    frames = synthetic_panel(("A", "B"), 300, seed=11)
    spy = SpyStrategy()
    result = run_backtest(frames, spy, initial_equity=10_000.0)

    index = frames["close"].index
    assert result.fills, "expected at least one fill"
    for fill in result.fills:
        # Reconstruct which decision bar produced this fill and require that
        # the fill is strictly later. Same-bar execution would mean the
        # engine traded on a price it could not have known.
        fill_loc = index.get_loc(fill.ts_filled)
        assert fill_loc >= 1
        assert index[fill_loc - 1] in spy.seen


def test_view_never_extends_past_the_decision_bar():
    frames = synthetic_panel(("A", "B"), 200, seed=12)
    spy = SpyStrategy()
    run_backtest(frames, spy, initial_equity=10_000.0)

    index = frames["close"].index
    for ts, length in zip(spy.seen, spy.view_lengths):
        # The view must hold exactly the bars up to and including `ts`.
        assert length == index.get_loc(ts) + 1
    # It must never have been shown the final bar, because there would be no
    # bar left to execute against.
    assert index[-1] not in spy.seen


def test_lookahead_would_be_caught_if_it_were_reintroduced():
    """A strategy given the future must vastly outperform an honest one.

    Stated as a *ratio*, deliberately. An absolute bound on the honest
    strategy would be testing noise: with 3000 hourly bars the standard error
    of an annualised Sharpe is ~1.7, so an honest strategy scoring 2 or 3 on
    one sample means nothing. The oracle's advantage, by contrast, is not
    subtle — Sharpe ~35 against ~1, final equity 40,000,000 against 11,000.

    If these margins ever narrow, the engine has begun leaking the future.
    """
    for seed in (13, 14, 15):
        frames = synthetic_panel(("A", "B", "C"), 3000, seed=seed, kind="random_walk")

        oracle = run_backtest(
            frames, FutureOracle(frames["close"]), initial_equity=10_000.0
        )
        honest = run_backtest(
            frames,
            CrossSectionalMomentum(lookback=48, n_long=1, rebalance_every=1),
            initial_equity=10_000.0,
        )

        oracle_sr = sharpe(oracle.returns, BARS_PER_YEAR)
        honest_sr = sharpe(honest.returns, BARS_PER_YEAR)

        assert oracle_sr > 20.0, (
            f"seed {seed}: one-bar foresight only scored Sharpe {oracle_sr:.2f}. "
            f"The cheating control is not cheating, so this test no longer "
            f"guards anything."
        )
        assert oracle_sr - honest_sr > 15.0, (
            f"seed {seed}: foresight beat the honest strategy by only "
            f"{oracle_sr - honest_sr:.2f} Sharpe. Either the oracle broke or the "
            f"honest path is seeing the future."
        )
        assert oracle.meta["final_equity"] > 100 * honest.meta["final_equity"]


def test_no_edge_on_a_random_walk():
    """Price is unpredictable by construction, so nothing should work.

    The threshold here is derived, not guessed, and that distinction matters.
    An annualised Sharpe estimated from n hourly bars carries a standard error
    of about sqrt(8760/n): at 1500 bars that is 2.4, so *any* assertion on a
    single short sample tests noise rather than behaviour. An earlier version
    of this test asserted `sharpe < 2.5` on 1500 bars and failed on 2 of 5
    seeds for exactly that reason.

    So: longer samples, averaged over independent seeds, and a bound of three
    standard errors of the mean.
    """
    n_bars, seeds = 6000, (1, 2, 3, 4, 5, 6, 7, 8)
    se_mean = sharpe_standard_error(n_bars, BARS_PER_YEAR) / math.sqrt(len(seeds))
    bound = 3.0 * se_mean

    for strategy in (
        CrossSectionalMomentum(lookback=72, n_long=1, rebalance_every=12),
        ZScoreMeanReversion(lookback=48, entry_z=-1.5),
    ):
        srs = []
        for seed in seeds:
            frames = synthetic_panel(
                ("A", "B", "C"), n_bars, seed=seed, kind="random_walk"
            )
            result = run_backtest(frames, strategy, initial_equity=10_000.0)
            srs.append(sharpe(result.returns, BARS_PER_YEAR))

        mean_sr = sum(srs) / len(srs)
        assert mean_sr < bound, (
            f"{strategy.name} averaged Sharpe {mean_sr:.2f} across {len(seeds)} "
            f"random-walk samples, above the {bound:.2f} bound (3 standard errors "
            f"of the mean). Consistent edge on unpredictable data means a bug, "
            f"most likely a lookahead leak."
        )
        # Catastrophic leaks show up on every seed, not on average.
        assert max(srs) < 8.0, f"{strategy.name} hit Sharpe {max(srs):.2f} on one seed"
