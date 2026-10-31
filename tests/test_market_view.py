"""MarketView's two backings, and the guarantee they both have to keep.

The backtest loop builds views from zero-copy numpy slices and the paper loop
builds them from DataFrames. That split exists purely for speed, so it earns
its place only if the two are indistinguishable to a strategy — otherwise the
backtest/paper parity that the whole design rests on would quietly depend on
which backing happened to be used.

`tests/test_no_lookahead.py` owns the engine-level version of the guarantee.
This file checks it at the level of the type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.data.synthetic import synthetic_panel
from tradebot.strategy import (
    BuyAndHold,
    CrossSectionalMomentum,
    EqualWeightRebalanced,
    ZScoreMeanReversion,
)
from tradebot.types import MarketView, OHLCV_FIELDS

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(SYMBOLS, 600, "1h", seed=4, kind="trending")


def frames_view(panel, upto: int) -> MarketView:
    return MarketView(frames={f: panel[f].iloc[: upto + 1] for f in OHLCV_FIELDS})


def arrays_view(panel, upto: int) -> MarketView:
    arrays = {
        f: panel[f].to_numpy(dtype="float64", copy=False)[: upto + 1]
        for f in OHLCV_FIELDS
    }
    return MarketView.from_arrays(arrays, panel["close"].index[: upto + 1], SYMBOLS)


@pytest.mark.parametrize("upto", [1, 50, 199, 599])
def test_the_two_backings_are_indistinguishable(panel, upto):
    a, b = frames_view(panel, upto), arrays_view(panel, upto)

    assert len(a) == len(b) == upto + 1
    assert a.now == b.now
    assert a.symbols == b.symbols
    for field in OHLCV_FIELDS:
        # The DataFrame built lazily from arrays must equal the sliced one.
        pd.testing.assert_frame_equal(a.frame(field), b.frame(field))
        np.testing.assert_array_equal(a.values(field), b.values(field))


@pytest.mark.parametrize(
    "strategy",
    [
        BuyAndHold(),
        EqualWeightRebalanced(),
        CrossSectionalMomentum(lookback=72, n_long=2, rebalance_every=24),
        ZScoreMeanReversion(lookback=72, entry_z=-1.0, max_positions=2),
    ],
    ids=lambda s: s.name,
)
def test_strategies_decide_identically_on_either_backing(panel, strategy):
    """Exact equality, including dict ordering.

    Ordering is not cosmetic here: the risk gate iterates the returned mapping,
    so a different order would produce violations in a different order, which
    the golden digests would catch as a behaviour change.
    """
    for upto in (100, 250, 400, 599):
        from_frames = strategy.target_weights(frames_view(panel, upto))
        from_arrays = strategy.target_weights(arrays_view(panel, upto))
        assert list(from_frames.items()) == list(from_arrays.items()), (
            f"{strategy.name} disagreed at bar {upto}"
        )


def test_a_view_cannot_reach_past_its_own_last_bar(panel):
    """The future is absent from the object, on both backings.

    Not a claim that a determined strategy cannot cheat — one that is handed
    the full panel through its constructor obviously can, which is exactly what
    `FutureOracle` in tests/test_no_lookahead.py does. The claim is narrower
    and is the one that matters in practice: nothing reachable through a view's
    public interface contains a bar after `now`.
    """
    upto = 300
    for view in (frames_view(panel, upto), arrays_view(panel, upto)):
        assert view.now == panel["close"].index[upto]
        assert len(view.values("close")) == upto + 1
        assert view.close.index[-1] == view.now

        # The next bar exists in the panel and is not in the view.
        assert len(panel["close"]) > upto + 1
        with pytest.raises(IndexError):
            _ = view.values("close")[upto + 1]
        with pytest.raises(IndexError):
            _ = view.close.iloc[upto + 1]


def test_a_view_needs_exactly_one_backing():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    arrays = {f: np.zeros((2, 1)) for f in OHLCV_FIELDS}
    frames = {f: pd.DataFrame(np.zeros((2, 1)), index=idx, columns=["A"]) for f in OHLCV_FIELDS}

    with pytest.raises(ValueError, match="exactly one backing"):
        MarketView()
    with pytest.raises(ValueError, match="exactly one backing"):
        MarketView(frames=frames, arrays=arrays, index=idx, columns=("A",))
    with pytest.raises(ValueError, match="needs index= and columns="):
        MarketView(arrays=arrays)


def test_array_slices_are_zero_copy(panel):
    """The reason the numpy backing exists at all.

    If each per-bar view copied its history, the backtest would be O(n^2) in
    memory traffic and the whole change would be pointless: at 51k bars that
    is tens of gigabytes of copying.
    """
    # One conversion of the panel, as the engine does it, then slices of it.
    arrays = {f: panel[f].to_numpy(dtype="float64", copy=False) for f in OHLCV_FIELDS}
    view = MarketView.from_arrays(
        {f: arrays[f][:300] for f in OHLCV_FIELDS},
        panel["close"].index[:300],
        SYMBOLS,
    )
    sliced = view.values("close")
    # `shares_memory`, not `.base is`: numpy points `.base` at the ultimate
    # owner of the buffer rather than the immediate parent, and `to_numpy(
    # copy=False)` is itself already a view onto pandas' internal block.
    assert np.shares_memory(sliced, arrays["close"]), "the slice copied"
    assert sliced.shape == (300, len(SYMBOLS))


def test_lazy_frames_are_built_once(panel):
    view = arrays_view(panel, 200)
    assert view.close is view.close
    assert view.values("close") is view.values("close")
