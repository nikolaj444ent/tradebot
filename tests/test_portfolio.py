"""Accounting invariants.

A backtest whose books do not balance can show any result at all, so these
are checked rather than assumed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradebot.costs import CostModel
from tradebot.data import synthetic_panel
from tradebot.engine import run_backtest
from tradebot.engine.portfolio import Portfolio, client_order_id
from tradebot.strategy import CrossSectionalMomentum, EqualWeightRebalanced
from tradebot.types import Fill, Side

TS = pd.Timestamp("2026-01-01", tz="UTC")


def test_cash_never_goes_negative_in_a_full_run():
    frames = synthetic_panel(("A", "B", "C"), 800, seed=21)
    # If sizing ever overspends, Portfolio.apply_fill raises, so a clean run
    # is itself the assertion. Run a high-turnover strategy to stress it.
    result = run_backtest(
        frames, EqualWeightRebalanced(), initial_equity=10_000.0, cash_buffer=0.0
    )
    assert result.meta["final_equity"] > 0


def test_cannot_sell_more_than_held():
    pf = Portfolio(cash=0.0, positions={"A": 1.0})
    with pytest.raises(ValueError, match="cannot short"):
        pf.apply_fill(
            Fill(
                client_order_id="x", ts_filled=TS, symbol="A",
                side=Side.SELL, qty=2.0, price=100.0, fee=0.0,
            )
        )


def test_cannot_spend_more_cash_than_held():
    pf = Portfolio(cash=50.0)
    with pytest.raises(ValueError, match="cash"):
        pf.apply_fill(
            Fill(
                client_order_id="x", ts_filled=TS, symbol="A",
                side=Side.BUY, qty=1.0, price=100.0, fee=0.0,
            )
        )


def test_equity_identity_after_a_fill():
    pf = Portfolio(cash=1000.0)
    pf.apply_fill(
        Fill(
            client_order_id="x", ts_filled=TS, symbol="A",
            side=Side.BUY, qty=2.0, price=100.0, fee=1.0,
        )
    )
    assert pf.cash == pytest.approx(1000.0 - 200.0 - 1.0)
    # Marked at cost, equity is down exactly the fee and nothing else.
    assert pf.equity(pd.Series({"A": 100.0})) == pytest.approx(999.0)
    assert pf.fees_paid == pytest.approx(1.0)


def test_order_ids_are_deterministic_so_replay_cannot_double_fill():
    a = client_order_id("run1", TS, "BTC/USDT", Side.BUY)
    b = client_order_id("run1", TS, "BTC/USDT", Side.BUY)
    c = client_order_id("run1", TS, "BTC/USDT", Side.SELL)
    d = client_order_id("run2", TS, "BTC/USDT", Side.BUY)
    assert a == b, "same decision must produce the same id, or a retry doubles up"
    assert a != c and a != d


def test_duplicate_order_ids_fill_only_once():
    from tradebot.execution.base import ExecutionContext
    from tradebot.execution.sim import SimExecution
    from tradebot.types import Order

    sim = SimExecution(costs=CostModel())
    order = Order(
        client_order_id="dup", ts_decided=TS, symbol="A", side=Side.BUY, qty=1.0
    )
    ctx = ExecutionContext(
        ts=TS, ref_price=pd.Series({"A": 100.0}),
        bar_volume_notional=pd.Series({"A": 1e6}),
    )
    assert len(sim.execute([order], ctx)) == 1
    assert sim.execute([order], ctx) == [], "replayed order must not fill twice"


def test_sells_fund_buys_within_one_bar():
    """A rebalance must not assume credit that does not exist."""
    pf = Portfolio(cash=0.0, positions={"A": 10.0})
    prices = pd.Series({"A": 100.0, "B": 100.0})
    orders = pf.orders_to_reach(
        {"B": 1.0}, prices, ts=TS, run_id="r",
        min_notional=10.0, taker_fee_bps=10.0, cash_buffer=0.0,
    )
    sides = [o.side for o in orders]
    assert sides and sides[0] is Side.SELL, "sells must be emitted before buys"
    buy = next(o for o in orders if o.side is Side.BUY)
    # Cash from the sale is 1000 less fees, so the buy must be smaller.
    assert buy.qty * 100.0 < 1000.0
