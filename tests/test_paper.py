"""The paper daemon's contract.

The load-bearing test is parity: driven over a replayed panel, the daemon must
reproduce `run_backtest` fill for fill and equity mark for equity mark. That
equivalence is what makes a paper-versus-backtest divergence *informative*
(DESIGN.md section 2) — it means the market or the assumptions changed, not
the code path.

The rest covers what a crash can break: idempotent restarts, journal/snapshot
reconciliation, persistent halts, and the risk gate's memory surviving a
restart.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradebot.costs import CostModel
from tradebot.data.synthetic import synthetic_panel
from tradebot.engine.backtest import run_backtest
from tradebot.paper import PaperHalted, PaperState, PaperTrader, ReconciliationError, ReplayFeed
from tradebot.risk import RiskLimits
from tradebot.strategy import CrossSectionalMomentum, EqualWeightRebalanced

SYMS = ("BTC/USDT", "ETH/USDT")


def wide_limits(**overrides) -> RiskLimits:
    """Limits that will not clamp or halt on well-behaved data, so the tests
    below measure the daemon, not the gate. halt_on_breach stays True: the
    daemon refuses anything else."""
    kw = dict(
        max_weight_per_symbol=1.0,
        max_gross_exposure=1.0,
        halt_on_breach=True,
        max_daily_loss_pct=0.95,
        max_drawdown_pct=0.95,
        symbol_allowlist=frozenset(SYMS),
    )
    kw.update(overrides)
    return RiskLimits(**kw)


def make_trader(tmp_path, frames, strategy, *, run_name="t", run_id="fixedrun01", limits=None,
                alerts=None):
    captured = alerts if alerts is not None else []
    return PaperTrader(
        feed=ReplayFeed(frames),
        strategy=strategy,
        costs=CostModel(),
        limits=limits or wide_limits(),
        state=PaperState.at(tmp_path, run_name),
        run_id=run_id,
        alert=lambda subject, body="": captured.append((subject, body)),
    )


@pytest.fixture
def trending():
    return synthetic_panel(SYMS, 400, "1h", seed=3, kind="trending")


def run_to_end(trader):
    trader.run_forever(sleep=lambda s: (_ for _ in ()).throw(AssertionError("replay slept")))


# ---------------------------------------------------------------- parity


def test_daemon_reproduces_the_backtest_exactly(tmp_path, trending):
    strategy = CrossSectionalMomentum(lookback=48, n_long=1, rebalance_every=12)
    limits = wide_limits()
    bt = run_backtest(trending, strategy, costs=CostModel(), limits=limits, run_id="fixedrun01")

    trader = make_trader(tmp_path, trending, strategy, limits=limits)
    run_to_end(trader)

    assert len(bt.fills) > 0, "a parity test with zero trades proves nothing"
    assert [f.client_order_id for f in trader.fills] == [f.client_order_id for f in bt.fills]
    for ours, theirs in zip(trader.fills, bt.fills):
        assert ours.symbol == theirs.symbol and ours.side == theirs.side
        assert ours.qty == pytest.approx(theirs.qty, rel=1e-12)
        assert ours.price == pytest.approx(theirs.price, rel=1e-12)
        assert ours.fee == pytest.approx(theirs.fee, rel=1e-12)
        assert ours.ts_filled == theirs.ts_filled

    pd.testing.assert_series_equal(
        trader.equity_series, bt.equity, check_names=False, rtol=1e-12, atol=0
    )


# ------------------------------------------------------------- restarts


def test_restart_resumes_without_double_fills(tmp_path, trending):
    strategy = CrossSectionalMomentum(lookback=48, n_long=1, rebalance_every=12)
    bt = run_backtest(trending, strategy, costs=CostModel(), limits=wide_limits(),
                      run_id="fixedrun01")

    first = make_trader(tmp_path, trending, strategy)
    first.start()
    first.run_forever(max_steps=120)
    fills_before = list(first.fills)

    # A new process: fresh objects, same state directory, same panel.
    second = make_trader(tmp_path, trending, strategy)
    run_to_end(second)

    combined = [f.client_order_id for f in fills_before + second.fills]
    assert len(combined) == len(set(combined)), "a restart double-filled an order"
    assert combined == [f.client_order_id for f in bt.fills]
    final_equity = second.pf.equity(trending["close"].iloc[-1])
    assert final_equity == pytest.approx(bt.meta["final_equity"], rel=1e-9)


def test_replaying_a_bar_after_a_crash_cannot_double_fill(tmp_path, trending):
    """The crash window DESIGN.md section 6 worries about: fills are in the
    journal, but the process died before the snapshot caught up. On restart
    the daemon re-decides the same bar; the deterministic order ids plus the
    seen-set reloaded from the journal must make that a no-op."""
    strategy = EqualWeightRebalanced()
    first = make_trader(tmp_path, trending, strategy)
    first.start()  # decides on the bootstrap bar and fills at the next open
    assert len(first.fills) == len(SYMS)

    # Roll the snapshot back to its pre-decision state, as a crash between
    # the fill journal writes and the snapshot rewrite would leave it.
    state = PaperState.at(tmp_path, "t")
    snap = json.loads(state.snapshot_path.read_text())
    snap.update({"last_bar": None, "n_fills": 0, "cash": 10_000.0, "positions": {}})
    state.snapshot_path.write_text(json.dumps(snap))

    second = make_trader(tmp_path, trending, strategy)
    second.start()  # re-decides the bootstrap bar
    assert {f.client_order_id for f in first.fills} <= second.adapter._seen
    second.run_forever(max_steps=3)
    ids = [f.client_order_id for f in first.fills + second.fills]
    assert len(ids) == len(set(ids))
    # The resumed book carries the journal's fills, not the stale snapshot's.
    assert second.pf.n_fills == len(first.fills) + len(second.fills)


def test_gate_memory_survives_a_restart(tmp_path, trending):
    strategy = EqualWeightRebalanced()
    first = make_trader(tmp_path, trending, strategy)
    first.start()
    first.run_forever(max_steps=50)
    hwm = first.gate._high_water
    assert hwm > 0

    second = make_trader(tmp_path, trending, strategy)
    second.start()
    assert second.gate._high_water == pytest.approx(hwm)
    assert second.gate._day_open_equity is not None


# -------------------------------------------------------- reconciliation


def test_tampered_snapshot_halts_on_startup(tmp_path, trending):
    strategy = EqualWeightRebalanced()
    first = make_trader(tmp_path, trending, strategy)
    first.start()
    first.run_forever(max_steps=30)

    state = PaperState.at(tmp_path, "t")
    snap = json.loads(state.snapshot_path.read_text())
    snap["cash"] += 123.45  # the book now claims money that no fill produced
    state.snapshot_path.write_text(json.dumps(snap))

    alerts: list = []
    second = make_trader(tmp_path, trending, strategy, alerts=alerts)
    with pytest.raises(ReconciliationError, match="cash mismatch"):
        second.start()
    assert state.halted_path.exists()
    assert alerts, "a reconciliation failure must reach a human"

    # And the halt is sticky: the next start refuses before even reconciling.
    third = make_trader(tmp_path, trending, strategy)
    with pytest.raises(PaperHalted):
        third.start()


def test_missing_snapshot_halts_on_startup(tmp_path, trending):
    strategy = EqualWeightRebalanced()
    first = make_trader(tmp_path, trending, strategy)
    first.start()
    first.run_forever(max_steps=5)

    state = PaperState.at(tmp_path, "t")
    state.snapshot_path.unlink()
    with pytest.raises(ReconciliationError, match="snapshot.json is missing"):
        make_trader(tmp_path, trending, strategy).start()


# ---------------------------------------------------------------- halts


def crash_panel():
    """Calm prices, then one bar loses 40%: unambiguous daily-loss breach."""
    frames = synthetic_panel(SYMS, 120, "1h", seed=5, kind="random_walk", vol_per_bar=0.001)
    out = {}
    for field in ("open", "high", "low", "close"):
        df = frames[field].copy()
        df.iloc[60:] *= 0.60
        out[field] = df
    out["volume"] = frames["volume"]
    return out


def test_breach_flattens_halts_and_refuses_to_resume(tmp_path):
    frames = crash_panel()
    limits = wide_limits(max_daily_loss_pct=0.15)
    alerts: list = []
    trader = make_trader(tmp_path, frames, EqualWeightRebalanced(), limits=limits, alerts=alerts)
    with pytest.raises(PaperHalted, match="daily loss"):
        run_to_end(trader)

    assert trader.pf.positions == {}, "a halt must flatten the book"
    state = PaperState.at(tmp_path, "t")
    assert state.halted_path.exists()
    assert alerts, "a halt must reach a human"
    events = [json.loads(l) for l in state.journal_path.read_text().splitlines()]
    assert any(e["type"] == "halt" for e in events)

    # Docker restarts the process; the marker must stop it cold.
    with pytest.raises(PaperHalted, match="HALTED"):
        make_trader(tmp_path, frames, EqualWeightRebalanced(), limits=limits).start()

    # A human clears the halt. The gate's memory survived, the day's loss is
    # still past the limit, so the circuit breaker fires again on the next
    # bar — by design, not by accident.
    assert state.clear_halt()
    revived = make_trader(tmp_path, frames, EqualWeightRebalanced(), limits=limits)
    with pytest.raises(PaperHalted, match="daily loss"):
        run_to_end(revived)


def test_paper_requires_a_working_circuit_breaker(tmp_path, trending):
    with pytest.raises(ValueError, match="halt_on_breach"):
        make_trader(tmp_path, trending, EqualWeightRebalanced(),
                    limits=wide_limits(halt_on_breach=False))


# ------------------------------------------------------------ housekeeping


def test_heartbeat_is_written(tmp_path, trending):
    trader = make_trader(tmp_path, trending, EqualWeightRebalanced())
    trader.start()
    trader.run_forever(max_steps=2)
    state = PaperState.at(tmp_path, "t")
    beat = json.loads(state.heartbeat_path.read_text())
    assert pd.Timestamp(beat["ts"]) > pd.Timestamp.now(tz="UTC") - pd.Timedelta("1min")
    assert beat["last_bar"] is not None


def test_changing_the_strategy_mid_run_is_refused(tmp_path, trending):
    first = make_trader(tmp_path, trending, EqualWeightRebalanced())
    first.start()
    swapped = make_trader(tmp_path, trending,
                          CrossSectionalMomentum(lookback=48, n_long=1, rebalance_every=12))
    with pytest.raises(ValueError, match="started with"):
        swapped.start()
