"""The dashboard's data layer must tell the same story as the run itself."""

from __future__ import annotations

import pytest

from tradebot.costs import CostModel
from tradebot.data.synthetic import synthetic_panel
from tradebot.paper import PaperState, PaperTrader, ReplayFeed
from tradebot.paper.report import all_runs, run_report
from tradebot.risk import RiskLimits

from tradebot.strategy import EqualWeightRebalanced

SYMS = ("BTC/USDT", "ETH/USDT")


def finished_run(tmp_path, name="demo"):
    frames = synthetic_panel(SYMS, 300, "1h", seed=3, kind="trending")
    trader = PaperTrader(
        feed=ReplayFeed(frames),
        strategy=EqualWeightRebalanced(),
        costs=CostModel(),
        limits=RiskLimits(
            max_weight_per_symbol=1.0, max_gross_exposure=1.0,
            halt_on_breach=True, max_daily_loss_pct=0.95, max_drawdown_pct=0.95,
            symbol_allowlist=frozenset(SYMS),
        ),
        state=PaperState.at(tmp_path, name),
        run_id="reportrun01",
        alert=lambda subject, body="": None,
    )
    trader.run_forever(sleep=lambda s: None)
    return trader


def test_report_tells_the_runs_story(tmp_path):
    trader = finished_run(tmp_path)
    runs = all_runs(tmp_path)
    assert [r["name"] for r in runs] == ["demo"]
    r = runs[0]

    assert r["run_id"] == "reportrun01"
    assert r["n_fills"] == len(trader.fills)
    assert r["n_marks"] == len(trader.equity_series)
    assert len(r["equity_curve"]) == len(trader.equity_series)
    assert r["equity"] == pytest.approx(float(trader.equity_series.iloc[-1]))
    assert r["pnl_pct"] == pytest.approx(r["equity"] / r["initial_equity"] - 1.0)
    assert r["cash"] == pytest.approx(trader.pf.cash)
    assert r["positions"].keys() == trader.pf.positions.keys()
    assert r["fees_paid"] == pytest.approx(trader.pf.fees_paid)
    assert r["halted"] is None
    assert r["heartbeat_age_s"] is not None and r["heartbeat_age_s"] < 60
    # Newest trade first, so the table reads top-down.
    assert r["fills"][0]["ts"] >= r["fills"][-1]["ts"]
    assert 0.0 <= r["max_drawdown"] < 1.0


def test_report_surfaces_a_halt(tmp_path):
    finished_run(tmp_path)
    state = PaperState.at(tmp_path, "demo")
    state.write_halted("daily loss 40.00% >= limit 15.00%")
    r = run_report(state)
    assert "daily loss" in r["halted"]


def test_no_runs_is_an_empty_list_not_an_error(tmp_path):
    assert all_runs(tmp_path) == []
