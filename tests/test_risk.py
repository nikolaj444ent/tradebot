"""The gate must only ever reduce risk, and must actually stop the run."""

from __future__ import annotations

import pandas as pd
import pytest

from tradebot.risk import RiskGate, RiskLimits

TS = pd.Timestamp("2026-01-01", tz="UTC")
ALLOW = frozenset({"A", "B", "C"})


def gate(**kw) -> RiskGate:
    return RiskGate(limits=RiskLimits(symbol_allowlist=ALLOW, **kw))


def test_empty_allowlist_is_a_config_error_not_permission():
    g = RiskGate(limits=RiskLimits())
    with pytest.raises(ValueError, match="allowlist"):
        g.clamp(TS, {"A": 0.5})


def test_symbols_outside_the_allowlist_are_dropped():
    out = gate().clamp(TS, {"A": 0.2, "ZZZ": 0.5})
    assert out == {"A": pytest.approx(0.2)}


def test_shorts_are_refused_on_a_spot_account():
    out = gate(max_weight_per_symbol=1.0).clamp(TS, {"A": -0.5})
    assert out == {}


def test_per_symbol_cap_is_applied():
    out = gate(max_weight_per_symbol=0.1).clamp(TS, {"A": 0.9})
    assert out["A"] == pytest.approx(0.1)


def test_gross_exposure_is_scaled_not_rejected():
    out = gate(max_weight_per_symbol=1.0, max_gross_exposure=1.0).clamp(
        TS, {"A": 0.8, "B": 0.8}
    )
    assert sum(out.values()) == pytest.approx(1.0)
    # Scaled proportionally, so relative conviction survives.
    assert out["A"] == pytest.approx(out["B"])


def test_a_fully_invested_book_logs_no_spurious_violation():
    """gross == 1.0 exactly must not trip the clamp on float noise.

    Without a tolerance this logged a "scaled by 1.000" no-op on every bar,
    which buries the violations that matter.
    """
    g = gate(max_weight_per_symbol=1.0, max_gross_exposure=1.0)
    g.clamp(TS, {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert [v.rule for v in g.violations] == []


def test_nan_targets_become_zero_and_are_recorded():
    g = gate()
    out = g.clamp(TS, {"A": float("nan"), "B": 0.1})
    assert "A" not in out
    assert any(v.rule == "nan_weight" for v in g.violations)


def test_daily_loss_halts_the_run():
    g = gate(max_daily_loss_pct=0.10)
    g.observe_equity(TS, 100.0)
    assert g.halt_reason(TS, 95.0) is None
    reason = g.halt_reason(TS, 89.0)
    assert reason is not None and "daily loss" in reason


def test_research_mode_records_the_breach_without_halting():
    """Otherwise every long-only crypto backtest truncates in week one."""
    g = gate(max_daily_loss_pct=0.05, halt_on_breach=False)
    g.observe_equity(TS, 100.0)
    assert g.halt_reason(TS, 80.0) is None
    assert any(v.rule == "breach_not_halted" for v in g.violations)


def test_drawdown_halt_uses_the_high_water_mark():
    g = gate(max_drawdown_pct=0.20, max_daily_loss_pct=0.99)
    for day, eq in [(1, 100.0), (2, 150.0), (3, 130.0)]:
        g.observe_equity(TS + pd.Timedelta(days=day), eq)
    reason = g.halt_reason(TS + pd.Timedelta(days=4), 115.0)
    assert reason is not None and "drawdown" in reason


def test_leverage_is_rejected_at_construction():
    with pytest.raises(ValueError, match="leverage"):
        RiskLimits(max_gross_exposure=2.0, allow_short=False)
