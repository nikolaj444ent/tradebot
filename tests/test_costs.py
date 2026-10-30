"""Costs must be monotone, directional, and impossible to switch off."""

from __future__ import annotations

import pytest

from tradebot.costs import CostModel


def test_impact_rises_with_participation():
    c = CostModel()
    small = c.impact_bps(100.0, 1_000_000.0)
    large = c.impact_bps(100_000.0, 1_000_000.0)
    assert small < large
    assert large <= c.impact_bps_at_full_bar


def test_unknown_volume_assumes_the_worst():
    """No volume data must not mean free execution."""
    c = CostModel()
    assert c.impact_bps(100.0, 0.0) == c.impact_bps_at_full_bar


def test_slippage_is_directional():
    c = CostModel()
    buy, _ = c.effective_price(100.0, True, 1000.0, 1e6)
    sell, _ = c.effective_price(100.0, False, 1000.0, 1e6)
    assert buy > 100.0 > sell, "buys pay up, sells receive less"


def test_round_trip_hurdle_is_what_a_signal_must_beat():
    c = CostModel(taker_fee_bps=10.0, half_spread_bps=2.0)
    hurdle = c.round_trip_bps(10_000.0, 1_000_000.0)
    # Two crossings of fee + spread alone is 24 bps, before any impact.
    assert hurdle > 24.0


def test_fees_scale_with_notional():
    c = CostModel(taker_fee_bps=10.0)
    assert c.fee(10_000.0) == pytest.approx(10.0)
