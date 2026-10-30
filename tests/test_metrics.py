"""Metric numerics, especially the ones that discount for searching."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradebot.validation.metrics import (
    _norm_ppf,
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe,
    sharpe_standard_error,
)
from tradebot.validation.pbo import probability_of_backtest_overfitting


def test_norm_ppf_matches_known_quantiles():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)


def test_sharpe_of_a_flat_series_is_zero_not_infinite():
    assert sharpe(pd.Series([0.0] * 50), 8760) == 0.0


def test_max_drawdown_on_a_known_path():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_standard_error_shrinks_with_sample_length():
    short = sharpe_standard_error(1500, 8760)
    long = sharpe_standard_error(87600, 8760)
    assert short > long
    # 62 days of hourly data cannot resolve a Sharpe of 2 from zero.
    assert short > 2.0


def test_deflated_sharpe_falls_as_trials_rise():
    """Same observed result, more searching, less credibility."""
    args = dict(observed_sr=0.05, n_obs=1000, skew=0.0, kurtosis=3.0)
    one = deflated_sharpe_ratio(**args, n_trials=1, trial_sr_variance=0.0)
    fifty = deflated_sharpe_ratio(**args, n_trials=50, trial_sr_variance=0.0004)
    many = deflated_sharpe_ratio(**args, n_trials=500, trial_sr_variance=0.0004)
    assert one > fifty > many


def test_psr_rises_with_a_stronger_result():
    weak = probabilistic_sharpe_ratio(0.01, 500, 0.0, 3.0)
    strong = probabilistic_sharpe_ratio(0.10, 500, 0.0, 3.0)
    assert strong > weak


def test_pbo_needs_more_than_one_candidate():
    df = pd.DataFrame({"only": np.random.default_rng(0).normal(size=500)})
    with pytest.raises(ValueError, match="at least 2 candidates"):
        probability_of_backtest_overfitting(df)


def test_pbo_is_high_for_pure_noise_candidates():
    """Choosing between worthless variants should show no skill.

    With independent noise columns, the in-sample winner is random, so its
    out-of-sample rank is uniform and PBO lands near 0.5.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(2000, 12)))
    result = probability_of_backtest_overfitting(df, n_blocks=10)
    assert 0.3 < result.pbo < 0.7, f"expected ~0.5 for noise, got {result.pbo}"


def test_pbo_is_low_when_one_candidate_is_genuinely_best():
    """A real, persistent edge must survive the split."""
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.01, size=(2000, 8))
    df = pd.DataFrame(noise)
    df[0] = df[0] + 0.004  # a consistent, sizeable edge in column 0
    result = probability_of_backtest_overfitting(df, n_blocks=10)
    assert result.pbo < 0.2, f"a real edge should give low PBO, got {result.pbo}"
