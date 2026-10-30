"""Deterministic synthetic market data.

Two uses, both load-bearing:

1. Tests run without network access or an exchange account.
2. `kind="random_walk"` generates a driftless martingale. On that data, *no*
   causal strategy can be reliably profitable after costs — the price is
   unpredictable by construction. So if the engine reports a strategy making
   good money on a random walk, the engine has a lookahead bug. That is the
   single most valuable test in this repo (`tests/test_no_lookahead.py`).

`kind="mean_reverting"` is the opposite control: an Ornstein-Uhlenbeck process
with a genuine, exploitable edge. A mean-reversion strategy that *fails* on it
means the strategy or the engine is broken, not that the idea is bad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def synthetic_panel(
    symbols: tuple[str, ...] | list[str] = ("BTC/USDT", "ETH/USDT"),
    n_bars: int = 2000,
    timeframe: str = "1h",
    *,
    seed: int = 0,
    kind: str = "random_walk",
    start_price: float = 100.0,
    vol_per_bar: float = 0.01,
    drift_per_bar: float = 0.0,
    reversion_strength: float = 0.05,
    base_volume: float = 1_000_000.0,
    start: str = "2022-01-01",
) -> dict[str, pd.DataFrame]:
    """Build an OHLCV panel with internally consistent bars.

    Returns the same {field: DataFrame} shape the real fetcher produces, so
    tests and live code travel the identical path.
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    rng = np.random.default_rng(seed)
    symbols = list(symbols)
    n_sym = len(symbols)

    index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        periods=n_bars,
        freq=f"{TIMEFRAME_MINUTES[timeframe]}min",
    )

    if kind == "random_walk":
        # Log-price is a martingale: increments are iid and mean-zero (unless
        # a drift is explicitly asked for). Nothing in the past predicts the
        # next increment.
        shocks = rng.normal(drift_per_bar, vol_per_bar, size=(n_bars, n_sym))
        log_price = np.log(start_price) + np.cumsum(shocks, axis=0)
    elif kind == "mean_reverting":
        # Ornstein-Uhlenbeck around a fixed log level: deviations decay, so
        # "buy what has fallen" is genuinely profitable before costs.
        log_price = np.empty((n_bars, n_sym))
        level = np.log(start_price)
        state = np.full(n_sym, level)
        for t in range(n_bars):
            state = state + reversion_strength * (level - state) + rng.normal(
                0.0, vol_per_bar, size=n_sym
            )
            log_price[t] = state
    elif kind == "trending":
        # Momentum control: persistent autocorrelated drift regimes.
        log_price = np.empty((n_bars, n_sym))
        state = np.full(n_sym, np.log(start_price))
        mom = np.zeros(n_sym)
        for t in range(n_bars):
            mom = 0.98 * mom + rng.normal(0.0, vol_per_bar * 0.15, size=n_sym)
            state = state + mom + rng.normal(0.0, vol_per_bar * 0.5, size=n_sym)
            log_price[t] = state
    else:
        raise ValueError(f"unknown kind {kind!r}")

    close = np.exp(log_price)

    # Continuous market: this bar opens where the last one closed.
    open_ = np.vstack([close[0:1], close[:-1]])

    # Intrabar extremes must bracket both open and close, or the data is
    # self-contradictory and would let a backtest fill at impossible prices.
    span = np.abs(rng.normal(0.0, vol_per_bar * 0.6, size=(n_bars, n_sym)))
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * (1.0 + span)
    low = lo_base * (1.0 - span)

    # Volume: lognormal, correlated with absolute return (busy bars are
    # volatile bars), so the impact model sees a realistic relationship.
    abs_ret = np.abs(np.diff(log_price, axis=0, prepend=log_price[0:1]))
    vol_mult = np.exp(rng.normal(0.0, 0.4, size=(n_bars, n_sym)))
    volume = base_volume * vol_mult * (1.0 + 20.0 * abs_ret)

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=index, columns=symbols).astype("float64")

    return {
        "open": frame(open_),
        "high": frame(high),
        "low": frame(low),
        "close": frame(close),
        "volume": frame(volume),
    }
