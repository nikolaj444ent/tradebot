"""Core value types.

Everything here is frozen. Mutable trading state lives in `engine.portfolio`,
and nowhere else, so there is exactly one place that can be wrong about what
we own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

# A target portfolio: symbol -> fraction of equity. Spot, long-only, so values
# are in [0, 1] and sum to <= 1. The leftover is cash.
TargetWeights = Mapping[str, float]

# One bar's worth of per-symbol numbers: prices, or notional volumes.
#
# Both a plain dict and a pandas Series satisfy this, and both are used. The
# backtest loop builds dicts out of numpy rows because a Series costs ~48us to
# construct and it needs four per bar; the paper loop already holds Series and
# passes those. Every consumer only does `.get(sym)`, `[sym]` and `.items()`,
# so neither has to know which it got.
PriceMap = Mapping[str, float]

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Order:
    """An intent to trade, sized in base units.

    `client_order_id` is the idempotency key. It is derived deterministically
    from (run_id, bar timestamp, symbol) so that replaying a bar after a crash
    produces the *same* id and the venue (or our paper book) rejects the
    duplicate instead of doubling our position. See DESIGN.md section 6.
    """

    client_order_id: str
    ts_decided: pd.Timestamp
    symbol: str
    side: Side
    qty: float  # base units, always positive
    reason: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"order qty must be positive, got {self.qty}")


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    ts_filled: pd.Timestamp
    symbol: str
    side: Side
    qty: float
    price: float  # all-in effective price, slippage already applied
    fee: float  # quote currency, always positive
    slippage_bps: float = 0.0

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side is Side.BUY else -self.qty

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True, slots=True)
class MarketView:
    """What a strategy is allowed to see: history up to and including `now`.

    This type is the no-lookahead guarantee. The engine builds it by slicing
    the panel at the current bar, so a strategy *cannot* reach a future bar —
    not because it promises not to, but because the future is not in the
    object. Any strategy that needs to cheat would have to reach outside its
    arguments, which `tests/test_no_lookahead.py` checks for with a control
    (`FutureOracle`) that is handed the full panel through its constructor.

    `close.iloc[-1]` is the close of the bar we are deciding on. Execution
    happens at the NEXT bar's open, which is deliberately not available here.

    **Two backings, one guarantee.** A view can be built from DataFrames
    (`from_frames`, used by the paper loop, which already holds them) or from
    numpy slices (`from_arrays`, used by the backtest loop). The second exists
    because profiling showed per-bar pandas object construction dominating
    runtime, and because `to_numpy()` per bar would be O(n^2) copying: numpy
    basic slices are zero-copy *views*, so the backtest converts the panel once
    and every bar's view is free. Indexing past the end of such a slice raises
    IndexError, exactly as with a DataFrame slice; both share storage with
    their parent through private internals, so neither is a defence against a
    strategy determined to cheat, and neither is meant to be. DataFrames are
    materialised lazily and cached, so `.close` still works on either backing.
    """

    #: DataFrame backing: field -> DataFrame(index=ts, cols=symbols).
    frames: Mapping[str, pd.DataFrame] | None = None
    #: numpy backing: field -> 2D array (bars x symbols), history only.
    arrays: Mapping[str, "np.ndarray"] | None = None
    index: pd.DatetimeIndex | None = None
    columns: tuple[str, ...] | None = None
    #: Memo for DataFrames built from `arrays`. Mutating a dict inside a frozen
    #: dataclass is allowed, and caching matters: a strategy may touch `.close`
    #: several times per bar.
    _built: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (self.frames is None) == (self.arrays is None):
            raise ValueError(
                "a MarketView needs exactly one backing: pass frames= or "
                "arrays= (with index= and columns=), not both and not neither."
            )
        if self.arrays is not None and (self.index is None or self.columns is None):
            raise ValueError("arrays= backing also needs index= and columns=")

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> "MarketView":
        return cls(frames=frames)

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, "np.ndarray"],
        index: pd.DatetimeIndex,
        columns: tuple[str, ...],
    ) -> "MarketView":
        return cls(arrays=arrays, index=index, columns=columns)

    # ---------------------------------------------------------------- accessors

    @property
    def now(self) -> pd.Timestamp:
        if self.index is not None:
            return self.index[-1]
        return self.close.index[-1]

    @property
    def symbols(self) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        return list(self.close.columns)

    def frame(self, field_name: str) -> pd.DataFrame:
        """The field as a DataFrame, built on demand under a numpy backing."""
        if self.frames is not None:
            return self.frames[field_name]
        built = self._built.get(field_name)
        if built is None:
            built = pd.DataFrame(
                self.arrays[field_name], index=self.index, columns=list(self.columns)
            )
            self._built[field_name] = built
        return built

    def values(self, field_name: str) -> "np.ndarray":
        """The field as a 2D numpy array (bars x symbols), history only.

        Strategies that only need row arithmetic should use this rather than
        `.close`: it avoids building a pandas object per bar, which profiling
        showed to be the single largest cost in a backtest.
        """
        if self.arrays is not None:
            return self.arrays[field_name]
        key = f"__values__{field_name}"
        arr = self._built.get(key)
        if arr is None:
            arr = self.frames[field_name].to_numpy(dtype="float64", copy=False)
            self._built[key] = arr
        return arr

    @property
    def open(self) -> pd.DataFrame:
        return self.frame("open")

    @property
    def high(self) -> pd.DataFrame:
        return self.frame("high")

    @property
    def low(self) -> pd.DataFrame:
        return self.frame("low")

    @property
    def close(self) -> pd.DataFrame:
        return self.frame("close")

    @property
    def volume(self) -> pd.DataFrame:
        return self.frame("volume")

    def __len__(self) -> int:
        if self.index is not None:
            return len(self.index)
        return len(self.close)

    def last_close(self) -> pd.Series:
        return self.close.iloc[-1]


@dataclass(frozen=True, slots=True)
class RiskViolation:
    """A record of the risk gate overriding the strategy.

    These are not warnings to be swallowed. A strategy that is constantly
    clamped is a strategy whose backtest no longer describes what it does.
    """

    ts: pd.Timestamp
    rule: str
    detail: str
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pd.Series  # mark-to-market equity, indexed by bar close ts
    returns: pd.Series  # simple per-bar returns of `equity`
    fills: list[Fill]
    violations: list[RiskViolation]
    weights: pd.DataFrame  # realised weights per bar, for attribution
    halted_at: pd.Timestamp | None = None
    halt_reason: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.fills)

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)
