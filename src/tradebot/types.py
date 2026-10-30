"""Core value types.

Everything here is frozen. Mutable trading state lives in `engine.portfolio`,
and nowhere else, so there is exactly one place that can be wrong about what
we own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import pandas as pd

# A target portfolio: symbol -> fraction of equity. Spot, long-only, so values
# are in [0, 1] and sum to <= 1. The leftover is cash.
TargetWeights = Mapping[str, float]

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
    arguments, which `tests/test_no_lookahead.py` checks for.

    `close.iloc[-1]` is the close of the bar we are deciding on. Execution
    happens at the NEXT bar's open, which is deliberately not available here.
    """

    frames: Mapping[str, pd.DataFrame]  # field -> DataFrame(index=ts, cols=symbols)

    @property
    def now(self) -> pd.Timestamp:
        return self.close.index[-1]

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def open(self) -> pd.DataFrame:
        return self.frames["open"]

    @property
    def high(self) -> pd.DataFrame:
        return self.frames["high"]

    @property
    def low(self) -> pd.DataFrame:
        return self.frames["low"]

    @property
    def close(self) -> pd.DataFrame:
        return self.frames["close"]

    @property
    def volume(self) -> pd.DataFrame:
        return self.frames["volume"]

    def __len__(self) -> int:
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
