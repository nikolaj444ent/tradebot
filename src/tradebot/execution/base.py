"""The execution boundary.

One interface, several implementations: a simulator for backtests, a paper
book against live quotes, and — not present, by design — a live adapter.
Everything above this line (strategy, risk, portfolio accounting) is identical
in all modes. That parity is the whole point: a paper run that diverges from
its backtest is then telling you something real about the market or your
assumptions, rather than about which code path you happened to take.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..types import Fill, Order


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Everything an adapter needs to price a fill.

    `ref_price` is the price orders execute *around* — in the backtest, the
    next bar's open. `bar_volume_notional` feeds the impact model.
    """

    ts: pd.Timestamp
    ref_price: pd.Series
    bar_volume_notional: pd.Series
    # Cash on hand when the bar opens. Orders were sized at the previous
    # bar's close, so by the time they execute the price has moved and a
    # fully-invested target can ask for more cash than exists. A real venue
    # partially fills or rejects; the simulator must do the same rather than
    # letting the books go negative.
    available_cash: float = float("inf")


class ExecutionAdapter(ABC):
    #: Human-readable mode, recorded in every result so a report can never be
    #: mistaken for a different mode's.
    mode: str = "abstract"

    @abstractmethod
    def execute(self, orders: list[Order], ctx: ExecutionContext) -> list[Fill]:
        """Attempt `orders`, returning the fills that actually happened.

        Implementations must be idempotent per `client_order_id`: submitting
        the same id twice fills once. A retry after a crash must not double a
        position.
        """
