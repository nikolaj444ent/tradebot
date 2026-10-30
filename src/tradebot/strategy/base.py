"""The strategy contract.

A strategy is a pure function of the past: `target_weights(view)` maps the
history it can see to the portfolio it wants to hold. It returns *targets*,
not orders — position sizing, cash management, costs and risk are the
engine's job, and keeping them out of the strategy is what lets the identical
strategy object run in a backtest, a paper run, and (one day, deliberately)
live.

Two rules, both load-bearing:

1. No state that is not derivable from `view`. If a strategy remembers
   something across calls, then replaying a bar after a restart gives a
   different answer, the backtest stops describing the live system, and
   crash recovery silently changes behaviour.
2. No I/O. No network, no clock, no files. A strategy that reads the wall
   clock behaves differently in backtest and paper for reasons that are
   invisible in its results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass

from ..types import MarketView


class Strategy(ABC):
    """Base class. Subclasses are expected to be frozen dataclasses."""

    #: Bars of history required before the first decision. The engine will not
    #: ask for targets until it can supply at least this many.
    warmup: int = 1

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def target_weights(self, view: MarketView) -> dict[str, float]:
        """Desired portfolio as symbol -> fraction of equity.

        Omitted symbols mean zero. Values are fractions of total equity, so
        {"BTC/USDT": 0.5} with 10k equity means hold 5k of BTC and the rest in
        cash. The risk gate will clamp whatever is returned, so a strategy
        need not enforce limits itself — but it should not rely on the gate to
        fix a sign error either.
        """

    def params(self) -> dict:
        """Parameters, for logging and walk-forward reporting.

        The parameter count matters: every tunable is a chance to overfit, and
        the walk-forward report prints this so the number is impossible to
        ignore.
        """
        if is_dataclass(self):
            return {
                k: v
                for k, v in asdict(self).items()
                if not k.startswith("_") and k != "warmup"
            }
        return {}

    def describe(self) -> str:
        p = self.params()
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(p.items()))
        return f"{self.name}({inner})"
