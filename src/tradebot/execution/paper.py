"""Paper execution: simulated fills against live quotes.

This adapter is deliberately thin, and the reason is the design rule in
DESIGN.md section 2: everything above the execution boundary must be identical
in every mode. If paper trading re-implemented fill pricing, a divergence
between a backtest and its paper run would be ambiguous — market, or code
path? — and that ambiguity is the one thing paper trading exists to remove.

So the fill arithmetic here is not a copy of the simulator's. It *is* the
simulator's: `PaperExecution` holds a `SimExecution` and delegates to it. What
it adds is the two things a long-running process needs and a backtest does
not:

1. **Idempotency across restarts.** The simulator's seen-set lives in memory,
   which is enough for a process that runs once. A daemon that restarts
   mid-bar must not refill an order it already filled, so the seen-set is
   seeded from the journal on startup. Combined with the deterministic
   `client_order_id` (DESIGN.md section 6), a replayed bar regenerates the
   same ids and this rejects them.

2. **A written record, before the fact.** Every order is journalled before it
   is executed and every fill after, so a crash leaves evidence rather than a
   mystery.

What it is *not* is a venue. No order reaches an exchange from this file; the
fills are modelled, priced off real quotes. The honest claim for a paper run
is therefore "these are the fills our cost model predicts against prices that
really happened", which is a far stronger claim than a backtest makes and a
far weaker one than live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..costs import CostModel
from ..types import Fill, Order
from .base import ExecutionAdapter, ExecutionContext
from .sim import SimExecution


@dataclass(slots=True)
class PaperExecution(ExecutionAdapter):
    """Simulated fills against live quotes, with durable idempotency.

    `journal` is a `tradebot.state.Journal` (typed loosely to keep this module
    free of a circular import). Passing None gives an unjournalled paper
    adapter, which is useful in tests and wrong in deployment.
    """

    costs: CostModel
    journal: object | None = None
    #: Order ids already filled, recovered from the journal on startup.
    known_filled: frozenset[str] = frozenset()
    mode: str = "paper"
    _sim: SimExecution = field(init=False)

    def __post_init__(self) -> None:
        self._sim = SimExecution(costs=self.costs, _seen=set(self.known_filled))

    @property
    def filled_ids(self) -> set[str]:
        return self._sim._seen

    def execute(self, orders: list[Order], ctx: ExecutionContext) -> list[Fill]:
        if self.journal is not None:
            for order in orders:
                if order.client_order_id in self._sim._seen:
                    # Not an error: this is idempotency working. It happens when
                    # a bar is replayed after a restart, and it is worth a line
                    # in the log precisely because the alternative was a double
                    # fill.
                    self.journal.append(
                        "order_suppressed",
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        qty=order.qty,
                        detail="already filled; duplicate rejected",
                    )
                else:
                    self.journal.log_order(order)

        fills = self._sim.execute(orders, ctx)

        if self.journal is not None:
            for fill in fills:
                self.journal.log_fill(fill)
        return fills

    def venue_positions(self) -> dict[str, float] | None:
        """What the venue says we hold — unanswerable in paper mode.

        Returning None rather than our own numbers is the point. A paper run
        has no independent counterparty to ask, and handing back our own book
        dressed up as a venue reply would turn the startup reconciliation into
        a tautology that passes while telling you nothing. A live adapter
        overrides this with a real positions query; until one exists, the
        daemon says plainly that this check is unavailable.
        """
        return None
