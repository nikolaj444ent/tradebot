"""Backtest fill simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..costs import CostModel
from ..types import Fill, Order, Side
from .base import ExecutionAdapter, ExecutionContext


@dataclass(slots=True)
class SimExecution(ExecutionAdapter):
    """Fills every order at the reference price plus modelled slippage.

    Deliberately optimistic in exactly one way — it assumes you always get
    filled — and deliberately pessimistic about the price you get. For liquid
    spot majors at retail size that is a fair trade. For anything illiquid it
    is not, and the impact term is where you should push back.
    """

    costs: CostModel
    mode: str = "sim"
    _seen: set[str] = field(default_factory=set)

    def execute(self, orders: list[Order], ctx: ExecutionContext) -> list[Fill]:
        fills: list[Fill] = []
        # Cash is consumed as we go. Orders arrive sells-first, so proceeds
        # from a sale are available to the buys behind it in the same bar —
        # which is what actually happens on a spot venue.
        cash = ctx.available_cash
        for o in orders:
            if o.client_order_id in self._seen:
                # Idempotency: a replayed bar must not trade twice.
                continue
            ref = float(ctx.ref_price.get(o.symbol, float("nan")))
            if ref != ref or ref <= 0:
                continue  # no tradeable price for this symbol on this bar
            notional = o.qty * ref
            bar_vol = float(ctx.bar_volume_notional.get(o.symbol, 0.0))
            price, slip_bps = self.costs.effective_price(
                ref, o.side is Side.BUY, notional, bar_vol
            )

            qty = o.qty
            if o.side is Side.BUY:
                # Clip to what the cash actually covers, including the fee.
                affordable = cash / (price * (1.0 + self.costs.taker_fee_bps * 1e-4))
                if affordable < qty:
                    qty = max(affordable, 0.0)
                if qty * price < max(self.costs.min_notional, 1e-9):
                    # Not enough cash left for a valid order: no partial fill.
                    continue

            fee = self.costs.fee(qty * price)
            cash += (qty * price - fee) if o.side is Side.SELL else -(qty * price + fee)

            self._seen.add(o.client_order_id)
            fills.append(
                Fill(
                    client_order_id=o.client_order_id,
                    ts_filled=ctx.ts,
                    symbol=o.symbol,
                    side=o.side,
                    qty=qty,
                    price=price,
                    fee=fee,
                    slippage_bps=slip_bps,
                )
            )
        return fills
