"""Position and cash accounting.

The only mutable trading state in the system. Everything else is frozen, so
when the books are wrong, this is where to look.

Spot semantics throughout: you cannot sell what you do not hold, and you
cannot spend cash you do not have. Both are enforced here rather than assumed,
because an accounting model that quietly permits negative cash produces
backtests with free leverage — one of the most common ways a paper result
turns out to be impossible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from ..types import Fill, Order, Side


# Floor on any order's notional, independent of the exchange minimum. A
# configured min_notional of 0 (used when measuring a zero-cost counterfactual)
# would otherwise let a delta of exactly 0.0 become a zero-quantity order,
# which Order rejects.
_EPS_NOTIONAL = 1e-9


def client_order_id(run_id: str, ts: pd.Timestamp, symbol: str, side: Side) -> str:
    """Deterministic idempotency key.

    Derived from the decision, not from a counter or a clock, so that replaying
    a bar after a crash regenerates the *same* id. The venue (or our paper
    book) then rejects the duplicate instead of doubling the position. A random
    uuid here would be a silent double-fill waiting for a restart.
    """
    raw = f"{run_id}|{ts.isoformat()}|{symbol}|{side.value}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


@dataclass(slots=True)
class Portfolio:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    fees_paid: float = 0.0
    n_fills: int = 0

    def position_qty(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def equity(self, prices: pd.Series) -> float:
        total = self.cash
        for sym, qty in self.positions.items():
            px = prices.get(sym)
            if px is not None and px == px:
                total += qty * float(px)
        return total

    def weights(self, prices: pd.Series) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        out = {}
        for sym, qty in self.positions.items():
            px = prices.get(sym)
            if px is not None and px == px and qty != 0:
                out[sym] = qty * float(px) / eq
        return out

    def apply_fill(self, fill: Fill) -> None:
        held = self.positions.get(fill.symbol, 0.0)
        if fill.side is Side.BUY:
            cost = fill.qty * fill.price + fill.fee
            if cost > self.cash + 1e-6:
                raise ValueError(
                    f"buy of {fill.symbol} costs {cost:.2f} but cash is "
                    f"{self.cash:.2f}. Order sizing let cash go negative — this is "
                    f"a bug in orders_to_reach, not a market event."
                )
            self.cash -= cost
            self.positions[fill.symbol] = held + fill.qty
        else:
            if fill.qty > held + 1e-9:
                raise ValueError(
                    f"sell of {fill.qty} {fill.symbol} but only {held} held. Spot "
                    f"accounts cannot short; this is a sizing bug."
                )
            self.cash += fill.qty * fill.price - fill.fee
            self.positions[fill.symbol] = held - fill.qty

        if abs(self.positions[fill.symbol]) < 1e-12:
            self.positions.pop(fill.symbol, None)
        self.fees_paid += fill.fee
        self.n_fills += 1

    def orders_to_reach(
        self,
        targets: dict[str, float],
        prices: pd.Series,
        *,
        ts: pd.Timestamp,
        run_id: str,
        min_notional: float,
        taker_fee_bps: float,
        cash_buffer: float = 0.0,
        reason: str = "",
    ) -> list[Order]:
        """Orders that move the book from where it is toward `targets`.

        Sells are emitted before buys and buys are scaled to the cash that will
        actually exist once the sells settle. That ordering is not cosmetic: a
        rebalance that assumes simultaneous settlement is assuming free
        intraday credit.
        """
        equity = self.equity(prices)
        if equity <= 0:
            return []

        deltas: dict[str, float] = {}
        for sym in set(targets) | set(self.positions):
            px = prices.get(sym)
            if px is None or px != px or float(px) <= 0:
                continue
            px = float(px)
            current_notional = self.positions.get(sym, 0.0) * px
            target_notional = float(targets.get(sym, 0.0)) * equity
            delta = target_notional - current_notional
            # Below the exchange minimum there is no order to place. Skipping
            # these also stops a strategy accumulating fees on dust.
            if abs(delta) >= max(min_notional, _EPS_NOTIONAL):
                deltas[sym] = delta

        sells = {s: d for s, d in deltas.items() if d < 0}
        buys = {s: d for s, d in deltas.items() if d > 0}

        fee_frac = taker_fee_bps * 1e-4
        proceeds = sum(-d for d in sells.values()) * (1.0 - fee_frac)
        available = self.cash + proceeds - cash_buffer * equity
        wanted = sum(buys.values()) * (1.0 + fee_frac)

        scale = 1.0
        if wanted > 0 and wanted > available:
            scale = max(available, 0.0) / wanted

        orders: list[Order] = []
        for sym, delta in sorted(sells.items()):
            px = float(prices[sym])
            qty = min(-delta / px, self.positions.get(sym, 0.0))
            if qty <= 0 or qty * px < max(min_notional, _EPS_NOTIONAL):
                continue
            orders.append(
                Order(
                    client_order_id=client_order_id(run_id, ts, sym, Side.SELL),
                    ts_decided=ts,
                    symbol=sym,
                    side=Side.SELL,
                    qty=qty,
                    reason=reason,
                )
            )
        for sym, delta in sorted(buys.items()):
            px = float(prices[sym])
            qty = (delta * scale) / px
            if qty <= 0 or qty * px < max(min_notional, _EPS_NOTIONAL):
                continue
            orders.append(
                Order(
                    client_order_id=client_order_id(run_id, ts, sym, Side.BUY),
                    ts_decided=ts,
                    symbol=sym,
                    side=Side.BUY,
                    qty=qty,
                    reason=reason,
                )
            )
        return orders

    def flatten_orders(
        self, prices: pd.Series, *, ts: pd.Timestamp, run_id: str, reason: str
    ) -> list[Order]:
        """Sell everything. Used when the risk gate halts the run."""
        orders = []
        for sym, qty in sorted(self.positions.items()):
            px = prices.get(sym)
            if px is None or px != px or qty <= 0:
                continue
            orders.append(
                Order(
                    client_order_id=client_order_id(run_id, ts, sym, Side.SELL),
                    ts_decided=ts,
                    symbol=sym,
                    side=Side.SELL,
                    qty=qty,
                    reason=reason,
                )
            )
        return orders
