"""Transaction cost model.

Costs are the single most common reason a promising backtest is worthless.
A 15 bps round-trip cost destroys any strategy whose average edge per trade
is 10 bps, and most simple technical signals are in exactly that range. So
this module is applied unconditionally in both the simulator and the paper
book, and there is no way to run a backtest with costs switched off.

Three components, all in basis points of traded notional:

  fee          Exchange commission. Crypto spot taker fees run ~10 bps
               (Binance) to ~26 bps (Kraken) before volume discounts.
  half_spread  You buy at the ask and sell at the bid; crossing costs you
               half the spread each way. ~1-3 bps for BTC/ETH, far more for
               small-cap alts.
  impact       Your own order moves the price. Modelled with the square-root
               law (Almgren et al.): impact scales with the square root of
               your participation rate in the bar's volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BPS = 1e-4


@dataclass(frozen=True, slots=True)
class CostModel:
    taker_fee_bps: float = 10.0
    half_spread_bps: float = 2.0
    # Impact in bps if a single order consumed 100% of the bar's volume.
    # The square-root law then gives, e.g., 1% participation -> coeff * 0.1.
    # 150 bps at full participation => 15 bps at 1%, which is deliberately
    # pessimistic for majors and roughly right for mid-caps.
    impact_bps_at_full_bar: float = 150.0
    # Exchange minimum order size in quote currency. Orders below this are
    # dropped rather than silently filled, which keeps rebalancing honest:
    # you cannot costlessly trade dust.
    min_notional: float = 10.0

    def impact_bps(self, notional: float, bar_notional_volume: float) -> float:
        """Price impact for trading `notional` into a bar that traded
        `bar_notional_volume`."""
        if bar_notional_volume <= 0 or notional <= 0:
            # No volume information: assume the worst rather than assume free.
            return self.impact_bps_at_full_bar
        participation = min(notional / bar_notional_volume, 1.0)
        return self.impact_bps_at_full_bar * math.sqrt(participation)

    def total_bps(self, notional: float, bar_notional_volume: float) -> float:
        """All-in one-way cost in bps."""
        return (
            self.taker_fee_bps
            + self.half_spread_bps
            + self.impact_bps(notional, bar_notional_volume)
        )

    def effective_price(
        self, ref_price: float, side_is_buy: bool, notional: float, bar_notional_volume: float
    ) -> tuple[float, float]:
        """Fill price after spread and impact, plus the slippage in bps.

        The exchange fee is charged separately (as `Fill.fee`) rather than
        folded into the price, so that fees and slippage stay separable in
        the post-trade report. Confusing the two hides which one is killing
        you.
        """
        slip_bps = self.half_spread_bps + self.impact_bps(notional, bar_notional_volume)
        direction = 1.0 if side_is_buy else -1.0
        return ref_price * (1.0 + direction * slip_bps * BPS), slip_bps

    def fee(self, notional: float) -> float:
        return abs(notional) * self.taker_fee_bps * BPS

    def round_trip_bps(self, notional: float, bar_notional_volume: float) -> float:
        """Convenience: the hurdle a signal must clear to be worth trading."""
        return 2.0 * self.total_bps(notional, bar_notional_volume)
