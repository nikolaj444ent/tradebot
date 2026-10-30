"""The risk gate.

Everything the strategy wants to do passes through here, in both backtest and
paper trading. The gate is not advisory: it clamps targets and it can halt the
run outright. It sits between strategy and execution precisely so that a bug
in a strategy — or an overconfident one — cannot express itself as an
unbounded position.

Design rule: the gate never *raises* position size and never invents a trade.
It only ever reduces exposure or stops trading. That makes it safe to apply
blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .types import RiskViolation, TargetWeights


@dataclass(frozen=True, slots=True)
class RiskLimits:
    # Spot, long-only: no single asset may exceed this share of equity.
    max_weight_per_symbol: float = 0.25
    # Sum of absolute weights. 1.0 means fully invested, never levered.
    max_gross_exposure: float = 1.0
    # Shorting requires margin, which spot accounts do not have. Keeping this
    # False means a sign error in a strategy becomes a clamp, not a short.
    allow_short: bool = False
    # Hard stop: if equity falls this far below the day's opening equity, the
    # run flattens and halts until a human restarts it.
    #
    # Calibrate this to the asset class or it will do more harm than good. A
    # long-only, fully-invested crypto book tracks the market, and crypto
    # routinely moves 5-10% in a day — so a 5% stop fires on *market* moves,
    # not on strategy errors, and permanently exits at the worst possible
    # moment. Set to 5% on 1h crypto data, a buy-and-hold benchmark halts
    # within the first day and every backtest silently truncates.
    #
    # So treat these two as a circuit breaker against bugs and runaway
    # behaviour — a strategy looping on orders, a sign error, a bad feed —
    # and not as a risk-management strategy. Position sizing is what manages
    # market risk.
    max_daily_loss_pct: float = 0.15
    # Hard stop on cumulative drawdown from the run's high-water mark. Crypto
    # buy-and-hold has repeatedly drawn down >70%, so a 20% limit here is a
    # decision to exit most bear markets permanently. That may be what you
    # want; it should be deliberate.
    max_drawdown_pct: float = 0.40
    # Research escape hatch. With halt_on_breach=False the gate still records
    # every breach but does not stop the run, so a backtest can measure the
    # full sample instead of truncating at the first breach. Paper and live
    # runs must leave this True — the whole point of a circuit breaker is that
    # it actually breaks the circuit.
    halt_on_breach: bool = True
    # Turnover cap per bar, as a fraction of equity. Catches a strategy that
    # starts thrashing: round-tripping the book every bar is a cost machine.
    max_turnover_per_bar: float = 1.0
    # Only these symbols may ever be traded. An empty set means "no allowlist
    # configured", which the gate treats as a configuration error, not as
    # permission for anything.
    symbol_allowlist: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0 < self.max_weight_per_symbol <= 1:
            raise ValueError("max_weight_per_symbol must be in (0, 1]")
        if self.max_gross_exposure > 1.0 and not self.allow_short:
            raise ValueError(
                "max_gross_exposure > 1 implies leverage, which a spot account "
                "does not have. Set allow_short/margin explicitly if you mean it."
            )
        if not 0 < self.max_daily_loss_pct < 1:
            raise ValueError("max_daily_loss_pct must be in (0, 1)")


@dataclass(slots=True)
class RiskGate:
    limits: RiskLimits
    violations: list[RiskViolation] = field(default_factory=list)
    _day_open_equity: float | None = None
    _day: object | None = None
    _high_water: float = 0.0

    def _record(self, ts: pd.Timestamp, rule: str, detail: str, symbol: str | None = None) -> None:
        self.violations.append(RiskViolation(ts=ts, rule=rule, detail=detail, symbol=symbol))

    def observe_equity(self, ts: pd.Timestamp, equity: float) -> None:
        """Track the daily and all-time reference points the stops key off."""
        day = ts.normalize() if isinstance(ts, pd.Timestamp) else None
        if self._day != day:
            self._day = day
            self._day_open_equity = equity
        self._high_water = max(self._high_water, equity)

    def breach_reason(self, ts: pd.Timestamp, equity: float) -> str | None:
        """Describe a limit breach, whether or not it will halt the run."""
        if self._day_open_equity and self._day_open_equity > 0:
            loss = 1.0 - equity / self._day_open_equity
            if loss >= self.limits.max_daily_loss_pct:
                return (
                    f"daily loss {loss:.2%} >= limit "
                    f"{self.limits.max_daily_loss_pct:.2%}"
                )
        if self._high_water > 0:
            dd = 1.0 - equity / self._high_water
            if dd >= self.limits.max_drawdown_pct:
                return f"drawdown {dd:.2%} >= limit {self.limits.max_drawdown_pct:.2%}"
        return None

    def halt_reason(self, ts: pd.Timestamp, equity: float) -> str | None:
        """A reason to stop trading entirely, or None to continue.

        In research mode (halt_on_breach=False) the breach is still recorded
        by the caller but never halts, so the sample stays intact.
        """
        reason = self.breach_reason(ts, equity)
        if reason is None:
            return None
        if not self.limits.halt_on_breach:
            self._record(ts, "breach_not_halted", reason)
            return None
        return reason

    def clamp(
        self,
        ts: pd.Timestamp,
        raw: TargetWeights,
        current_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Reduce `raw` to something the limits permit. Never increases risk."""
        if not self.limits.symbol_allowlist:
            raise ValueError(
                "risk.symbol_allowlist is empty. Configure the universe explicitly; "
                "an empty allowlist is a config error, not permission to trade anything."
            )

        out: dict[str, float] = {}
        for sym, w in raw.items():
            if sym not in self.limits.symbol_allowlist:
                self._record(ts, "symbol_allowlist", f"{sym} not in allowlist", sym)
                continue
            if w != w:  # NaN
                self._record(ts, "nan_weight", f"{sym} target was NaN -> 0", sym)
                continue
            if w < 0 and not self.limits.allow_short:
                self._record(ts, "no_short", f"{sym} target {w:+.3f} -> 0", sym)
                w = 0.0
            cap = self.limits.max_weight_per_symbol
            if abs(w) > cap:
                self._record(ts, "max_weight_per_symbol", f"{sym} {w:+.3f} -> {cap:+.3f}", sym)
                w = cap if w > 0 else -cap
            if w != 0.0:
                out[sym] = float(w)

        gross = sum(abs(w) for w in out.values())
        # Tolerance matters: a fully-invested book sums to 1.0 with float
        # noise, and without this every single bar logs a "scaled by 1.000"
        # no-op violation, which buries the real ones.
        if gross > self.limits.max_gross_exposure * (1 + 1e-9) and gross > 0:
            scale = self.limits.max_gross_exposure / gross
            self._record(
                ts, "max_gross_exposure", f"gross {gross:.3f} scaled by {scale:.3f}"
            )
            out = {s: w * scale for s, w in out.items()}

        if current_weights is not None:
            turnover = sum(
                abs(out.get(s, 0.0) - current_weights.get(s, 0.0))
                for s in set(out) | set(current_weights)
            )
            cap = self.limits.max_turnover_per_bar
            if turnover > cap > 0:
                # Move only part of the way toward the target rather than
                # refusing the trade: a partial step is closer to the intent
                # than standing still.
                scale = cap / turnover
                self._record(
                    ts, "max_turnover_per_bar", f"turnover {turnover:.3f} scaled by {scale:.3f}"
                )
                out = {
                    s: current_weights.get(s, 0.0)
                    + (out.get(s, 0.0) - current_weights.get(s, 0.0)) * scale
                    for s in set(out) | set(current_weights)
                }
                out = {s: w for s, w in out.items() if abs(w) > 1e-12}

        return out
