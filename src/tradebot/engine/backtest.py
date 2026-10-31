"""The backtest loop.

The ordering inside the loop is the whole design. Read it as a contract:

    at bar i:  the strategy sees history up to and including bar i's close
               the risk gate clamps what it asked for
    at bar i+1: the resulting orders fill at the OPEN, with costs
               equity is marked at bar i+1's close

Nothing decided at bar i can use any number from bar i+1. That is enforced by
construction — `MarketView` is built from a slice that ends at i, so bar i+1
is not reachable from inside a strategy — rather than by review. The companion
check is `tests/test_no_lookahead.py`, which runs strategies on a driftless
random walk: if the engine leaked the future, they would make money on data
that is unpredictable by definition.

The one-bar delay is not conservatism, it is the truth: at the moment a bar
closes you know its close, and the soonest you can act is the next bar.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from ..config import Config
from ..costs import CostModel
from ..execution.base import ExecutionAdapter, ExecutionContext
from ..execution.sim import SimExecution
from ..risk import RiskGate, RiskLimits
from ..strategy.base import Strategy
from ..types import BacktestResult, Fill, MarketView, OHLCV_FIELDS, RiskViolation
from .portfolio import Portfolio


def _array_view_at(
    arrays: dict[str, "np.ndarray"],
    index: pd.DatetimeIndex,
    columns: tuple[str, ...],
    upto: int,
) -> MarketView:
    """History ending at bar `upto`, inclusive. The future is simply absent.

    Backed by numpy rather than DataFrames: `arr[: upto + 1]` is a zero-copy
    numpy view, so this allocates nothing per bar, while still ending at
    `upto`. The guarantee is unchanged — indexing past the end raises, and a
    strategy would have to reach outside its arguments to see bar `upto + 1`.
    `MarketView` materialises DataFrames on demand for strategies that want
    them, so `view.close` still works.
    """
    return MarketView.from_arrays(
        {f: arrays[f][: upto + 1] for f in OHLCV_FIELDS}, index[: upto + 1], columns
    )


def run_backtest(
    frames: dict[str, pd.DataFrame],
    strategy: Strategy,
    *,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    initial_equity: float = 10_000.0,
    cash_buffer: float = 0.02,
    adapter: ExecutionAdapter | None = None,
    run_id: str | None = None,
    snapshot_id: str = "unknown",
) -> BacktestResult:
    close = frames["close"]
    volume = frames["volume"]
    n = len(close)

    costs = costs or CostModel()
    if limits is None:
        # Permissive but never levered. A per-symbol cap here would silently
        # make some portfolios unreachable: with a 0.25 cap and 3 symbols, an
        # equal-weight target of 0.333 each gets clamped to 0.25 each, and two
        # different strategies can come out byte-identical. Real limits belong
        # in the config file, where `Config.validate()` checks that the cap and
        # the universe size are mutually consistent.
        limits = RiskLimits(
            max_weight_per_symbol=1.0,
            max_gross_exposure=1.0,
            # A bare run_backtest is a measurement: breaches are recorded but
            # must not truncate the sample. Paper runs load limits from config,
            # where halt_on_breach stays True.
            halt_on_breach=False,
            symbol_allowlist=frozenset(close.columns),
        )
    adapter = adapter or SimExecution(costs=costs)
    run_id = run_id or uuid.uuid4().hex[:12]

    gate = RiskGate(limits=limits)
    pf = Portfolio(cash=initial_equity)

    warmup = max(int(strategy.warmup), 1)
    if n < warmup + 2:
        raise ValueError(
            f"{n} bars is too few for {strategy.name} (warmup {warmup}); "
            f"need at least {warmup + 2}."
        )

    # Notional volume per bar, for the impact model: price * units traded.
    bar_notional = (volume * close).fillna(0.0)

    # The panel is converted to numpy ONCE, here, and the loop below reads rows
    # out of it as plain dicts. The reason is measured rather than assumed:
    # profiling a 6,000-bar run showed each `df.iloc[i]` row costing ~48us in
    # pandas object construction, and the loop needs four of them per bar. The
    # arithmetic is identical — same float64 values, same order of operations —
    # so results are unchanged, which `tests/test_engine_golden.py` pins
    # exactly. Everything downstream (portfolio, risk gate, execution adapter)
    # only ever does `.get(symbol)`, `[symbol]` and `.items()`, which a dict
    # supports and a Series supports; nothing had to learn a new interface.
    symbols: tuple[str, ...] = tuple(close.columns)
    panel = {
        f: frames[f].to_numpy(dtype="float64", copy=False) for f in OHLCV_FIELDS
    }
    close_arr = panel["close"]
    open_arr = panel["open"]
    notional_arr = bar_notional.to_numpy(dtype="float64", copy=False)
    bar_index = close.index

    def row(arr: np.ndarray, i: int) -> dict[str, float]:
        return dict(zip(symbols, arr[i].tolist()))

    equity_ts: list[pd.Timestamp] = []
    equity_vals: list[float] = []
    weight_rows: list[dict[str, float]] = []
    all_fills: list[Fill] = []
    halted_at: pd.Timestamp | None = None
    halt_reason: str | None = None

    n_clamped_decisions = 0
    start = warmup - 1
    # Stop at n-2: the final decision bar needs a bar after it to fill in.
    for i in range(start, n - 1):
        ts_decide = close.index[i]
        ts_fill = close.index[i + 1]
        px_decide = row(close_arr, i)
        px_fill_ref = row(open_arr, i + 1)
        px_mark = row(close_arr, i + 1)

        equity = pf.equity(px_decide)
        gate.observe_equity(ts_decide, equity)

        ctx = ExecutionContext(
            ts=ts_fill,
            ref_price=px_fill_ref,
            bar_volume_notional=row(notional_arr, i + 1),
            available_cash=pf.cash,
        )

        reason = gate.halt_reason(ts_decide, equity)
        if reason is not None:
            # Flatten at the next open and stop. A halted run does not quietly
            # resume: restarting is a human decision.
            orders = pf.flatten_orders(
                px_fill_ref, ts=ts_decide, run_id=run_id, reason=f"halt: {reason}"
            )
            for f in adapter.execute(orders, ctx):
                pf.apply_fill(f)
                all_fills.append(f)
            halted_at, halt_reason = ts_decide, reason
            gate.violations.append(
                RiskViolation(ts=ts_decide, rule="halt", detail=reason)
            )
            equity_ts.append(ts_fill)
            equity_vals.append(pf.equity(px_mark))
            weight_rows.append({})
            break

        view = _array_view_at(panel, bar_index, symbols, i)
        raw = strategy.target_weights(view)
        before = len(gate.violations)
        targets = gate.clamp(ts_decide, raw, pf.weights(px_decide))
        if len(gate.violations) > before:
            n_clamped_decisions += 1

        orders = pf.orders_to_reach(
            targets,
            px_decide,
            ts=ts_decide,
            run_id=run_id,
            min_notional=costs.min_notional,
            taker_fee_bps=costs.taker_fee_bps,
            cash_buffer=cash_buffer,
            reason=strategy.name,
        )
        for f in adapter.execute(orders, ctx):
            pf.apply_fill(f)
            all_fills.append(f)

        equity_ts.append(ts_fill)
        equity_vals.append(pf.equity(px_mark))
        weight_rows.append(pf.weights(px_mark))

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_ts), name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    weights = pd.DataFrame(weight_rows, index=equity.index).fillna(0.0)

    gross_traded = sum(f.notional for f in all_fills)

    # How often the gate overrode the strategy. This is reported, not hidden:
    # a high clamp rate means the backtest describes the gate's behaviour
    # rather than the strategy's, and the result should not be trusted as a
    # measurement of the idea.
    n_decisions = max(len(equity), 1)
    by_rule: dict[str, int] = {}
    for v in gate.violations:
        by_rule[v.rule] = by_rule.get(v.rule, 0) + 1
    # Two different numbers, previously conflated into one mislabelled ratio.
    # A single decision can trip several rules, so counting violations and
    # calling the result a percentage of decisions produced "overrode 167% of
    # decisions". clamp_rate is now a genuine fraction in [0, 1].
    clamp_rate = n_clamped_decisions / n_decisions
    violations_per_decision = (
        sum(c for r, c in by_rule.items() if r not in ("halt", "breach_not_halted"))
        / n_decisions
    )
    return BacktestResult(
        equity=equity,
        returns=returns,
        fills=all_fills,
        violations=list(gate.violations),
        weights=weights,
        halted_at=halted_at,
        halt_reason=halt_reason,
        meta={
            "strategy": strategy.describe(),
            "params": strategy.params(),
            "n_params": len(strategy.params()),
            "run_id": run_id,
            "mode": adapter.mode,
            "snapshot_id": snapshot_id,
            "initial_equity": initial_equity,
            "final_equity": float(equity.iloc[-1]) if len(equity) else initial_equity,
            "fees_paid": pf.fees_paid,
            "gross_traded_notional": gross_traded,
            # Costs as a share of starting capital: the number that tells you
            # whether you are running a strategy or a fee-generation scheme.
            "cost_drag_pct_of_initial": pf.fees_paid / initial_equity if initial_equity else 0.0,
            "first_bar": str(close.index[0]),
            "last_bar": str(close.index[-1]),
            "n_bars": n,
            "violations_by_rule": by_rule,
            "clamp_rate": clamp_rate,
            "violations_per_decision": violations_per_decision,
            # Bars actually traded, which is less than n_bars when a halt
            # truncates the run. Every metric below is computed on this.
            "effective_bars": len(equity),
            "truncated": len(equity) < (n - warmup),
        },
    )


def run_from_config(
    frames: dict[str, pd.DataFrame],
    strategy: Strategy,
    cfg: Config,
    *,
    snapshot_id: str = "unknown",
    run_id: str | None = None,
) -> BacktestResult:
    return run_backtest(
        frames,
        strategy,
        costs=cfg.costs,
        limits=cfg.risk,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
        run_id=run_id,
        snapshot_id=snapshot_id,
    )
