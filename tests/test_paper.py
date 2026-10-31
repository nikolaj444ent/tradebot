"""The paper-trading daemon.

The first test is the important one. Everything else here checks that the
daemon survives being restarted, tampered with, or fed bad data; the parity
test checks the claim the whole design rests on — that a paper run and a
backtest of the same strategy on the same bars are the *same computation*, so
that when they later disagree on live data, the disagreement is information
about the market rather than about which code path ran.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tradebot.config import BacktestConfig, Config, DataConfig
from tradebot.costs import CostModel
from tradebot.data.quotes import (
    CcxtQuoteSource,
    Quote,
    ReplayQuoteSource,
    StaleFeed,
    _drop_unclosed,
)
from tradebot.data.synthetic import synthetic_panel
from tradebot.engine.backtest import run_backtest
from tradebot.engine.paper import (
    PaperStartupRefused,
    result_from_journal,
    run_paper_loop,
    start_paper_trader,
)
from tradebot.engine.portfolio import Portfolio
from tradebot.risk import RiskLimits
from tradebot.state import Journal, reconcile
from tradebot.strategy import BuyAndHold, CrossSectionalMomentum
from tradebot.execution.base import ExecutionContext
from tradebot.execution.paper import PaperExecution
from tradebot.types import Order, Side

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")


def make_config(**overrides) -> Config:
    risk = RiskLimits(
        max_weight_per_symbol=0.4,
        max_gross_exposure=1.0,
        max_daily_loss_pct=0.15,
        max_drawdown_pct=0.40,
        max_turnover_per_bar=1.0,
        halt_on_breach=True,
        symbol_allowlist=frozenset(SYMBOLS),
    )
    cfg = Config(
        data=DataConfig(symbols=SYMBOLS, timeframe="1h"),
        costs=CostModel(),
        risk=risk,
        backtest=BacktestConfig(initial_equity=10_000.0, cash_buffer=0.02),
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    cfg.validate()
    return cfg


def wide_config() -> Config:
    """Limits loose enough that the circuit breaker stays out of the way.

    Worth stating why this exists: on synthetic 1%-per-bar data the configured
    15% daily stop genuinely fires within a couple of days, exactly as
    DESIGN.md section 5 warns. That is correct behaviour and it is tested
    elsewhere; tests about reconciliation and bookkeeping need a run that is
    still holding a position at the end, so they widen the stops rather than
    quietly depend on not tripping them.
    """
    return make_config(
        risk=RiskLimits(
            max_weight_per_symbol=0.4,
            max_gross_exposure=1.0,
            max_daily_loss_pct=0.90,
            max_drawdown_pct=0.95,
            halt_on_breach=True,
            symbol_allowlist=frozenset(SYMBOLS),
        )
    )


def panel(kind="trending", n=900, seed=1, **kwargs):
    return synthetic_panel(SYMBOLS, n, "1h", seed=seed, kind=kind, **kwargs)


def drive(cfg, strategy, journal, frames, *, max_bars=None, volume_mode="actual_next"):
    trader = start_paper_trader(cfg, strategy, journal)
    source = ReplayQuoteSource(
        frames,
        timeframe=cfg.data.timeframe,
        start_index=max(int(strategy.warmup), 1) - 1,
        volume_mode=volume_mode,
    )
    outcome = run_paper_loop(trader, source, max_bars=max_bars)
    return trader, outcome


# --------------------------------------------------------------------- parity


@pytest.mark.parametrize(
    "kind,strategy_factory",
    [
        ("trending", lambda: CrossSectionalMomentum(lookback=72, n_long=2, rebalance_every=24)),
        ("random_walk", lambda: CrossSectionalMomentum(lookback=48, n_long=1, rebalance_every=12)),
        ("mean_reverting", BuyAndHold),
    ],
)
def test_replayed_paper_run_reproduces_the_backtest(tmp_path, kind, strategy_factory):
    """The parity claim, as an assertion rather than a promise.

    Same bars, same strategy, same costs, same gate: the paper loop must
    produce the backtest's equity curve bar for bar, and the same fills. If
    this ever fails, every paper-versus-backtest comparison downstream is
    measuring a code difference and nobody can tell.
    """
    cfg = make_config()
    frames = panel(kind=kind)

    expected = run_backtest(
        frames,
        strategy_factory(),
        costs=cfg.costs,
        limits=cfg.risk,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
    )

    journal = Journal(tmp_path, "parity")
    drive(cfg, strategy_factory(), journal, frames)
    actual = journal.equity_curve()

    assert len(actual) == len(expected.equity), (
        f"paper recorded {len(actual)} marks, backtest {len(expected.equity)}"
    )
    assert list(actual.index) == list(expected.equity.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.equity.to_numpy(), rtol=1e-12)

    # And the trades themselves, not just the curve they produce.
    def trades(fills):
        return [(f.symbol, f.side.value, round(f.qty, 10), round(f.price, 8)) for f in fills]

    assert trades(journal.fills()) == trades(expected.fills)


def test_parity_holds_for_a_run_that_halts(tmp_path):
    """A halt must land on the same bar, for the same reason, in both modes."""
    cfg = make_config(
        risk=RiskLimits(
            max_weight_per_symbol=1.0,
            max_gross_exposure=1.0,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.05,
            halt_on_breach=True,
            symbol_allowlist=frozenset(SYMBOLS),
        )
    )
    frames = panel(kind="random_walk", n=400, seed=7, drift_per_bar=-0.004)

    expected = run_backtest(
        frames,
        BuyAndHold(),
        costs=cfg.costs,
        limits=cfg.risk,
        initial_equity=cfg.backtest.initial_equity,
        cash_buffer=cfg.backtest.cash_buffer,
    )
    assert expected.halted_at is not None, "test data should trip the stop"

    journal = Journal(tmp_path, "halting")
    _, outcome = drive(cfg, BuyAndHold(), journal, frames)

    assert outcome.halted
    assert outcome.bar_ts == expected.halted_at
    assert outcome.halt_reason == expected.halt_reason

    # A mark is "equity at this bar's close", so a halt must not add a second
    # one at a timestamp that already has it.
    curve = journal.equity_curve()
    assert not curve.index.has_duplicates
    assert curve.index.is_monotonic_increasing
    # The curve up to the halt is the backtest's, bar for bar; the backtest
    # adds one final point at the next bar's close, which a halted live run
    # never reaches.
    shared = min(len(curve), len(expected.equity))
    assert list(curve.index[:shared]) == list(expected.equity.index[:shared])
    np.testing.assert_allclose(
        curve.to_numpy()[:shared], expected.equity.to_numpy()[:shared], rtol=1e-12
    )
    flattened = [e for e in journal.replay() if e.get("event") == "halt_flattened"]
    assert len(flattened) == 1
    assert flattened[0]["positions_before"]


# ------------------------------------------------------------------ restarts


def test_restart_resumes_rather_than_retrading(tmp_path):
    """Re-presenting bars already processed must change nothing.

    This is the crash-recovery case: the daemon comes back up, the quote source
    hands it history that includes bars it has already acted on, and it must
    neither re-mark equity nor re-trade.
    """
    cfg = make_config()
    frames = panel(n=600)
    strategy = CrossSectionalMomentum(lookback=48, n_long=2, rebalance_every=12)

    journal = Journal(tmp_path, "restart")
    first, _ = drive(cfg, strategy, journal, frames, max_bars=200)
    curve_before = journal.equity_curve()
    fills_before = journal.fills()
    assert len(fills_before) > 0, "test needs a run that actually traded"

    # A fresh process, same journal, replayed from the very beginning.
    second = start_paper_trader(cfg, strategy, Journal(tmp_path, "restart"))
    assert second.last_bar_ts == first.last_bar_ts
    assert second.portfolio.positions == first.portfolio.positions
    assert second.portfolio.cash == pytest.approx(first.portfolio.cash)

    journal2 = Journal(tmp_path, "restart")
    trader2 = start_paper_trader(cfg, strategy, journal2)
    source = ReplayQuoteSource(frames, start_index=strategy.warmup - 1)
    outcome = run_paper_loop(trader2, source, max_bars=200)

    curve_after = journal2.equity_curve()
    fills_after = journal2.fills()
    # Bars already seen are skipped, so the only new marks are genuinely new
    # bars; none of the old ones are duplicated.
    assert not curve_after.index.has_duplicates
    assert list(curve_after.index[: len(curve_before)]) == list(curve_before.index)
    np.testing.assert_allclose(
        curve_after.to_numpy()[: len(curve_before)], curve_before.to_numpy(), rtol=1e-12
    )
    assert len(fills_after) >= len(fills_before)
    assert outcome is not None


def test_a_replayed_bar_cannot_double_fill(tmp_path):
    """Two independent defences against refilling, tested separately.

    The first is accounting: the portfolio is rebuilt from recorded fills, so a
    re-decided bar finds the target already held and emits no order at all. The
    second is the idempotency key, which is what remains if the first ever
    fails — a crash between placing an order and recording its fill would leave
    exactly that gap, and on a real venue the order would already be live.
    """
    cfg = make_config()
    frames = panel(n=400)
    strategy = BuyAndHold()

    journal = Journal(tmp_path, "idem")
    trader = start_paper_trader(cfg, strategy, journal)
    quotes = list(ReplayQuoteSource(frames, start_index=0).quotes())[:3]
    for quote in quotes:
        trader.on_quote(quote)
    first_fills = journal.fills()
    assert first_fills, "BuyAndHold should buy on its first bar"
    positions_before = dict(trader.portfolio.positions)

    # Defence one. Restart, rewind the bar watermark to force a re-decision of
    # bars already traded, and confirm nothing is bought twice.
    journal2 = Journal(tmp_path, "idem")
    trader2 = start_paper_trader(cfg, strategy, journal2)
    assert trader2.adapter.filled_ids == {f.client_order_id for f in first_fills}
    trader2.last_bar_ts = None
    for quote in quotes:
        trader2.on_quote(quote)

    ids = [f.client_order_id for f in journal2.fills()]
    assert len(ids) == len(set(ids)), "a replayed bar produced a second fill"
    for sym, qty in positions_before.items():
        assert trader2.portfolio.position_qty(sym) == pytest.approx(qty, rel=1e-9)

    # Defence two. The same order id, handed to a restarted adapter that has
    # seen it before, must not fill — and must leave a trace.
    order = Order(
        client_order_id=first_fills[0].client_order_id,
        ts_decided=quotes[0].bar_ts,
        symbol=first_fills[0].symbol,
        side=Side.BUY,
        qty=1.0,
    )
    journal3 = Journal(tmp_path, "idem")
    adapter = PaperExecution(
        costs=cfg.costs,
        journal=journal3,
        known_filled=frozenset({order.client_order_id}),
    )
    ctx = ExecutionContext(
        ts=quotes[0].fill_ts,
        ref_price=quotes[0].ref_price,
        bar_volume_notional=quotes[0].bar_volume_notional,
        available_cash=1e9,
    )
    assert adapter.execute([order], ctx) == []
    suppressed = [e for e in journal3.replay() if e.get("event") == "order_suppressed"]
    assert suppressed, "the duplicate should be recorded, not silently dropped"
    assert suppressed[-1]["client_order_id"] == order.client_order_id


def test_drawdown_high_water_mark_survives_a_restart(tmp_path):
    """A crash-looping process must not keep resetting its own stops.

    The drawdown stop keys off a high-water mark. Rebuild the gate from nothing
    on each start and equity could fall arbitrarily far in a series of short
    runs without ever breaching, because every restart would declare the
    current, lower equity to be the new peak.
    """
    cfg = make_config()
    journal = Journal(tmp_path, "hwm")
    # Fabricate a run that peaked at 20k and has since fallen to 12k.
    journal.append("run_started", run_id="x", initial_equity=10_000.0)
    for ts, equity in [
        ("2024-01-01T00:00:00+00:00", 10_000.0),
        ("2024-01-02T00:00:00+00:00", 20_000.0),
        ("2024-01-03T00:00:00+00:00", 12_000.0),
    ]:
        journal.log_mark(pd.Timestamp(ts), equity, {})

    trader = start_paper_trader(cfg, BuyAndHold(), Journal(tmp_path, "hwm"))
    assert trader.gate._high_water == pytest.approx(20_000.0)
    # 12k against a 20k peak is a 40% drawdown: already at the limit.
    assert trader.gate.breach_reason(pd.Timestamp("2024-01-03T01:00:00+00:00"), 12_000.0)


# ------------------------------------------------------- startup refusals


def test_refuses_to_start_when_the_breaker_is_disabled(tmp_path):
    cfg = make_config(
        risk=RiskLimits(
            max_weight_per_symbol=0.4,
            halt_on_breach=False,
            symbol_allowlist=frozenset(SYMBOLS),
        )
    )
    with pytest.raises(PaperStartupRefused, match="halt_on_breach"):
        start_paper_trader(cfg, BuyAndHold(), Journal(tmp_path, "nobreaker"))


def test_a_halt_is_sticky_until_a_human_clears_it(tmp_path):
    cfg = make_config(
        risk=RiskLimits(
            max_weight_per_symbol=1.0,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.03,
            halt_on_breach=True,
            symbol_allowlist=frozenset(SYMBOLS),
        )
    )
    frames = panel(kind="random_walk", n=300, seed=11, drift_per_bar=-0.005)

    journal = Journal(tmp_path, "sticky")
    trader, outcome = drive(cfg, BuyAndHold(), journal, frames)
    assert outcome.halted

    # Flattened: a halt exits the position, it does not just stop deciding.
    assert trader.portfolio.positions == {}
    assert journal.halt_path.exists()

    with pytest.raises(PaperStartupRefused, match="halted previously"):
        start_paper_trader(cfg, BuyAndHold(), Journal(tmp_path, "sticky"))

    journal.clear_halt()
    resumed = start_paper_trader(cfg, BuyAndHold(), Journal(tmp_path, "sticky"))
    assert resumed.portfolio.positions == {}


def test_a_tampered_book_blocks_the_start(tmp_path):
    """The journal is the authority; disagreement means state is unknown."""
    cfg = wide_config()
    frames = panel(n=300)
    journal = Journal(tmp_path, "tamper")
    drive(cfg, BuyAndHold(), journal, frames, max_bars=50)

    book = journal.read_book()
    assert book and book["positions"]
    symbol = sorted(book["positions"])[0]
    book["positions"][symbol] = float(book["positions"][symbol]) * 1.5
    journal.book_path.write_text(json.dumps(book, default=str))

    with pytest.raises(PaperStartupRefused, match="does NOT reconcile"):
        start_paper_trader(cfg, BuyAndHold(), Journal(tmp_path, "tamper"))
    # And it halts, rather than leaving an unexplained state for the next start.
    assert Journal(tmp_path, "tamper").halted() is not None


def test_mixing_two_capital_bases_in_one_journal_is_refused(tmp_path):
    cfg = make_config()
    journal = Journal(tmp_path, "capital")
    journal.append("run_started", run_id="x", initial_equity=10_000.0)
    bigger = dataclasses.replace(
        cfg, backtest=BacktestConfig(initial_equity=50_000.0, cash_buffer=0.02)
    )
    with pytest.raises(ValueError, match="initial_equity"):
        start_paper_trader(bigger, BuyAndHold(), Journal(tmp_path, "capital"))


# --------------------------------------------------------- state and journal


def test_journal_rebuilds_the_portfolio_from_fills_alone(tmp_path):
    cfg = wide_config()
    frames = panel(n=500)
    # BuyAndHold rather than momentum: momentum is allowed to sit in cash when
    # nothing is rising, and a flat book is a weaker test than a held one.
    strategy = BuyAndHold()
    journal = Journal(tmp_path, "rebuild")
    trader, outcome = drive(cfg, strategy, journal, frames, max_bars=120)
    assert not outcome.halted
    assert trader.portfolio.positions

    rebuilt = Journal(tmp_path, "rebuild").rebuild(cfg.backtest.initial_equity)
    assert rebuilt.portfolio.cash == pytest.approx(trader.portfolio.cash)
    assert rebuilt.portfolio.positions.keys() == trader.portfolio.positions.keys()
    for sym, qty in trader.portfolio.positions.items():
        assert rebuilt.portfolio.position_qty(sym) == pytest.approx(qty)
    assert reconcile(
        rebuilt.portfolio, None, book=journal.read_book()
    ).ok


def test_a_torn_final_line_costs_one_bar_not_the_run(tmp_path):
    journal = Journal(tmp_path, "torn")
    journal.append("run_started", run_id="x", initial_equity=10_000.0)
    journal.log_mark(pd.Timestamp("2024-01-01T00:00:00+00:00"), 10_000.0, {})
    with open(journal.path, "a") as fh:
        fh.write('{"seq": 3, "event": "mark", "ts": "2024-01-01T01:00')

    rebuilt = Journal(tmp_path, "torn").rebuild(10_000.0)
    assert len(rebuilt.marks) == 1


def test_corruption_that_is_not_a_torn_write_raises(tmp_path):
    journal = Journal(tmp_path, "corrupt")
    journal.append("run_started", run_id="x", initial_equity=10_000.0)
    lines = journal.path.read_text().splitlines()
    journal.path.write_text(lines[0] + "\n{ this is not json\n" + lines[0] + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        list(Journal(tmp_path, "corrupt").rebuild(10_000.0).marks)


def test_duplicate_fills_in_the_journal_are_fatal(tmp_path):
    journal = Journal(tmp_path, "dupe")
    journal.append("run_started", run_id="x", initial_equity=10_000.0)
    for _ in range(2):
        journal.append(
            "fill",
            client_order_id="same-id",
            ts_filled="2024-01-01T00:00:00+00:00",
            symbol="BTC/USDT",
            side="buy",
            qty=1.0,
            price=100.0,
            fee=0.1,
            slippage_bps=1.0,
        )
    with pytest.raises(ValueError, match="duplicate fill"):
        Journal(tmp_path, "dupe").rebuild(10_000.0)


def test_reconcile_reports_a_venue_mismatch(tmp_path):
    pf = Portfolio(cash=100.0, positions={"BTC/USDT": 2.0})
    assert reconcile(pf, {"BTC/USDT": 2.0}).ok
    bad = reconcile(pf, {"BTC/USDT": 1.0})
    assert not bad.ok
    assert "venue holds" in bad.problems[0]
    # Paper mode cannot run the venue check, and must say so rather than
    # quietly passing a check it never performed.
    assert reconcile(pf, None).detail["venue_checked"] is False


def test_heartbeat_age_and_halt_sentinel(tmp_path):
    journal = Journal(tmp_path, "hb")
    assert journal.heartbeat_age() is None
    journal.write_heartbeat(
        bar_ts=pd.Timestamp("2024-01-01T00:00:00+00:00"), equity=1.0, interval_s=3600
    )
    age = journal.heartbeat_age()
    assert age is not None and age < 60
    assert journal.halted() is None
    journal.halt("because", ts=pd.Timestamp("2024-01-01T00:00:00+00:00"))
    assert journal.halted()["reason"] == "because"
    assert journal.clear_halt() is True
    assert journal.clear_halt() is False


# ------------------------------------------------------------- quote sources


def test_the_unclosed_bar_is_never_served():
    """An in-progress candle must not reach a strategy.

    Its close keeps changing, so feeding it in makes decisions irreproducible
    and silently breaks parity with the backtest. The bar indexed at 02:00
    covers 02:00-03:00 and is only closed once the clock passes 03:00.
    """
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "close": [1.5, 2.5, 3.5]}, index=idx
    )
    closed, pending = _drop_unclosed(df, "1h", pd.Timestamp("2024-01-01 02:30", tz="UTC"))
    assert list(closed.index) == list(idx[:2])
    assert pending is not None and pending["open"] == 3.0

    closed, pending = _drop_unclosed(df, "1h", pd.Timestamp("2024-01-01 03:00", tz="UTC"))
    assert list(closed.index) == list(idx)
    assert pending is None


def test_ccxt_source_uses_the_new_bars_open_and_rejects_a_stale_feed():
    """The fill price is the next bar's open — fixed, and never revised."""

    class FakeExchange:
        rateLimit = 0

        def __init__(self, rows):
            self.rows = rows
            self.ticker_calls = 0

        def fetch_ohlcv(self, symbol, timeframe, limit=None):
            return self.rows

        def fetch_ticker(self, symbol):
            self.ticker_calls += 1
            return {"last": 999.0}

    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    rows = [
        [int((base + pd.Timedelta(hours=i)).timestamp() * 1000), 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i, 100.0]
        for i in range(4)
    ]
    source = CcxtQuoteSource(
        ["BTC/USDT"], timeframe="1h", clock=lambda: base + pd.Timedelta(hours=3, minutes=10)
    )
    fake = FakeExchange(rows)
    source._ex = fake

    quote = source._poll_once()
    assert quote is not None
    # Bars 0-2 are closed at 03:10; bar 3 is in progress.
    assert quote.bar_ts == base + pd.Timedelta(hours=2)
    assert len(quote) == 3
    # ref price is the in-progress bar's open (13.0), not the ticker.
    assert quote.ref_price["BTC/USDT"] == pytest.approx(13.0)
    assert fake.ticker_calls == 0
    assert quote.volume_is_estimate is True

    stale = CcxtQuoteSource(
        ["BTC/USDT"], timeframe="1h", max_stale_bars=2,
        clock=lambda: base + pd.Timedelta(hours=12),
    )
    stale._ex = FakeExchange(rows)
    with pytest.raises(StaleFeed, match="stale feed"):
        stale._poll_once()


def test_a_stale_feed_halts_the_run(tmp_path):
    class Exploding:
        timeframe = "1h"

        def quotes(self):
            yield from ()
            raise StaleFeed("feed is 9 bars behind")

        def close(self):
            return None

    cfg = make_config()
    journal = Journal(tmp_path, "stale")
    trader = start_paper_trader(cfg, BuyAndHold(), journal)
    with pytest.raises(StaleFeed):
        run_paper_loop(trader, Exploding())
    halted = Journal(tmp_path, "stale").halted()
    assert halted is not None and "stale feed" in halted["reason"]


def test_worst_case_volume_is_never_cheaper_than_the_estimate(tmp_path):
    """Unknown volume must cost more, not less. DESIGN.md section 4."""
    cfg = make_config()
    frames = panel(n=400)

    costs = []
    for mode in ("actual_next", "worst_case"):
        journal = Journal(tmp_path, f"vol-{mode}")
        drive(cfg, BuyAndHold(), journal, frames, max_bars=60, volume_mode=mode)
        fills = journal.fills()
        assert fills
        costs.append(float(np.mean([f.slippage_bps for f in fills])))
    assert costs[1] >= costs[0]


def test_replay_source_serves_the_bar_it_decides_on():
    frames = panel(n=50)
    quotes = list(ReplayQuoteSource(frames, start_index=10).quotes())
    assert len(quotes) == 40
    first = quotes[0]
    assert isinstance(first, Quote)
    assert len(first) == 11
    assert first.bar_ts == frames["close"].index[10]
    # Fills happen at the NEXT bar's open, which the decision bar cannot see.
    assert first.ref_price.equals(frames["open"].iloc[11])
    # Past the end of the data there is no next open, so nothing can fill.
    assert quotes[-1].ref_price.isna().all()


# ------------------------------------------------------------------ reporting


def test_result_from_journal_is_summarizable(tmp_path):
    cfg = make_config()
    frames = panel(n=600)
    strategy = CrossSectionalMomentum(lookback=48, n_long=2, rebalance_every=12)
    journal = Journal(tmp_path, "report")
    drive(cfg, strategy, journal, frames, max_bars=200)

    from tradebot.validation.metrics import summarize

    result = result_from_journal(journal, cfg.backtest.initial_equity, strategy=strategy)
    summary = summarize(result, cfg.bars_per_year)
    assert result.meta["mode"] == "paper"
    assert 0.0 <= result.meta["clamp_rate"] <= 1.0
    assert summary["effective_bars"] == len(result.equity)
    assert np.isfinite(summary["sharpe_annual"])
    assert all(f.side in (Side.BUY, Side.SELL) for f in result.fills)


def test_cli_watchdog_reports_a_dead_run(tmp_path, capsys):
    from tradebot.run.paper import main

    args = ["--run-name", "wd", "--state-dir", str(tmp_path), "--watchdog"]
    assert main(args) == 3
    assert "NO HEARTBEAT" in capsys.readouterr().out

    journal = Journal(tmp_path, "wd")
    journal.write_heartbeat(
        bar_ts=pd.Timestamp("2024-01-01", tz="UTC"), equity=10_000.0, interval_s=3600
    )
    journal.write_book(
        Portfolio(cash=10_000.0), pd.Timestamp("2024-01-01", tz="UTC"), 10_000.0
    )
    assert main(args) == 0
    assert "alive" in capsys.readouterr().out

    # Zero tolerance for silence: the same heartbeat now reads as stale.
    assert main(args + ["--watchdog-max-age", "0"]) == 3
    assert "STALE" in capsys.readouterr().out

    journal.halt("drawdown 41% >= limit 40%", ts=pd.Timestamp("2024-01-02", tz="UTC"))
    assert main(args) == 3
    assert "HALTED" in capsys.readouterr().out


def test_cli_replay_runs_end_to_end(tmp_path, capsys):
    from tradebot.run.paper import main

    code = main(
        [
            "--run-name", "cli",
            "--state-dir", str(tmp_path),
            "--replay",
            "--synthetic", "trending",
            "--synthetic-bars", "400",
            "--strategy", "hold",
            "--max-bars", "100",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "REPLAY (no network)" in out
    assert Journal(tmp_path, "cli").equity_curve().size > 0


def test_the_halt_record_names_what_it_was_holding(tmp_path):
    """Positions at the moment of the breach, not after the liquidation.

    After a halt the book is flat by construction, so recording the position
    afterwards records nothing. The useful fact is what it was holding when the
    breaker tripped.
    """
    cfg = make_config(
        risk=RiskLimits(
            max_weight_per_symbol=1.0,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.03,
            halt_on_breach=True,
            symbol_allowlist=frozenset(SYMBOLS),
        )
    )
    frames = panel(kind="random_walk", n=300, seed=11, drift_per_bar=-0.005)
    journal = Journal(tmp_path, "halt-record")
    trader, outcome = drive(cfg, BuyAndHold(), journal, frames)

    assert outcome.halted
    assert trader.portfolio.positions == {}, "a halt flattens the book"
    held = journal.halted()["positions_at_halt"]
    assert held, "the halt record must say what was held when it tripped"
    assert set(held) <= set(SYMBOLS)


def test_a_report_names_the_strategy_that_actually_ran(tmp_path):
    """The journal is the authority, not whatever --strategy says today.

    Reporting a paper curve under the name of a strategy that did not produce
    it is the precise failure mode this repository exists to prevent, and it is
    easy to hit: `--status` takes a `--strategy` flag with a default.
    """
    cfg = wide_config()
    frames = panel(n=300)
    journal = Journal(tmp_path, "attrib")
    drive(cfg, BuyAndHold(), journal, frames, max_bars=40)

    honest = result_from_journal(
        journal, cfg.backtest.initial_equity, strategy=BuyAndHold()
    )
    assert honest.meta["strategy"] == "BuyAndHold()"
    assert honest.meta["strategy_matches_journal"] is True

    mismatched = result_from_journal(
        journal, cfg.backtest.initial_equity, strategy=CrossSectionalMomentum()
    )
    assert mismatched.meta["strategy"] == "BuyAndHold()"
    assert mismatched.meta["strategy_matches_journal"] is False


def test_already_seen_bars_do_not_consume_the_bar_budget(tmp_path):
    """A resumed run must make progress, not spend --max-bars on old bars."""
    cfg = wide_config()
    frames = panel(n=400)
    strategy = BuyAndHold()

    journal = Journal(tmp_path, "budget")
    drive(cfg, strategy, journal, frames, max_bars=30)
    first = journal.equity_curve()

    journal2 = Journal(tmp_path, "budget")
    drive(cfg, strategy, journal2, frames, max_bars=30)
    second = journal2.equity_curve()

    assert len(second) > len(first), "the resumed run made no progress"
    assert not second.index.has_duplicates


@pytest.mark.parametrize(
    "module",
    [
        "tradebot.state",
        "tradebot.engine.paper",
        "tradebot.execution.paper",
        "tradebot.data.quotes",
        "tradebot.run.paper",
    ],
)
def test_every_module_can_be_the_first_one_imported(module):
    """No import cycles, whichever module a caller reaches for first.

    This is a real bug that was caught here rather than in review: `state`
    holds the persistence layer and `engine` sits above it, so the moment
    `state` imported the engine package at module scope, `import
    tradebot.state` in a fresh interpreter failed — while the test suite kept
    passing, because something else had always imported the engine first. It
    has to be a subprocess for the same reason: within one interpreter the
    cycle is hidden by whatever is already in sys.modules.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}; print('ok')"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"importing {module} first failed:\n{proc.stderr}"
