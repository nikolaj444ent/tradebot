"""Quote sources: where a live paper run gets its bars and its fill price.

A backtest reads a pinned panel and knows every number in advance. A paper run
polls a venue and has to be careful about two things that cannot go wrong
quietly.

**The in-progress bar must never be used.** `fetch_ohlcv` returns the current,
unfinished candle as its last element. Its close changes every few seconds, so
a strategy fed that bar produces a different answer each time it is asked the
same question — decisions stop being reproducible, the paper run stops
matching its backtest, and the cause is invisible in the results. Worse, on
some venues that bar is revised after the fact. So `_drop_unclosed` removes it,
every time, and the daemon additionally refuses to act twice on the same bar.

**The fill price should be the next bar's open, not the last trade.** The
backtest contract (DESIGN.md section 3) fills at bar `i+1`'s open. Once bar
`i+1` has started, its *open* is already fixed and will never change, even
though its close has not happened yet. Reading just that one field from the
unfinished bar is therefore both faithful to the contract and reproducible,
which the ticker's last price is not. The ticker is only a fallback for the
seconds before the new bar has printed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from ..types import OHLCV_FIELDS

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class StaleFeed(RuntimeError):
    """The newest closed bar is too old to trade on."""


@dataclass(frozen=True, slots=True)
class Quote:
    """One poll's worth of market state.

    `bars` ends at the newest *closed* bar — the bar being decided on.
    `ref_price` is what fills execute around: the open of the bar that has
    just started. `bar_volume_notional` feeds the impact model and is an
    estimate in live mode, which is noted on `volume_is_estimate` rather than
    hidden.
    """

    bars: dict[str, pd.DataFrame]
    ref_price: pd.Series
    bar_volume_notional: pd.Series
    fill_ts: pd.Timestamp
    volume_is_estimate: bool = False

    @property
    def bar_ts(self) -> pd.Timestamp:
        """Close timestamp of the newest closed bar: the decision bar."""
        return self.bars["close"].index[-1]

    def __len__(self) -> int:
        return len(self.bars["close"])


class QuoteSource(ABC):
    """Supplies quotes to the paper loop, live or replayed."""

    timeframe: str = "1h"

    @abstractmethod
    def quotes(self) -> Iterator[Quote]:
        """Yield one quote per closed bar, blocking until each bar closes."""

    def close(self) -> None:
        return None


def _drop_unclosed(
    df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split a frame into its closed bars and the unfinished one, if any.

    A bar indexed at time `t` covers `[t, t + interval)`, so it has closed only
    once `now >= t + interval`. Anything else is still being written.
    """
    if df.empty:
        return df, None
    step = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    now = now or pd.Timestamp.now(tz="UTC")
    closed_mask = df.index + step <= now
    unclosed = None
    if not closed_mask.all():
        pending = df[~closed_mask]
        if len(pending):
            unclosed = pending.iloc[0]
    return df[closed_mask], unclosed


class ReplayQuoteSource(QuoteSource):
    """Drives the paper loop from a pinned panel, as fast as it can run.

    This is how the paper daemon is tested, and it is the one piece of
    machinery that makes the parity claim checkable: replaying a snapshot
    through the paper loop must reproduce the backtest's equity curve on the
    same data, bar for bar. If it ever does not, the two code paths have
    drifted and every paper-versus-backtest comparison after that is noise.

    It is also useful outside tests — `--replay` gives a dry run of the whole
    daemon, journal and reconciliation included, without waiting an hour per
    bar.
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        timeframe: str = "1h",
        start_index: int = 0,
        volume_mode: str = "actual_next",
    ) -> None:
        if volume_mode not in ("actual_next", "trailing", "worst_case"):
            raise ValueError(f"unknown volume_mode {volume_mode!r}")
        self.frames = frames
        self.timeframe = timeframe
        self.start_index = max(int(start_index), 0)
        # "actual_next" uses the volume of the bar being filled into, which is
        # what the backtest does and therefore what parity requires. The other
        # two exist to measure, offline, what the live estimate costs: in
        # production that number is not knowable at fill time.
        self.volume_mode = volume_mode

    def quotes(self) -> Iterator[Quote]:
        close = self.frames["close"]
        open_ = self.frames["open"]
        notional = (self.frames["volume"] * close).fillna(0.0)
        n = len(close)
        nan_row = pd.Series(float("nan"), index=close.columns)

        for k in range(self.start_index, n):
            has_next = k + 1 < n
            if self.volume_mode == "worst_case":
                vol = pd.Series(0.0, index=close.columns)
            elif self.volume_mode == "trailing" or not has_next:
                vol = notional.iloc[k]
            else:
                vol = notional.iloc[k + 1]
            yield Quote(
                bars={f: self.frames[f].iloc[: k + 1] for f in OHLCV_FIELDS},
                # Past the end of the panel there is no next open, so there is
                # no price to fill at. That is the truth of the situation at
                # the newest bar in live too, until the next bar prints.
                ref_price=open_.iloc[k + 1] if has_next else nan_row,
                bar_volume_notional=vol,
                fill_ts=close.index[k + 1] if has_next else close.index[k],
                volume_is_estimate=self.volume_mode != "actual_next",
            )


class CcxtQuoteSource(QuoteSource):
    """Public OHLCV polling via ccxt. No credentials, no order endpoints.

    Only public market-data endpoints are touched. This class cannot place an
    order; it has no key and no code to do so.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] | list[str],
        *,
        exchange: str = "binance",
        timeframe: str = "1h",
        history_bars: int = 500,
        settle_seconds: float = 20.0,
        poll_seconds: float = 15.0,
        max_stale_bars: int = 2,
        volume_mode: str = "trailing",
        sleeper=time.sleep,
        clock=None,
    ) -> None:
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        self.symbols = list(symbols)
        self.exchange_name = exchange
        self.timeframe = timeframe
        self.history_bars = history_bars
        # Venues do not publish a bar the instant it closes. Polling at exactly
        # the boundary gets a missing or half-aggregated bar, so wait a little.
        self.settle_seconds = settle_seconds
        self.poll_seconds = poll_seconds
        # A feed this far behind is not slow, it is broken. Trading on a stale
        # bar is trading on a price that no longer exists.
        self.max_stale_bars = max_stale_bars
        self.volume_mode = volume_mode
        self._sleep = sleeper
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._ex = None

    # -------------------------------------------------------------- internals

    def _exchange(self):
        if self._ex is None:
            import ccxt  # lazy: tests and backtests need no network stack

            self._ex = getattr(ccxt, self.exchange_name)({"enableRateLimit": True})
        return self._ex

    def _poll_once(self) -> Quote | None:
        ex = self._exchange()
        now = self._clock()
        step = pd.Timedelta(seconds=TIMEFRAME_SECONDS[self.timeframe])

        closed: dict[str, pd.DataFrame] = {}
        pending: dict[str, pd.Series] = {}
        for sym in self.symbols:
            raw = ex.fetch_ohlcv(sym, self.timeframe, limit=self.history_bars)
            if not raw:
                return None
            df = pd.DataFrame(
                raw, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df = df.drop_duplicates(subset="ts").sort_values("ts")
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").astype("float64")
            done, unclosed = _drop_unclosed(df, self.timeframe, now)
            if done.empty:
                return None
            closed[sym] = done
            if unclosed is not None:
                pending[sym] = unclosed

        # Inner join, for the same reason the fetcher does it: a symbol missing
        # a bar would otherwise contribute a NaN that becomes a zero price, and
        # a zero price is a 100% loss nobody took.
        common = None
        for df in closed.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is None or len(common) == 0:
            return None

        frames = {
            field: pd.DataFrame(
                {s: closed[s].loc[common, field] for s in self.symbols}, index=common
            ).astype("float64")
            for field in OHLCV_FIELDS
        }

        bar_ts = common[-1]
        staleness = (now - (bar_ts + step)) / step
        if staleness > self.max_stale_bars:
            raise StaleFeed(
                f"newest closed bar is {bar_ts}, which is {staleness:.1f} bars "
                f"behind {now}. Refusing to trade on a stale feed."
            )

        # The open of the bar now in progress: already fixed, never revised.
        # Falls back to the last trade only in the seconds before it prints.
        ref: dict[str, float] = {}
        for sym in self.symbols:
            row = pending.get(sym)
            if row is not None and float(row["open"]) > 0:
                ref[sym] = float(row["open"])
            else:
                try:
                    ref[sym] = float(ex.fetch_ticker(sym)["last"])
                except Exception:
                    ref[sym] = float("nan")

        if self.volume_mode == "worst_case":
            vol = pd.Series(0.0, index=frames["close"].columns)
        else:
            # The volume of the bar we are filling into is unknowable at fill
            # time; the bar just closed is the best estimator available, and it
            # is a real estimate rather than an assumption of free execution.
            # Where paper slippage diverges from backtest slippage, this is the
            # first place to look — see DESIGN.md section 11.
            vol = (frames["volume"] * frames["close"]).fillna(0.0).iloc[-1]

        return Quote(
            bars=frames,
            ref_price=pd.Series(ref),
            bar_volume_notional=vol,
            fill_ts=now,
            volume_is_estimate=True,
        )

    # ------------------------------------------------------------------ public

    def quotes(self) -> Iterator[Quote]:
        step = TIMEFRAME_SECONDS[self.timeframe]
        last_bar: pd.Timestamp | None = None
        while True:
            quote = self._poll_once()
            if quote is not None and quote.bar_ts != last_bar:
                last_bar = quote.bar_ts
                yield quote
                # Sleep until just past the next boundary rather than busy
                # polling: one decision per bar is the contract.
                now = self._clock()
                next_close = quote.bar_ts + pd.Timedelta(seconds=2 * step)
                wait = (next_close - now).total_seconds() + self.settle_seconds
                self._sleep(max(wait, self.poll_seconds))
            else:
                self._sleep(self.poll_seconds)

