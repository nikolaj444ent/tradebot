"""Market data feeds for the paper daemon.

The daemon is deliberately ignorant of where bars come from. A ``MarketFeed``
yields *closed* bars only — the forming bar is never visible, for the same
reason ``MarketView`` hides the future in backtests — plus an execution
reference: the price you could actually trade at right now.

Two implementations:

- ``LiveCcxtFeed`` polls public OHLCV endpoints (no keys) and prices fills
  off the live ticker.
- ``ReplayFeed`` walks a pre-built panel bar by bar. It exists so the daemon
  can be driven deterministically in tests — and so the whole daemon can be
  proven equivalent to ``run_backtest`` on the same data, which is the parity
  DESIGN.md section 2 depends on. On a replay, the execution reference is the
  next bar's open: exactly the backtest's fill price.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..data.fetch import TIMEFRAME_MS
from ..types import OHLCV_FIELDS


@dataclass(frozen=True, slots=True)
class ExecRef:
    """What an order would fill against, right now.

    ``bar_volume_notional`` feeds the impact model. On a replay it is the fill
    bar's true volume (as in the backtest); live it is an *estimate* from the
    last closed bar, because the fill bar's volume does not exist yet.
    """

    ts: pd.Timestamp
    ref_price: pd.Series
    bar_volume_notional: pd.Series


class MarketFeed(ABC):
    timeframe: str = "1h"
    #: True once the feed can never produce another bar (replay only).
    exhausted: bool = False

    @abstractmethod
    def bootstrap(
        self,
        min_bars: int,
        origin: pd.Timestamp | None = None,
        upto: pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """History of closed bars to start from.

        With ``origin`` (a resumed run), history must start at that bar so the
        strategy's view is the same function of the data it was before the
        restart. ``upto`` is the resume point; live feeds ignore it and return
        everything up to now.
        """

    @abstractmethod
    def next_bar(self) -> dict[str, pd.DataFrame] | None:
        """The panel including the next closed bar, or None if none yet.

        Non-blocking. A live feed returns None until the next bar has closed
        and been published; a replay returns None only at the end of the
        panel, after setting ``exhausted``.
        """

    @abstractmethod
    def execution_ref(self) -> ExecRef | None:
        """Price/volume to fill at right now, or None if nothing can fill."""


class ReplayFeed(MarketFeed):
    """Walk a fixed panel, one closed bar at a time."""

    def __init__(self, frames: dict[str, pd.DataFrame], timeframe: str = "1h"):
        self._frames = frames
        self._n = len(frames["close"])
        self._cursor: int = -1
        self.timeframe = timeframe
        self.exhausted = False

    def bootstrap(self, min_bars, origin=None, upto=None):
        idx = self._frames["close"].index
        if origin is not None and idx[0] != origin:
            raise ValueError(
                f"replay panel starts at {idx[0]}, but the run's origin is {origin}. "
                f"A resumed run must see the same history it saw before."
            )
        self._cursor = int(idx.get_loc(upto)) if upto is not None else min_bars - 1
        if self._cursor + 1 < min_bars:
            raise ValueError(
                f"panel has only {self._cursor + 1} bars before the resume point; "
                f"the strategy needs {min_bars}."
            )
        return self._slice()

    def _slice(self) -> dict[str, pd.DataFrame]:
        return {f: self._frames[f].iloc[: self._cursor + 1] for f in OHLCV_FIELDS}

    def next_bar(self):
        if self._cursor + 1 >= self._n:
            self.exhausted = True
            return None
        self._cursor += 1
        return self._slice()

    def execution_ref(self):
        j = self._cursor + 1
        if j >= self._n:
            # The last bar of a replay has no bar after it to fill in, exactly
            # as the backtest loop stops one bar early.
            return None
        return ExecRef(
            ts=self._frames["close"].index[j],
            ref_price=self._frames["open"].iloc[j],
            bar_volume_notional=(
                self._frames["volume"].iloc[j] * self._frames["close"].iloc[j]
            ).fillna(0.0),
        )


class LiveCcxtFeed(MarketFeed):
    """Closed bars from public OHLCV endpoints; fills priced off the ticker.

    No credentials: this reads the same public endpoints as the fetcher. The
    forming bar is dropped on every read — only bars whose close time has
    passed (plus a grace period for the exchange to finalise the candle) are
    ever visible to the strategy.
    """

    def __init__(
        self,
        exchange: str,
        symbols: tuple[str, ...] | list[str],
        timeframe: str = "1h",
        *,
        grace_seconds: int = 15,
    ):
        import ccxt  # lazy: tests never touch the network stack

        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        self._ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.grace = pd.Timedelta(seconds=grace_seconds)
        self._step = pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe])
        self._frames: dict[str, pd.DataFrame] | None = None
        self.exhausted = False

    def bootstrap(self, min_bars, origin=None, upto=None):
        from ..data.fetch import fetch_panel

        now = pd.Timestamp.now(tz="UTC")
        if origin is not None:
            start = origin
        else:
            # A few spare bars so that clock skew or a thin first candle
            # cannot leave us one bar short of the warmup.
            start = now - (min_bars + 5) * self._step
        frames = fetch_panel(
            self.symbols,
            self.timeframe,
            start=start.tz_convert("UTC").tz_localize(None).isoformat(),
            end=None,
            exchange=self._ex.id,
        )
        # Drop any bar that has not closed yet: its close time is in the future.
        idx = frames["close"].index
        closed = idx[idx + self._step <= now]
        frames = {f: frames[f].loc[closed] for f in OHLCV_FIELDS}
        if len(frames["close"]) < min_bars:
            raise RuntimeError(
                f"bootstrap returned {len(frames['close'])} closed bars; the "
                f"strategy needs {min_bars}. Widen the window or check the exchange."
            )
        self._frames = frames
        return frames

    def next_bar(self):
        assert self._frames is not None, "bootstrap() first"
        idx = self._frames["close"].index
        last = idx[-1]
        target = last + self._step
        now = pd.Timestamp.now(tz="UTC")
        if now < target + self._step + self.grace:
            return None  # the next bar has not closed yet
        now_ms = int(now.timestamp() * 1000)
        since = int(target.timestamp() * 1000)
        step_ms = TIMEFRAME_MS[self.timeframe]

        per: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            try:
                batch = self._ex.fetch_ohlcv(sym, self.timeframe, since=since, limit=200)
            except Exception as exc:  # transient network/exchange error: retry next poll
                print(f"[feed] fetch_ohlcv({sym}) failed, will retry: {exc}")
                return None
            rows = [r for r in batch if r[0] + step_ms <= now_ms]  # closed only
            if not rows:
                return None  # exchange has not published the candle yet
            df = pd.DataFrame(
                rows, columns=["ts", "open", "high", "low", "close", "volume"]
            ).drop_duplicates(subset="ts")
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            per[sym] = df.set_index("ts").astype("float64")

        common = None
        for df in per.values():
            common = df.index if common is None else common.intersection(df.index)
        common = common[common > last].sort_values()
        if len(common) == 0:
            return None
        for f in OHLCV_FIELDS:
            new = pd.DataFrame(
                {s: per[s].loc[common, f] for s in self.symbols}, index=common
            ).astype("float64")
            self._frames[f] = pd.concat([self._frames[f], new])
        return {f: self._frames[f] for f in OHLCV_FIELDS}

    def execution_ref(self):
        assert self._frames is not None, "bootstrap() first"
        try:
            tickers = self._ex.fetch_tickers(self.symbols)
            last = {s: float(tickers[s]["last"]) for s in self.symbols}
        except Exception as exc:
            print(f"[feed] fetch_tickers failed, cannot price an execution: {exc}")
            return None
        return ExecRef(
            ts=pd.Timestamp.now(tz="UTC"),
            ref_price=pd.Series(last, dtype="float64"),
            # Estimate: the fill bar's volume does not exist yet, so use the
            # last closed bar's. The impact model treats low volume as high
            # impact, which errs on the expensive side.
            bar_volume_notional=(
                self._frames["volume"].iloc[-1] * self._frames["close"].iloc[-1]
            ).fillna(0.0),
        )
