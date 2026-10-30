"""OHLCV download via ccxt.

Kept deliberately thin: it paginates, respects rate limits, aligns symbols
onto a common index, and hands the result to SnapshotStore. It makes no
trading decisions and holds no credentials — public OHLCV endpoints need none.

Note on running this: some networks (including Claude's cloud sandbox by
default) block exchange API hosts. Run it on the host that will run the
harness.
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from ..types import OHLCV_FIELDS
from .store import SnapshotStore

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _fetch_symbol(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                  limit: int = 1000, max_retries: int = 5) -> pd.DataFrame:
    rows: list[list] = []
    cursor = since_ms
    step = TIMEFRAME_MS[timeframe]
    while cursor < until_ms:
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
                break
            except Exception as exc:  # ccxt raises a wide family of errors
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"failed fetching {symbol} at {cursor}: {exc}"
                    ) from exc
                time.sleep(2**attempt)
        if not batch:
            break
        rows.extend(batch)
        advanced = batch[-1][0] + step
        if advanced <= cursor:  # exchange refused to move forward
            break
        cursor = advanced
        time.sleep(max(ex.rateLimit, 200) / 1000.0)

    if not rows:
        raise RuntimeError(f"no data returned for {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df = df[df["ts"] < until_ms]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype("float64")


def fetch_panel(
    symbols: Iterable[str],
    timeframe: str = "1h",
    start: str = "2021-01-01",
    end: str | None = None,
    exchange: str = "binance",
) -> dict[str, pd.DataFrame]:
    import ccxt  # imported lazily so tests need no network stack

    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe {timeframe!r}")

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    until_ms = int(
        (pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp()
        * 1000
    )

    per_symbol = {
        sym: _fetch_symbol(ex, sym, timeframe, since_ms, until_ms)
        for sym in symbols
    }

    # Inner join on timestamps. A symbol that was not listed for part of the
    # window would otherwise introduce NaNs that quietly become zeros, and a
    # zero price is a 100% loss the strategy never actually took.
    common = None
    for df in per_symbol.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise RuntimeError("symbols share no common timestamps")

    syms = list(per_symbol)
    return {
        field: pd.DataFrame(
            {s: per_symbol[s].loc[common, field] for s in syms}, index=common
        ).astype("float64")
        for field in OHLCV_FIELDS
    }


def fetch_and_save(
    symbols: Iterable[str],
    timeframe: str = "1h",
    start: str = "2021-01-01",
    end: str | None = None,
    exchange: str = "binance",
    snapshot_dir: str = "data/snapshots",
    strict: bool = True,
):
    frames = fetch_panel(symbols, timeframe, start, end, exchange)
    store = SnapshotStore(snapshot_dir)
    return store.save(
        frames,
        exchange=exchange,
        timeframe=timeframe,
        source=f"ccxt:{exchange}:{start}..{end or 'now'}",
        strict=strict,
    )
