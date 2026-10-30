"""Pinned market-data snapshots.

Why snapshots instead of fetching fresh each run: exchange history is not
immutable. Candles get revised, delisted pairs vanish, and APIs backfill gaps.
One published backtesting project found its in-sample CAGR moved from -0.7% to
+1.8% between two downloads of nominally the same dataset, months apart. If
your data silently changes, you cannot tell a real improvement from a data
artefact, and every comparison you make against an older result is void.

So: fetch once, hash it, write a manifest, and have every backtest record
which snapshot it ran against. The parquet files are gitignored (large, and
rebuildable); the manifest is committed, so a result stays reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..types import OHLCV_FIELDS

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Snapshot:
    """An immutable, content-addressed panel of OHLCV data."""

    snapshot_id: str
    frames: dict[str, pd.DataFrame]  # field -> DataFrame(index=ts UTC, cols=symbols)
    manifest: dict

    @property
    def symbols(self) -> list[str]:
        return list(self.frames["close"].columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frames["close"].index

    def __len__(self) -> int:
        return len(self.index)

    def slice(self, start: int, stop: int) -> "Snapshot":
        return Snapshot(
            snapshot_id=self.snapshot_id,
            frames={k: v.iloc[start:stop] for k, v in self.frames.items()},
            manifest=self.manifest,
        )


def _frame_digest(frames: dict[str, pd.DataFrame]) -> str:
    """Stable content hash over the actual numbers, not the file bytes.

    Hashing the parquet file would be fragile: pyarrow writes different bytes
    for identical data across versions. Hashing the values means the same data
    always gets the same id.
    """
    h = hashlib.sha256()
    h.update(f"v{SCHEMA_VERSION}".encode())
    for field in OHLCV_FIELDS:
        df = frames[field]
        h.update(field.encode())
        h.update(",".join(map(str, df.columns)).encode())
        # int64 nanoseconds: unambiguous across pandas versions
        h.update(df.index.view("int64").tobytes() if hasattr(df.index, "view")
                 else df.index.astype("int64").to_numpy().tobytes())
        h.update(df.to_numpy(dtype="float64").tobytes())
    return h.hexdigest()[:16]


def validate_panel(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Structural checks. Returns a list of problems; empty means clean.

    These are the data bugs that quietly produce fake profits, so they are
    errors rather than warnings.
    """
    problems: list[str] = []
    missing = set(OHLCV_FIELDS) - set(frames)
    if missing:
        return [f"missing fields: {sorted(missing)}"]

    ref = frames["close"]
    for field, df in frames.items():
        if not df.index.equals(ref.index):
            problems.append(f"{field}: index differs from close")
        if list(df.columns) != list(ref.columns):
            problems.append(f"{field}: columns differ from close")

    if not ref.index.is_monotonic_increasing:
        problems.append("index is not monotonically increasing")
    if ref.index.has_duplicates:
        problems.append("index has duplicate timestamps")
    if getattr(ref.index, "tz", None) is None:
        problems.append("index is tz-naive; must be UTC")

    o, h, l, c = (frames[f] for f in ("open", "high", "low", "close"))
    if (h < l).to_numpy().any():
        problems.append("high < low on some bars")
    for name, df in (("open", o), ("close", c)):
        above = (df > h + 1e-9).to_numpy().any()
        below = (df < l - 1e-9).to_numpy().any()
        if above or below:
            problems.append(f"{name} outside [low, high] on some bars")
    if (frames["volume"] < 0).to_numpy().any():
        problems.append("negative volume")
    if (c <= 0).to_numpy().any():
        problems.append("non-positive close price")

    # A gap in a continuously traded market means missing bars, which shifts
    # every lookback window. Detect it rather than silently reindexing.
    if len(ref.index) > 2:
        deltas = pd.Series(ref.index).diff().dropna()
        if deltas.nunique() > 1:
            modal = deltas.mode().iloc[0]
            n_gaps = int((deltas != modal).sum())
            problems.append(
                f"{n_gaps} irregular bar interval(s); modal interval is {modal}"
            )
    return problems


class SnapshotStore:
    def __init__(self, root: str | Path = "data/snapshots") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, snapshot_id: str) -> tuple[Path, Path]:
        return (
            self.root / f"{snapshot_id}.parquet",
            self.root / f"{snapshot_id}.manifest.json",
        )

    def save(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        exchange: str,
        timeframe: str,
        source: str,
        strict: bool = True,
    ) -> Snapshot:
        problems = validate_panel(frames)
        if problems and strict:
            raise ValueError(
                "market data failed validation:\n  - " + "\n  - ".join(problems)
            )

        snapshot_id = _frame_digest(frames)
        ref = frames["close"]
        manifest = {
            "snapshot_id": snapshot_id,
            "schema_version": SCHEMA_VERSION,
            "exchange": exchange,
            "timeframe": timeframe,
            "source": source,
            "symbols": list(ref.columns),
            "n_bars": int(len(ref)),
            "first_bar": ref.index[0].isoformat(),
            "last_bar": ref.index[-1].isoformat(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "validation_problems": problems,
        }

        pq_path, mf_path = self._paths(snapshot_id)
        long = pd.concat(
            {f: frames[f] for f in OHLCV_FIELDS}, names=["field", "ts"]
        )
        long.to_parquet(pq_path)
        mf_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return Snapshot(snapshot_id=snapshot_id, frames=dict(frames), manifest=manifest)

    def load(self, snapshot_id: str) -> Snapshot:
        pq_path, mf_path = self._paths(snapshot_id)
        if not pq_path.exists():
            raise FileNotFoundError(
                f"snapshot {snapshot_id} has no parquet at {pq_path}. The manifest "
                f"is committed but the data is not; re-fetch it with "
                f"`python -m tradebot.run.fetch --snapshot {snapshot_id}`."
            )
        long = pd.read_parquet(pq_path)
        frames = {
            f: long.xs(f, level="field").sort_index() for f in OHLCV_FIELDS
        }
        manifest = json.loads(mf_path.read_text()) if mf_path.exists() else {}
        # Re-verify: a snapshot that no longer hashes to its own id has been
        # tampered with or written by an incompatible version.
        actual = _frame_digest(frames)
        if actual != snapshot_id:
            raise ValueError(
                f"snapshot {snapshot_id} hashes to {actual}: data on disk does not "
                f"match its id. Delete it and re-fetch."
            )
        return Snapshot(snapshot_id=snapshot_id, frames=frames, manifest=manifest)

    def list_snapshots(self) -> list[dict]:
        out = []
        for mf in sorted(self.root.glob("*.manifest.json")):
            try:
                out.append(json.loads(mf.read_text()))
            except json.JSONDecodeError:
                continue
        return out
