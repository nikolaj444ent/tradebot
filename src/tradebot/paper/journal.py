"""Append-only paper-trading state.

The journal is the venue. In live trading the exchange remembers your orders
and positions; in paper trading nothing does unless we write it down. So every
decision, order, fill and equity mark is appended to a JSONL journal *before*
the in-memory book moves on, and the book can always be rebuilt by replaying
the fills from the top.

Two files, two jobs:

- ``journal.jsonl`` — append-only truth. Never rewritten, never edited.
- ``snapshot.json`` — "where the book should be", rewritten atomically after
  every step. On startup the daemon replays the journal and checks that the
  snapshot is a prefix-consistent view of it; a mismatch means state was
  corrupted or hand-edited, and the daemon halts rather than trade a book it
  cannot vouch for (DESIGN.md section 6).

Plus two small markers:

- ``heartbeat`` — touched every loop tick, so an outside watchdog can tell a
  quiet daemon from a dead one. The failure that actually costs money is not
  a bad trade, it is a silent death while holding a position.
- ``HALTED`` — written when the risk gate stops the run. While it exists the
  daemon refuses to start: a halted run resumes by human decision
  (``--clear-halt``), not by Docker restart policy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..types import Fill, Side


@dataclass(frozen=True, slots=True)
class PaperState:
    """The on-disk home of one named paper run."""

    root: Path

    @classmethod
    def at(cls, state_dir: str | Path, run_name: str) -> "PaperState":
        return cls(root=Path(state_dir) / "paper" / run_name)

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def snapshot_path(self) -> Path:
        return self.root / "snapshot.json"

    @property
    def heartbeat_path(self) -> Path:
        return self.root / "heartbeat"

    @property
    def halted_path(self) -> Path:
        return self.root / "HALTED"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def heartbeat(self, last_bar: pd.Timestamp | None = None) -> None:
        payload = {
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "last_bar": last_bar.isoformat() if last_bar is not None else None,
        }
        self.heartbeat_path.write_text(json.dumps(payload) + "\n")

    def halted_reason(self) -> str | None:
        if self.halted_path.exists():
            return self.halted_path.read_text().strip() or "halted (no reason recorded)"
        return None

    def write_halted(self, reason: str) -> None:
        self.ensure()
        self.halted_path.write_text(reason + "\n")

    def clear_halt(self) -> bool:
        """Remove the halt marker. Returns True if one existed.

        This is the human decision DESIGN.md section 5 talks about. Note that
        the risk-gate state (high-water mark, day-open equity) is persisted in
        the snapshot and survives the clear — if the book is still past a
        limit, the daemon will halt again on the next bar, which is the
        circuit breaker working, not a bug.
        """
        if self.halted_path.exists():
            self.halted_path.unlink()
            return True
        return False


class Journal:
    """Append-only JSONL event log. One line per event, fsynced."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, event: dict) -> None:
        line = json.dumps(event, separators=(",", ":"), sort_keys=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        lines = self.path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # A crash mid-append can truncate the final line. Every
                    # earlier line was fsynced whole, so a torn line anywhere
                    # else means real corruption.
                    break
                raise ValueError(
                    f"{self.path}: corrupt journal at line {i + 1}. The journal is "
                    f"append-only truth; do not edit it by hand."
                )
        return events


def fill_to_event(fill: Fill) -> dict:
    return {
        "type": "fill",
        "id": fill.client_order_id,
        "ts": fill.ts_filled.isoformat(),
        "symbol": fill.symbol,
        "side": fill.side.value,
        "qty": float(fill.qty),
        "price": float(fill.price),
        "fee": float(fill.fee),
        "slippage_bps": float(fill.slippage_bps),
    }


def fill_from_event(e: dict) -> Fill:
    return Fill(
        client_order_id=e["id"],
        ts_filled=pd.Timestamp(e["ts"]),
        symbol=e["symbol"],
        side=Side(e["side"]),
        qty=e["qty"],
        price=e["price"],
        fee=e["fee"],
        slippage_bps=e.get("slippage_bps", 0.0),
    )


def write_snapshot(path: Path, data: dict) -> None:
    """Atomic replace: a crash mid-write leaves the old snapshot, never half a file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
