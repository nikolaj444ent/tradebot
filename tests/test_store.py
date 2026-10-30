"""Snapshots must be content-addressed and must reject broken data."""

from __future__ import annotations

import pytest

from tradebot.data import SnapshotStore, synthetic_panel
from tradebot.data.store import validate_panel


def test_identical_data_hashes_identically(tmp_path):
    a = synthetic_panel(("A", "B"), 200, seed=5)
    b = synthetic_panel(("A", "B"), 200, seed=5)
    store = SnapshotStore(tmp_path)
    assert store.save(a, exchange="x", timeframe="1h", source="t").snapshot_id == \
           store.save(b, exchange="x", timeframe="1h", source="t").snapshot_id


def test_different_data_hashes_differently(tmp_path):
    store = SnapshotStore(tmp_path)
    one = store.save(
        synthetic_panel(("A",), 200, seed=5), exchange="x", timeframe="1h", source="t"
    )
    two = store.save(
        synthetic_panel(("A",), 200, seed=6), exchange="x", timeframe="1h", source="t"
    )
    assert one.snapshot_id != two.snapshot_id


def test_round_trip_preserves_values(tmp_path):
    store = SnapshotStore(tmp_path)
    saved = store.save(
        synthetic_panel(("A", "B"), 150, seed=7),
        exchange="x", timeframe="1h", source="t",
    )
    loaded = store.load(saved.snapshot_id)
    assert loaded.snapshot_id == saved.snapshot_id
    for field in ("open", "high", "low", "close", "volume"):
        assert loaded.frames[field].equals(saved.frames[field])


def test_validation_catches_high_below_low():
    frames = synthetic_panel(("A",), 100, seed=8)
    frames["high"], frames["low"] = frames["low"], frames["high"]
    assert any("high < low" in p for p in validate_panel(frames))


def test_validation_catches_a_tz_naive_index():
    frames = synthetic_panel(("A",), 100, seed=8)
    for f in frames:
        frames[f].index = frames[f].index.tz_localize(None)
    assert any("tz-naive" in p for p in validate_panel(frames))


def test_validation_catches_missing_bars():
    """A gap shifts every lookback window, so it must not pass silently."""
    frames = synthetic_panel(("A",), 100, seed=8)
    keep = list(range(0, 40)) + list(range(50, 100))
    frames = {f: df.iloc[keep] for f, df in frames.items()}
    assert any("irregular bar interval" in p for p in validate_panel(frames))


def test_saving_invalid_data_raises_by_default(tmp_path):
    frames = synthetic_panel(("A",), 100, seed=9)
    frames["close"].iloc[5, 0] = -1.0
    with pytest.raises(ValueError, match="failed validation"):
        SnapshotStore(tmp_path).save(
            frames, exchange="x", timeframe="1h", source="t"
        )
