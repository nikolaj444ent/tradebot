"""Shared CLI plumbing."""

from __future__ import annotations

import argparse

import pandas as pd

from ..config import Config, load_config
from ..data import SnapshotStore, synthetic_panel
from ..strategy import (
    BuyAndHold,
    CrossSectionalMomentum,
    EqualWeightRebalanced,
    ZScoreMeanReversion,
)

STRATEGIES = {
    "hold": BuyAndHold,
    "eqw": EqualWeightRebalanced,
    "momentum": CrossSectionalMomentum,
    "meanrev": ZScoreMeanReversion,
}

# Grids for walk-forward. Kept small on purpose: every extra combination is
# another trial to deflate against, and a grid of 500 points buys you a worse
# deflated Sharpe, not a better strategy.
PARAM_GRIDS = {
    "momentum": {
        "lookback": [72, 168, 336],
        "n_long": [1, 2],
        "rebalance_every": [24],
    },
    "meanrev": {
        "lookback": [48, 72, 168],
        "entry_z": [-1.0, -1.5, -2.0],
        "max_positions": [2],
    },
    "hold": {},
    "eqw": {},
}


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config/default.yaml")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--snapshot", help="pinned snapshot id from data/snapshots")
    src.add_argument(
        "--synthetic",
        choices=["random_walk", "trending", "mean_reverting"],
        help="generate data instead of loading a snapshot (for smoke tests; "
        "results on synthetic data prove plumbing, not profitability)",
    )
    p.add_argument("--synthetic-bars", type=int, default=6000)
    p.add_argument("--synthetic-seed", type=int, default=0)


def resolve_data(args, cfg: Config) -> tuple[dict[str, pd.DataFrame], str]:
    if args.synthetic:
        frames = synthetic_panel(
            cfg.data.symbols,
            args.synthetic_bars,
            cfg.data.timeframe,
            seed=args.synthetic_seed,
            kind=args.synthetic,
        )
        return frames, f"synthetic:{args.synthetic}:seed{args.synthetic_seed}"

    store = SnapshotStore(cfg.data.snapshot_dir)
    if args.snapshot:
        snap = store.load(args.snapshot)
        return snap.frames, snap.snapshot_id

    available = store.list_snapshots()
    if not available:
        raise SystemExit(
            "No pinned snapshots found.\n"
            "  Fetch one:   python -m tradebot.run.fetch\n"
            "  Or smoke-test without network:\n"
            "               python -m tradebot.run.backtest --synthetic trending\n"
            "Note: exchange APIs must be reachable from this host."
        )
    if len(available) > 1:
        ids = "\n".join(
            f"    {m['snapshot_id']}  {m['timeframe']:>4}  {m['n_bars']:>7} bars  "
            f"{m['first_bar'][:10]}..{m['last_bar'][:10]}"
            for m in available
        )
        raise SystemExit(
            f"Several snapshots available; choose one with --snapshot:\n{ids}"
        )
    snap = store.load(available[0]["snapshot_id"])
    return snap.frames, snap.snapshot_id


def load_cfg(args) -> Config:
    return load_config(args.config)
