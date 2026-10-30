"""Pin a market-data snapshot.

    python -m tradebot.run.fetch
    python -m tradebot.run.fetch --start 2020-01-01 --timeframe 4h

Needs the exchange API reachable from this host.
"""

from __future__ import annotations

import argparse

from ..config import load_config
from ..data.fetch import fetch_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--timeframe")
    p.add_argument("--exchange")
    p.add_argument(
        "--allow-invalid",
        action="store_true",
        help="save even if validation finds problems (records them in the "
        "manifest). Use only when you understand the specific problem.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    d = cfg.data

    exchange = args.exchange or d.exchange
    timeframe = args.timeframe or d.timeframe
    start = args.start or d.start
    end = args.end or d.end

    print(f"Fetching {list(d.symbols)} {timeframe} from {exchange}, {start}..{end or 'now'}")
    try:
        snap = fetch_and_save(
            d.symbols, timeframe, start, end, exchange,
            str(d.snapshot_dir), strict=not args.allow_invalid,
        )
    except Exception as exc:
        raise SystemExit(
            f"\nFetch failed: {exc}\n\n"
            f"If this is a network or TLS error, the exchange API is probably not "
            f"reachable from this host. Public OHLCV needs no API key, but it does "
            f"need outbound access to {exchange}."
        ) from exc

    print(f"\nSnapshot {snap.snapshot_id}")
    print(f"  bars    {snap.manifest['n_bars']}")
    print(f"  range   {snap.manifest['first_bar']} .. {snap.manifest['last_bar']}")
    print(f"  symbols {snap.manifest['symbols']}")
    if snap.manifest["validation_problems"]:
        print("  PROBLEMS:")
        for prob in snap.manifest["validation_problems"]:
            print(f"    - {prob}")
    print(f"\nCommit data/snapshots/{snap.snapshot_id}.manifest.json to pin this result.")


if __name__ == "__main__":
    main()
