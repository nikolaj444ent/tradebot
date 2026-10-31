"""Exact-output pins for the engine and the baseline strategies.

This file exists to make performance work safe. Optimising a backtest engine is
unusually dangerous: a change that is 10x faster and 0.3% different is worse
than useless, because every result recorded before it silently stops being
comparable, and nothing in a returns series announces that it moved. Threshold
assertions ("Sharpe above 1") do not catch that. Exact equality does.

So each case below is run through the engine and reduced to a digest over the
*exact* float repr of every equity point and every fill. The digests were
generated from the engine as it stood when it passed 78 tests, before any
vectorisation. If a refactor changes a single bit of a single fill, these fail.

The summary numbers stored next to each digest are not the assertion — they are
there so that a failure tells you roughly *how* the behaviour moved, rather
than only that it did.

To regenerate deliberately, after an intended behaviour change:

    python tests/test_engine_golden.py --update

That is a decision to abandon comparability with every result recorded before
it, so the diff on this file belongs in the commit message.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tradebot.costs import CostModel
from tradebot.data.synthetic import synthetic_panel
from tradebot.engine.backtest import run_backtest
from tradebot.risk import RiskLimits
from tradebot.strategy import (
    BuyAndHold,
    CrossSectionalMomentum,
    EqualWeightRebalanced,
    ZScoreMeanReversion,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "engine.json"

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")


def _strategy(spec: dict):
    kinds = {
        "hold": BuyAndHold,
        "eqw": EqualWeightRebalanced,
        "momentum": CrossSectionalMomentum,
        "meanrev": ZScoreMeanReversion,
    }
    params = {k: v for k, v in spec.items() if k != "kind"}
    return kinds[spec["kind"]](**params)


#: One case per behaviour worth pinning. Deliberately spread across data
#: regimes, strategy families and limit configurations, because a refactor can
#: easily be exact on a trending panel and wrong on a halting one.
CASES: list[dict] = [
    {
        "name": "momentum-trending-capped",
        "data": {"kind": "trending", "n_bars": 2500, "seed": 0},
        "strategy": {"kind": "momentum", "lookback": 168, "n_long": 2, "rebalance_every": 24},
        "limits": {"max_weight_per_symbol": 0.30, "halt_on_breach": False},
    },
    {
        "name": "momentum-randomwalk-uncapped",
        "data": {"kind": "random_walk", "n_bars": 2000, "seed": 3},
        "strategy": {"kind": "momentum", "lookback": 72, "n_long": 1, "rebalance_every": 12},
        "limits": {"max_weight_per_symbol": 1.0, "halt_on_breach": False},
    },
    {
        "name": "meanrev-meanreverting",
        "data": {"kind": "mean_reverting", "n_bars": 2000, "seed": 5},
        "strategy": {"kind": "meanrev", "lookback": 72, "entry_z": -1.5, "max_positions": 2},
        "limits": {"max_weight_per_symbol": 0.30, "halt_on_breach": False},
    },
    {
        "name": "hold-trending",
        "data": {"kind": "trending", "n_bars": 1500, "seed": 1},
        "strategy": {"kind": "hold"},
        "limits": {"max_weight_per_symbol": 1.0, "halt_on_breach": False},
    },
    {
        "name": "eqw-trending-capped",
        "data": {"kind": "trending", "n_bars": 1500, "seed": 1},
        "strategy": {"kind": "eqw"},
        "limits": {"max_weight_per_symbol": 0.30, "halt_on_breach": False},
    },
    {
        # A run that actually trips the breaker: halting changes the loop's
        # exit path, the final mark and the flatten orders.
        "name": "hold-halting",
        "data": {"kind": "random_walk", "n_bars": 1200, "seed": 7, "drift_per_bar": -0.004},
        "strategy": {"kind": "hold"},
        "limits": {
            "max_weight_per_symbol": 1.0,
            "max_daily_loss_pct": 0.03,
            "max_drawdown_pct": 0.05,
            "halt_on_breach": True,
        },
        "expect_halt": True,
    },
    {
        # Turnover clamping takes a different branch in the gate, which
        # rewrites every target rather than just scaling it.
        "name": "eqw-turnover-clamped",
        "data": {"kind": "random_walk", "n_bars": 1200, "seed": 9},
        "strategy": {"kind": "eqw"},
        "limits": {
            "max_weight_per_symbol": 0.30,
            "max_turnover_per_bar": 0.05,
            "halt_on_breach": False,
        },
    },
]


def run_case(case: dict) -> dict:
    """Run one case and reduce it to an exactly-comparable digest."""
    frames = synthetic_panel(
        SYMBOLS,
        case["data"]["n_bars"],
        "1h",
        seed=case["data"]["seed"],
        kind=case["data"]["kind"],
        **{k: v for k, v in case["data"].items() if k not in ("kind", "n_bars", "seed")},
    )
    limits = RiskLimits(symbol_allowlist=frozenset(SYMBOLS), **case["limits"])
    result = run_backtest(
        frames,
        _strategy(case["strategy"]),
        costs=CostModel(),
        limits=limits,
        initial_equity=10_000.0,
        cash_buffer=0.02,
        run_id="golden",  # fixed, so client_order_ids are comparable too
    )

    h = hashlib.sha256()
    for ts, value in result.equity.items():
        h.update(f"{ts.isoformat()}|{value!r}".encode())
    for fill in result.fills:
        h.update(
            f"{fill.client_order_id}|{fill.ts_filled.isoformat()}|{fill.symbol}|"
            f"{fill.side.value}|{fill.qty!r}|{fill.price!r}|{fill.fee!r}|"
            f"{fill.slippage_bps!r}".encode()
        )
    for violation in result.violations:
        h.update(f"{violation.ts.isoformat()}|{violation.rule}|{violation.detail}".encode())

    return {
        "digest": h.hexdigest(),
        # Diagnostics only. Never asserted on their own — they exist so a
        # failure says how far behaviour moved, not merely that it did.
        "summary": {
            "n_equity_points": int(len(result.equity)),
            "n_fills": int(len(result.fills)),
            "n_violations": int(len(result.violations)),
            "final_equity": float(result.equity.iloc[-1]) if len(result.equity) else None,
            "fees_paid": float(result.total_fees),
            "halted_at": str(result.halted_at) if result.halted_at is not None else None,
            "halt_reason": result.halt_reason,
            "clamp_rate": float(result.meta["clamp_rate"]),
        },
    }


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"no golden file at {GOLDEN_PATH}; run this file with --update")
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_engine_output_is_bit_for_bit_unchanged(case):
    golden = _load_golden()
    assert case["name"] in golden, (
        f"{case['name']} is a new case with no recorded baseline; regenerate "
        f"the golden file deliberately."
    )
    expected = golden[case["name"]]
    actual = run_case(case)

    if actual["digest"] != expected["digest"]:
        lines = [
            f"engine output changed for {case['name']}.",
            "This is only acceptable if you MEANT to change behaviour; every",
            "result recorded before this point stops being comparable.",
            "",
            f"{'field':<22}{'expected':>20}{'actual':>20}",
        ]
        for key in expected["summary"]:
            lines.append(
                f"{key:<22}{str(expected['summary'][key]):>20}"
                f"{str(actual['summary'][key]):>20}"
            )
        pytest.fail("\n".join(lines))


def test_the_halting_case_actually_halts():
    """Guards the pin itself: a case that stopped halting would pin nothing."""
    for case in CASES:
        if case.get("expect_halt"):
            assert run_case(case)["summary"]["halted_at"] is not None


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        print(__doc__)
        print("Refusing to regenerate without --update.")
        raise SystemExit(1)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for case in CASES:
        print(f"  running {case['name']} ...", flush=True)
        out[case["name"]] = run_case(case)
        print(f"    {out[case['name']]['digest'][:16]}  {out[case['name']]['summary']}")
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {GOLDEN_PATH}")
