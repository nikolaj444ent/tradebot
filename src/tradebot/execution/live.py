"""Live trading: not implemented, on purpose.

This file exists so that the absence of live trading is a visible, deliberate
decision rather than an oversight someone fills in casually.

Enabling live trading is not a config flag. It is a reviewed change that must,
at minimum, come after:

  1. A walk-forward result with out-of-sample Sharpe that survives the
     deflated-Sharpe adjustment for the number of variants actually tried.
  2. At least one to three months of paper trading spanning a regime change,
     with live tracking error against the backtest that is explainable.
  3. A reconciliation loop that halts on any mismatch between internal state
     and venue state.
  4. Alerting that pages a human when the process dies holding a position —
     the failure that actually costs money is not a bad trade, it is a silent
     death with an open book.
  5. A deliberate decision about size, written down, that you would be
     content to lose entirely.

Until then, importing this module is an error rather than a foot-gun.
"""

from __future__ import annotations


class LiveTradingDisabled(RuntimeError):
    pass


_MESSAGE = (
    "Live trading is not implemented in this repository. This is deliberate: see "
    "the checklist in src/tradebot/execution/live.py and DESIGN.md section 7. "
    "Run `--mode paper` instead."
)


def __getattr__(name: str):
    raise LiveTradingDisabled(_MESSAGE)


raise LiveTradingDisabled(_MESSAGE)
