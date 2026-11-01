# Research log

**Every variant you test goes here, including the ones you abandon.**

This is not bookkeeping. The deflated Sharpe ratio needs an honest count of
how many things you tried, because the right benchmark is not zero — it is the
expected maximum Sharpe of that many worthless strategies. The same observed
per-bar Sharpe of 0.05 is worth 0.94 after one trial and 0.36 after five
hundred. Nothing about the strategy changed; only how hard you looked.

So the count only works if it is complete. A variant you tried, disliked, and
deleted still counts. Keeping this file is the cheapest form of intellectual
honesty available here.

## How to log

One row per *distinct configuration you evaluated*. A walk-forward over a
6-point grid across 7 folds is 42 trials, not 1 — record the grid size and let
the tool report `n_trials`.

| Date | Idea | Variants | Data snapshot | OOS Sharpe | Deflated Sharpe | PBO | Verdict |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | harness smoke test: cross-sectional momentum on synthetic trend data | 6-point grid | `synthetic:trending:seed0` | n/a | n/a | n/a | Not a finding. Momentum scoring well on trend-generated data is a tautology; this only proved the plumbing runs end to end. |
| 2026-11-01 | cross-sectional momentum, walk-forward over the 6-point grid | 192 trials (6 combos × 32 folds) | `729e29a33b276ac9` (51,128 1h bars, 2021..2026-11) | 0.68 ± 0.44 (t=1.56) | **0.00** | 0.60 | **No edge.** OOS Sharpe not distinguishable from zero over 5.26 years; deflated Sharpe zero after 192 trials; PBO ≈ 0.6 means the in-sample winner picks *worse* than a coin flip out of sample; lookback churns across folds (CV 0.49); OOS max drawdown 85%. The harness did its job: idea killed in 12 minutes. The live paper run of momentum continues as a *daemon* validation exercise only, not a promoted strategy. |

## Running total

- **Ideas evaluated on real data:** 1
- **Cumulative trials on real data:** 192
- **Promoted to paper trading:** 0 (the running momentum paper run validates the daemon, not the strategy)
- **Promoted to live:** 0 (live trading is not implemented; see DESIGN.md §7)

## Notes on what counts as a new idea

Changing a lookback from 168 to 336 is a **variant**. Changing from
"hold the strongest asset" to "hold the one with the lowest volatility" is a
**new idea** with its own trial count. When in doubt, count it as a variant —
it discounts your result more, which is the safer error.
