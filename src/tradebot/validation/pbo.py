"""Probability of Backtest Overfitting, via CSCV.

Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado,
Zhu). The question it answers is not "is my strategy good?" but "when I pick
the best of my candidates on one half of the data, how often is it below
median on the other half?"

If that happens about half the time, your selection procedure has no skill:
you are choosing noise. PBO is that frequency. Above ~0.5 means the search
itself is worthless, and the honest response is to stop tuning rather than to
tune differently.

Note what this measures: the *procedure*, not one strategy. So feed it the
whole family of variants you considered, including the ones you rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    n_combinations: int
    n_candidates: int
    n_blocks: int
    logits: np.ndarray
    # Median OOS performance of the IS-selected candidate, in Sharpe units.
    median_oos_sharpe_of_selected: float

    def verdict(self) -> str:
        if math.isnan(self.pbo):
            return "indeterminate"
        if self.pbo < 0.1:
            return "selection looks robust"
        if self.pbo < 0.3:
            return "some overfitting; treat the chosen variant with caution"
        if self.pbo < 0.5:
            return "substantial overfitting"
        return "the selection procedure has no demonstrable skill"


def _block_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (bars x candidates) block, unannualised."""
    if block.shape[0] < 2:
        return np.zeros(block.shape[1])
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    out = np.zeros_like(mu)
    ok = (sd > 0) & np.isfinite(sd)
    out[ok] = mu[ok] / sd[ok]
    return out


def probability_of_backtest_overfitting(
    returns: pd.DataFrame, n_blocks: int = 10, max_combinations: int = 2000
) -> PBOResult:
    """`returns`: one column per candidate variant, rows aligned in time."""
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even so it can split in half")
    if returns.shape[1] < 2:
        raise ValueError(
            "PBO needs at least 2 candidates: it measures whether *choosing* "
            "between variants has skill. With one variant there is no choice."
        )
    data = returns.dropna(how="any").to_numpy(dtype="float64")
    n_rows, n_cand = data.shape
    if n_rows < n_blocks * 2:
        raise ValueError(f"{n_rows} rows is too few for {n_blocks} blocks")

    # Contiguous blocks preserve autocorrelation within each block, which
    # random row sampling would destroy.
    edges = np.linspace(0, n_rows, n_blocks + 1, dtype=int)
    blocks = [data[edges[i] : edges[i + 1]] for i in range(n_blocks)]

    half = n_blocks // 2
    combos = list(combinations(range(n_blocks), half))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    logits: list[float] = []
    oos_of_selected: list[float] = []
    for is_idx in combos:
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_data = np.vstack([blocks[i] for i in is_idx])
        oos_data = np.vstack([blocks[i] for i in oos_idx])

        is_sr = _block_sharpe(is_data)
        oos_sr = _block_sharpe(oos_data)

        best = int(np.argmax(is_sr))
        oos_of_selected.append(float(oos_sr[best]))

        # Relative rank of the selected candidate in the OOS ordering.
        order = np.argsort(oos_sr)  # ascending
        rank = int(np.where(order == best)[0][0]) + 1
        omega = rank / (n_cand + 1.0)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))

    arr = np.asarray(logits)
    # lambda < 0 means the IS-best landed below the OOS median.
    pbo = float((arr < 0).mean()) if arr.size else float("nan")
    return PBOResult(
        pbo=pbo,
        n_combinations=len(combos),
        n_candidates=n_cand,
        n_blocks=n_blocks,
        logits=arr,
        median_oos_sharpe_of_selected=float(np.median(oos_of_selected))
        if oos_of_selected
        else float("nan"),
    )
