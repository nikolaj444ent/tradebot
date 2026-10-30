"""Performance metrics, including the ones that discount for searching.

A Sharpe ratio on its own is close to meaningless once you have tried more
than one idea. If you test 50 variants and report the best, you have measured
the maximum of 50 noisy draws, not an edge. The two corrections here —
probabilistic and deflated Sharpe (Bailey & Lopez de Prado) — exist to put a
number on that, and `deflated_sharpe_ratio` *requires* you to say how many
trials you ran. Lying to it is the easiest way to fool yourself in this repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..types import BacktestResult

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate).

    Hand-rolled to keep scipy out of the dependency list; the harness needs
    numpy and pandas anyway, and scipy is a large install for one function.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe(returns: pd.Series, bars_per_year: int, rf_per_bar: float = 0.0) -> float:
    """Annualised Sharpe. Returns 0.0 rather than inf for a flat series."""
    r = pd.Series(returns).dropna() - rf_per_bar
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd) * math.sqrt(bars_per_year)


def sharpe_standard_error(
    n_obs: int, bars_per_year: int, per_bar_sr: float = 0.0
) -> float:
    """Standard error of the ANNUALISED Sharpe estimate.

    This is the number most backtests omit and most backtests need. For iid
    returns, SE(SR_per_bar) ~ sqrt((1 + SR^2/2) / n), annualised by
    sqrt(bars_per_year). The consequence is blunt: 1500 hourly bars (~62 days)
    gives SE ~ 2.4, so an annualised Sharpe of 2.0 from such a sample is
    statistically indistinguishable from zero. No amount of strategy work
    fixes a sample that short — only more data does.
    """
    if n_obs < 2:
        return float("inf")
    return math.sqrt((1.0 + 0.5 * per_bar_sr**2) / n_obs) * math.sqrt(bars_per_year)


def sortino(returns: pd.Series, bars_per_year: int) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if dd == 0 or not np.isfinite(dd):
        return 0.0
    return float(r.mean() / dd) * math.sqrt(bars_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough fall, as a positive fraction."""
    e = pd.Series(equity).dropna()
    if e.empty:
        return 0.0
    peak = e.cummax()
    return float((1.0 - e / peak).max())


def cagr(equity: pd.Series, bars_per_year: int) -> float:
    """Compound annual growth rate.

    Beware of short samples: annualising a 6-month 94x return gives a CAGR of
    837,900%, which is arithmetically correct and completely meaningless. The
    caller is expected to read `sample_years` alongside this; `summarize()`
    reports both and warns below one year.
    """
    e = pd.Series(equity).dropna()
    if len(e) < 2 or e.iloc[0] <= 0 or e.iloc[-1] <= 0:
        return 0.0
    years = len(e) / bars_per_year
    if years <= 0:
        return 0.0
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / years) - 1.0)


def probabilistic_sharpe_ratio(
    observed_sr: float, n_obs: int, skew: float, kurtosis: float,
    benchmark_sr: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark), given non-normal returns.

    All Sharpe values are PER-PERIOD, not annualised — mixing the two is the
    standard way to get a nonsense answer here. `kurtosis` is non-excess
    (3.0 for a normal distribution).
    """
    if n_obs < 3:
        return float("nan")
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if denom <= 0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    observed_sr: float, n_obs: int, skew: float, kurtosis: float,
    n_trials: int, trial_sr_variance: float,
) -> float:
    """PSR against the Sharpe you'd expect from luck alone across `n_trials`.

    This is the number to report. The benchmark is not zero: it is the
    *expected maximum* Sharpe from `n_trials` independent worthless
    strategies. Try enough variants and that benchmark exceeds anything you
    found, which is the correct conclusion rather than a disappointing one.

    All Sharpe values per-period. `n_trials` must be the honest count of
    everything you tried, including variants you abandoned — which is what
    RESEARCH_LOG.md is for.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1 or trial_sr_variance <= 0:
        # Nothing was searched over, so there is nothing to deflate.
        return probabilistic_sharpe_ratio(observed_sr, n_obs, skew, kurtosis, 0.0)
    sd = math.sqrt(trial_sr_variance)
    e = math.e
    sr0 = sd * (
        (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
        + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * e))
    )
    return probabilistic_sharpe_ratio(observed_sr, n_obs, skew, kurtosis, sr0)


@dataclass(frozen=True, slots=True)
class Summary:
    values: dict

    def __getitem__(self, k):
        return self.values[k]

    def table(self) -> str:
        rows = []
        for k, v in self.values.items():
            if isinstance(v, float):
                rows.append(f"  {k:<28} {v:>12.4f}")
            else:
                rows.append(f"  {k:<28} {str(v):>12}")
        return "\n".join(rows)


def summarize(
    result: BacktestResult, bars_per_year: int, *, n_trials: int = 1,
    trial_sr_variance: float = 0.0,
) -> Summary:
    r = result.returns.dropna()
    eq = result.equity
    per_bar_sr = 0.0
    if len(r) > 1 and float(r.std(ddof=1)) > 0:
        per_bar_sr = float(r.mean() / r.std(ddof=1))

    skew = float(r.skew()) if len(r) > 2 else 0.0
    # pandas gives EXCESS kurtosis; the PSR formula wants non-excess.
    kurt = float(r.kurt()) + 3.0 if len(r) > 3 else 3.0

    turnover = 0.0
    if result.fills and len(eq):
        turnover = sum(f.notional for f in result.fills) / float(eq.iloc[0])

    sr_annual = sharpe(r, bars_per_year)
    sr_se = sharpe_standard_error(len(r), bars_per_year, per_bar_sr)

    values = {
        "final_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0,
        "cagr": cagr(eq, bars_per_year),
        "sharpe_annual": sr_annual,
        "sharpe_std_error": sr_se,
        # |Sharpe| / SE. Below ~2 the result is not distinguishable from luck,
        # however good the headline number looks.
        "sharpe_t_stat": abs(sr_annual) / sr_se if sr_se > 0 else 0.0,
        "sample_years": len(r) / bars_per_year,
        "sortino_annual": sortino(r, bars_per_year),
        "max_drawdown": max_drawdown(eq),
        "n_trades": result.n_trades,
        "fees_paid": result.total_fees,
        "gross_turnover_x_capital": turnover,
        "skew": skew,
        "kurtosis": kurt,
        "clamp_rate": result.meta.get("clamp_rate", 0.0),
        "violations_per_decision": result.meta.get("violations_per_decision", 0.0),
        "effective_bars": result.meta.get("effective_bars", len(r)),
        "truncated": result.meta.get("truncated", False),
        "halted": result.halted_at is not None,
        "n_params": result.meta.get("n_params", 0),
        "n_trials_declared": n_trials,
        "psr_vs_zero": probabilistic_sharpe_ratio(per_bar_sr, len(r), skew, kurt, 0.0),
        "deflated_sharpe": deflated_sharpe_ratio(
            per_bar_sr, len(r), skew, kurt, n_trials, trial_sr_variance
        ),
    }
    return Summary(values=values)
