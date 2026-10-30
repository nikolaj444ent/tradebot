"""Typed configuration loading.

Config is validated on load, not on use. A harness that discovers its universe
is misconfigured three hours into a paper run has wasted three hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .costs import CostModel
from .risk import RiskLimits

# Bars per year, used to annualise Sharpe and vol. Crypto trades continuously,
# so unlike equities there is no 252-day convention: a day is a full 24h.
BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


@dataclass(frozen=True, slots=True)
class DataConfig:
    exchange: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframe: str = "1h"
    start: str = "2021-01-01"
    end: str | None = None
    snapshot_dir: Path = Path("data/snapshots")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    # Fraction of equity held back as cash buffer, so that a rebalance never
    # needs more cash than exists after fees.
    cash_buffer: float = 0.02


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_bars: int = 24 * 180  # 180 days at 1h
    test_bars: int = 24 * 60  # 60 days at 1h
    step_bars: int = 24 * 60  # non-overlapping test windows
    min_folds: int = 3


@dataclass(frozen=True, slots=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskLimits = field(default_factory=RiskLimits)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    @property
    def bars_per_year(self) -> int:
        tf = self.data.timeframe
        if tf not in BARS_PER_YEAR:
            raise ValueError(f"unknown timeframe {tf!r}; add it to BARS_PER_YEAR")
        return BARS_PER_YEAR[tf]

    def validate(self) -> None:
        if not self.data.symbols:
            raise ValueError("data.symbols is empty")
        allow = self.risk.symbol_allowlist
        extra = set(self.data.symbols) - set(allow)
        if extra:
            raise ValueError(
                f"symbols {sorted(extra)} are in the universe but not in "
                f"risk.symbol_allowlist. The allowlist is the authority; add them "
                f"there deliberately."
            )
        _ = self.bars_per_year
        n_sym = len(self.data.symbols)
        if self.risk.max_weight_per_symbol * n_sym < self.risk.max_gross_exposure:
            raise ValueError(
                f"{n_sym} symbols capped at {self.risk.max_weight_per_symbol:.2f} each "
                f"cannot reach gross exposure {self.risk.max_gross_exposure:.2f}; the "
                f"strategy could never be fully invested."
            )


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    data_raw = dict(raw.get("data", {}))
    if "symbols" in data_raw:
        data_raw["symbols"] = tuple(data_raw["symbols"])
    if "snapshot_dir" in data_raw:
        data_raw["snapshot_dir"] = Path(data_raw["snapshot_dir"])

    risk_raw = dict(raw.get("risk", {}))
    # The allowlist defaults to the configured universe, but must be written
    # down in the file to be non-empty — see RiskGate.clamp.
    if "symbol_allowlist" in risk_raw:
        risk_raw["symbol_allowlist"] = frozenset(risk_raw["symbol_allowlist"])

    cfg = Config(
        data=DataConfig(**data_raw),
        costs=CostModel(**raw.get("costs", {})),
        risk=RiskLimits(**risk_raw),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        walkforward=WalkForwardConfig(**raw.get("walkforward", {})),
    )
    cfg.validate()
    return cfg
