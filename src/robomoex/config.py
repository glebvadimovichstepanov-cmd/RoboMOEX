"""Explicit validated configuration. No credentials are needed by stages 1 and 2."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class StrategyConfig:
    vol_1d_mult: float = 0.5
    vol_15_mult: float = 0.5
    vol_1m_mult: float = 1.0
    rsi_1d_thresh: float = 40.0
    rsi_15_thresh: float = 40.0
    rsi_1m_thresh: float = 35.0
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 1.5
    doji_thresh: float = 0.3

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key.startswith("rsi_") and value > 100:
                raise ValueError(f"{key} must be <= 100")
        if self.doji_thresh > 1:
            raise ValueError("doji_thresh must be <= 1")


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    lot_size: int = 10
    quantity_lots: int = 1
    tick_size: float = 0.01

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        for key in ("lot_size", "quantity_lots"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.initial_cash <= 0 or self.tick_size <= 0 or self.fee_rate >= 1:
            raise ValueError("Invalid cash, tick size or fee")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be < 10000")


def load_config(path: str | Path | None) -> tuple[StrategyConfig, ExecutionConfig]:
    if path is None:
        return StrategyConfig(), ExecutionConfig()
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) - {"strategy", "execution"}:
        raise ValueError("Unknown configuration section")
    result = []
    for name, cls in (("strategy", StrategyConfig), ("execution", ExecutionConfig)):
        values = data.get(name, {})
        if set(values) - {field.name for field in fields(cls)}:
            raise ValueError(f"Unknown {name} parameter")
        result.append(cls(**values))
    return tuple(result)
