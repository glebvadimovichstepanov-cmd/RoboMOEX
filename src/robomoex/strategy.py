"""Causal long-only multi-timeframe confluence, adapted from source run_strategy_v2."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import aggregate


def indicators(bars: pd.DataFrame, median_window: int) -> pd.DataFrame:
    c, h, lo = bars.close, bars.high, bars.low
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    return pd.DataFrame(
        {
            "close_time": bars.close_time,
            "close": c,
            "ema10": c.ewm(span=10, min_periods=10, adjust=False).mean(),
            "ema32": c.ewm(span=32, min_periods=32, adjust=False).mean(),
            "rsi": rsi,
            "atr": atr,
            "atr_med": atr.rolling(median_window, min_periods=median_window).median(),
            "body_ratio": (c - bars.open).abs() / (h - lo).replace(0, np.nan),
        }
    )


def signals(
    bars: pd.DataFrame, sessions: pd.DataFrame, config: StrategyConfig | None = None
) -> pd.DataFrame:
    config = config or StrategyConfig()
    base = indicators(bars, 60)
    aligned = {}
    for label, length, median in (("m15", 15, 20), ("d1", None, 2)):
        higher = aggregate(bars, sessions, length)
        if higher.empty:
            # No complete higher bars means no entries, including warm-up.
            return pd.DataFrame({"enter": False, "atr": base.atr}, index=bars.index)
        feature = indicators(higher, median)
        aligned[label] = pd.merge_asof(
            base[["close_time"]], feature, on="close_time", direction="backward"
        )
    daily, m15 = aligned["d1"], aligned["m15"]
    ready = pd.Series(True, index=bars.index)
    for frame in (base, daily, m15):
        ready &= frame[["rsi", "atr", "atr_med", "ema32", "body_ratio"]].notna().all(axis=1)
    trend = (daily.close > daily.ema10) & (daily.ema10 > daily.ema32)
    volatility = (
        (daily.atr > config.vol_1d_mult * daily.atr_med)
        & (m15.atr > config.vol_15_mult * m15.atr_med)
        & (base.atr > config.vol_1m_mult * base.atr_med)
    )
    rsi_filter = (
        (daily.rsi < config.rsi_1d_thresh)
        | (m15.rsi < config.rsi_15_thresh)
        | (base.rsi < config.rsi_1m_thresh)
    )
    entry = ready & trend & volatility & rsi_filter & (base.body_ratio >= config.doji_thresh)
    result = pd.DataFrame({"enter": entry, "atr": base.atr}, index=bars.index)
    result.attrs["strategy"] = {"name": "confluence-long-v1", **asdict(config)}
    return result
