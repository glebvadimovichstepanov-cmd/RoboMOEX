"""Explainable, causal signal and risk engine for daily SNGS research.

The engine accepts market data plus optional external context. Missing context is
explicitly represented as unavailable and can only reduce a signal. All rolling
statistics use observations strictly before the decision row.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.003
    max_position_weight: float = 0.05
    max_daily_loss: float = 0.015
    max_monthly_drawdown: float = 0.05
    max_drawdown_stop: float = 0.12
    stop_atr: float = 2.5
    trailing_atr: float = 3.5
    time_stop_bars: int = 7
    max_spread_bps: float = 25.0
    min_avg_turnover: float = 30_000_000.0
    target_volatility: float = 0.14


@dataclass(frozen=True)
class CompositeConfig:
    buy_threshold: float = 0.50
    exit_threshold: float = 0.10
    short_threshold: float = -0.50
    trend_weight: float = 0.25
    momentum_weight: float = 0.20
    oil_weight: float = 0.20
    fx_weight: float = 0.10
    rate_weight: float = 0.10
    relative_weight: float = 0.10
    liquidity_weight: float = 0.05


def _winsor(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    q = series.dropna().quantile([low, high])
    return series.clip(q.iloc[0], q.iloc[1]) if len(q) == 2 else series


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=max(20, window // 2)).mean()
    std = prior.rolling(window, min_periods=max(20, window // 2)).std(ddof=1)
    return _winsor((series - mean) / std.replace(0, np.nan)).clip(-4, 4) / 4


def build_features(daily: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build causal daily features from OHLCV and optional date-indexed context."""
    required = {"open_time", "close_time", "open", "high", "low", "close", "volume"}
    if required - set(daily.columns):
        raise ValueError("daily OHLCV columns are required")
    x = daily.copy().reset_index(drop=True)
    close, high, low = x.close.astype(float), x.high.astype(float), x.low.astype(float)
    ret = close.pct_change()
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    ema20 = close.ewm(span=20, min_periods=20, adjust=False).mean()
    ema50 = close.ewm(span=50, min_periods=50, adjust=False).mean()
    ema200 = close.ewm(span=200, min_periods=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).clip(0, 100)
    x["log_ret_1"] = np.log(close).diff()
    x["log_ret_5"] = np.log(close).diff(5)
    x["log_ret_20"] = np.log(close).diff(20)
    x["ema20"], x["ema50"], x["ema200"] = ema20, ema50, ema200
    x["rsi14"] = rsi
    x["atr14"] = atr
    x["price_z20"] = _zscore(close, 20)
    x["realized_vol20"] = ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    x["volume_ratio20"] = x.volume / x.volume.shift(1).rolling(20, min_periods=20).mean()
    x["turnover"] = close * x.volume
    if context is not None:
        c = context.copy()
        if "date" not in c:
            c["date"] = (
                pd.to_datetime(c["timestamp"]).dt.date
                if "timestamp" in c
                else pd.to_datetime(c["open_time"]).dt.date
            )
        x["date"] = pd.to_datetime(x.close_time).dt.tz_convert("Europe/Moscow").dt.date
        x = x.merge(c, on="date", how="left", suffixes=("", "_ctx"))
        for name in (
            "imoex",
            "oil",
            "usd_rub",
            "cny_rub",
            "rgbI",
            "rate",
            "spread_bps",
            "event_penalty",
            "event_risk",
        ):
            if name in x:
                x[f"{name}_ret5"] = (
                    x[name].pct_change(5)
                    if name not in ("rate", "spread_bps", "event_penalty", "event_risk")
                    else x[name].diff(5)
                )
        if "imoex" in x:
            x["relative_imoex20"] = (close.pct_change(20) - x.imoex.pct_change(20)).replace(
                [np.inf, -np.inf], np.nan
            )
        if "oil" in x:
            x["oil_corr60"] = ret.rolling(60, min_periods=30).corr(x.oil.pct_change())
        if "usd_rub" in x:
            x["fx_corr60"] = ret.rolling(60, min_periods=30).corr(x.usd_rub.pct_change())
    return x


def _score(value: float | None) -> float:
    return 0.0 if value is None or not np.isfinite(value) else float(np.tanh(value))


def position_size(
    equity: float, price: float, atr: float, annual_vol: float | None, risk: RiskLimits
) -> float:
    if min(equity, price, atr) <= 0:
        return 0.0
    by_risk = equity * risk.risk_per_trade / (risk.stop_atr * atr)
    by_vol = (
        equity * risk.target_volatility / (annual_vol * np.sqrt(252)) / price
        if annual_vol and annual_vol > 0
        else np.inf
    )
    return max(0.0, min(by_risk, by_vol, equity * risk.max_position_weight / price))


def make_signal(
    row: pd.Series,
    *,
    equity: float,
    position: str = "FLAT",
    config: CompositeConfig | None = None,
    risk: RiskLimits | None = None,
) -> dict:
    config, risk = config or CompositeConfig(), risk or RiskLimits()
    trend = _score(
        (
            (row.get("close", np.nan) / row.get("ema20", np.nan) - 1)
            + (row.get("ema20", np.nan) / row.get("ema50", np.nan) - 1)
            + (row.get("ema50", np.nan) / row.get("ema200", np.nan) - 1)
        )
        * 12
    )
    momentum = _score(
        (row.get("log_ret_5", np.nan) + row.get("log_ret_20", np.nan)) * 8
        + (row.get("rsi14", 50) - 50) / 20
    )
    oil = _score(
        row.get("oil_ret5", np.nan)
        * (1 if abs(row.get("oil_corr60", 0)) < 0.1 else np.sign(row.get("oil_corr60", 0)))
        * 8
    )
    fx = _score(
        row.get("usd_rub_ret5", np.nan)
        * (1 if abs(row.get("fx_corr60", 0)) < 0.1 else np.sign(row.get("fx_corr60", 0)))
        * 8
    )
    rate = _score(-row.get("rate_ret5", row.get("rgbI_ret5", np.nan)) * 4)
    relative = _score(row.get("relative_imoex20", np.nan) * 8)
    liquidity_ok = bool(
        row.get("turnover", 0) >= risk.min_avg_turnover
        and row.get("spread_bps", 0) <= risk.max_spread_bps
    )
    liquidity = 1.0 if liquidity_ok else -1.0
    penalty = float(np.clip(row.get("event_penalty", 1.0), 0, 1))
    score = float(
        np.clip(
            (
                config.trend_weight * trend
                + config.momentum_weight * momentum
                + config.oil_weight * oil
                + config.fx_weight * fx
                + config.rate_weight * rate
                + config.relative_weight * relative
                + config.liquidity_weight * liquidity
            )
            * penalty,
            -1,
            1,
        )
    )
    event = str(row.get("event_risk", "LOW")).upper()
    high_event = event in {"HIGH", "EXTREME"}
    regime = "RISK_ON" if trend > 0.15 else "RISK_OFF" if trend < -0.15 else "NEUTRAL"
    action = "NO_TRADE"
    if not liquidity_ok or high_event:
        action = "EXIT" if position != "FLAT" else "NO_TRADE"
    elif position == "LONG" and score < config.exit_threshold:
        action = "EXIT"
    elif position == "FLAT" and score > config.buy_threshold and regime == "RISK_ON":
        action = "BUY"
    elif position == "SHORT" and score > config.exit_threshold:
        action = "EXIT"
    elif position == "FLAT" and score < config.short_threshold:
        action = "NO_TRADE"  # borrow availability is not assumed
    reasons = [f"trend={trend:.3f}", f"momentum={momentum:.3f}", f"relative={relative:.3f}"]
    warnings = [] if liquidity_ok else ["liquidity limits not met"]
    if high_event:
        warnings.append("high event risk")
    return {
        "timestamp": str(row.get("close_time")),
        "ticker": "SNGS",
        "action": action,
        "signal_score": score,
        "confidence": float(min(1, abs(score))),
        "position_weight_target": position_size(
            equity,
            float(row.get("close", 0)),
            float(row.get("atr14", 0) or 0),
            row.get("realized_vol20"),
            risk,
        ),
        "stop_price": float(row.close - risk.stop_atr * row.atr14)
        if action == "BUY" and pd.notna(row.get("atr14"))
        else None,
        "trailing_stop_distance_atr": risk.trailing_atr if action == "BUY" else None,
        "time_stop_bars": risk.time_stop_bars if action == "BUY" else None,
        "market_regime": regime,
        "liquidity_ok": liquidity_ok,
        "event_risk_level": event if event in {"LOW", "MEDIUM", "HIGH", "EXTREME"} else "LOW",
        "reasons": reasons,
        "warnings": warnings,
    }


def signal_frame(
    features: pd.DataFrame,
    *,
    equity: float = 100_000,
    position: str = "FLAT",
    config: CompositeConfig | None = None,
    risk: RiskLimits | None = None,
) -> list[dict]:
    return [
        make_signal(row, equity=equity, position=position, config=config, risk=risk)
        for _, row in features.iterrows()
    ]

