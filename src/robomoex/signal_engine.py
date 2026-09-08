"""Explainable, causal signal and risk engine for daily SNGS research.

The engine accepts market data plus optional external context. Missing context is
explicitly represented as unavailable and can only reduce a signal. All rolling
statistics use observations strictly before the decision row.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import validate_bars


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
    max_entry_volatility: float = 0.60
    max_participation: float = 0.01


@dataclass(frozen=True)
class CompositeConfig:
    require_context: bool = True
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
    microstructure_weight: float = 0.0  # opt-in until walk-forward evidence supports allocation


def _winsor(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    prior = series.shift(1).expanding(min_periods=20)
    return series.clip(prior.quantile(low), prior.quantile(high))


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=max(20, window // 2)).mean()
    std = prior.rolling(window, min_periods=max(20, window // 2)).std(ddof=1)
    return _winsor((series - mean) / std.replace(0, np.nan)).clip(-4, 4) / 4


def build_features(
    daily: pd.DataFrame,
    context: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal daily features from OHLCV and optional date-indexed context."""
    required = {"open_time", "close_time", "open", "high", "low", "close", "volume"}
    if required - set(daily.columns):
        raise ValueError("daily OHLCV columns are required")
    x = validate_bars(daily)
    if corporate_actions is not None:
        from .corporate_actions import adjust_prices

        x = adjust_prices(x, corporate_actions)
    close, high, low = x.close.astype(float), x.high.astype(float), x.low.astype(float)
    raw_close = close
    close = x.get("signal_close", close).astype(float)
    ret = close.pct_change(fill_method=None)
    tr = pd.concat(
        [high - low, (high - raw_close.shift()).abs(), (low - raw_close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    ema20 = close.ewm(span=20, min_periods=20, adjust=False).mean()
    ema50 = close.ewm(span=50, min_periods=50, adjust=False).mean()
    ema200 = close.ewm(span=200, min_periods=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).clip(0, 100)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    x["log_ret_1"] = np.log(close).diff()
    x["log_ret_5"] = np.log(close).diff(5)
    x["log_ret_20"] = np.log(close).diff(20)
    x["ema20"], x["ema50"], x["ema200"] = ema20, ema50, ema200
    x["rsi14"] = rsi
    x["macd"] = (
        close.ewm(span=12, min_periods=12, adjust=False).mean()
        - close.ewm(span=26, min_periods=26, adjust=False).mean()
    )
    x["macd_signal"] = x.macd.ewm(span=9, min_periods=9, adjust=False).mean()
    x["macd_histogram"] = x.macd - x.macd_signal
    x["atr14"] = atr
    x["price_z20"] = _zscore(close, 20)
    x["realized_vol20"] = ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    x["volume_ratio20"] = x.volume / x.volume.shift(1).rolling(20, min_periods=20).mean()
    x["turnover"] = raw_close * x.volume
    x["avg_turnover20"] = x.turnover.shift(1).rolling(20, min_periods=20).mean()
    if context is not None:
        c = context.copy()
        if "date" not in c:
            c["date"] = (
                pd.to_datetime(c["timestamp"]).dt.date
                if "timestamp" in c
                else pd.to_datetime(c["open_time"]).dt.date
            )
        x["date"] = pd.to_datetime(x.close_time).dt.tz_convert("Europe/Moscow").dt.date
        c["date"] = pd.to_datetime(c["date"]).dt.date
        if c.date.duplicated().any():
            raise ValueError("Duplicate context dates")
        # Daily market observations are conservatively usable the following Moscow day.
        c["date"] = c.date + pd.Timedelta(days=1)
        x = x.merge(c, on="date", how="left", suffixes=("", "_ctx"), validate="one_to_one")
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
                if name == "event_risk":
                    continue
                x[f"{name}_ret5"] = (
                    x[name].pct_change(5, fill_method=None)
                    if name not in ("rate", "spread_bps", "event_penalty", "event_risk")
                    else x[name].diff(5)
                )
        if "imoex" in x:
            x["relative_imoex20"] = (
                close.pct_change(20, fill_method=None) - x.imoex.pct_change(20, fill_method=None)
            ).replace([np.inf, -np.inf], np.nan)
        if "oil" in x:
            x["oil_corr60"] = ret.rolling(60, min_periods=30).corr(
                x.oil.pct_change(fill_method=None)
            )
            x["oil_corr120"] = ret.rolling(120, min_periods=60).corr(
                x.oil.pct_change(fill_method=None)
            )
        if "usd_rub" in x:
            x["fx_corr60"] = ret.rolling(60, min_periods=30).corr(
                x.usd_rub.pct_change(fill_method=None)
            )
            x["fx_corr120"] = ret.rolling(120, min_periods=60).corr(
                x.usd_rub.pct_change(fill_method=None)
            )
        for name in ("cny_rub", "rgbI"):
            if name in x:
                x[f"{name}_ret20"] = x[name].pct_change(20, fill_method=None)
        peers = [p for p in ("lkoh", "rosn", "tatn", "bane", "gazp", "nvtk") if p in x]
        if peers:
            peer_returns = x[peers].pct_change(20, fill_method=None)
            median = peer_returns.median(axis=1).where(peer_returns.notna().sum(axis=1) >= 4)
            x["relative_peers20"] = close.pct_change(20, fill_method=None) - median
    return x


def _score(value: float | None) -> float:
    return 0.0 if value is None or not np.isfinite(value) else float(np.tanh(value))


def _flag(value):
    return isinstance(value, bool | np.bool_) and bool(value)


def position_size(
    equity: float, price: float, atr: float, annual_vol: float | None, risk: RiskLimits
) -> float:
    if not np.isfinite([equity, price, atr]).all() or min(equity, price, atr) <= 0:
        return 0.0
    by_risk = equity * risk.risk_per_trade / (risk.stop_atr * atr)
    by_vol = (
        equity * risk.target_volatility / annual_vol / price
        if annual_vol is not None and np.isfinite(annual_vol) and annual_vol > 0
        else 0.0
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
            (row.get("signal_close", row.get("close", np.nan)) / row.get("ema20", np.nan) - 1)
            + (row.get("ema20", np.nan) / row.get("ema50", np.nan) - 1)
            + (row.get("ema50", np.nan) / row.get("ema200", np.nan) - 1)
        )
        * 12
    )
    momentum = _score(
        (row.get("log_ret_5", np.nan) + row.get("log_ret_20", np.nan)) * 8
        + (row.get("rsi14", 50) - 50) / 20
    )
    oil = _score(row.get("oil_ret5", np.nan) * row.get("oil_corr60", 0) * 8)
    fx = _score(row.get("usd_rub_ret5", np.nan) * row.get("fx_corr60", 0) * 8)
    rate_change = row.get("rate_ret5", np.nan)
    rate = _score((-rate_change if np.isfinite(rate_change) else row.get("rgbI_ret5", np.nan)) * 4)
    relative = _score(row.get("relative_imoex20", np.nan) * 8)
    liquidity_ok = bool(
        row.get("avg_turnover20", 0) >= risk.min_avg_turnover
        and 0 <= row.get("spread_bps", np.nan) <= risk.max_spread_bps
    )
    liquidity = 1.0 if liquidity_ok else -1.0
    penalty = float(row.get("event_penalty", np.nan))
    penalty = float(np.clip(penalty, 0, 1)) if np.isfinite(penalty) else 0.0
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
                + config.microstructure_weight * _score(row.get("microstructure_score", np.nan))
            )
            * penalty,
            -1,
            1,
        )
    )
    event = str(row.get("event_risk", "UNKNOWN")).upper()
    high_event = event not in {"LOW", "MEDIUM"}
    required = (
        "close",
        "ema20",
        "ema50",
        "ema200",
        "atr14",
        "realized_vol20",
        "log_ret_5",
        "log_ret_20",
        "rsi14",
        "event_penalty",
    )
    missing = [name for name in required if not np.isfinite(row.get(name, np.nan))]
    if config.require_context:
        for weight, names in (
            (config.oil_weight, ("oil_ret5", "oil_corr60")),
            (config.fx_weight, ("usd_rub_ret5", "fx_corr60")),
            (config.relative_weight, ("relative_imoex20",)),
        ):
            if weight:
                missing.extend(name for name in names if not np.isfinite(row.get(name, np.nan)))
        if config.rate_weight and not any(
            np.isfinite(row.get(name, np.nan)) for name in ("rate_ret5", "rgbI_ret5")
        ):
            missing.append("rate_ret5_or_rgbI_ret5")
    dividend_block = _flag(row.get("dividend_entry_block", False)) or _flag(
        row.get("is_dividend_cutoff", False)
    )
    event_flags = any(
        _flag(row.get(name, False))
        for name in (
            "sanctions_event",
            "geopolitical_event",
            "tax_event",
            "opec_event",
            "report_event",
        )
    )
    high_volatility = row.get("realized_vol20", np.inf) > risk.max_entry_volatility
    regime = "RISK_ON" if trend > 0.15 else "RISK_OFF" if trend < -0.15 else "NEUTRAL"
    action = "NO_TRADE"
    if not liquidity_ok or high_event or missing or dividend_block or event_flags:
        action = "EXIT" if position != "FLAT" else "NO_TRADE"
    elif position == "LONG" and score < config.exit_threshold:
        action = "EXIT"
    elif (
        position == "FLAT"
        and score > config.buy_threshold
        and regime == "RISK_ON"
        and not high_volatility
    ):
        action = "BUY"
    elif position == "SHORT" and score > -config.exit_threshold:
        action = "EXIT"
    elif position == "FLAT" and score < config.short_threshold:
        action = "NO_TRADE"  # borrow availability is not assumed
    reasons = [f"trend={trend:.3f}", f"momentum={momentum:.3f}", f"relative={relative:.3f}"]
    if np.isfinite(row.get("microstructure_score", np.nan)):
        reasons.append(f"microstructure={row['microstructure_score']:.3f}")
    warnings = [] if liquidity_ok else ["liquidity limits not met"]
    if high_event:
        warnings.append("high event risk")
    if missing:
        warnings.append("missing required data: " + ",".join(missing))
    if dividend_block:
        warnings.append("dividend entry blackout")
    if high_volatility:
        warnings.append("entry volatility limit")
    if event_flags:
        warnings.append("scheduled material event")
    return {
        "timestamp": str(row.get("close_time")),
        "ticker": "SNGS",
        "short_allowed": False,
        "borrow_status": "UNKNOWN_REQUIRES_BROKER",
        "is_dividend_cutoff": _flag(row.get("is_dividend_cutoff", False)),
        "dividend_entry_block": dividend_block,
        "missing_data": missing,
        "components": dict(
            trend=trend,
            momentum=momentum,
            oil=oil,
            fx=fx,
            rate=rate,
            relative=relative,
            liquidity=liquidity,
            event_penalty=penalty,
        ),
        "action": action,
        "signal_score": score,
        "confidence": float(min(1, abs(score))),
        "position_weight_target": (
            position_size(
                equity,
                float(row.get("close", 0)),
                float(row.get("atr14", 0) or 0),
                row.get("realized_vol20"),
                risk,
            )
            * float(row.get("close", 0))
            / equity
        )
        if action == "BUY" and equity > 0
        else 0.0,
        "stop_price": float(row.close - risk.stop_atr * row.atr14)
        if action == "BUY" and pd.notna(row.get("atr14"))
        else None,
        "trailing_stop_distance_atr": risk.trailing_atr if action == "BUY" else None,
        "time_stop_bars": risk.time_stop_bars if action == "BUY" else None,
        "market_regime": regime,
        "liquidity_ok": liquidity_ok,
        "event_risk_level": event if event in {"LOW", "MEDIUM", "HIGH", "EXTREME"} else "UNKNOWN",
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
