"""Causal research features from AlgoPack; no synthetic historical tick spreads."""

import numpy as np
import pandas as pd

from .algopack_cache import read_rows


def daily_features(db, ticker="SNGS"):
    """Aggregate only observations published by the end of each Moscow day.

    These are end-of-day research factors, available the next Moscow day.
    SYSTIME is a provider timestamp, not proof of historical revision availability.
    """
    pieces = []
    for metric in ("tradestats", "obstats", "orderstats", "hi2", "alerts"):
        rows = read_rows(db, f"eq.{ticker}.{metric}")
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        timestamp = pd.to_datetime(frame.tradedate + " " + frame.tradetime).dt.tz_localize(
            "Europe/Moscow"
        )
        published = pd.to_datetime(frame.SYSTIME).dt.tz_localize("Europe/Moscow")
        # Extra minute prevents boundary leakage when the provider timestamp is absent/early.
        available = pd.concat([timestamp + pd.Timedelta(minutes=1), published], axis=1).max(axis=1)
        frame["date"] = pd.to_datetime(frame.tradedate)
        end_of_day = frame.date.dt.tz_localize("Europe/Moscow") + pd.Timedelta(days=1)
        frame = frame[available < end_of_day].copy()
        group = frame.groupby("date")
        if metric == "tradestats":
            total = group[["val", "val_b", "val_s"]].sum(min_count=1)
            part = pd.DataFrame(
                {
                    "algopack_turnover": total.val,
                    "trade_imbalance": (total.val_b - total.val_s)
                    / (total.val_b + total.val_s).replace(0, np.nan),
                }
            )
        elif metric == "obstats":
            part = pd.DataFrame(
                {
                    "spread_bps": group.spread_bbo.quantile(0.95),
                    "book_imbalance": group.imbalance_val.mean(),
                    "depth_spread_bps": group.spread_1mio.quantile(0.95),
                    "book_observations": group.size(),
                }
            )
        elif metric == "orderstats":
            total = group[["put_val_b", "put_val_s", "cancel_val_b", "cancel_val_s"]].sum(
                min_count=1
            )
            gross = (total.put_val_b + total.put_val_s).replace(0, np.nan)
            part = pd.DataFrame(
                {
                    "order_imbalance": (
                        (total.put_val_b - total.cancel_val_b)
                        - (total.put_val_s - total.cancel_val_s)
                    )
                    / gross,
                    "cancel_ratio": (total.cancel_val_b + total.cancel_val_s) / gross,
                }
            )
        elif metric == "hi2":
            part = frame.pivot_table(
                index="date", columns="metric", values="value", aggfunc="last"
            ).add_prefix("hi2_")
        else:
            part = group.size().to_frame("market_alert_count")
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=["available_at"])
    result = pd.concat(pieces, axis=1).sort_index()
    result["available_at"] = result.index.tz_localize("Europe/Moscow") + pd.Timedelta(days=1)
    return result.reset_index(drop=True)


def attach_features(features, factors):
    """Backward as-of with 36h maximum age; weekends/gaps do not imply fresh liquidity."""
    if factors.empty:
        return features.copy()
    left = features.copy()
    left["close_time"] = pd.to_datetime(left.close_time, utc=True)
    right = factors.copy()
    right["available_at"] = pd.to_datetime(right.available_at, utc=True)
    overlap = (set(left) & set(right)) - {"close_time"}
    left = left.drop(columns=list(overlap))
    result = pd.merge_asof(
        left.sort_values("close_time"),
        right.sort_values("available_at"),
        left_on="close_time",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(hours=36),
    )
    if "algopack_turnover" in result:
        # Turnover is already known prior to the decision timestamp; never convert lots as shares.
        result["avg_turnover20"] = result.algopack_turnover.rolling(20, min_periods=20).mean()
    result["microstructure_score"] = (
        result.reindex(columns=["trade_imbalance", "book_imbalance", "order_imbalance"])
        .clip(-1, 1)
        .mean(axis=1, skipna=False)
    )
    return result
