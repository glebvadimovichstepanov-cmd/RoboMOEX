"""Small deterministic SYNTHETIC fixture. These are not exchange prices or sessions."""

import math

import pandas as pd


def dataset(days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, sessions = [], []
    for day in range(days):
        opening = pd.Timestamp("2025-01-06T10:00:00+03:00") + pd.Timedelta(days=day)
        sessions.append({"open_time": opening, "close_time": opening + pd.Timedelta(minutes=60)})
        for minute in range(60):
            price = 100 + day * 0.2 + math.sin(minute / 3) * 0.8
            close = 100 + day * 0.2 + math.sin((minute + 1) / 3) * 0.8
            start = opening + pd.Timedelta(minutes=minute)
            rows.append(
                {
                    "open_time": start,
                    "close_time": start + pd.Timedelta(minutes=1),
                    "open": price,
                    "high": max(price, close) + 0.04,
                    "low": min(price, close) - 0.04,
                    "close": close,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(sessions)
