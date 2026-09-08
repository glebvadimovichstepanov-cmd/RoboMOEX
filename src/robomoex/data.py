"""OHLCV contracts. A row is available at close_time, never at open_time."""

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
BAR_COLUMNS = ["open_time", "close_time", *PRICE_COLUMNS, "volume"]


def utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamp must have an explicit timezone")
    return stamp.tz_convert("UTC")


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or set(BAR_COLUMNS) - set(frame.columns):
        raise ValueError("Nonempty OHLCV with open_time and close_time is required")
    data = frame[BAR_COLUMNS].copy()
    for column in ("open_time", "close_time"):
        data[column] = pd.DatetimeIndex([utc(value) for value in data[column]])
        if data[column].duplicated().any() or not data[column].is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered {column}")
    if (data.close_time <= data.open_time).any():
        raise ValueError("close_time must follow open_time")
    if (data.open_time.iloc[1:].array < data.close_time.iloc[:-1].array).any():
        raise ValueError("Overlapping bars")
    for col in [*PRICE_COLUMNS, "volume"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Nonfinite {col}; missing prices are never filled")
    if (data[PRICE_COLUMNS] <= 0).any().any() or (data.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if (data.high < data[["open", "close", "low"]].max(axis=1)).any() or (
        data.low > data[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Invalid OHLC bounds")
    return data.reset_index(drop=True)


def validate_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or set(("open_time", "close_time")) - set(frame.columns):
        raise ValueError("Explicit exchange session schedule is required")
    result = frame[["open_time", "close_time"]].copy()
    for col in result:
        result[col] = pd.DatetimeIndex([utc(value) for value in result[col]])
        if (result[col] != result[col].dt.floor("min")).any():
            raise ValueError("Sessions must start and end on minute boundaries")
    if not result.open_time.is_monotonic_increasing or result.open_time.duplicated().any():
        raise ValueError("Session schedule must be ordered and unique")
    duration = result.close_time - result.open_time
    if (duration <= pd.Timedelta(0)).any() or (duration > pd.Timedelta(days=1)).any():
        raise ValueError("Invalid session duration")
    if ((duration / pd.Timedelta(minutes=1)) % 1 != 0).any():
        raise ValueError("Sessions must contain whole minutes")
    if (result.open_time.iloc[1:].array < result.close_time.iloc[:-1].array).any():
        raise ValueError("Overlapping sessions")
    # v1 supports exactly one continuous selected session per Moscow trading date.
    dates = result.open_time.dt.tz_convert("Europe/Moscow").dt.date
    if dates.duplicated().any():
        raise ValueError("v1 requires one selected continuous session per trading date")
    return result.reset_index(drop=True)


def closed_minutes(frame: pd.DataFrame, sessions: pd.DataFrame, as_of) -> pd.DataFrame:
    bars, schedule, cutoff = validate_bars(frame), validate_sessions(sessions), utc(as_of)
    if ((bars.close_time - bars.open_time) != pd.Timedelta(minutes=1)).any():
        raise ValueError("Only one-minute source bars are accepted")
    bars = bars[bars.close_time <= cutoff].copy()
    expected = pd.DatetimeIndex([], tz="UTC")
    for row in schedule.itertuples():
        end = min(row.close_time, cutoff.floor("min"))
        if end > row.open_time:
            expected = expected.append(
                pd.date_range(row.open_time, end, freq="min", inclusive="left")
            )
    actual = pd.DatetimeIndex(bars.open_time)
    if len(actual.difference(expected)):
        raise ValueError("Bars outside the supplied session schedule")
    if len(expected.difference(actual)):
        raise ValueError("Missing closed minutes; incomplete datasets are rejected")
    if bars.empty:
        raise ValueError("No closed bars")
    return bars.reset_index(drop=True)


def aggregate(frame: pd.DataFrame, sessions: pd.DataFrame, minutes: int | None) -> pd.DataFrame:
    """Aggregate complete session-anchored buckets; None means selected session bar."""
    bars, schedule = validate_bars(frame), validate_sessions(sessions)
    if minutes is not None and (type(minutes) is not int or minutes < 1):
        raise ValueError("minutes must be positive")
    records = []
    for session in schedule.itertuples():
        start = session.open_time
        while start < session.close_time:
            end = (
                session.close_time
                if minutes is None
                else min(start + pd.Timedelta(minutes=minutes), session.close_time)
            )
            chunk = bars[(bars.open_time >= start) & (bars.close_time <= end)]
            expected = pd.date_range(start, end, freq="min", inclusive="left")
            if pd.DatetimeIndex(chunk.open_time).equals(expected):
                records.append(
                    dict(
                        open_time=start,
                        close_time=end,
                        open=chunk.open.iloc[0],
                        high=chunk.high.max(),
                        low=chunk.low.min(),
                        close=chunk.close.iloc[-1],
                        volume=chunk.volume.sum(),
                    )
                )
            start = end
    return pd.DataFrame(records, columns=BAR_COLUMNS)


def assert_fresh(
    frame: pd.DataFrame, sessions: pd.DataFrame, now, max_age: pd.Timedelta | None = None
) -> None:
    current = utc(now)
    if max_age is None:
        max_age = pd.Timedelta(minutes=2)
    if max_age <= pd.Timedelta(0):
        raise ValueError("max_age must be positive")
    schedule = validate_sessions(sessions)
    if not ((schedule.open_time <= current) & (current < schedule.close_time)).any():
        raise ValueError("Outside the explicit trading session")
    latest = validate_bars(frame).close_time.iloc[-1]
    if latest > current or current - latest > max_age:
        raise ValueError("Stale or future market data")
