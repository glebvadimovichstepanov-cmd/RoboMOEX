import numpy as np
import pandas as pd
import pytest

from robomoex.data import aggregate, assert_fresh, closed_minutes, validate_bars, validate_sessions
from robomoex.demo import dataset


@pytest.fixture
def sample():
    bars, sessions = dataset(2)
    return validate_bars(bars), validate_sessions(sessions)


@pytest.mark.parametrize(
    "defect", ["nan", "inf", "negative", "ohlc", "duplicate", "order", "naive"]
)
def test_bad_market_data_rejected(sample, defect):
    bars, _ = sample
    if defect in ("nan", "inf", "negative"):
        bars.loc[0, "close"] = {"nan": np.nan, "inf": np.inf, "negative": -1}[defect]
    elif defect == "ohlc":
        bars.loc[0, "high"] = 1
    elif defect == "duplicate":
        bars = pd.concat([bars, bars.tail(1)], ignore_index=True)
    elif defect == "order":
        bars = bars.iloc[::-1]
    else:
        bars["open_time"] = bars.open_time.dt.tz_localize(None)
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_closed_only_and_gap_detection(sample):
    bars, sessions = sample
    cutoff = bars.close_time.iloc[16]
    filtered = closed_minutes(bars, sessions, cutoff)
    assert len(filtered) == 17
    with pytest.raises(ValueError, match="Missing"):
        closed_minutes(bars.drop(index=5), sessions, cutoff)
    # A forming 15m bar must not be emitted.
    higher = aggregate(filtered, sessions, 15)
    assert len(higher) == 1
    assert higher.close_time.iloc[0] == bars.close_time.iloc[14]
    assert higher.volume.iloc[0] == bars.volume.iloc[:15].sum()
    assert aggregate(filtered, sessions, None).empty


def test_daily_available_only_after_session(sample):
    bars, sessions = sample
    before = closed_minutes(bars, sessions, bars.close_time.iloc[58])
    assert aggregate(before, sessions, None).empty
    after = closed_minutes(bars, sessions, bars.close_time.iloc[59])
    daily = aggregate(after, sessions, None)
    assert len(daily) == 1
    assert daily.close_time.iloc[0] == sessions.close_time.iloc[0]


def test_timezone_and_session_freshness(sample):
    bars, sessions = sample
    prefix = bars.iloc[:10]
    assert_fresh(prefix, sessions, prefix.close_time.iloc[-1] + pd.Timedelta(seconds=30))
    with pytest.raises(ValueError, match="Stale"):
        assert_fresh(prefix, sessions, prefix.close_time.iloc[-1] + pd.Timedelta(minutes=5))
    with pytest.raises(ValueError, match="Outside"):
        assert_fresh(prefix, sessions, "2025-01-06T00:00:00Z")
    with pytest.raises(ValueError, match="future"):
        assert_fresh(bars, sessions, prefix.close_time.iloc[-1])


def test_outside_session_and_unsupported_schedule(sample):
    bars, sessions = sample
    with pytest.raises(ValueError, match="outside"):
        closed_minutes(bars, sessions.iloc[:1], bars.close_time.iloc[-1])
    with pytest.raises(ValueError):
        validate_sessions(pd.concat([sessions, sessions.iloc[:1]]))


def test_missing_bar_in_aggregate_is_not_fabricated(sample):
    bars, sessions = sample
    higher = aggregate(bars.drop(index=4), sessions, 15)
    assert len(higher) == 7  # one of eight buckets is incomplete
