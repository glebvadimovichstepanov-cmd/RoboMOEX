import pandas as pd
import pytest

from robomoex.config import StrategyConfig
from robomoex.data import closed_minutes, validate_bars, validate_sessions
from robomoex.demo import dataset
from robomoex.strategy import indicators, signals


@pytest.fixture(scope="module")
def history():
    bars, sessions = dataset()
    return validate_bars(bars), validate_sessions(sessions)


def test_warmup_not_backfilled(history):
    bars, _ = history
    result = indicators(bars, 60)
    assert result.rsi.iloc[:14].isna().all()
    assert result.ema32.iloc[:31].isna().all()
    assert result.atr_med.iloc[:72].isna().all()


def test_rsi_monotonic_and_flat():
    bars, _ = dataset(1)
    for increasing, expected in ((True, 100), (False, 50)):
        bars["close"] = [100 + i if increasing else 100 for i in range(len(bars))]
        bars["open"] = bars.close
        bars["high"] = bars.close + 1
        bars["low"] = bars.close - 1
        assert indicators(bars, 2).rsi.iloc[-1] == expected


@pytest.mark.parametrize("cut", [1919, 1920, 1937, 2305, 2698])
def test_prefix_invariance(history, cut):
    bars, sessions = history
    prefix = closed_minutes(bars, sessions, bars.close_time.iloc[cut])
    actual = signals(prefix, sessions)
    whole = signals(bars, sessions).iloc[: cut + 1]
    pd.testing.assert_frame_equal(actual, whole)


def test_future_mutation_cannot_change_old_decisions(history):
    bars, sessions = history
    cut = 2207
    changed = bars.copy()
    changed.loc[cut + 1 :, ["open", "high", "low", "close"]] *= 2
    original = signals(bars, sessions)
    modified = signals(changed, sessions)
    pd.testing.assert_frame_equal(original.iloc[: cut + 1], modified.iloc[: cut + 1])


def test_real_parameters_used_and_warmup_has_no_orders(history):
    bars, sessions = history
    original = signals(bars, sessions)
    strict = signals(bars, sessions, StrategyConfig(doji_thresh=1))
    assert original.enter.sum() > 0
    assert strict.enter.sum() == 0
    assert not original.enter.iloc[: 31 * 60].any()
