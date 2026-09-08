import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
from test_signal_engine import daily_fixture

from robomoex.cache import merge_bars
from robomoex.context import SOURCES, download_moex_daily, update_context_cache
from robomoex.research import replay, walk_forward
from robomoex.signal_engine import CompositeConfig, build_features, make_signal


def test_appending_future_does_not_change_features():
    bars = daily_fixture(300)
    before = build_features(bars.iloc[:230])
    after = build_features(bars)
    pd.testing.assert_frame_equal(before, after.iloc[:230])


def test_context_duplicates_are_rejected():
    with pytest.raises(ValueError, match="Duplicate context"):
        build_features(daily_fixture(), pd.DataFrame({"date": ["2024-01-02"] * 2}))


def test_unknown_event_or_spread_blocks_entry_and_weight():
    row = build_features(daily_fixture()).iloc[-1].copy()
    row["event_risk"], row["event_penalty"] = "LOW", 1.0
    signal = make_signal(row, equity=100000, config=CompositeConfig(buy_threshold=0))
    assert signal["action"] == "NO_TRADE"
    assert signal["position_weight_target"] == 0
    row["spread_bps"], row["event_risk"] = 10, "UNKNOWN"
    assert make_signal(row, equity=100000)["action"] == "NO_TRADE"


def test_all_ohlcv_duplicate_conflicts_rejected():
    bars = daily_fixture(2)
    other = bars.copy()
    other.loc[0, "volume"] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        merge_bars(bars, other)


def test_empty_weekend_is_cached_and_malformed_response_preserves_file(tmp_path):
    path = tmp_path / "x.json"
    calls = []

    def empty(url):
        calls.append(url)
        return {"candles": {"columns": ["begin", "close"], "data": []}}

    update_context_cache(path, SOURCES[0], "2025-01-01", "2025-01-02", empty)
    old = path.read_bytes()
    update_context_cache(path, SOURCES[0], "2025-01-01", "2025-01-02", empty)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Invalid candles"):
        update_context_cache(path, SOURCES[0], "2025-01-01", "2025-01-03", lambda _: {})
    assert path.read_bytes() == old


def test_daily_pagination_and_stalls():
    calls = []

    def page(url):
        offset = int(parse_qs(urlparse(url).query)["start"][0])
        calls.append(offset)
        return {
            "candles": {
                "columns": ["begin", "close"],
                "data": [["2025-01-01", 10]] if offset == 0 else [["2025-01-02", 11]],
            }
        }

    assert len(download_moex_daily(SOURCES[0], "2025-01-01", "2025-01-02", page)) == 2
    assert calls == [0, 1]
    with pytest.raises(ValueError, match="advance"):
        download_moex_daily(SOURCES[0], "2025-01-01", "2025-01-03", page)


def test_replay_next_open_position_cap_and_first_loss_drawdown():
    frame = build_features(daily_fixture(270)).iloc[250:].reset_index(drop=True)
    frame["spread_bps"], frame["event_risk"], frame["event_penalty"] = 10, "LOW", 1.0
    frame["avg_turnover20"] = 50000000
    # Synthetic price is 125, so a 100-share lot exceeds the 5% allocation cap.
    assert (
        replay(frame, CompositeConfig(buy_threshold=0, require_context=False))["metrics"]["trades"]
        == 0
    )
    result = replay(frame, CompositeConfig(buy_threshold=0, require_context=False), lot=10)
    assert result["equity"][0]["quantity"] == 0
    assert result["equity"][1]["quantity"] > 0
    assert result["equity"][1]["quantity"] * frame.open.iloc[1] <= 5000
    assert result["metrics"]["max_drawdown"] > 0
    json.dumps(result, allow_nan=False)


def test_walkforward_holdout_does_not_change_thresholds():
    frame = build_features(daily_fixture(400))
    first = walk_forward(frame, train=260, test=60, holdout=60)
    changed = frame.copy()
    changed.loc[340:, "close"] *= 1.1
    second = walk_forward(changed, train=260, test=60, holdout=60)
    assert first["windows"] == second["windows"]
    assert first["holdout"]["threshold"] == second["holdout"]["threshold"]
    assert np.isfinite(first["holdout"]["metrics"]["total_return"])
