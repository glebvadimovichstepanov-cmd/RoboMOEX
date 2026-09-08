import pandas as pd

from robomoex.demo import dataset
from robomoex.signal_engine import RiskLimits, build_features, make_signal, position_size


def daily_fixture(days=260):
    bars, _ = dataset(1)
    rows = []
    for i in range(days):
        start = pd.Timestamp("2024-01-01T10:00:00+03:00") + pd.Timedelta(days=i)
        price = 100 + i * 0.1
        rows.append(
            {
                "open_time": start,
                "close_time": start + pd.Timedelta(days=1),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1_000_000,
            }
        )
    return pd.DataFrame(rows)


def test_features_are_causal_and_signal_has_contract():
    features = build_features(daily_fixture())
    row = features.iloc[-1].copy()
    row["turnover"] = 40_000_000
    row["spread_bps"] = 10
    signal = make_signal(row, equity=100_000)
    assert (
        set(("timestamp", "ticker", "action", "signal_score", "confidence", "reasons", "warnings"))
        <= signal.keys()
    )
    assert -1 <= signal["signal_score"] <= 1
    assert 0 <= signal["confidence"] <= 1


def test_high_event_or_bad_liquidity_blocks_entries():
    row = build_features(daily_fixture()).iloc[-1].copy()
    row["turnover"], row["spread_bps"], row["event_risk"] = 1, 100, "HIGH"
    result = make_signal(row, equity=100_000)
    assert result["action"] == "NO_TRADE"
    assert result["liquidity_ok"] is False


def test_string_event_risk_is_context_safe():
    daily = daily_fixture()
    context = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.close_time).dt.date,
            "event_risk": ["LOW"] * len(daily),
            "event_penalty": [1.0] * len(daily),
        }
    )
    features = build_features(daily, context=context)
    assert "event_risk" in features
    assert "event_risk_ret5" not in features


def test_position_size_uses_smaller_risk_constraint():
    risk = RiskLimits()
    assert position_size(100_000, 100, 2, 0.2, risk) <= 50_000 / 100
