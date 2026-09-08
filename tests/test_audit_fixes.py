import io
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_signal_engine import daily_fixture

from robomoex.cbr import download, update
from robomoex.corporate_actions import adjust_prices, flags, validate_events
from robomoex.disclosure import event_links, ingest, parse_event, refresh
from robomoex.instrument_info import resolve
from robomoex.model_validation import evaluate, purged_windows, safe_prediction
from robomoex.research import replay
from robomoex.signal_engine import CompositeConfig, RiskLimits, build_features, make_signal


def event(**kwargs):
    return dict(
        event_id="div-1",
        event_type="DIVIDEND",
        ticker="SNGS",
        effective_at="2025-01-06T00:00:00+03:00",
        known_at="2025-01-01T10:00:00+03:00",
        source_url="https://www.e-disclosure.ru/portal/event.aspx?EventId=test",
        amount_rub=10.0,
        verified=True,
        **kwargs,
    )


def test_dividend_blackout_uses_known_events_and_trading_dates():
    events = validate_events([event()])
    sessions = ["2025-01-03", "2025-01-06"]
    assert flags(events, "2025-01-02T19:00:00+03:00", sessions=sessions)["dividend_entry_block"]
    assert flags(events, "2025-01-06T19:00:00+03:00")["is_dividend_cutoff"]
    assert not flags(events, "2024-12-31T19:00:00+03:00")["dividend_entry_block"]


def test_unverified_and_naive_events_rejected():
    row = event()
    row["verified"] = "true"
    with pytest.raises(ValueError, match="unverified"):
        validate_events([row])
    row = event()
    row["known_at"] = "2025-01-01"
    with pytest.raises(ValueError, match="timezone"):
        validate_events([row])


def test_forward_adjustment_preserves_raw_prices_and_prior_history():
    bars = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2025-01-03T10:00:00+03:00", "2025-01-06T10:00:00+03:00"]),
            "close_time": pd.to_datetime(
                ["2025-01-03T19:00:00+03:00", "2025-01-06T19:00:00+03:00"]
            ),
            "close": [100.0, 90.0],
        }
    )
    result = adjust_prices(bars, validate_events([event()]))
    assert result.signal_close.tolist() == [100.0, 100.0]
    assert result.close.tolist() == [100.0, 90.0]
    row = event()
    row["known_at"] = "2025-01-07T10:00:00+03:00"
    with pytest.raises(ValueError, match="not known"):
        adjust_prices(bars, validate_events([row]))


def healthy_row():
    row = build_features(daily_fixture(270)).iloc[-1].copy()
    row["spread_bps"], row["avg_turnover20"] = 10.0, 50000000.0
    row["event_risk"], row["event_penalty"] = "LOW", 1.0
    for name in ("oil_ret5", "usd_rub_ret5", "relative_imoex20", "rate_ret5"):
        row[name] = 0.0
    row["oil_corr60"], row["fx_corr60"] = 0.2, 0.2
    return row


def test_missing_required_input_and_dividend_or_volatility_block():
    row = healthy_row()
    config = CompositeConfig(buy_threshold=0)
    assert make_signal(row, equity=100000, config=config)["action"] == "BUY"
    for field, value in (
        ("atr14", np.nan),
        ("realized_vol20", 0.8),
        ("dividend_entry_block", np.bool_(True)),
    ):
        changed = row.copy()
        changed[field] = value
        assert make_signal(changed, equity=100000, config=config)["action"] == "NO_TRADE"


def test_actual_correlation_magnitude_and_required_macro_context():
    row = healthy_row()
    row["oil_ret5"], row["oil_corr60"] = 0.02, -0.4
    signal = make_signal(row, equity=100000)
    assert signal["components"]["oil"] == pytest.approx(np.tanh(-0.064))
    row["oil_corr60"] = np.nan
    result = make_signal(row, equity=100000, config=CompositeConfig(buy_threshold=0))
    assert result["action"] == "NO_TRADE"
    assert "oil_corr60" in result["missing_data"]


def test_macro_features_do_not_fill_missing_prices():
    bars = daily_fixture(270)
    dates = pd.to_datetime(bars.close_time).dt.date
    context = pd.DataFrame(
        {
            "date": dates,
            "cny_rub": np.arange(270.0) + 10,
            **{
                name: np.arange(270.0) + 100
                for name in ("lkoh", "rosn", "tatn", "bane", "gazp", "nvtk")
            },
        }
    )
    context.loc[250, "cny_rub"] = np.nan
    result = build_features(bars, context)
    assert pd.isna(result.cny_rub_ret5.iloc[251])
    assert {"macd", "macd_signal", "macd_histogram", "relative_peers20", "cny_rub_ret20"} <= set(
        result
    )


def test_liquidity_participation_caps_fills():
    frame = build_features(daily_fixture(270)).iloc[250:].reset_index(drop=True)
    frame["spread_bps"], frame["avg_turnover20"] = 10.0, 50000000.0
    frame["event_risk"], frame["event_penalty"] = "LOW", 1.0
    risk = replace(RiskLimits(), max_participation=0.00001)
    result = replay(
        frame, CompositeConfig(buy_threshold=0, require_context=False), lot=1, risk=risk
    )
    assert result["equity"][1]["quantity"] * frame.open.iloc[1] <= 500
    assert result["metrics"]["traded_notional"] > 0


def test_account_drawdown_floor_causes_exit_and_alert():
    frame = build_features(daily_fixture(270)).iloc[250:255].reset_index(drop=True)
    frame["spread_bps"], frame["avg_turnover20"] = 10.0, 50000000.0
    frame["event_risk"], frame["event_penalty"] = "LOW", 1.0
    frame["atr14"] = 20.0
    frame.loc[1, "low"] = 80
    limits = replace(RiskLimits(), max_daily_loss=0.001, risk_per_trade=0.1)
    result = replay(
        frame, CompositeConfig(buy_threshold=0, require_context=False), lot=1, risk=limits
    )
    assert result["trades"][0]["reason"] == "account_risk"
    assert result["alerts"]


HTML = (
    '<meta property="article:published_time" content="2025-01-01T12:00:00+03:00">'
    "<h1>Сургутнефтегаз</h1><p>Решение о дивидендах</p>"
)
URL = "https://www.e-disclosure.ru/portal/event.aspx?EventId=test"


def test_disclosure_idempotent_revisions_and_403_preserve_cache(tmp_path):
    first = ingest(tmp_path, URL, HTML)
    assert ingest(tmp_path, URL, HTML)["observed_at"] == first["observed_at"]
    second = ingest(tmp_path, URL, HTML + "<p>Исправление</p>")
    assert second["known_at"] == second["observed_at"]
    assert second["date_verified"] is False

    def denied(_):
        raise ValueError("HTTP 403")

    result = refresh(tmp_path, denied)
    assert not result["archive_complete"]
    assert result["status"] == "unavailable"
    assert len(list((tmp_path / "events").rglob("*.json"))) == 2


def test_disclosure_rejects_unlabelled_dates_or_wrong_issuer():
    with pytest.raises(ValueError, match="timestamp"):
        parse_event("Сургутнефтегаз 01.01.2025 12:00", URL)
    with pytest.raises(ValueError, match="identity"):
        event_links("<h1>Other issuer</h1>")


def test_cbr_schema_cache_and_failure_preservation(tmp_path):
    raw = b"<KeyRate><KR><DT>2025-01-03T00:00:00+03:00</DT><Rate>21.00</Rate></KR></KeyRate>"

    def transport(request, timeout):
        assert request.get_header("Authorization") is None
        return io.BytesIO(raw)

    assert download("2025-01-01", "2025-01-03", transport).rate.iloc[0] == 21.0
    path = tmp_path / "rate.json"
    update(path, "2025-01-01", "2025-01-03", transport)
    original = json.loads(path.read_text())
    replayed = update(
        path, "2025-01-01", "2025-01-03", lambda *_: pytest.fail("No network expected")
    )
    assert replayed["sha256"] == original["sha256"]
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update(path, "2025-01-01", "2025-01-04", lambda *_, **kw: io.BytesIO(b"<Fault/>"))
    assert path.read_bytes() == before


def test_instrument_confusable_ticker_rejected_without_network():
    with pytest.raises(ValueError, match="ASCII"):
        resolve("SNGSб", transport=lambda _: pytest.fail("No network"))


def test_purging_and_model_training_are_future_independent():
    for training, testing in purged_windows(400):
        assert training[-1] + 5 < testing[0] - 2
    features = build_features(daily_fixture(400))
    first = evaluate(features)
    changed = features.copy()
    changed.loc[260:, "close"] *= 2
    second = evaluate(changed)
    assert first["folds"][0]["coefficients"] == second["folds"][0]["coefficients"]
    assert safe_prediction(None, [np.nan])["status"] == "RULES_FALLBACK"


def execution_frame():
    rows = []
    for i in range(4):
        row = healthy_row()
        row["open_time"] = pd.Timestamp("2025-01-02T10:00:00+03:00") + pd.Timedelta(days=i)
        row["close_time"] = row.open_time + pd.Timedelta(hours=8)
        for name in ("open", "high", "low", "close"):
            row[name] = 15.0
        row["ema20"], row["ema50"], row["ema200"] = 14.0, 13.0, 12.0
        row["atr14"], row["realized_vol20"] = 1.0, 0.2
        row["dividend_cash_per_share"], row["split_ratio"] = 0.0, 1.0
        row["dividend_payment_at"] = ""
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def test_dividend_receivable_is_not_cash_and_entitlement_precedes_ex_date():
    frame = execution_frame()
    frame.loc[2:, ["open", "high", "low", "close"]] = 14.0
    frame.loc[2, "dividend_cash_per_share"] = 1.0
    frame.loc[2, "dividend_payment_at"] = "2025-02-01T10:00:00+03:00"
    result = replay(frame, CompositeConfig(buy_threshold=0), fee=0, slippage=0)
    # Spread costs are still applied even when fixed slippage is zero.
    assert result["metrics"]["ending_dividend_receivable"] > 0
    assert result["trades"][0]["gross_dividends"] == result["metrics"]["ending_dividend_receivable"]
    assert abs(result["trades"][0]["net_pnl"]) < 10
    frame.loc[2, "dividend_cash_per_share"] = 0
    frame.loc[1, "dividend_cash_per_share"] = 1
    frame.loc[1, "dividend_payment_at"] = "2025-02-01T10:00:00+03:00"
    assert (
        replay(frame, CompositeConfig(buy_threshold=0))["metrics"]["ending_dividend_receivable"]
        == 0
    )


def test_split_preserves_economic_position():
    frame = execution_frame()
    frame.loc[2, "split_ratio"] = 2.0
    frame.loc[2:, ["open", "high", "low", "close"]] = 7.5
    frame.loc[2:, ["ema20", "ema50", "ema200"]] = [7.0, 6.5, 6.0]
    result = replay(frame, CompositeConfig(buy_threshold=0), fee=0, slippage=0)
    assert result["trades"][0]["quantity"] == result["equity"][1]["quantity"] * 2
    assert abs(result["trades"][0]["net_pnl"]) < 10
