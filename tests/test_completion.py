import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest
from test_signal_engine import daily_fixture

from robomoex.daily_iss import download, update
from robomoex.evaluation import benchmark
from robomoex.external_feeds import connect, oil_rows, update_news, update_oil
from robomoex.operations import Journal, redact
from robomoex.signal_engine import build_features


def candle(begin, close=25):
    return [begin + " 00:00:00", begin + " 23:59:59", 25, 26, 24, close, 1000000]


def response(rows):
    return {
        "candles": {
            "columns": ["begin", "end", "open", "high", "low", "close", "volume"],
            "data": rows,
        }
    }


def test_native_daily_preserves_date_and_detects_pagination_loop():
    calls = []

    def transport(url):
        calls.append(url)
        return response([candle("2025-01-03")]) if len(calls) == 1 else response([])

    bars = download("2025-01-03", "2025-01-03", transport)
    assert str(bars.close_time.iloc[0].tz_convert("Europe/Moscow").date()) == "2025-01-03"
    with pytest.raises(ValueError, match="advance"):
        download("2025-01-03", "2025-01-03", lambda url: response([candle("2025-01-03")]))


def test_incremental_daily_refresh_replaces_revision_and_caps_today(tmp_path):
    requests = []
    price = [25]

    def transport(url):
        query = parse_qs(urlsplit(url).query)
        requests.append(query)
        return (
            response([candle("2025-01-03", price[0])])
            if query["start"] == ["0"] and query["from"][0] <= "2025-01-03" <= query["till"][0]
            else response([])
        )

    path = tmp_path / "daily.json"
    update(path, "2025-01-01", "2025-01-04", transport, now="2025-01-04T18:00:00+03:00")
    assert requests[0]["till"] == ["2025-01-03"]
    price[0] = 25.5
    bars = update(path, "2025-01-01", "2025-01-04", transport, now="2025-01-04T18:00:00+03:00")
    assert len(bars) == 1 and bars.close.iloc[0] == 25.5


def test_asof_weekend_is_causal_and_expires():
    bars = daily_fixture(270)
    # Friday observation is available Saturday; next bars remain independent of future context.
    dates = pd.to_datetime(bars.close_time, utc=True).dt.tz_convert("Europe/Moscow").dt.date
    context = pd.DataFrame(dict(date=dates.iloc[:260], oil=np.arange(260) + 100.0))
    context = context[~pd.to_datetime(context.date).dt.dayofweek.isin([5, 6])]
    features = build_features(bars, context)
    assert features.oil.iloc[-1] != features.oil.iloc[-1]  # expired after four days
    altered = context.copy()
    altered.loc[altered.index[-1], "oil"] = 99999
    changed = build_features(bars, altered)
    pd.testing.assert_frame_equal(features.iloc[:250], changed.iloc[:250])


def test_fred_negative_wti_and_missing_values_valid():
    rows = oil_rows(
        b"observation_date,DCOILWTICO\n2020-04-20,-36.98\n2020-04-21,.\n",
        "DCOILWTICO",
        pd.Timestamp("2020-04-20").date(),
        pd.Timestamp("2020-04-21").date(),
    )
    assert rows[0]["value"] == -36.98 and rows[1]["value"] is None


def test_fred_historical_cache_incremental_and_revision_preserved(tmp_path):
    requests = []

    def transport(url):
        query = parse_qs(urlsplit(url).query)
        requests.append(query)
        series = query["id"][0]
        return f"observation_date,{series}\n2025-01-02,70\n".encode()

    with connect(tmp_path / "external.db") as db:
        update_oil(db, "2025-01-01", "2025-01-03", transport, now="2026-09-09T00:00:00Z")
        report = update_oil(db, "2025-01-01", "2025-01-03", transport, now="2026-09-09T00:00:00Z")
        assert len(requests) == 2 and report["wti"]["status"] == "cached"
        assert (
            db.execute("SELECT COUNT(*) FROM observations WHERE source='fred.wti'").fetchone()[0]
            == 1
        )


def test_news_failure_does_not_advance_coverage(tmp_path):
    def fail(url):
        raise HTTPError(url, 429, "rate limit", {}, None)

    with connect(tmp_path / "news.db") as db:
        report = update_news(db, "2026-09-08", "2026-09-08", fail, now="2026-09-09T00:00:00Z")
        assert report[0]["http_status"] == 429
        assert db.execute("SELECT COUNT(*) FROM coverage").fetchone()[0] == 0


def test_news_keeps_observation_time_and_flags_truncation(tmp_path):
    articles = [
        dict(url=f"https://example.org/{i}", title="Oil", seendate="20260908T120000Z")
        for i in range(250)
    ]
    with connect(tmp_path / "news.db") as db:
        report = update_news(
            db,
            "2026-09-08",
            "2026-09-08",
            lambda url: json.dumps(dict(articles=articles)).encode(),
            now="2026-09-09T00:00:00Z",
        )
        assert report[0]["status"] == "truncated"
        payload = json.loads(
            db.execute(
                "SELECT payload FROM observations WHERE key=?", ("https://example.org/0",)
            ).fetchone()[0]
        )
        assert payload["known_at"].startswith("2026-09-09")
        assert payload["event_risk"] == "UNKNOWN"


def test_journal_deduplicates_alerts_and_redacts(tmp_path):
    with Journal(tmp_path) as journal:
        for _ in range(2):
            journal.emit("failed", "ERROR", authorization="Bearer abc", token="hidden")
        assert journal.db.execute("SELECT occurrences FROM alerts").fetchone()[0] == 2
    raw = (tmp_path / "runtime.jsonl").read_text()
    assert "hidden" not in raw and "Bearer abc" not in raw
    assert "eyJ" not in redact("eyJabc.def.ghi")


def test_benchmark_costs_and_lot_rounding():
    frame = pd.DataFrame(dict(open=[100, 100], close=[100, 100]))
    result = benchmark(frame)
    assert result["total_return"] < 0
    assert result["shares"] == 900
