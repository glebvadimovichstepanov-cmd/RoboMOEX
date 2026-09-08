from datetime import date
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from robomoex.algopack_cache import Dataset, connect, fetch, read_rows, snapshot, update
from robomoex.algopack_features import attach_features

SOURCE = Dataset("eq.SNGS.tradestats", "datashop/algopack/eq/tradestats/SNGS.json")


def response(rows, index=0, total=None):
    return {
        "data": {"columns": ["tradedate", "tradetime", "secid", "val"], "data": rows},
        "data.cursor": {
            "columns": ["INDEX", "TOTAL", "PAGESIZE"],
            "data": [[index, len(rows) if total is None else total, 1]],
        },
    }


def test_pagination_is_complete():
    def transport(url):
        offset = int(parse_qs(urlsplit(url).query)["start"][0])
        return response([["2025-01-03", f"10:0{offset}:00", "SNGS", 100]], offset, 2)

    assert len(fetch(SOURCE, "2025-01-03", "2025-01-03", transport)["data"]) == 2


def test_failed_page_does_not_mark_range_complete(tmp_path):
    def transport(url):
        if "start=1" in url:
            raise ValueError("Malformed second page")
        return response([["2025-01-03", "10:00:00", "SNGS", 100]], total=2)

    with connect(tmp_path / "cache.sqlite") as db:
        with pytest.raises(ValueError, match="second page"):
            update(db, SOURCE, "2025-01-03", "2025-01-03", transport)
        assert db.execute("SELECT count(*) FROM windows").fetchone()[0] == 0


def test_replay_empty_coverage_and_incremental_tail(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        return response([])

    with connect(tmp_path / "cache.sqlite") as db:
        update(db, SOURCE, "2025-01-01", "2025-01-07", transport)
        update(db, SOURCE, "2025-01-01", "2025-01-07", transport)
        assert len(calls) == 1
        update(db, SOURCE, "2025-01-01", "2025-01-08", transport)
        assert "from=2025-01-08" in calls[-1]


def test_revision_deletion_and_checksum(tmp_path):
    with connect(tmp_path / "cache.sqlite") as db:
        update(
            db,
            SOURCE,
            "2025-01-03",
            "2025-01-03",
            lambda _: response([["2025-01-03", "10:00:00", "SNGS", 100]]),
        )
        assert len(read_rows(db, SOURCE.name)) == 1
        update(db, SOURCE, "2025-01-03", "2025-01-03", lambda _: response([]), refresh=True)
        assert read_rows(db, SOURCE.name) == []
        assert db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 2
        db.execute("UPDATE windows SET sha256='bad'")
        with pytest.raises(ValueError, match="checksum"):
            read_rows(db, SOURCE.name)


def test_recent_days_refreshed(tmp_path):
    calls = []

    def transport(url):
        calls.append(url)
        return response([])

    with connect(tmp_path / "cache.sqlite") as db:
        for _ in range(2):
            update(db, SOURCE, "2025-01-01", "2025-01-03", transport, today=date(2025, 1, 3))
        assert len(calls) == 2
        assert "from=2025-01-02" in calls[-1]


def test_wrong_range_and_repeated_cursor_rejected():
    with pytest.raises(ValueError, match="date range"):
        fetch(
            SOURCE,
            "2025-01-01",
            "2025-01-02",
            lambda _: response([["2026-01-01", "10:00:00", "SNGS", 100]]),
        )
    with pytest.raises(ValueError, match="repeated"):
        fetch(
            SOURCE,
            "2025-01-03",
            "2025-01-03",
            lambda _: response([["2025-01-03", "10:00:00", "SNGS", 100]], total=2),
        )


def test_snapshot_never_becomes_historical_coverage(tmp_path):
    with connect(tmp_path / "cache.sqlite") as db:
        source = Dataset("snapshot.book", "test.json", dated=False, cursor=False)
        snapshot(db, source, lambda _: response([]))
        assert db.execute("SELECT count(*) FROM windows").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 1


def test_asof_does_not_use_future_or_stale_factors():
    features = pd.DataFrame(
        {
            "close_time": pd.to_datetime(
                ["2025-01-03T18:00:00Z", "2025-01-04T18:00:00Z", "2025-01-07T18:00:00Z"]
            ),
            "spread_bps": [float("nan")] * 3,
        }
    )
    factors = pd.DataFrame(
        {
            "available_at": [pd.Timestamp("2025-01-03T21:00:00Z")],
            "spread_bps": [12],
            "trade_imbalance": [0.5],
            "book_imbalance": [0.2],
            "order_imbalance": [0.2],
        }
    )
    result = attach_features(features, factors)
    assert pd.isna(result.spread_bps.iloc[0])
    assert result.spread_bps.iloc[1] == 12
    assert pd.isna(result.spread_bps.iloc[2])
    assert result.microstructure_score.iloc[1] == pytest.approx(0.3)
