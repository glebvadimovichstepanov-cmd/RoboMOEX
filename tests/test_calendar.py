import pandas as pd
import pytest

from robomoex.calendar import download_stock_calendar


def response(rows):
    return {"off_days": {"columns": ["tradedate", "is_traded", "reason"], "data": rows}}


def test_calendar_uses_transfered_days_and_skips_holidays():
    schedule = download_stock_calendar(
        "2025-01-01T00:00:00+03:00",
        "2025-01-09T23:00:00+03:00",
        transport=lambda url: response(
            [
                ["2025-01-01", 0, "H"],
                ["2025-01-04", 1, "T"],
                ["2025-01-05", 0, "W"],
                ["2025-01-08", 1, "N"],
            ]
        ),
    )
    assert len(schedule) == 2
    assert schedule.open_time.iloc[0] == pd.Timestamp("2025-01-04T06:50:00Z")
    assert schedule.close_time.iloc[1] == pd.Timestamp("2025-01-08T15:50:00Z")


def test_calendar_rejects_empty_and_invalid_hours():
    def empty(url):
        return response([["2025-01-01", 0, "H"]])

    with pytest.raises(ValueError, match="no trading sessions"):
        download_stock_calendar(
            "2025-01-01T00:00:00+03:00", "2025-01-02T00:00:00+03:00", transport=empty
        )
    with pytest.raises(ValueError, match="open_at"):
        download_stock_calendar(
            "2025-01-01T00:00:00+03:00",
            "2025-01-02T00:00:00+03:00",
            open_at="19:00",
            close_at="18:00",
            transport=empty,
        )


def test_calendar_requires_expected_table():
    with pytest.raises(ValueError, match="no off_days"):
        download_stock_calendar(
            "2025-01-01T00:00:00+03:00",
            "2025-01-02T00:00:00+03:00",
            transport=lambda url: {},
        )


def test_calendar_clips_partial_requested_range():
    schedule = download_stock_calendar(
        "2025-01-08T10:00:30+03:00",
        "2025-01-08T10:05:30+03:00",
        transport=lambda url: response([["2025-01-08", 1, "N"]]),
    )
    assert schedule.open_time.iloc[0] == pd.Timestamp("2025-01-08T07:01:00Z")
    assert schedule.close_time.iloc[0] == pd.Timestamp("2025-01-08T07:05:00Z")

