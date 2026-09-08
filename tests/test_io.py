import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from robomoex.cache import load_cache, save_cache
from robomoex.demo import dataset
from robomoex.moex import download_minutes


def test_cache_roundtrip_and_corruption(tmp_path):
    bars, _ = dataset(1)
    path, query = tmp_path / "cache.json", {"symbol": "SBER", "end": "fixed"}
    save_cache(path, bars, query)
    first = load_cache(path, query)
    pd.testing.assert_frame_equal(first, load_cache(path, query))
    with pytest.raises(ValueError, match="query"):
        load_cache(path, {"symbol": "GAZP"})
    payload = json.loads(path.read_text())
    payload["csv"] += "corruption"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_cache(path, query)


def block(minutes):
    return {
        "candles": {
            "columns": ["begin", "open", "high", "low", "close", "volume"],
            "data": [[f"2025-01-06 10:{i:02}:00", 100, 101, 99, 100, 1] for i in minutes],
        }
    }


def test_moex_pagination_timezone_and_complete_end():
    offsets = []

    def transport(url):
        offset = int(parse_qs(urlparse(url).query)["start"][0])
        offsets.append(offset)
        return block([0, 1] if offset == 0 else [2])

    frame = download_minutes(
        "SBER", "2025-01-06T10:00:00+03:00", "2025-01-06T10:03:00+03:00", transport=transport
    )
    assert offsets == [0, 2]
    assert len(frame) == 3
    assert frame.open_time.iloc[0] == pd.Timestamp("2025-01-06T07:00:00Z")
    assert frame.close_time.iloc[-1] == pd.Timestamp("2025-01-06T07:03:00Z")


def test_pagination_stall_rejected():
    with pytest.raises(ValueError, match="advance"):
        download_minutes(
            "SBER",
            "2025-01-06T10:00:00+03:00",
            "2025-01-06T10:10:00+03:00",
            transport=lambda url: block([0, 1]),
        )


def test_failed_page_is_not_silently_cached_as_complete():
    def transport(url):
        offset = int(parse_qs(urlparse(url).query)["start"][0])
        if offset:
            raise TimeoutError("simulated outage")
        return block([0, 1])

    with pytest.raises(TimeoutError):
        download_minutes(
            "SBER", "2025-01-06T10:00:00+03:00", "2025-01-06T10:10:00+03:00", transport=transport
        )


def test_invalid_symbol_never_reaches_transport():
    def forbidden(url):
        pytest.fail("Unexpected request")

    with pytest.raises(ValueError, match="symbol"):
        download_minutes(
            "../SBER", "2025-01-06T10:00:00Z", "2025-01-06T11:00:00Z", transport=forbidden
        )
