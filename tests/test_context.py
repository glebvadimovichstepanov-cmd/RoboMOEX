import pandas as pd

from robomoex.context import MoexDailySource, load_context_cache, update_context_cache


def test_context_cache_extends_only_missing_edges(tmp_path):
    source = MoexDailySource("imoex", "stock", "index", "SNGX", "IMOEX")
    calls = []

    def transport(url):
        calls.append(url)
        return {
            "candles": {
                "columns": ["begin", "close"],
                "data": [["2025-01-01 00:00:00", 1.0], ["2025-01-02 00:00:00", 2.0]],
            }
        }

    path = tmp_path / "imoex.json"
    update_context_cache(path, source, "2025-01-01", "2025-01-02", transport)
    update_context_cache(path, source, "2025-01-01", "2025-01-02", transport)
    assert len(calls) == 1
    frame, first, last = load_context_cache(
        path,
        {
            "provider": "moex-iss-daily-v1",
            "source": "imoex",
            "engine": "stock",
            "market": "index",
            "board": "SNGX",
            "security": "IMOEX",
        },
    )
    assert len(frame) == 2
    assert first == pd.Timestamp("2025-01-01").date()
    assert last == pd.Timestamp("2025-01-02").date()
