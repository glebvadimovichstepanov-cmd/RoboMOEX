from robomoex.events import download_dividends, event_context, load_event_cache, save_event_cache


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


def test_dividend_feed_and_event_context(tmp_path):
    body = b"<table><tr><td>17.07.2025</td><td>0.90 RUB</td></tr></table>"
    frame = download_dividends(
        "https://example.test/sngs", lambda *_args, **_kwargs: Response(body)
    )
    assert frame.iloc[0].amount_rub == 0.9
    context = event_context(frame)
    assert context.iloc[0].event_risk == "UNKNOWN"
    path = tmp_path / "events.json"
    identity = {"provider": "test", "ticker": "SNGS"}
    save_event_cache(path, frame, identity)
    loaded = load_event_cache(path, identity)
    assert loaded.equals(frame.reset_index(drop=True))
