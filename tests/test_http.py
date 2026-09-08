import io
from urllib.error import HTTPError, URLError

import pytest

from robomoex.moex import get_json


def test_transient_http_retry_bounded(monkeypatch):
    calls = []

    def request(url, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise HTTPError(url, 503, "temporary", {}, None)
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("robomoex.moex.urlopen", request)
    monkeypatch.setattr("robomoex.moex.time.sleep", lambda seconds: None)
    assert get_json("https://iss.moex.com/test") == {"ok": True}
    assert calls == [15, 15, 15]


def test_permanent_error_not_retried(monkeypatch):
    calls = []

    def request(url, timeout):
        calls.append(url)
        raise HTTPError(url, 403, "forbidden", {}, None)

    monkeypatch.setattr("robomoex.moex.urlopen", request)
    with pytest.raises(HTTPError):
        get_json("https://iss.moex.com/test")
    assert len(calls) == 1


def test_connection_failure_is_reported_after_three_attempts(monkeypatch):
    calls = []

    def request(url, timeout):
        calls.append(url)
        raise URLError("offline")

    monkeypatch.setattr("robomoex.moex.urlopen", request)
    monkeypatch.setattr("robomoex.moex.time.sleep", lambda seconds: None)
    with pytest.raises(URLError):
        get_json("https://iss.moex.com/test")
    assert len(calls) == 3
