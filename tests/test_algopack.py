import io
import ssl

import pytest

from robomoex.algopack import NoRedirect, get_authenticated_json


def test_token_never_sent_to_other_hosts(monkeypatch):
    def forbidden():
        raise AssertionError("Token must not be read")

    monkeypatch.setattr("robomoex.algopack.get_token", forbidden)
    for url in (
        "http://apim.moex.com/iss/x",
        "https://example.com/iss/x",
        "https://apim.moex.com.evil.test/iss/x",
    ):
        with pytest.raises(ValueError, match="restricted"):
            get_authenticated_json(url)


def test_authenticated_redirect_is_blocked():
    with pytest.raises(ValueError, match="redirects"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")


def test_authentication_keeps_tls_and_hostname_checks(monkeypatch):
    monkeypatch.setattr("robomoex.algopack.get_token", lambda: "test-token")

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://apim.moex.com/iss/engines.json"
            assert request.get_header("Authorization") == "Bearer test-token"
            return io.BytesIO(b'{"ok":true}')

    def opener(*handlers):
        context = handlers[1]._context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        return Opener()

    monkeypatch.setattr("robomoex.algopack.build_opener", opener)
    assert get_authenticated_json("https://iss.moex.com/iss/engines.json") == {"ok": True}
