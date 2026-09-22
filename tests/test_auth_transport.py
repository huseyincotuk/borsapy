"""Tests for the per-request ``auth_transport`` live-quote field.

All tests are fully mocked: no network access, no real credentials. The
WebSocket layer is replaced with a fake ``WebSocketApp`` that replays a
single ``qsd`` packet after invoking the real ``on_open`` handler, so the
token-selection code path runs exactly as in production.
"""

import json
from unittest.mock import patch

import pytest

from borsapy._providers import tradingview as tv_module
from borsapy._providers.tradingview import TradingViewProvider
from borsapy.ticker import Ticker

SECRET_TOKEN = "super-secret-auth-token-value-123456"


# Sent messages of the most recent FakeWebSocketApp instance, so tests can
# verify what was actually placed on the wire.
_last_sent_messages: list[str] = []


class FakeWebSocketApp:
    """Minimal WebSocketApp replacement that drives the real callbacks."""

    def __init__(self, url, on_open, on_message, on_error, header):
        self._on_open = on_open
        self._on_message = on_message
        self.sent_messages = []
        global _last_sent_messages
        _last_sent_messages = self.sent_messages

    def send(self, data):
        self.sent_messages.append(data)

    def run_forever(self):
        # Execute the real on_open (token selection + sends) ...
        self._on_open(self)
        # ... then deliver a minimal qsd packet so the quote completes.
        payload = json.dumps(
            {
                "m": "qsd",
                "p": [
                    "qs",
                    {
                        "s": "BIST:THYAO",
                        "v": {"lp": 100.5, "lp_time": 1700000000},
                    },
                ],
            }
        )
        self._on_message(self, f"~m~{len(payload)}~m~{payload}")

    def close(self):
        pass


@pytest.fixture
def tv_auth_state():
    """Isolate and restore the module-level auth credentials."""
    saved = tv_module._auth_credentials
    tv_module._auth_credentials = None
    yield tv_module
    tv_module._auth_credentials = saved


def _get_quote():
    with patch("websocket.WebSocketApp", FakeWebSocketApp):
        provider = TradingViewProvider()
        return provider.get_quote("THYAO")


def _set_credentials(tv_module, auth_token):
    tv_module._auth_credentials = {
        "session": "test-session-id",
        "session_sign": "test-session-sign",
        "auth_token": auth_token,
        "user": {"id": 1},
    }


def test_no_credentials_anonymous_token(tv_auth_state):
    """No configured credentials -> auth_transport == 'anonymous_token'."""
    result = _get_quote()
    assert result["auth_transport"] == "anonymous_token"


def test_truthy_auth_token(tv_auth_state):
    """Configured truthy auth_token -> auth_transport == 'auth_token'."""
    _set_credentials(tv_auth_state, SECRET_TOKEN)
    result = _get_quote()
    assert result["auth_transport"] == "auth_token"


def test_none_auth_token_anonymous(tv_auth_state):
    """Configured auth_token=None -> auth_transport == 'anonymous_token'."""
    _set_credentials(tv_auth_state, None)
    result = _get_quote()
    assert result["auth_transport"] == "anonymous_token"


def test_empty_auth_token_anonymous(tv_auth_state):
    """Configured auth_token='' -> auth_transport == 'anonymous_token'."""
    _set_credentials(tv_auth_state, "")
    result = _get_quote()
    assert result["auth_transport"] == "anonymous_token"


def test_real_token_sent_but_never_exposed(tv_auth_state):
    """The real token is sent over the wire but never appears in the result."""
    _set_credentials(tv_auth_state, SECRET_TOKEN)
    result = _get_quote()

    # The provider DID select and send the real token (behavior preserved)...
    sent = json.dumps(_last_sent_messages)
    assert SECRET_TOKEN in sent
    # ...and classified the request as auth_token...
    assert result["auth_transport"] == "auth_token"
    # ...but the token value is nowhere in the returned quote.
    assert SECRET_TOKEN not in json.dumps(result)
    assert "unauthorized_user_token" not in json.dumps(result)


def test_anonymous_token_sent_when_unconfigured(tv_auth_state):
    """Token selection behavior is preserved: placeholder sent when unconfigured."""
    result = _get_quote()
    sent = json.dumps(_last_sent_messages)
    assert "unauthorized_user_token" in sent
    assert result["auth_transport"] == "anonymous_token"
    assert result["auth_transport"] in ("auth_token", "anonymous_token", "unknown")


def test_secret_leak_guard_provider(tv_auth_state):
    """Provider quote dict exposes no credential material."""
    _set_credentials(tv_auth_state, SECRET_TOKEN)
    result = _get_quote()

    for key in ("auth_token", "session", "session_sign", "cookies", "user"):
        assert key not in result
    dumped = json.dumps(result)
    assert SECRET_TOKEN not in dumped
    assert "test-session-id" not in dumped
    assert "test-session-sign" not in dumped


def test_ticker_exposes_auth_transport():
    """Ticker.get_live_quote() exposes auth_transport when provider supplies it."""
    ticker = Ticker.__new__(Ticker)
    ticker._symbol = "THYAO"
    ticker._tradingview = None

    from unittest.mock import MagicMock

    mock_provider = MagicMock()
    mock_provider.get_quote.return_value = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 100.5,
        "auth_transport": "auth_token",
    }
    ticker._tradingview = mock_provider

    result = ticker.get_live_quote()
    assert result["auth_transport"] == "auth_token"
    assert result["last"] == 100.5


def test_ticker_omits_auth_transport_when_missing():
    """Ticker.get_live_quote() omits auth_transport when provider lacks it."""
    from unittest.mock import MagicMock

    ticker = Ticker.__new__(Ticker)
    ticker._symbol = "THYAO"
    mock_provider = MagicMock()
    mock_provider.get_quote.return_value = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 100.5,
    }
    ticker._tradingview = mock_provider

    result = ticker.get_live_quote()
    assert "auth_transport" not in result
    assert result == {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 100.5,
    }


def test_ticker_secret_leak_guard():
    """Public live quote never exposes auth_token/session/session_sign keys."""
    from unittest.mock import MagicMock

    ticker = Ticker.__new__(Ticker)
    ticker._symbol = "THYAO"
    mock_provider = MagicMock()
    mock_provider.get_quote.return_value = {
        "symbol": "THYAO",
        "last": 100.5,
        "auth_transport": "auth_token",
        "auth_token": SECRET_TOKEN,
        "session": "test-session-id",
        "session_sign": "test-session-sign",
    }
    ticker._tradingview = mock_provider

    result = ticker.get_live_quote()
    for key in ("auth_token", "session", "session_sign"):
        assert key not in result
    assert SECRET_TOKEN not in json.dumps(result)


def test_per_request_classification(tv_auth_state):
    """Classification is captured per request, not recomputed from later state."""
    # Request 1: no credentials.
    result1 = _get_quote()
    assert result1["auth_transport"] == "anonymous_token"

    # Global auth state changes AFTER request 1 completed.
    _set_credentials(tv_auth_state, SECRET_TOKEN)
    # The already-returned quote keeps its original classification.
    assert result1["auth_transport"] == "anonymous_token"

    # Request 2 (new credentials) gets its own classification.
    result2 = _get_quote()
    assert result2["auth_transport"] == "auth_token"

    # Global state changes again; result2 remains stable.
    tv_auth_state._auth_credentials = None
    assert result2["auth_transport"] == "auth_token"
    assert result1["auth_transport"] == "anonymous_token"
