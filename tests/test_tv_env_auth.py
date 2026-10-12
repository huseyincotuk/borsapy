"""Offline tests for the TradingView environment-based auth bootstrap.

All tests are fully mocked: no network access, no real credentials. The
WebSocket layer is replaced with a fake ``WebSocketApp`` that replays a single
``qsd`` packet after invoking the real ``on_open`` handler, so the token
selection path (``TradingViewProvider._get_auth_token``) runs exactly as in
production. ``get_user`` is monkeypatched so the lazy environment bootstrap
never touches the wire.
"""

import json
from unittest.mock import MagicMock

import pytest

from borsapy._providers import tradingview as tv_module
from borsapy._providers.tradingview import TradingViewProvider
from borsapy.exceptions import AuthenticationError

ENV_SESSION = "env-session-id-secret-value"
ENV_SESSION_SIGN = "env-session-sign-secret-value"
AUTH_TOKEN = "resolved-auth-token-secret-value"
EXPLICIT_TOKEN = "explicit-auth-token-secret-value"


class FakeWebSocketApp:
    """Minimal WebSocketApp replacement that drives the real callbacks."""

    def __init__(self, url, on_open, on_message, on_error, header):
        self._on_open = on_open
        self._on_message = on_message
        self.sent_messages = []

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
                    {"s": "BIST:THYAO", "v": {"lp": 100.5, "lp_time": 1700000000}},
                ],
            }
        )
        self._on_message(self, f"~m~{len(payload)}~m~{payload}")

    def close(self):
        pass


@pytest.fixture
def tv_env_state(monkeypatch):
    """Isolate the module auth store, bootstrap flag and environment.

    Restores the original module state afterwards and installs the offline
    WebSocket fake. No environment variable is left set.
    """
    saved_credentials = tv_module._auth_credentials
    saved_flag = tv_module._env_bootstrap_attempted
    tv_module._auth_credentials = None
    tv_module._env_bootstrap_attempted = False
    monkeypatch.delenv("TRADINGVIEW_SESSION", raising=False)
    monkeypatch.delenv("TRADINGVIEW_SESSION_SIGN", raising=False)
    monkeypatch.setattr("websocket.WebSocketApp", FakeWebSocketApp)
    yield tv_module
    tv_module._auth_credentials = saved_credentials
    tv_module._env_bootstrap_attempted = saved_flag


def _get_quote():
    return TradingViewProvider().get_quote("THYAO")


def _install_get_user(monkeypatch, *, result=None, error=None):
    """Patch ``TradingViewProvider.get_user`` and return the mock."""
    mock = MagicMock()
    if error is not None:
        mock.side_effect = error
    else:
        mock.return_value = result if result is not None else {"auth_token": AUTH_TOKEN}
    monkeypatch.setattr(TradingViewProvider, "get_user", mock)
    return mock


def test_environment_absent_selects_anonymous(tv_env_state, monkeypatch):
    """No env, no explicit creds -> anonymous token, get_user untouched."""
    get_user = _install_get_user(monkeypatch)

    quote = _get_quote()

    assert quote["auth_transport"] == "anonymous_token"
    assert get_user.call_count == 0


def test_environment_session_selects_auth_token(tv_env_state, monkeypatch):
    """Present env session -> bootstrap resolves and sends the auth token."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    get_user = _install_get_user(monkeypatch)

    quote = _get_quote()

    assert quote["auth_transport"] == "auth_token"
    assert get_user.call_count == 1
    stored = tv_env_state._auth_credentials
    assert stored is not None
    assert stored["auth_token"] == AUTH_TOKEN
    assert stored["session"] == ENV_SESSION


def test_authentication_error_falls_back_anonymous(tv_env_state, monkeypatch):
    """A bootstrap AuthenticationError must not raise; anonymous fallback."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    _install_get_user(monkeypatch, error=AuthenticationError("invalid session"))

    quote = _get_quote()

    assert quote["auth_transport"] == "anonymous_token"
    assert tv_env_state._auth_credentials is None


def test_bootstrap_attempted_at_most_once(tv_env_state, monkeypatch):
    """Two consecutive acquisitions invoke get_user exactly once."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    get_user = _install_get_user(monkeypatch)

    first = _get_quote()
    second = _get_quote()

    assert first["auth_transport"] == "auth_token"
    assert second["auth_transport"] == "auth_token"
    assert get_user.call_count == 1


def test_explicit_auth_wins_over_environment(tv_env_state, monkeypatch):
    """Explicit set_tradingview_auth wins; the env get_user mock is untouched."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    get_user = _install_get_user(monkeypatch)
    login_user = MagicMock(
        return_value={
            "session": "explicit-session",
            "session_sign": "explicit-sign",
            "auth_token": EXPLICIT_TOKEN,
        }
    )
    monkeypatch.setattr(TradingViewProvider, "login_user", login_user)

    tv_module.set_tradingview_auth(username="user", password="pass")
    quote = _get_quote()

    assert login_user.call_count == 1
    assert quote["auth_transport"] == "auth_token"
    # The explicit path disables the environment bootstrap entirely.
    assert get_user.call_count == 0
    assert tv_env_state._auth_credentials["auth_token"] == EXPLICIT_TOKEN


def test_clear_auth_resets_bootstrap(tv_env_state, monkeypatch):
    """clear_tradingview_auth resets the flag so bootstrap runs again."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    get_user = _install_get_user(monkeypatch)

    first = _get_quote()
    assert first["auth_transport"] == "auth_token"
    assert get_user.call_count == 1

    tv_module.clear_tradingview_auth()
    second = _get_quote()

    assert second["auth_transport"] == "auth_token"
    assert get_user.call_count == 2


def test_empty_session_behaves_as_absent(tv_env_state, monkeypatch):
    """An empty-string env session is treated as absent (no get_user call)."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", "")
    get_user = _install_get_user(monkeypatch)

    quote = _get_quote()

    assert quote["auth_transport"] == "anonymous_token"
    assert get_user.call_count == 0


def test_whitespace_session_behaves_as_absent(tv_env_state, monkeypatch):
    """A whitespace-only env session is treated as absent."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", "   ")
    get_user = _install_get_user(monkeypatch)

    quote = _get_quote()

    assert quote["auth_transport"] == "anonymous_token"
    assert get_user.call_count == 0


def test_no_credential_leakage(tv_env_state, monkeypatch):
    """Neither the quote dict nor auth_transport leaks credential material."""
    monkeypatch.setenv("TRADINGVIEW_SESSION", ENV_SESSION)
    monkeypatch.setenv("TRADINGVIEW_SESSION_SIGN", ENV_SESSION_SIGN)
    _install_get_user(monkeypatch, result={"auth_token": AUTH_TOKEN})

    quote = _get_quote()
    dumped = json.dumps(quote)

    for secret in (ENV_SESSION, ENV_SESSION_SIGN, AUTH_TOKEN):
        assert secret not in dumped
    assert quote["auth_transport"] in ("auth_token", "anonymous_token")
    for key in ("auth_token", "session", "session_sign", "user"):
        assert key not in quote
