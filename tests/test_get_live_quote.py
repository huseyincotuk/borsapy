"""Tests for Ticker.get_live_quote public API.

The tests mock the internal TradingView provider to avoid network calls and verify
that the public method returns only the documented fields, preserves ``None``
values, and does not leak internal keys.
"""

from unittest.mock import MagicMock

from borsapy.ticker import Ticker


def _make_ticker(mock_quote: dict) -> Ticker:
    """Create a Ticker instance with a mocked TradingView provider.

    The function bypasses the normal provider initialization by constructing the
    object via ``__new__`` and injecting a ``MagicMock`` that returns ``mock_quote``
    from ``get_quote``.
    """
    ticker = Ticker.__new__(Ticker)
    ticker._symbol = "THYAO"
    ticker._tradingview = MagicMock()
    ticker._tradingview.get_quote.return_value = mock_quote
    return ticker


def test_get_live_quote_all_fields():
    """All allowed fields are returned and extra fields are omitted."""
    mock_quote = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 100.5,
        "bid": 100.0,
        "ask": 101.0,
        "volume": 12345,
        "timestamp": 1700000000,
        # internal/private fields that should not be exposed
        "description": "some description",
        "currency": "TRY",
    }
    ticker = _make_ticker(mock_quote)
    result = ticker.get_live_quote()
    expected = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 100.5,
        "bid": 100.0,
        "ask": 101.0,
        "volume": 12345,
        "timestamp": 1700000000,
    }
    assert result == expected
    # Ensure the internal provider was called with the correct symbol.
    ticker._tradingview.get_quote.assert_called_once_with("THYAO")


def test_get_live_quote_missing_and_none():
    """Missing fields are omitted; ``None`` values are preserved."""
    mock_quote = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": None,
        "bid": None,
        "ask": 101.0,
        # ``volume`` is omitted entirely
        "timestamp": None,
        "extra": "should not appear",
    }
    ticker = _make_ticker(mock_quote)
    result = ticker.get_live_quote()
    expected = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": None,
        "bid": None,
        "ask": 101.0,
        "timestamp": None,
    }
    assert result == expected
    # No ``volume`` key because it was not present in the provider result.
    assert "volume" not in result
    # Ensure internal keys are not leaked.
    assert "extra" not in result


def test_get_live_quote_empty_result():
    """When the provider returns an empty dict, the public method returns an empty dict."""
    ticker = _make_ticker({})
    result = ticker.get_live_quote()
    assert result == {}
    ticker._tradingview.get_quote.assert_called_once_with("THYAO")
