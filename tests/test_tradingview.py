"""Tests for TradingView provider authentication."""

from unittest.mock import Mock, patch

import httpx

from borsapy._providers.tradingview import TradingViewProvider


class TestTradingViewAuthentication:

    @patch("borsapy._providers.tradingview.httpx.Client")
    def test_get_user_uses_cookie_jar(self, mock_client_cls):
        response = Mock(spec=httpx.Response)
        response.text = '{"auth_token":"test-token","id":123}'

        client = Mock()
        client.get.return_value = response
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        mock_client_cls.return_value = client

        provider = TradingViewProvider()
        result = provider.get_user("test-session", "test-sign")

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["cookies"]["sessionid"] == "test-session"
        assert kwargs["cookies"]["sessionid_sign"] == "test-sign"
        assert kwargs["follow_redirects"] is True
        assert "Cookie" not in kwargs["headers"]
        assert result["auth_token"] == "test-token"
