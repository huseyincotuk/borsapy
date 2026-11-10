"""Offline pin tests for KAP fund objId extraction (get_fund_id case fix).

KAP's fund info page may expose the 32-hex ``objId`` in lowercase or uppercase
hex. Because the returned id is used verbatim in the disclosure filter URL path
(which is case-sensitive), the source casing must survive unchanged.

All tests serve canned HTML/JSON through a fake transport; no network access is
performed.
"""

from __future__ import annotations

import httpx
import pytest

from borsapy._providers.kap_holdings import (
    PORTFOLIO_REPORT_DISCLOSURE_TYPE,
    KAPHoldingsProvider,
)
from borsapy.exceptions import DataNotAvailableError

# Prep evidence prefix (2026-11-10): KAP serves lowercase objIds today.
LOWERCASE_ID = "4028328c8b2f3d4e5f60718293a4b5c6"
# Legacy uppercase form, pinned from the previous example comment.
UPPERCASE_ID = "33E5FED7E50300EAE0530A4A622B2AEA"

KAP_LINK = "https://www.kap.org.tr/tr/fon-bilgileri/genel/4028e4a1"


class _FakeResponse:
    def __init__(
        self,
        text: str = "",
        payload: object | None = None,
        status_exc: Exception | None = None,
    ):
        self.text = text
        self._payload = payload
        self._status_exc = status_exc

    def raise_for_status(self) -> None:
        if self._status_exc is not None:
            raise self._status_exc

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal httpx.Client stand-in that records calls and serves canned data."""

    def __init__(self, text: str = "", get_exc: Exception | None = None, handler=None):
        self._text = text
        self._get_exc = get_exc
        self._handler = handler
        self.calls = 0
        self.urls: list[str] = []

    def get(self, url, **kwargs):  # noqa: ARG002
        self.calls += 1
        self.urls.append(url)
        if self._get_exc is not None:
            raise self._get_exc
        if self._handler is not None:
            return self._handler(url)
        return _FakeResponse(text=self._text)


class _FakeTefas:
    def __init__(self, kap_link: str | None):
        self._kap_link = kap_link
        self.calls = 0

    def get_fund_detail(self, fund_code, fund_type="YAT"):  # noqa: ARG002
        self.calls += 1
        return {"code": fund_code, "kap_link": self._kap_link}


def _make_provider(monkeypatch, client, kap_link=KAP_LINK):
    provider = KAPHoldingsProvider()
    provider._client = client
    tefas = _FakeTefas(kap_link)
    monkeypatch.setattr(
        "borsapy._providers.tefas.get_tefas_provider", lambda: tefas
    )
    return provider, tefas


def _escaped_page(fund_id: str) -> str:
    """Next.js payload shape: objId\\":\\"<hex>\\"."""
    return '{"props":{"pageProps":{"objId\\":\\"' + fund_id + '\\""}}}'


def _plain_page(fund_id: str) -> str:
    """Plain JSON shape: "objId":"<hex>"."""
    return '{"props":{"pageProps":{"objId":"' + fund_id + '"}}}'


# --------------------------------------------------------------------------
# REQ-002.1 / REQ-002.2 — casing preserved exactly as received
# --------------------------------------------------------------------------


def test_lowercase_objid_captured_unchanged(monkeypatch):
    client = _FakeClient(text=_escaped_page(LOWERCASE_ID))
    provider, _ = _make_provider(monkeypatch, client)

    assert provider.get_fund_id("BHE") == LOWERCASE_ID
    assert provider.get_fund_id("BHE") == "4028328c8b2f3d4e5f60718293a4b5c6"


def test_uppercase_objid_captured_unchanged(monkeypatch):
    client = _FakeClient(text=_escaped_page(UPPERCASE_ID))
    provider, _ = _make_provider(monkeypatch, client)

    result = provider.get_fund_id("BHE")

    assert result == UPPERCASE_ID
    assert result == "33E5FED7E50300EAE0530A4A622B2AEA"


# --------------------------------------------------------------------------
# REQ-002.3 — escaped-JSON form works for both casings (plus plain JSON)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fund_id", [LOWERCASE_ID, UPPERCASE_ID])
@pytest.mark.parametrize("builder", [_escaped_page, _plain_page])
def test_objid_form_works_for_both_cases(monkeypatch, builder, fund_id):
    client = _FakeClient(text=builder(fund_id))
    provider, _ = _make_provider(monkeypatch, client)

    assert provider.get_fund_id("BHE") == fund_id


# --------------------------------------------------------------------------
# Negatif örtü — fail-closed paths
# --------------------------------------------------------------------------


def test_missing_objid_returns_none(monkeypatch):
    client = _FakeClient(text="<html><body>objId yok</body></html>")
    provider, _ = _make_provider(monkeypatch, client)

    assert provider.get_fund_id("BHE") is None


def test_missing_kap_link_returns_none(monkeypatch):
    client = _FakeClient(text=_escaped_page(LOWERCASE_ID))
    provider, _ = _make_provider(monkeypatch, client, kap_link=None)

    assert provider.get_fund_id("BHE") is None
    assert client.calls == 0


def test_broken_page_returns_none(monkeypatch):
    client = _FakeClient(get_exc=httpx.ConnectError("boom"))
    provider, _ = _make_provider(monkeypatch, client)

    assert provider.get_fund_id("BHE") is None


def test_http_error_returns_none(monkeypatch):
    def handler(url):  # noqa: ARG001
        return _FakeResponse(status_exc=RuntimeError("bad status"))

    provider, _ = _make_provider(monkeypatch, _FakeClient(handler=handler))

    assert provider.get_fund_id("BHE") is None


@pytest.mark.parametrize(
    "bad_id",
    [
        LOWERCASE_ID[:31],  # 31 hex chars
        LOWERCASE_ID + "a",  # 33 hex chars
    ],
)
def test_non_32_hex_values_rejected(monkeypatch, bad_id):
    client = _FakeClient(text=_escaped_page(bad_id))
    provider, _ = _make_provider(monkeypatch, client)

    assert provider.get_fund_id("BHE") is None


# --------------------------------------------------------------------------
# REQ-002.5 — cache: second call for the same fund_code hits no transport
# --------------------------------------------------------------------------


def test_cache_prevents_second_request(monkeypatch):
    client = _FakeClient(text=_escaped_page(LOWERCASE_ID))
    provider, tefas = _make_provider(monkeypatch, client)

    first = provider.get_fund_id("BHE")
    second = provider.get_fund_id("BHE")

    assert first == second == LOWERCASE_ID
    assert client.calls == 1
    assert tefas.calls == 1


# --------------------------------------------------------------------------
# REQ-002.6 — get_disclosures forwards the fund_id verbatim into the URL
# --------------------------------------------------------------------------


def test_get_disclosures_uses_fund_id_verbatim(monkeypatch):
    payload = [
        {
            "disclosureBasic": {
                "disclosureId": 123,
                "disclosureIndex": 456,
                "publishDate": "2026-10-31",
                "title": "Portföy Dağılım Raporu",
                "summary": "",
                "year": 2026,
                "donem": 10,
                "attachmentCount": 1,
            }
        }
    ]

    def handler(url):
        if url.startswith(KAPHoldingsProvider.DISCLOSURE_FILTER_URL):
            return _FakeResponse(payload=payload)
        return _FakeResponse(text=_escaped_page(LOWERCASE_ID))

    client = _FakeClient(handler=handler)
    provider, _ = _make_provider(monkeypatch, client)

    disclosures = provider.get_disclosures("BHE")

    expected_url = (
        f"{KAPHoldingsProvider.DISCLOSURE_FILTER_URL}/"
        f"{LOWERCASE_ID}/"
        f"{PORTFOLIO_REPORT_DISCLOSURE_TYPE}/365"
    )
    assert client.urls == [KAP_LINK, expected_url]
    assert disclosures[0]["disclosure_id"] == 123
    assert disclosures[0]["period"] == 10


def test_get_disclosures_raises_when_fund_id_missing(monkeypatch):
    client = _FakeClient(text="<html>objId yok</html>")
    provider, _ = _make_provider(monkeypatch, client)

    with pytest.raises(DataNotAvailableError):
        provider.get_disclosures("BHE")
