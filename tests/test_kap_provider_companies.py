"""Offline pin tests for the KAP BIST company decode flow (FA-07).

The fixtures under ``tests/fixtures/kap/`` are the sealed 2026-11-03 captures
and must stay byte-identical. All tests here serve the fixture HTML through a
fake transport, so no real network access is performed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest

import borsapy._providers.kap as kap_module
from borsapy._providers.kap import (
    KAPProvider,
    _decode_member_blocks,
    _extract_next_f_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kap"
HTML_FIXTURE = FIXTURE_DIR / "fa07_bist_sirketler_2026_11_03.html"
JSON_FIXTURE = FIXTURE_DIR / "fa07_bist_sirketler_ssr_2026_11_03.json"

HTML_SHA256 = "500cb49c4051b78251ea4e1f6ed915d64a455513ac9f1a3e55fc6a5773c2723f"
JSON_SHA256 = "d693224cbdf9db4e39b6f4ab7ec4b57c29110998605a5f025c8ee39e682fda8d"

THYAO_OID = "4028e4a140f2ed720140f376bebb01a7"

FORMERLY_MISSED = [
    "ADB",
    "ADBNK",
    "AYTEM",
    "BNPPI",
    "DKYVK",
    "DMLKT",
    "DOCO",
    "DRPHN",
    "GRM",
    "GRYAT",
    "GSIPD",
    "MSY",
    "MSYBN",
    "PKENT",
    "USHOL",
    "YYA",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def html() -> str:
    return HTML_FIXTURE.read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, text: str | None = None, exc: Exception | None = None):
        self._text = text
        self._exc = exc
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._text or "")


def _provider(html_text: str) -> tuple[KAPProvider, _FakeClient]:
    provider = KAPProvider()
    client = _FakeClient(text=html_text)
    provider._client = client
    return provider, client


def test_fixture_checksums():
    assert HTML_FIXTURE.stat().st_size == 1540543
    assert _sha256(HTML_FIXTURE) == HTML_SHA256
    assert _sha256(JSON_FIXTURE) == JSON_SHA256


def test_payload_has_three_push_chunks(html):
    chunks = re.findall(
        r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html, re.DOTALL
    )
    assert len(chunks) == 3
    payload = _extract_next_f_payload(html)
    assert '"mkkMemberOid"' in payload


def test_decode_member_blocks_pins(html):
    records = _decode_member_blocks(html)

    assert len(records) == 755
    assert len({record["oid"] for record in records}) == 755

    tickers = [ticker for record in records for ticker in record["tickers"]]
    assert len(tickers) == 804
    assert len(set(tickers)) == 804

    multi_ticker = [record for record in records if len(record["tickers"]) > 1]
    assert len(multi_ticker) == 45

    with_city = [record for record in records if record["city"]]
    assert len(with_city) == 747

    assert set(records[0]) == {"oid", "name", "tickers", "city"}


def test_get_companies_rows(html):
    provider, client = _provider(html)
    df = provider.get_companies()

    assert client.calls == 1
    assert list(df.columns) == ["ticker", "name", "city"]
    assert len(df) == 804
    assert df["ticker"].nunique() == 804

    garan = df[df["ticker"] == "GARAN"].iloc[0]
    tgb = df[df["ticker"] == "TGB"].iloc[0]
    assert garan["name"] == tgb["name"]
    assert garan["city"] == tgb["city"]

    ykb = df[df["ticker"] == "YKB"].iloc[0]
    ykbnk = df[df["ticker"] == "YKBNK"].iloc[0]
    assert ykb["name"] == ykbnk["name"]
    assert ykb["city"] == ykbnk["city"]


def test_thyao_oid(html):
    provider, _client = _provider(html)

    df = provider.get_companies()
    thyao = df[df["ticker"] == "THYAO"]
    assert len(thyao) == 1
    assert thyao.iloc[0]["name"] == "TÜRK HAVA YOLLARI A.O."
    assert thyao.iloc[0]["city"] == "İSTANBUL"

    assert provider.get_member_oid("THYAO") == THYAO_OID


@pytest.mark.parametrize("code", FORMERLY_MISSED)
def test_formerly_missed_codes_resolve(html, code):
    provider, _client = _provider(html)
    assert provider.get_member_oid(code) is not None


def test_fail_loud_on_missing_blocks():
    with pytest.raises(RuntimeError):
        _decode_member_blocks("<html></html>")

    provider, _client = _provider("<html></html>")
    with pytest.raises(RuntimeError):
        provider.get_companies()


def test_get_member_oid_parse_and_network_failures():
    provider, _client = _provider("<html></html>")
    with pytest.raises(RuntimeError):
        provider.get_member_oid("THYAO")

    failing = KAPProvider()
    failing._client = _FakeClient(exc=RuntimeError("network down"))
    assert failing.get_member_oid("THYAO") is None


def test_company_cache_prevents_refetch(html):
    provider, client = _provider(html)

    first = provider.get_companies()
    second = provider.get_companies()

    assert client.calls == 1
    assert first is second


def test_search_contract(html):
    provider, _client = _provider(html)

    result = provider.search("THYAO")
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "THYAO"

    empty = provider.search("")
    assert isinstance(empty, pd.DataFrame)
    assert empty.empty


def test_unknown_symbol_returns_none(html):
    provider, _client = _provider(html)
    assert provider.get_member_oid("YABANCI") is None


def test_extract_payload_without_chunks_raises():
    with pytest.raises(RuntimeError):
        _extract_next_f_payload("<html></html>")


def test_module_has_no_excel_path():
    assert not hasattr(KAPProvider, "EXCEL_URL")
    assert not hasattr(kap_module, "io")
