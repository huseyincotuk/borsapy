"""Offline pin tests for the KAP company structure parser (W1.75 Parsel A).

The fixture under ``tests/fixtures/kap/`` is the sealed 2026-11-09 capture of
the KAP "Genel Bilgiler" page for PENTA (the reference case of
``KAP_genel_sayfa_FD_dogrulama.md``) and must stay byte-identical. All tests
serve that HTML through a fake transport; no real network access is performed.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from borsapy._providers.kap import (
    DATA_UNAVAILABLE,
    KAPProvider,
    _extract_item_object,
    _extract_named_object,
    _extract_next_f_payload,
    _parse_company_structure,
    _parse_dmy_date,
    _parse_tr_decimal,
)
from borsapy.exceptions import KAPStructureError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kap"
HTML_FIXTURE = FIXTURE_DIR / "w175_penta_genel_2026_11_09.html"

HTML_SHA256 = "1a1b1ef1dcbdb928251c5753713fdb072022f2572bbd47a1164526add6d6ff50"
HTML_SIZE = 336971

PENTA_OID = "4028e4a14184e9c9014198033c094251"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def html() -> str:
    return HTML_FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def payload(html: str) -> str:
    return _extract_next_f_payload(html)


@pytest.fixture(scope="module")
def structure(payload: str) -> dict:
    return _parse_company_structure(
        payload,
        symbol="PENTA",
        member_oid=PENTA_OID,
        source_url=f"https://kap.org.tr/tr/sirket-bilgileri/genel/{PENTA_OID}",
        retrieved_at="2026-11-09T00:00:00+00:00",
    )


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, text: str = "", exc: Exception | None = None):
        self._text = text
        self._exc = exc
        self.calls = 0
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.calls += 1
        self.urls.append(url)
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._text)


def _provider(html_text: str) -> tuple[KAPProvider, _FakeClient]:
    provider = KAPProvider()
    client = _FakeClient(text=html_text)
    provider._client = client
    provider.get_member_oid = lambda symbol: PENTA_OID  # noqa: ARG005
    return provider, client


# --------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------


def test_fixture_checksum():
    assert HTML_FIXTURE.exists()
    assert HTML_FIXTURE.stat().st_size == HTML_SIZE
    assert _sha256(HTML_FIXTURE) == HTML_SHA256


def test_payload_has_capital_structure_items(payload):
    for item_key in (
        "kpy41_acc5_odenmis_sermaye",
        "kpy41_acc5_kayitli_sermaye_tavani",
        "kpy41_acc5_sermayede_dogrudan",
        "kpy41_acc5_son_durum_sermayeye",
        "kpy41_acc5_sermayeyi_temsil_eden",
        "kpy41_acc5_fiili_dolasimdaki_pay",
    ):
        assert _extract_item_object(payload, item_key) is not None, item_key

    assert _extract_named_object(payload, "actualSharesOutstandingGrid") is not None


# --------------------------------------------------------------------------
# Turkish number / date parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("393.516.000", Decimal("393516000")),
        ("154.873.002,50", Decimal("154873002.50")),
        ("0,00", Decimal("0.00")),
        ("39,35", Decimal("39.35")),
        ("106418079", Decimal("106418079")),
        ("-1.204,5", Decimal("-1204.5")),
        # KAP mixes formats: some rows carry plain decimals with a dot.
        ("1379999999.99", Decimal("1379999999.99")),
        ("678773422.299", Decimal("678773422.299")),
        ("0.01", Decimal("0.01")),
        # A lone dot with exactly three digits is a Turkish thousands group.
        ("1.234", Decimal("1234")),
    ],
)
def test_parse_tr_decimal_valid(raw, expected):
    parsed = _parse_tr_decimal(raw)
    assert isinstance(parsed, Decimal)
    assert parsed == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "-", "Bilgi Mevcut Değil", "n/a", "12.34.56", []],
)
def test_parse_tr_decimal_unavailable_or_invalid(raw):
    assert _parse_tr_decimal(raw) is None


def test_parse_dmy_date():
    assert _parse_dmy_date("08/11/2026") == "2026-11-08"
    assert _parse_dmy_date("24/03/2023 05:08:10") == "2023-03-24"
    assert _parse_dmy_date("20261108") == "2026-11-08"
    assert _parse_dmy_date(None) is None
    assert _parse_dmy_date("2026-11-08") is None


# --------------------------------------------------------------------------
# PENTA pins (spec §"Doğrulanan alanlar" + three reconciliations)
# --------------------------------------------------------------------------


def test_capital_pins(structure):
    assert structure["paid_in_capital"] == Decimal("393516000")
    assert structure["registered_capital_ceiling"] == Decimal("750000000")
    assert isinstance(structure["paid_in_capital"], Decimal)
    assert structure["raw"]["paid_in_capital"] == "393.516.000"
    assert structure["raw"]["registered_capital_ceiling"] == "750.000.000"


def test_direct_shareholders_pins(structure):
    block = structure["direct_shareholders"]
    assert block["status"] == "OK"
    assert block["as_of"] == "2026-09-26"

    rows = block["rows"]
    assert len(rows) == 6
    assert [row["name"] for row in rows] == [
        "GÖZDE GİRİŞİM SERMAYESİ YATIRIM ORTAKLIĞI ANONİM ŞİRKETİ",
        "MUSTAFA ERGÜN",
        "SİNAN GÜÇLÜ",
        "MÜRSEL ÖZÇELİK",
        "BÜLENT KORAY AKSOY",
        "DİĞER",
    ]
    assert [row["ratio_percent"] for row in rows] == [
        Decimal("32.21"),
        Decimal("9.72"),
        Decimal("6.61"),
        Decimal("6.24"),
        Decimal("5.84"),
        Decimal("39.38"),
    ]
    assert rows[0]["share_tl"] == Decimal("126753831")
    assert rows[0]["raw"]["shareInCapital"] == "126.753.831"

    total = block["reported_total"]
    assert total["name"] == "TOPLAM"
    assert total["share_tl"] == Decimal("393516000")
    # The section header says "%3 veya daha fazla"; the item name says "%5".
    # Both are kept verbatim, the header wins for display.
    assert block["section_title"].startswith("Sermayede Doğrudan %5")
    assert len(block["raw_rows"]) == 7


def test_float_pins(structure):
    block = structure["float"]
    assert block["as_of"] == "2026-11-08"
    assert block["as_of_raw"] == "20261108"
    assert block["isin"] == "PENTA"
    assert block["total_shares"] == Decimal("393516000.00")
    assert block["non_traded_shares"] == Decimal("212446224.00")
    assert block["traded_shares"] == Decimal("181069776.00")
    assert block["excluded_shares_c"] == Decimal("26196773.50")
    assert block["excluded_shares_de"] == Decimal("0.00")
    assert block["actual_float_shares"] == Decimal("154873002.50")
    assert block["actual_float_ratio_percent"] == Decimal("39.35")
    assert block["raw_row"]["actualSharesOutstanding"] == "154.873.002,50"


def test_share_groups_pins(structure):
    block = structure["share_groups"]
    assert block["status"] == "OK"
    assert [(row["group"], row["nominal_value"]) for row in block["rows"]] == [
        ("A", Decimal("106418079")),
        ("B", Decimal("287097921")),
    ]
    assert block["total_nominal"] == Decimal("393516000")
    assert block["matches_paid_in_capital"] is True
    assert block["raw_rows"][0]["nominalValueOfShares"] == "106418079"


def test_mixed_number_format_group_rows(payload):
    mixed = payload.replace('"nominalValueOfShares":"106418079"', '"nominalValueOfShares":"106418079.00"')
    mixed = mixed.replace('"nominalValueOfShares":"287097921"', '"nominalValueOfShares":"287097921.00"')
    assert mixed != payload

    result = _parse_company_structure(
        mixed, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
    )
    assert [row["nominal_value"] for row in result["share_groups"]["rows"]] == [
        Decimal("106418079.00"),
        Decimal("287097921.00"),
    ]
    assert result["share_groups"]["matches_paid_in_capital"] is True


def test_share_group_exchange_flag_is_flattened(payload):
    flagged = payload.replace(
        '"exchangeTradedOrNot":null',
        '"exchangeTradedOrNot":{"key":"1","text":"İşlem Görüyor"}',
    )
    assert flagged != payload

    result = _parse_company_structure(
        flagged, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
    )
    assert [row["exchange_traded"] for row in result["share_groups"]["rows"]] == [
        "İşlem Görüyor",
        "İşlem Görüyor",
    ]


def test_three_part_whole_reconciliations(structure):
    checks = {item["name"]: item for item in structure["reconciliations"]}
    assert len(structure["reconciliations"]) == 3

    first = checks["traded_plus_non_traded_equals_total"]
    assert first["ok"] is True
    assert first["left"] == Decimal("393516000.00") == first["right"]

    second = checks["traded_minus_exclusions_equals_float"]
    assert second["ok"] is True
    assert second["left"] == Decimal("154873002.50") == second["right"]

    third = checks["float_over_total_equals_reported_ratio"]
    assert third["ok"] is True
    assert third["right"] == Decimal("39.35")
    assert third["tolerance_pp"] == Decimal("0.01")
    assert first["difference"] == 0
    assert second["difference"] == 0
    assert abs(third["difference"]) <= third["tolerance_pp"]
    assert structure["reconciliations_all_ok"] is True


def test_float_methodology_credit(structure):
    methodology = structure["methodology"]
    assert "Fiili dolaşımdaki payların belirlenmesinde" in methodology["note"]
    assert methodology["effective_from"] == "2026-09-11"
    assert methodology["rule"] == (
        "SPK 03.09.2026 tarihli ve 54/1626 sayılı İlke Kararı"
    )
    assert methodology["applies_to"] == "actualSharesOutstandingGrid"
    assert structure["as_of"] == "2026-11-08"
    assert structure["source"]["url"].endswith(PENTA_OID)
    assert structure["source"]["retrieved_at"] == "2026-11-09T00:00:00+00:00"


def test_source_rounding_is_reported_not_hidden(payload):
    # KCHOL case (2026-11-08): float arithmetic is off by 0.01 TL in the source.
    rounded = payload.replace('"actualSharesOutstanding":"154.873.002,50"', '"actualSharesOutstanding":"154.873.002,51"')
    assert rounded != payload

    result = _parse_company_structure(
        rounded, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
    )
    check = {
        item["name"]: item for item in result["reconciliations"]
    }["traded_minus_exclusions_equals_float"]
    assert check["ok"] is False
    assert check["difference"] == Decimal("-0.01")
    assert result["reconciliations_all_ok"] is False
    assert result["float"]["actual_float_shares"] == Decimal("154873002.51")


def test_indirect_shareholders_is_data_unavailable(structure):
    block = structure["indirect_shareholders"]
    assert block["status"] == DATA_UNAVAILABLE
    assert block["reason"] == "kaynak ifadesi: Bilgi Mevcut Değil"
    assert block["rows"] == []
    assert block["raw_value"] == []
    assert any(
        entry["field"] == "indirect_shareholders"
        and entry["status"] == DATA_UNAVAILABLE
        for entry in structure["missing"]
    )


# --------------------------------------------------------------------------
# Fail-closed and degraded-input behaviour
# --------------------------------------------------------------------------


def test_missing_mandatory_block_fails_closed(payload):
    stripped = payload.replace('"itemKey":"kpy41_acc5_odenmis_sermaye"', '"itemKey":"x"')
    assert stripped != payload
    with pytest.raises(KAPStructureError):
        _parse_company_structure(
            stripped, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
        )


def test_missing_float_block_fails_closed(payload):
    stripped = payload.replace(
        '"itemKey":"kpy41_acc5_fiili_dolasimdaki_pay"', '"itemKey":"x"'
    )
    assert stripped != payload
    with pytest.raises(KAPStructureError):
        _parse_company_structure(
            stripped, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
        )


def test_unparseable_paid_in_capital_fails_closed(payload):
    stripped = payload.replace(
        '{"no":32,"itemName":"Ödenmiş/Çıkarılmış Sermaye",'
        '"itemKey":"kpy41_acc5_odenmis_sermaye","value":"393.516.000"',
        '{"no":32,"itemName":"Ödenmiş/Çıkarılmış Sermaye",'
        '"itemKey":"kpy41_acc5_odenmis_sermaye","value":"Bilgi Mevcut Değil"',
    )
    assert stripped != payload
    with pytest.raises(KAPStructureError):
        _parse_company_structure(
            stripped, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
        )


def test_missing_methodology_note_degrades_to_data_unavailable(payload):
    stripped = payload.replace("actualSharesOutstandingGrid", "renamedFloatGrid")
    assert stripped != payload
    result = _parse_company_structure(
        stripped, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
    )
    assert result["methodology"]["note"] == DATA_UNAVAILABLE
    assert result["methodology"]["effective_from"] is None
    assert result["float"]["actual_float_ratio_percent"] == Decimal("39.35")
    assert any(entry["field"] == "float_methodology" for entry in result["missing"])


def test_capital_ceiling_absent_degrades_to_data_unavailable(payload):
    stripped = payload.replace(
        '"itemKey":"kpy41_acc5_kayitli_sermaye_tavani","value":"750.000.000"',
        '"itemKey":"kpy41_acc5_kayitli_sermaye_tavani","value":"Bilgi Mevcut Değil"',
    )
    assert stripped != payload
    result = _parse_company_structure(
        stripped, symbol="PENTA", member_oid=PENTA_OID, source_url="u"
    )
    assert result["registered_capital_ceiling"] is None
    assert any(
        entry["field"] == "registered_capital_ceiling"
        for entry in result["missing"]
    )


# --------------------------------------------------------------------------
# Provider wiring (fetch, cache, error handling)
# --------------------------------------------------------------------------


def test_provider_single_fetch_and_cache(html):
    provider, client = _provider(html)

    first = provider.get_company_structure("PENTA")
    second = provider.get_company_structure("PENTA")

    assert client.calls == 1
    assert first is second
    assert client.urls == [
        f"https://kap.org.tr/tr/sirket-bilgileri/genel/{PENTA_OID}"
    ]
    assert first["paid_in_capital"] == Decimal("393516000")


def test_provider_shares_general_page_fetch_with_business_summary(html):
    provider, client = _provider(html)

    summary = provider._get_business_summary(PENTA_OID)  # noqa: SLF001
    structure = provider.get_company_structure("PENTA")

    assert client.calls == 1
    assert summary and summary.startswith("Bilgisayar, donanım ve yazılım")
    assert structure["float"]["actual_float_ratio_percent"] == Decimal("39.35")


def test_provider_normalizes_symbol_suffix(html):
    provider, client = _provider(html)

    result = provider.get_company_structure("penta.is")

    assert result["symbol"] == "PENTA"
    assert client.calls == 1


def test_provider_missing_oid_fails_closed():
    provider = KAPProvider()
    provider._client = _FakeClient(text="")
    provider.get_member_oid = lambda symbol: None  # noqa: ARG005

    with pytest.raises(KAPStructureError):
        provider.get_company_structure("PENTA")


def test_provider_fetch_failure_fails_closed():
    provider = KAPProvider()
    provider._client = _FakeClient(exc=RuntimeError("network down"))
    provider.get_member_oid = lambda symbol: PENTA_OID  # noqa: ARG005

    with pytest.raises(KAPStructureError):
        provider.get_company_structure("PENTA")


def test_provider_marked_up_page_fails_closed():
    provider, _client = _provider("<html></html>")

    with pytest.raises(KAPStructureError):
        provider.get_company_structure("PENTA")
