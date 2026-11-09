"""Offline pin tests for the VBTS status module and the disclosure-class column.

W1.75 Parsel C (borsapy). Fixtures under ``tests/fixtures/kap/`` are the sealed
2026-11-09 captures from ``vbts_dkb_arastirma_2026_11_09/kanit/`` and must stay
byte-identical. All tests serve those pages through a fake transport, so no real
network access happens.

Golden cases come from the sealed research evidence (``RAPOR.md`` §2.1/§2.2):
231 VBTS records / 192 symbols / 47 symbols inside the 90-day window.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from borsapy._providers.kap import (
    BIST_MEMBER_OID,
    DEFAULT_VBTS_WINDOW_DAYS,
    VBTS_SUMMARY,
    VBTS_TITLE,
    KAPProvider,
    _escaped_string_field,
    _parse_disclosure_records,
    _parse_kap_publish_datetime,
    _split_escaped_records,
)
from borsapy.exceptions import KAPStructureError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kap"
BIST_FIXTURE = FIXTURE_DIR / "w175c_bist_member_disclosures.html"
PENTA_FIXTURE = FIXTURE_DIR / "w175c_penta_disclosures.html"

BIST_SHA256 = "5584707eefce9c301dbc60ee876a6a64372c45053f6924e4601c8ea07eaae41c"
BIST_SIZE = 5994648
PENTA_SHA256 = "4637d8f4ae3594f13a58fbdf9efaccd28fdd77115e3afa31ef8b3a169a894917"
PENTA_SIZE = 238240

#: Reference date of the sealed capture (window pins are relative to it).
AS_OF = "2026-11-09"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bist_html() -> str:
    return BIST_FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def penta_html() -> str:
    return PENTA_FIXTURE.read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, text: str = "", status_code: int = 200, exc: Exception | None = None):
        self._text = text
        self._status = status_code
        self._exc = exc
        self.calls = 0
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.calls += 1
        self.urls.append(url)
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._text, self._status)


def _vbts_provider(html_text: str, **kwargs) -> tuple[KAPProvider, _FakeClient]:
    provider = KAPProvider()
    client = _FakeClient(text=html_text, **kwargs)
    provider._client = client
    return provider, client


def _disclosure_provider(html_text: str) -> tuple[KAPProvider, _FakeClient]:
    provider = KAPProvider()
    client = _FakeClient(text=html_text)
    provider._client = client
    provider.get_member_oid = lambda symbol: "oid-" + symbol  # noqa: ARG005
    return provider, client


# --------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------


def test_fixture_checksums():
    assert BIST_FIXTURE.stat().st_size == BIST_SIZE
    assert _sha256(BIST_FIXTURE) == BIST_SHA256
    assert PENTA_FIXTURE.stat().st_size == PENTA_SIZE
    assert _sha256(PENTA_FIXTURE) == PENTA_SHA256


# --------------------------------------------------------------------------
# Parsing primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("09.11.2026 18:28:45", ("2026-11-09", "09.11.2026 18:28:45")),
        ("01.03.2026", ("2026-03-01", "01.03.2026")),
        ("31.12.2025 18:33:13", ("2025-12-31", "31.12.2025 18:33:13")),
    ],
)
def test_parse_publish_datetime_valid(raw, expected):
    assert _parse_kap_publish_datetime(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "2026-11-09", "9.11.2026", "abc"])
def test_parse_publish_datetime_invalid(raw):
    iso, raw_text = _parse_kap_publish_datetime(raw)
    assert iso is None


def test_split_records_isolates_each_record():
    marker = 'publishDate\\":\\"'
    text = (
        '{"' + marker + '01.01.2026 10:00:00\\",\\"disclosureClass\\":\\"FR\\"}'
        '{"' + marker + '02.01.2026 10:00:00\\"}'
    )
    chunks = _split_escaped_records(text)
    assert len(chunks) == 2
    # A record without the class must NOT borrow the previous record's class.
    assert _escaped_string_field(chunks[1], "disclosureClass") is None
    assert _escaped_string_field(chunks[0], "disclosureClass") == "FR"


# --------------------------------------------------------------------------
# Parça 1 — disclosureClass column (additive surface)
# --------------------------------------------------------------------------


def test_disclosure_columns_are_additive_and_ordered(penta_html):
    provider, _client = _disclosure_provider(penta_html)
    frame = provider.get_disclosures("PENTA", limit=200)

    assert list(frame.columns) == ["Date", "Title", "URL", "DisclosureClass"]
    assert len(frame) == 54
    assert frame["DisclosureClass"].tolist().count("DKB") == 14
    assert set(frame["DisclosureClass"]) == {"ODA", "DKB", "FR", "DG"}


def test_penta_class_golden_counts(penta_html):
    provider, _client = _disclosure_provider(penta_html)
    frame = provider.get_disclosures("PENTA", limit=500)

    counts = frame["DisclosureClass"].value_counts().to_dict()
    # Golden: PENTA ODA 20 / DKB 14 / FR 13 / DG 7 (capture 2026-11-09 19:46).
    assert counts == {"ODA": 20, "DKB": 14, "FR": 13, "DG": 7}


def test_existing_columns_keep_their_contract(penta_html):
    provider, _client = _disclosure_provider(penta_html)
    frame = provider.get_disclosures("PENTA", limit=3)

    assert len(frame) == 3
    first = frame.iloc[0]
    assert first["Date"] == "03.11.2026 19:18:45"
    assert first["Title"] == "Sorumluluk Beyanı (Konsolide)"
    assert first["URL"] == "https://www.kap.org.tr/tr/Bildirim/1691930"


def test_class_filter_is_client_side_and_needs_no_extra_request(penta_html):
    provider, client = _disclosure_provider(penta_html)

    filtered = provider.get_disclosures("PENTA", limit=500, disclosure_class="DKB")
    assert len(filtered) == 14
    assert set(filtered["DisclosureClass"]) == {"DKB"}
    assert client.calls == 1  # one fetch, filtered in memory

    # Filter is case-insensitive and normalized.
    lower = provider.get_disclosures("PENTA", limit=500, disclosure_class="fr")
    assert len(lower) == 13
    assert client.calls == 2  # the filter itself never issues a request


def test_class_filter_respects_limit_after_filtering(penta_html):
    provider, _client = _disclosure_provider(penta_html)

    frame = provider.get_disclosures("PENTA", limit=3, disclosure_class="ODA")
    assert len(frame) == 3
    assert set(frame["DisclosureClass"]) == {"ODA"}


def test_class_filter_unknown_value_returns_empty_frame(penta_html):
    provider, _client = _disclosure_provider(penta_html)

    frame = provider.get_disclosures("PENTA", disclosure_class="ZZZ")
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert list(frame.columns) == ["Date", "Title", "URL", "DisclosureClass"]


def test_unknown_symbol_keeps_new_column_shape():
    provider = KAPProvider()
    provider._client = _FakeClient()
    provider.get_member_oid = lambda symbol: None  # noqa: ARG005

    frame = provider.get_disclosures("NOPE")
    assert frame.empty
    assert list(frame.columns) == ["Date", "Title", "URL", "DisclosureClass"]


def test_missing_class_is_none_not_guessed():
    """A record without disclosureClass yields None and is reported in attrs."""
    html = (
        'lead{\\"data\\":['
        '{\\"disclosureBasic\\":{\\"publishDate\\":\\"01.10.2026 10:00:00\\",'
        '\\"disclosureIndex\\":111,\\"title\\":\\"A Bildirimi\\",'
        '\\"relatedStocks\\":null,\\"disclosureClass\\":\\"FR\\"}},'
        '{\\"disclosureBasic\\":{\\"publishDate\\":\\"30.09.2026 10:00:00\\",'
        '\\"disclosureIndex\\":112,\\"title\\":\\"B Bildirimi\\"}}'
        "]}"
    )
    provider, _client = _disclosure_provider(html)
    frame = provider.get_disclosures("SYN", limit=10)

    assert len(frame) == 2
    assert frame["DisclosureClass"].tolist() == ["FR", None]
    # The class-less record is not an orphan (it has date/index/title).
    assert frame.attrs["unparsed_class_records"] == []


def test_unparseable_records_are_reported():
    """A record missing publishDate cannot be placed in time and is reported."""
    html = (
        'lead{\\"data\\":['
        '{\\"disclosureBasic\\":{\\"disclosureIndex\\":113,\\"title\\":\\"Tarihsiz\\",'
        '\\"disclosureClass\\":\\"FR\\"}}'
        '{"publishDate\\":\\"01.10.2026 10:00:00\\",'
        '\\"disclosureIndex\\":114,\\"title\\":\\"C Bildirimi\\",'
        '\\"disclosureClass\\":\\"DG\\"}}'
        "]}"
    )
    provider, _client = _disclosure_provider(html)
    frame = provider.get_disclosures("SYN", limit=10)

    assert "C Bildirimi" in frame["Title"].tolist()
    assert "Tarihsiz" not in frame["Title"].tolist()
    assert len(frame.attrs["unparsed_class_records"]) == 1


# --------------------------------------------------------------------------
# Parça 2 — VBTS status (whole market, one request)
# --------------------------------------------------------------------------


def test_vbts_counts_golden(bist_html):
    provider, client = _vbts_provider(bist_html)
    result = provider.get_vbts_status(as_of=AS_OF)

    counts = result["counts"]
    assert counts["records_total"] == 2000
    assert counts["dkb_total"] == 2000  # every record on the BIST member page is DKB
    assert counts["vbts_matched"] == 231
    assert counts["symbols_total"] == 192
    assert counts["symbols_in_window"] == 47
    assert counts["unparseable_records"] == 0
    assert client.calls == 1  # one request for the whole market


def test_vbts_uses_member_oid_and_reports_provenance(bist_html):
    provider, client = _vbts_provider(bist_html)
    result = provider.get_vbts_status(as_of=AS_OF)

    assert client.urls == [
        f"https://www.kap.org.tr/tr/bildirim-sorgu-sonuc?member={BIST_MEMBER_OID}"
    ]
    source = result["source"]
    assert source["member_oid"] == BIST_MEMBER_OID
    assert "veri-türetimi" in source["member_oid_provenance"]
    assert source["sha256"] == BIST_SHA256
    assert source["http_status"] == 200
    assert source["bytes"] == BIST_SIZE


def test_vbts_symbol_golden_cases(bist_html):
    provider, _client = _vbts_provider(bist_html)
    symbols = provider.get_vbts_status(as_of=AS_OF)["symbols"]

    kartn = symbols["KARTN"]
    assert kartn["vbts_var"] is True
    assert kartn["kayit_sayisi"] == 3
    assert kartn["son_tarih"] == "2026-11-09"
    assert kartn["within_window"] is True
    assert kartn["disclosure_index"] == [1695497, 1641564, 1597764]

    tera = symbols["TERA"]
    assert tera["kayit_sayisi"] == 3
    assert tera["son_tarih"] == "2026-06-15"
    assert tera["within_window"] is False  # outside the 90-day window

    assert symbols["OZATD"]["son_tarih"] == "2026-09-02"
    assert symbols["OZATD"]["within_window"] is True

    for clean in ("PENTA", "THYAO", "GARAN", "ASELS", "TUPRS", "TUKAS"):
        assert clean not in symbols


def test_vbts_multi_symbol_split(bist_html):
    provider, _client = _vbts_provider(bist_html)
    symbols = provider.get_vbts_status(as_of=AS_OF)["symbols"]

    ahsgy = symbols["AHSGY"]
    kzgyo = symbols["KZGYO"]
    shared = 1663194  # "AHSGY, KZGYO" record
    assert shared in ahsgy["disclosure_index"]
    assert kzgyo["disclosure_index"] == [shared]
    assert ahsgy["ham_related"] == "AHSGY, KZGYO"
    assert kzgyo["ham_related"] == "AHSGY, KZGYO"


def test_spk_measure_is_not_vbts(bist_html, penta_html):
    """IEYHO's 'Sermaye Piyasası Kurulu Tedbir Kararı' is SPK, not VBTS."""
    provider, _client = _vbts_provider(bist_html)
    result = provider.get_vbts_status(as_of=AS_OF)

    assert "IEYHO" not in result["symbols"]

    # The same page carries 2,000 DKB notices of which only 231 are VBTS.
    assert result["counts"]["dkb_total"] - result["counts"]["vbts_matched"] == 1769


def test_vbts_window_parameter_changes_result(bist_html):
    provider, _client = _vbts_provider(bist_html)

    narrow = provider.get_vbts_status(days=7, as_of=AS_OF)
    wide = provider.get_vbts_status(days=365, as_of=AS_OF)

    assert narrow["window"]["days"] == 7
    assert narrow["window"]["from"] == "2026-11-02"
    assert narrow["window"]["provenance"] == "PARAMETRE+VARSAYILAN (K0-a c)"

    assert narrow["counts"]["symbols_in_window"] < wide["counts"]["symbols_in_window"]
    assert wide["counts"]["symbols_in_window"] == wide["counts"]["symbols_total"]
    assert wide["counts"]["symbols_total"] == 192


def test_vbts_default_window_is_90_days(bist_html):
    provider, _client = _vbts_provider(bist_html)
    result = provider.get_vbts_status(as_of=AS_OF)

    assert DEFAULT_VBTS_WINDOW_DAYS == 90
    assert result["window"]["days"] == 90
    assert result["window"]["from"] == "2026-08-11"
    assert result["window"]["to"] == AS_OF


def test_vbts_measure_type_is_never_derived(bist_html):
    provider, _client = _vbts_provider(bist_html)
    result = provider.get_vbts_status(as_of=AS_OF)

    assert result["tedbir_turu"].startswith("VERİ YOK")
    for entry in result["symbols"].values():
        assert set(entry) == {
            "vbts_var",
            "kayit_sayisi",
            "son_tarih",
            "disclosure_index",
            "within_window",
            "ham_related",
        }


def test_vbts_cache_prevents_refetch(bist_html):
    provider, client = _vbts_provider(bist_html)

    first = provider.get_vbts_status(days=90, as_of=AS_OF)
    second = provider.get_vbts_status(days=30, as_of=AS_OF)

    assert client.calls == 1  # page parsed once, window re-applied per call
    assert first["window"]["from"] != second["window"]["from"]
    assert first["counts"]["symbols_total"] == second["counts"]["symbols_total"]


@pytest.mark.parametrize("bad_days", [0, -5, None])
def test_vbts_invalid_days_rejected(bist_html, bad_days):
    provider, _client = _vbts_provider(bist_html)
    with pytest.raises(ValueError):
        provider.get_vbts_status(days=bad_days, as_of=AS_OF)


def test_vbts_invalid_as_of_rejected(bist_html):
    provider, _client = _vbts_provider(bist_html)
    with pytest.raises(ValueError):
        provider.get_vbts_status(as_of="09.11.2026")


# --- fail-closed ----------------------------------------------------------


def test_vbts_http_failure_fails_closed(bist_html):
    provider, _client = _vbts_provider(bist_html, status_code=503)
    with pytest.raises(KAPStructureError):
        provider.get_vbts_status(as_of=AS_OF)


def test_vbts_transport_failure_fails_closed():
    provider = KAPProvider()
    provider._client = _FakeClient(exc=RuntimeError("network down"))
    with pytest.raises(KAPStructureError):
        provider.get_vbts_status(as_of=AS_OF)


def test_vbts_empty_body_is_data_unavailable_not_clear():
    """Throttle lesson: an empty 200 body must NOT read as 'no VBTS measures'."""
    provider, _client = _vbts_provider("")
    with pytest.raises(KAPStructureError):
        provider.get_vbts_status(as_of=AS_OF)


def test_vbts_garbage_page_fails_closed():
    provider, _client = _vbts_provider("<html><body>maintenance</body></html>")
    with pytest.raises(KAPStructureError):
        provider.get_vbts_status(as_of=AS_OF)


# --------------------------------------------------------------------------
# Parser-level guard: the three-field match is what makes 231 out of 2,000
# --------------------------------------------------------------------------


def test_three_field_match_is_required(bist_html):
    records = _parse_disclosure_records(bist_html)
    three_field = [
        item
        for item in records
        if item["disclosure_class"] == "DKB"
        and item["title"] == VBTS_TITLE
        and item["summary"] == VBTS_SUMMARY
    ]
    title_only = [item for item in records if item["title"] == VBTS_TITLE]

    assert len(three_field) == 231
    # Title alone over-counts ~3x: 696 notices share the BISTECH title.
    assert len(title_only) == 696
