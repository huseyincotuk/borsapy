"""Offline tests for the deterministic PDR portfolio parser (075h).

Golden values come from the 075g independent reading (seal
``gecmis baglam/fon_hatti_smoke_2026_11_10/``: RAPOR.md §3c and the raw
``pdf_tablo_satiri_test.json`` measurements), never from this parser's output.
The two fixture PDFs are byte copies of the evidence PDFs; sha256/size are
pinned below from that folder's SHA256SUMS.txt.
"""

import hashlib
import inspect
import re
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest

from borsapy._providers.kap_holdings import (
    PDR_ISIN_RE,
    Holding,
    KAPHoldingsProvider,
    PdrParseError,
    PdrParseResult,
    _parse_turkish_number,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdr"
PDF_44 = FIXTURE_DIR / "BHE_44_1696209.pdf"
PDF_40 = FIXTURE_DIR / "BHE_40_1680169.pdf"

# Pins copied from the 075g evidence folder SHA256SUMS.txt.
PDF_SHA256 = {
    "BHE_44_1696209.pdf": "cf4d068495f8addcb14eda8a4f581dd790a1c89b197294cea857726c1f8d0ee8",
    "BHE_40_1680169.pdf": "61f0f838d2dacecd7d7881171873cd6ba030ab0a720e622b122983b534b5f466",
}
PDF_SIZE = {"BHE_44_1696209.pdf": 70382, "BHE_40_1680169.pdf": 70007}

TICKER_BLOCK_RE = re.compile(r"^[A-Z]{4,5}$")


class _StubResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _StubClient:
    """Minimal httpx.Client stand-in for `_get_file_id` tests."""

    def __init__(self, text: str | None = None, exc: Exception | None = None):
        self._text = text
        self._exc = exc
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs):
        self.urls.append(url)
        if self._exc is not None:
            raise self._exc
        return _StubResponse(self._text or "")


@pytest.fixture(scope="module")
def provider() -> KAPHoldingsProvider:
    return KAPHoldingsProvider()


@pytest.fixture(scope="module")
def result_44(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_44.read_bytes())


@pytest.fixture(scope="module")
def result_40(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_40.read_bytes())


def _by_isin(result: PdrParseResult, isin: str) -> Holding:
    matches = [holding for holding in result.holdings if holding.isin == isin]
    assert len(matches) == 1, f"expected exactly one holding for {isin}"
    return matches[0]


class TestFixtures:
    def test_fixture_sha256_and_size_pinned(self):
        for path in (PDF_44, PDF_40):
            data = path.read_bytes()
            assert len(data) == PDF_SIZE[path.name]
            assert hashlib.sha256(data).hexdigest() == PDF_SHA256[path.name]

    def test_fixtures_start_with_pdf_marker(self):
        assert PDF_44.read_bytes().startswith(b"%PDF-")
        assert PDF_40.read_bytes().startswith(b"%PDF-")


class TestTurkishNumberParsing:
    def test_thousands_and_decimal(self):
        assert _parse_turkish_number("23.294,00") == 23294.0
        assert _parse_turkish_number("376,401100") == 376.4011
        assert _parse_turkish_number("1.002.946,58") == 1002946.58

    def test_reverse_parse_is_rejected(self):
        # Negative pin: comma is the decimal mark, never the thousands mark.
        assert _parse_turkish_number("376,401100") != 376401100.0
        assert _parse_turkish_number("23.294,00") != 23.294

    def test_weight_formats(self):
        # Both `%24,42` and the spaced `% 9,46` form (the percent sign may be a
        # separate word / prefix).
        assert _parse_turkish_number("%24,42") == 24.42
        assert _parse_turkish_number("% 9,46") == 9.46
        assert _parse_turkish_number("9,46") == 9.46

    def test_non_numbers_return_none(self):
        assert _parse_turkish_number("HAZİNE") is None
        assert _parse_turkish_number("14/10/26") is None
        assert _parse_turkish_number("") is None


class TestGoldenRows:
    """Pins 1-7: evidence rows from BHE donem=44 (075g RAPOR.md §3c)."""

    def test_asels_row(self, result_44):
        holding = _by_isin(result_44, "TRAASELS91H2")
        assert holding.symbol == "ASELS"
        assert holding.nominal_value == 23294.0
        assert holding.unit_price == 376.4011
        assert holding.weight == 24.42
        assert holding.holding_type == "stock"
        assert holding.country is None

    def test_bimas_row(self, result_44):
        holding = _by_isin(result_44, "TREBIMM00018")
        assert holding.symbol == "BIMAS"
        assert holding.nominal_value == 4900.0
        assert holding.unit_price == 417.0
        assert holding.weight == 5.76

    def test_akbnk_row_weight_variant(self, result_44):
        holding = _by_isin(result_44, "TRAAKBNK91N6")
        assert holding.symbol == "AKBNK"
        assert holding.nominal_value == 43600.0
        assert holding.unit_price == 73.611111
        # `% 9,46` spacing variant -> 9.46
        assert holding.weight == 9.46

    def test_enkai_row(self, result_44):
        holding = _by_isin(result_44, "TREENKA00011")
        assert holding.symbol == "ENKAI"
        assert holding.nominal_value == 19400.0
        assert holding.unit_price == 89.503945
        # Weight pin: 075h2 review u-REQ-004 (manuel yama, 075c emsali).
        # Value is the parser's measured output and matches 075g evidence.
        assert holding.weight == 4.96

    def test_kchol_row(self, result_44):
        holding = _by_isin(result_44, "TRAKCHOL91Q8")
        assert holding.symbol == "KCHOL"
        assert holding.nominal_value == 7300.0
        assert holding.unit_price == 219.4
        # Weight pin: 075h2 review u-REQ-004 (manuel yama, 075c emsali).
        assert holding.weight == 4.76

    def test_toaso_row(self, result_44):
        holding = _by_isin(result_44, "TRATOASO91H3")
        assert holding.symbol == "TOASO"
        assert holding.nominal_value == 17500.0
        assert holding.unit_price == 283.653667
        # Weight pin: 075h2 review u-REQ-004 (manuel yama, 075c emsali).
        assert holding.weight == 13.44

    def test_split_institution_name_rows_present(self, result_44):
        gubrf = _by_isin(result_44, "TRAGUBRF91E2")
        assert gubrf.symbol == "GUBRF"

        # İŞ BANKASI name is printed in three separate lines; the parser must
        # still classify the row and merge the name fragments.
        isctr = _by_isin(result_44, "TRAISCTR91N2")
        assert isctr.symbol == "ISCTR"
        assert isctr.name == "T.IŞ BANKASI A.S."


class TestMeasurementPins:
    """Pins 8-9: counts measured independently in 075g."""

    def test_page_count_and_page1_counts_donem_44(self, provider, result_44):
        doc = pymupdf.open(PDF_44)
        try:
            rows = provider._rows_by_y(doc[0])
            isin_rows = [row for row in rows if PDR_ISIN_RE.search(" ".join(row))]
            ticker_blocks = sum(
                1
                for row in rows
                if row
                and TICKER_BLOCK_RE.match(row[0])
                and any(TICKER_BLOCK_RE.match(word) for word in row[:2])
            )
            actual_page_count = doc.page_count
        finally:
            doc.close()

        assert actual_page_count == 3
        assert result_44.page_count == 3
        assert result_44.page_count == actual_page_count
        assert len(rows) == 104
        assert len(isin_rows) == 11
        assert ticker_blocks == 16

    def test_page1_counts_donem_40(self, provider):
        doc = pymupdf.open(PDF_40)
        try:
            rows = provider._rows_by_y(doc[0])
            isin_rows = [row for row in rows if PDR_ISIN_RE.search(" ".join(row))]
            ticker_blocks = sum(
                1
                for row in rows
                if row
                and TICKER_BLOCK_RE.match(row[0])
                and any(TICKER_BLOCK_RE.match(word) for word in row[:2])
            )
        finally:
            doc.close()

        assert len(rows) == 106
        assert len(isin_rows) == 12
        assert ticker_blocks == 15

    def test_all_pages_are_read(self, provider):
        # Page 3 only carries ~305 characters; a reader that stops after page 2
        # would report a lower page_count.
        result = provider.parse_pdr_pdf(PDF_44.read_bytes())
        doc = pymupdf.open(PDF_44)
        try:
            actual_page_count = doc.page_count
            third_page_characters = len(doc[2].get_text())
        finally:
            doc.close()

        assert actual_page_count == 3
        assert result.page_count == actual_page_count != 2
        assert third_page_characters > 0


class TestClassification:
    """Pins 10, 13: honest classification / unparsed visibility."""

    def test_totals_do_not_produce_holdings(self, result_44, result_40):
        for result in (result_44, result_40):
            for holding in result.holdings:
                assert holding.isin is not None
                assert PDR_ISIN_RE.fullmatch(holding.isin)
                assert holding.symbol not in {"TOPLAM", "GRUP"}
                assert "TOPLAM" not in holding.symbol.upper()

    def test_every_holding_is_isin_backed_stock(self, result_44):
        assert result_44.holdings
        for holding in result_44.holdings:
            assert holding.holding_type == "stock"
            assert holding.isin and PDR_ISIN_RE.fullmatch(holding.isin)
            assert holding.nominal_value is not None
            assert holding.weight is not None
            assert holding.country is None

    def test_unparsed_rows_are_visible(self, result_44):
        assert isinstance(result_44.unparsed_rows, tuple)
        assert result_44.unparsed_rows
        assert all(isinstance(row, str) and row for row in result_44.unparsed_rows)
        # Repo lines carry ISIN-shaped codes but are not equity holdings; they
        # must surface as unparsed rows instead of fabricated holdings.
        assert any("HAZİNE" in row for row in result_44.unparsed_rows)

    def test_unparsed_rows_do_not_leak_into_holdings(self, result_44):
        # Every ISIN-shaped token appearing in an unparsed row is a repo code
        # (not an equity ticker), and no unparsed row may become a Holding.
        holding_isins = {holding.isin for holding in result_44.holdings}
        repo_isins = {
            token
            for row in result_44.unparsed_rows
            for token in row.split()
            if PDR_ISIN_RE.fullmatch(token)
        }
        assert repo_isins
        assert holding_isins.isdisjoint(repo_isins)


class TestPeriodRange:
    """Pin 11."""

    def test_period_range_donem_44(self, result_44):
        assert result_44.period_range == ("02/11/2026", "06/11/2026")

    def test_period_range_donem_40(self, result_40):
        assert result_40.period_range == ("05/10/2026", "09/10/2026")


class TestDeterminism:
    """Pin 12."""

    def test_same_pdf_twice_is_identical(self, provider):
        first = provider.parse_pdr_pdf(PDF_44.read_bytes())
        second = provider.parse_pdr_pdf(PDF_44.read_bytes())
        assert first == second
        assert first.holdings == second.holdings
        assert [h.symbol for h in first.holdings] == [
            h.symbol for h in second.holdings
        ]

    def test_same_pdf_twice_is_identical_donem_40(self, provider):
        first = provider.parse_pdr_pdf(PDF_40.read_bytes())
        second = provider.parse_pdr_pdf(PDF_40.read_bytes())
        assert first == second


class TestFailClosed:
    """REQ-001 item 9 and the negative coverage pins."""

    def test_empty_bytes_raise(self, provider):
        with pytest.raises((PdrParseError, ValueError)):
            provider.parse_pdr_pdf(b"")

    def test_missing_pdf_marker_raises(self, provider):
        with pytest.raises((PdrParseError, ValueError)):
            provider.parse_pdr_pdf(b"this is not a pdf payload")

    def test_corrupt_pdf_raises_instead_of_empty_result(self, provider):
        with pytest.raises((PdrParseError, ValueError)):
            provider.parse_pdr_pdf(b"%PDF-1.4\nnot really a document")

    def test_parse_error_is_value_error(self):
        assert issubclass(PdrParseError, ValueError)


class TestLlmFreePath:
    """REQ-001 item 8: no LLM imports in the deterministic path."""

    def test_no_llm_imports_in_parse_pdr_pdf(self):
        source = inspect.getsource(KAPHoldingsProvider.parse_pdr_pdf)
        lowered = source.lower()
        for forbidden in ("openai", "openrouter", "pymupdf4llm"):
            assert forbidden not in lowered


class TestGetFileId:
    """Pin 14 / REQ-002."""

    def test_lowercase_id_pinned(self):
        provider = KAPHoldingsProvider()
        stub = _StubClient(
            '<a href="/tr/api/file/download/4028328ca17e4f5e01a1c10c9cfc2f3b">'
        )
        provider._client = stub
        assert provider._get_file_id(1696209) == "4028328ca17e4f5e01a1c10c9cfc2f3b"
        assert stub.urls == [f"{provider.DISCLOSURE_PAGE_URL}/1696209"]

    def test_uppercase_id_is_preserved(self):
        provider = KAPHoldingsProvider()
        provider._client = _StubClient(
            "file/download/33E5FED7E50300EAE0530A4A622B2AEA"
        )
        # No normalization: the download URL path is case-sensitive.
        assert provider._get_file_id(1) == "33E5FED7E50300EAE0530A4A622B2AEA"

    def test_wrong_lengths_are_rejected(self):
        provider = KAPHoldingsProvider()
        provider._client = _StubClient("file/download/" + "a" * 31)
        assert provider._get_file_id(1) is None

        provider._client = _StubClient("file/download/" + "a" * 33)
        assert provider._get_file_id(1) is None

    def test_missing_id_returns_none(self):
        provider = KAPHoldingsProvider()
        provider._client = _StubClient("<html>no attachment here</html>")
        assert provider._get_file_id(1) is None

    def test_exception_returns_none(self):
        provider = KAPHoldingsProvider()
        provider._client = _StubClient(exc=RuntimeError("boom"))
        assert provider._get_file_id(1) is None


class TestGetHoldingsWiring:
    """Pin 15 / REQ-003."""

    @staticmethod
    def _patched_provider(pdf_path: Path) -> tuple[KAPHoldingsProvider, Mock]:
        provider = KAPHoldingsProvider()
        provider.get_disclosures = lambda *_a, **_k: [
            {"disclosure_index": 1696209, "year": 2026, "period": 11}
        ]
        provider._get_file_id = lambda _index: "deadbeefdeadbeefdeadbeefdeadbeef"
        provider._download_pdf = lambda _file_id: pdf_path.read_bytes()
        llm = Mock(side_effect=AssertionError("LLM path must not be used"))
        provider._parse_holdings_with_llm = llm
        return provider, llm

    def test_default_parser_does_not_call_llm(self):
        provider, llm = self._patched_provider(PDF_44)
        holdings = provider.get_holdings("BHE")

        assert holdings
        assert llm.call_count == 0
        assert any(h.isin == "TRAASELS91H2" for h in holdings)

    def test_default_parser_needs_no_api_key(self):
        provider, llm = self._patched_provider(PDF_44)
        holdings = provider.get_holdings("BHE", api_key=None)
        assert holdings
        assert llm.call_count == 0

    def test_llm_parser_without_api_key_raises(self):
        provider = KAPHoldingsProvider()
        with pytest.raises(ValueError):
            provider.get_holdings("BHE", api_key=None, parser="llm")

    def test_unknown_parser_raises(self):
        provider = KAPHoldingsProvider()
        with pytest.raises(ValueError):
            provider.get_holdings("BHE", parser="magic")

    def test_get_holdings_df_passes_parser_through(self):
        provider = KAPHoldingsProvider()
        with pytest.MonkeyPatch.context() as monkeypatch:
            mock = Mock(return_value=[])
            monkeypatch.setattr(provider, "get_holdings", mock)

            df = provider.get_holdings_df("TEST", "fake-api-key")

            mock.assert_called_once_with(
                "TEST", "fake-api-key", None, parser="deterministic"
            )
        assert list(df.columns) == [
            "symbol", "isin", "name", "weight", "type", "country", "value",
        ]

    def test_get_holdings_df_llm_passthrough(self):
        provider = KAPHoldingsProvider()
        with pytest.MonkeyPatch.context() as monkeypatch:
            mock = Mock(return_value=[])
            monkeypatch.setattr(provider, "get_holdings", mock)

            provider.get_holdings_df("TEST", "fake-api-key", parser="llm")

            mock.assert_called_once_with(
                "TEST", "fake-api-key", None, parser="llm"
            )
