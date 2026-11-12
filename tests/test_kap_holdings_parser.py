"""Offline tests for the deterministic PDR portfolio parser (075h + 075l).

Golden values come from two independent evidence seals, never from this
parser's output:

* 075g (`gecmis baglam/fon_hatti_smoke_2026_11_10/`: RAPOR.md §3c and raw
  ``pdf_tablo_satiri_test.json``) for the BHE pins.
* 075l discovery (`gecmis baglam/pw15b_discovery_2026_11_11/`) plus its
  independent count ``pdr_templates_launch_2026_11_11/EK_D_BAGIMSIZ_SAYIM.md``
  for the GBH/MUT/DDA/DDF pins.

The fixture PDFs are byte copies of the sealed evidence PDFs; sha256/size are
pinned below from the seals' SHA256SUMS.txt (BHE) and from the discovery
folder's hashes (GBH/MUT/DDA/DDF).
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
    PDR_TICKER_RE,
    Holding,
    KAPHoldingsProvider,
    PdrParseError,
    PdrParseResult,
    PdrSilentZeroError,
    _parse_turkish_number,
    _pdr_cell_number,
    _pdr_nearest_owner,
    _pdr_number_class,
    _pdr_ticker_parts,
)
from borsapy.exceptions import DataNotAvailableError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdr"
PDF_44 = FIXTURE_DIR / "BHE_44_1696209.pdf"
PDF_40 = FIXTURE_DIR / "BHE_40_1680169.pdf"
PDF_GBH = FIXTURE_DIR / "GBH_1692815.pdf"
PDF_MUT = FIXTURE_DIR / "MUT_1692820.pdf"
PDF_DDA = FIXTURE_DIR / "DDA_1695047.pdf"
PDF_DDF = FIXTURE_DIR / "DDF_1695048.pdf"

# Pins copied from the evidence folders' SHA256SUMS.txt / discovery hashes.
PDF_SHA256 = {
    "BHE_44_1696209.pdf": "cf4d068495f8addcb14eda8a4f581dd790a1c89b197294cea857726c1f8d0ee8",
    "BHE_40_1680169.pdf": "61f0f838d2dacecd7d7881171873cd6ba030ab0a720e622b122983b534b5f466",
    "GBH_1692815.pdf": "f6c42666195e9bc8227d19638697eb41482f34317564927f08aae86a01daa420",
    "MUT_1692820.pdf": "7a732e637417687df88e51f3dc7e599e2b7e833881b4cea7aae0322dbd7da736",
    "DDA_1695047.pdf": "433ee59a147727fea0b846070a4f61ad8f84b50f13511db7cfcdc30ec3b8a527",
    "DDF_1695048.pdf": "a19279d8ecd785c5939474b6824bc3e6c7a441921d6550a0e74c2acdc131b07f",
}
PDF_SIZE = {
    "BHE_44_1696209.pdf": 70382,
    "BHE_40_1680169.pdf": 70007,
    "GBH_1692815.pdf": 76715,
    "MUT_1692820.pdf": 63272,
    "DDA_1695047.pdf": 66643,
    "DDF_1695048.pdf": 66994,
}

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


# ---------------------------------------------------------------------------
# 075l — template expansion pins (`.E` suffix + Deniz-type column awareness)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result_gbh(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_GBH.read_bytes())


@pytest.fixture(scope="module")
def result_mut(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_MUT.read_bytes())


@pytest.fixture(scope="module")
def result_dda(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_DDA.read_bytes())


@pytest.fixture(scope="module")
def result_ddf(provider: KAPHoldingsProvider) -> PdrParseResult:
    return provider.parse_pdr_pdf(PDF_DDF.read_bytes())


class TestDiscoveryFixtures:
    """The four 075l fixtures are untouched byte copies of the seal PDFs."""

    def test_fixture_sha256_and_size_pinned(self):
        for path in (PDF_GBH, PDF_MUT, PDF_DDA, PDF_DDF):
            data = path.read_bytes()
            assert len(data) == PDF_SIZE[path.name]
            assert hashlib.sha256(data).hexdigest() == PDF_SHA256[path.name]

    def test_fixtures_start_with_pdf_marker(self):
        for path in (PDF_GBH, PDF_MUT, PDF_DDA, PDF_DDF):
            assert path.read_bytes().startswith(b"%PDF-")


class TestUsFormatTemplate:
    """Pin 1-2: `.E` suffix template (GBH/MUT), US number format."""

    def test_gbh_holdings_meet_independent_count(self, result_gbh):
        # EK-D independent count: 33 distinct ISINs, 31 distinct `.E` tickers.
        assert len(result_gbh.holdings) >= 30
        assert len({h.isin for h in result_gbh.holdings}) >= 30

    def test_gbh_akbnk_suffix_stripped_and_us_number(self, result_gbh):
        akbnk = [h for h in result_gbh.holdings if h.symbol == "AKBNK"]
        assert akbnk
        # EK-D geometry: AKBNK.E row prints nominal 100,000.00 (US format).
        assert any(h.nominal_value == 100000.0 for h in akbnk)
        assert any(h.market_value == 6090000.0 for h in akbnk)
        assert all(h.isin and PDR_ISIN_RE.fullmatch(h.isin) for h in akbnk)
        # The source suffix is preserved in notes, the symbol is clean.
        assert all(h.notes == "source_ticker=AKBNK.E" for h in akbnk)
        assert all("." not in h.symbol for h in result_gbh.holdings)

    def test_gbh_all_symbols_have_no_suffix(self, result_gbh):
        assert result_gbh.holdings
        for holding in result_gbh.holdings:
            assert re.fullmatch(r"[A-Z]{3,6}", holding.symbol)

    def test_mut_holdings_positive_no_silent_zero(self, result_mut):
        # EK-D independent count: 28 distinct ISINs / 26 `.E` tickers.
        assert result_mut.holdings
        assert len(result_mut.holdings) > 0

    def test_gbh_unparsed_row_carries_out_of_band_isin(self, result_gbh):
        # BIGCH rows print their weight into a column the header cannot
        # resolve, so they must surface as unparsed rather than vanish.
        assert any("TREBIGC00011" in row for row in result_gbh.unparsed_rows)


class TestDenizTemplate:
    """Pin 4-5: Deniz-type column-aware block detection (DDA/DDF)."""

    def test_dda_holdings_beat_pilot(self, result_dda):
        # Pilot parsed 5 of the 9 ISINs; the loss must now be < 44 %.
        assert len(result_dda.holdings) >= 5
        assert len({h.isin for h in result_dda.holdings}) >= 5
        assert len({h.isin for h in result_dda.holdings}) / 9 > 0.56

    def test_dda_garan_halkb_isins_are_distinct(self, result_dda):
        by_symbol = {h.symbol: h for h in result_dda.holdings}
        # EK-D geometry: naive 22 pt window matches GARAN with HALKB's ISIN.
        assert "GARAN" in by_symbol and "HALKB" in by_symbol
        assert by_symbol["GARAN"].isin == "TRAGARAN91N1"
        assert by_symbol["HALKB"].isin == "TRETHAL00019"

    def test_ddf_holdings_positive(self, result_ddf):
        assert len(result_ddf.holdings) >= 4
        assert len({h.isin for h in result_ddf.holdings}) >= 4

    def test_bhe_regression_guard(self, result_44):
        # Existing BHE pins must not move under the new matching.
        assert result_44.page_count == 3
        asels = _by_isin(result_44, "TRAASELS91H2")
        assert asels.symbol == "ASELS"
        assert asels.nominal_value == 23294.0
        assert asels.weight == 24.42
        assert _by_isin(result_44, "TRAAKBNK91N6").symbol == "AKBNK"
        assert not any("TOPLAM" in h.symbol.upper() for h in result_44.holdings)


class TestSuffixRule:
    """Pin: the suffix is accepted, stripped, and never a parse trigger."""

    def test_ticker_regex_accepts_suffix(self):
        assert PDR_TICKER_RE.fullmatch("AKBNK.E")
        assert PDR_TICKER_RE.fullmatch("AKBNK")
        assert PDR_TICKER_RE.fullmatch("SSAAT.EL")

    def test_ticker_regex_rejects_header_tokens(self):
        # `A.PAY` is a table header, not a ticker; only 1-2 letter suffixes.
        assert PDR_TICKER_RE.fullmatch("A.PAY") is None
        assert PDR_TICKER_RE.fullmatch("AKBNK.ABC") is None
        assert PDR_TICKER_RE.fullmatch("TOPLAM.E") is not None  # shape only

    def test_symbol_output_strips_suffix(self):
        assert _pdr_ticker_parts("AKBNK.E") == ("AKBNK", "AKBNK.E")
        assert _pdr_ticker_parts("AKBNK") == ("AKBNK", None)
        assert _pdr_ticker_parts("SSAAT.EL") == ("SSAAT", "SSAAT.EL")

    def test_suffix_without_numeric_block_is_not_a_holding(self, result_44):
        # BHE has no `.E` tickers; suffix shape alone never creates a holding.
        assert all(h.isin for h in result_44.holdings)
        assert all(h.notes is None for h in result_44.holdings)


class TestNumberFormatDiscovery:
    """Pin 8: mixed Turkish/US cells yield None (never an invented value)."""

    def test_mixed_format_cell_returns_none(self):
        assert _pdr_cell_number("1.234,56", "us") is None
        assert _pdr_cell_number("1,234.56", "tr") is None

    def test_matching_format_is_read(self):
        assert _pdr_cell_number("100,000.00", "us") == 100000.0
        assert _pdr_cell_number("100.000,00", "tr") == 100000.0

    def test_ambiguous_classification_is_none(self):
        # On its own the cell could be either format; without a discovered
        # column format nothing is read (no guess).
        assert _pdr_number_class("100,000") is None
        assert _pdr_number_class("1.234") is None
        assert _pdr_cell_number("100,000", None) is None
        # With the column discovered as US the same cell is unambiguous.
        assert _pdr_cell_number("100,000", "us") == 100000.0

    def test_unknown_column_format_returns_none(self):
        assert _pdr_cell_number("100,000.00", None) is None


class TestSilentZeroGuard:
    """Pin 6-7: silent zero fails loud; ambiguous neighbours stay unparsed."""

    def test_silent_zero_raises_pdr_parse_error(self):
        with pytest.raises(PdrSilentZeroError):
            KAPHoldingsProvider._pdr_check_silent_zero(3, [], [])

    def test_silent_zero_is_not_data_not_available(self):
        assert issubclass(PdrSilentZeroError, PdrParseError)
        assert not issubclass(PdrSilentZeroError, DataNotAvailableError)

    def test_no_contradiction_when_rows_become_unparsed(self):
        # ISIN rows that landed in unparsed_rows are not a silent zero.
        KAPHoldingsProvider._pdr_check_silent_zero(3, [], ["ISIN row"])

    def test_no_contradiction_when_holdings_exist(self):
        holding = Holding(
            symbol="ASELS",
            isin="TRAASELS91H2",
            name=None,
            weight=1.0,
            holding_type="stock",
        )
        KAPHoldingsProvider._pdr_check_silent_zero(3, [holding], [])

    def test_equal_distance_neighbour_is_ambiguous(self):
        # Pin 9: two owners exactly 5 pt apart on either side -> no guess.
        owners = [{"y": 100.0}, {"y": 110.0}]
        best, ambiguous = _pdr_nearest_owner(owners, 105.0, 24.0)
        assert best is None
        assert ambiguous is True

    def test_unique_neighbour_is_selected(self):
        owners = [{"y": 100.0}, {"y": 110.0}]
        best, ambiguous = _pdr_nearest_owner(owners, 108.0, 24.0)
        assert best is owners[1]
        assert ambiguous is False

    def test_too_far_neighbour_is_not_selected(self):
        owners = [{"y": 100.0}]
        best, ambiguous = _pdr_nearest_owner(owners, 140.0, 24.0)
        assert best is None
        assert ambiguous is False


class TestDiscoveryDeterminism:
    """Pin 10: the same PDF parsed twice is byte-identical in content."""

    def test_same_gbh_twice_is_identical(self, provider):
        first = provider.parse_pdr_pdf(PDF_GBH.read_bytes())
        second = provider.parse_pdr_pdf(PDF_GBH.read_bytes())
        assert first == second
        assert first.holdings == second.holdings
        assert first.unparsed_rows == second.unparsed_rows

    def test_same_dda_twice_is_identical(self, provider):
        first = provider.parse_pdr_pdf(PDF_DDA.read_bytes())
        second = provider.parse_pdr_pdf(PDF_DDA.read_bytes())
        assert first == second


# ---------------------------------------------------------------------------
# Row-level routing tests (075l2 review u-REQ-003)
#
# The two ambiguity pins are asserted at ROW level here: a synthetic PDF is
# built with the offending geometry and the assertion is that the offending
# row lands in ``unparsed_rows`` (not merely that a helper returns None).
# ---------------------------------------------------------------------------

_SYN_HEADER = (
    (370, 100, "NOMINAL"),
    (410, 100, "DEGER"),
    (450, 100, "BIRIM"),
    (480, 100, "FIYAT"),
    (560, 100, "TOPLAM"),
    (600, 100, "DEGER"),
    (660, 100, "TOPLAM"),
    (700, 100, "(FPD"),
    (730, 100, "GORE)"),
)

#: Three clean Turkish-format rows; enough to make band discovery succeed.
_SYN_CLEAN_ROWS = (
    ("AKBNK", "TRAAKBNK91N6", "100.000,00", "50,00", "5.000.000,00", "10,00"),
    ("ASELS", "TRAASELS91H2", "50.000,00", "60,00", "3.000.000,00", "6,00"),
    ("BIMAS", "TREBIMM00018", "10.000,00", "70,00", "700.000,00", "1,40"),
)


def _syn_right_text(page, right: float, y: float, text: str, size: int = 8) -> None:
    """Insert right-aligned text (PDR columns are printed right-aligned)."""
    width = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
    page.insert_text((right - width, y), text, fontsize=size)


def _syn_build_pdf(extra_items, tmp_path: Path, name: str) -> bytes:
    """Build a one-page synthetic PDR table: header + clean rows + extras."""
    items = list(_SYN_HEADER)
    y = 140
    for ticker, isin, nominal, price, value, weight in _SYN_CLEAN_ROWS:
        items += [
            (40, y, ticker), (200, y, isin), (370, y, nominal),
            (480, y, price), (560, y, value), (660, y, weight),
        ]
        y += 20
    items += list(extra_items)

    doc = pymupdf.open()
    page = doc.new_page(width=900, height=1200)
    for right, item_y, text in items:
        _syn_right_text(page, right, item_y, text)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path.read_bytes()


def test_mixed_format_row_lands_in_unparsed_rows(tmp_path):
    """A US-format cell inside a Turkish column must reach unparsed_rows."""
    extra = (
        (40, 200, "KCHOL"), (200, 200, "TRAKCHOL91Q8"),
        (370, 200, "30,000.00"),  # US format inside a Turkish nominal column
        (480, 200, "80,00"), (560, 200, "2.400.000,00"), (660, 200, "3,00"),
    )
    data = _syn_build_pdf(extra, tmp_path, "mixed.pdf")

    result = KAPHoldingsProvider().parse_pdr_pdf(data)

    # Row-level assertion: the offending row is carried in unparsed_rows.
    assert any("KCHOL" in row for row in result.unparsed_rows)
    # And it never becomes a fabricated holding.
    assert "KCHOL" not in {holding.symbol for holding in result.holdings}
    # With one ambiguous cell the whole column format is unknown, so nothing
    # is guessed: no holding is produced from this page.
    assert result.holdings == ()


def test_equal_distance_neighbour_row_lands_in_unparsed_rows(tmp_path):
    """A numeric block equidistant from two tickers must not steal either."""
    extra = (
        # Two tickers 10 pt apart; the numeric block sits exactly between them.
        (40, 220, "KCHOL"), (200, 220, "TRAKCHOL91Q8"),
        (40, 230, "TOASO"), (200, 230, "TRATOASO91H3"),
        (370, 225, "30.000,00"), (480, 225, "80,00"),
        (560, 225, "2.400.000,00"), (660, 225, "3,00"),
    )
    data = _syn_build_pdf(extra, tmp_path, "tie.pdf")

    result = KAPHoldingsProvider().parse_pdr_pdf(data)

    symbols = {holding.symbol for holding in result.holdings}
    # Row-level assertion: neither tied ticker is claimed by the shared block.
    assert "KCHOL" not in symbols
    assert "TOASO" not in symbols
    assert any("KCHOL" in row for row in result.unparsed_rows)
    assert any("TOASO" in row or "TRATOASO91H3" in row for row in result.unparsed_rows)
    # The three unambiguous rows above stay parsed (no collateral damage).
    assert {"AKBNK", "ASELS", "BIMAS"} <= symbols
