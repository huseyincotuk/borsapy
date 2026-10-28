"""Offline fixture tests for the embedded İş Yatırım malitablo parser (FA-01a).

The fixtures under ``tests/fixtures/isyatirim/`` are byte-identical copies of
the deterministic ``fa01a_prep.py`` outputs and must never be edited by hand.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from borsapy._providers.isyatirim import IsYatirimProvider
from borsapy.exceptions import APIError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "isyatirim"
THYAO_FIXTURE = FIXTURE_DIR / "fa01a_thyao_malitablo.html"
AKBNK_FIXTURE = FIXTURE_DIR / "fa01a_akbnk_malitablo.html"

# Full values from gecmis baglam/.../fa01a_prep_goldens.json
THYAO_SHA256 = "696ce2640b224384d8e29989f45ff1d3b3c310c2061c1c8de9e58466c53fddb1"
AKBNK_SHA256 = "7ef0f06a7e85d74f935701b43fcb5439bf6279c2fd87de58742a275ddf57896b"

THYAO_PERIODS = ["2026Q2", "2026Q1", "2025Q4", "2025Q3"]
AKBNK_PERIODS = ["2026Q3", "2026Q2", "2026Q1", "2025Q4"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path, statement_type: str, quarterly: bool = True) -> pd.DataFrame:
    provider = IsYatirimProvider()
    return provider._parse_embedded_statement(
        _read(path), quarterly=quarterly, statement_type=statement_type
    )


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    """Minimal httpx.Client stand-in for offline page/postback tests."""

    def __init__(self, get_text: str, post_text: str | None = None):
        self.get_text = get_text
        self.post_text = post_text if post_text is not None else get_text
        self.get_calls: list[str] = []
        self.post_calls: list[dict] = []

    def get(self, url, timeout=None, **kwargs):
        self.get_calls.append(url)
        return _FakeResponse(self.get_text)

    def post(self, url, data=None, timeout=None, **kwargs):
        self.post_calls.append(dict(data or {}))
        return _FakeResponse(self.post_text)


# =============================================================================
# C0 — fixture placement and integrity
# =============================================================================


class TestFixtures:
    def test_thyao_fixture_sha256(self):
        digest = hashlib.sha256(THYAO_FIXTURE.read_bytes()).hexdigest()
        assert digest == THYAO_SHA256

    def test_akbnk_fixture_sha256(self):
        digest = hashlib.sha256(AKBNK_FIXTURE.read_bytes()).hexdigest()
        assert digest == AKBNK_SHA256

    def test_fixtures_are_extracted_tables(self):
        for path in (THYAO_FIXTURE, AKBNK_FIXTURE):
            text = _read(path)
            assert text.lstrip().startswith("<table")
            assert 'data-csvname="malitablo"' in text


# =============================================================================
# C1 — THYAO section structure and pins
# =============================================================================


class TestThyaoStructure:
    def test_balance_sheet_row_count(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        assert len(df) == 70
        assert list(df.columns) == THYAO_PERIODS

    def test_income_stmt_is_gelir_plus_dipnot(self):
        df = _parse(THYAO_FIXTURE, "income_stmt")
        assert len(df) == 51  # 43 Gelir Tablosu + 8 Dipnot
        assert list(df.columns) == THYAO_PERIODS
        # Dipnot rows come after the income statement rows (source order)
        assert list(df.index).index("Amortisman Giderleri") > list(df.index).index(
            "Hisse Başına Kazanç"
        )

    def test_cashflow_row_count(self):
        df = _parse(THYAO_FIXTURE, "cashflow")
        assert len(df) == 26
        assert list(df.columns) == THYAO_PERIODS

    def test_statement_type_none_returns_all_sections(self):
        df = _parse(THYAO_FIXTURE, None)
        assert len(df) == 70 + 43 + 8 + 26

    def test_duplicate_row_names_preserved_in_source_order(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        positions = [i for i, name in enumerate(df.index) if name == "Ticari Alacaklar"]
        assert len(positions) == 2
        assert positions[0] < positions[1]
        # First occurrence is the current-asset line, second is non-current (0).
        assert df.iloc[positions[0]]["2026Q2"] == 69062000000.0
        assert df.iloc[positions[1]]["2026Q2"] == 0.0

    def test_row_names_are_stripped(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        assert "(Ara Toplam)" in df.index
        assert all(name == name.strip() for name in df.index)

    def test_zero_parses_to_float(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        assert df.loc["Canlı Varlıklar"].iloc[0].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_restated_decimal_parses_to_float(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        assert df.loc["Dönen Varlıklar", "2025Q4"] == 513912099203.0


class TestThyaoPins:
    def test_balance_sheet_pins(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet")
        assert df.loc["Dönen Varlıklar", "2026Q2"] == 512191000000.0

    def test_income_pins(self):
        df = _parse(THYAO_FIXTURE, "income_stmt")
        assert df.loc["Satış Gelirleri", "2026Q2"] == 585069000000.0
        assert df.loc["FAALİYET KARI (ZARARI)", "2026Q2"] == -5098000000.0
        assert df.loc["Amortisman Giderleri", "2026Q2"] == 58665000000.0
        assert df.loc["Ana Ortaklık Payları", "2025Q4"] == 139199933601.0
        assert df.loc["Azınlık Payları", "2026Q2"] == -98000000.0

    def test_cashflow_pin(self):
        df = _parse(THYAO_FIXTURE, "cashflow")
        assert df.loc["Serbest Nakit Akım", "2026Q2"] == 13887000000.0


# =============================================================================
# C1/C4 — annual mode period labelling
# =============================================================================


class TestAnnualMode:
    def test_only_december_columns_emitted(self):
        df = _parse(THYAO_FIXTURE, "balance_sheet", quarterly=False)
        assert list(df.columns) == ["2025"]


# =============================================================================
# C2 — AKBNK (UFRS/BDDK) format-agnostic parsing
# =============================================================================


class TestAkbnkStructure:
    def test_balance_sheet_rows(self):
        df = _parse(AKBNK_FIXTURE, "balance_sheet")
        assert len(df) == 130
        assert list(df.columns) == AKBNK_PERIODS

    def test_income_rows_and_roman_numerals(self):
        df = _parse(AKBNK_FIXTURE, "income_stmt")
        assert len(df) == 62
        assert "III. NET FAİZ GELİRİ/GİDERİ (I - II)" in df.index
        assert df.loc["III. NET FAİZ GELİRİ/GİDERİ (I - II)", "2026Q3"] == 130373000000.0

    def test_balance_sheet_roman_numerals(self):
        df = _parse(AKBNK_FIXTURE, "balance_sheet")
        assert df.loc["VI. KREDİLER", "2026Q3"] == 2334451000000.0

    def test_bank_has_no_cashflow_section(self):
        df = _parse(AKBNK_FIXTURE, "cashflow")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# =============================================================================
# C1 — fetch path / error handling / group + period postback
# =============================================================================


def _page_with_form(table_html: str, group_value: str | None = None) -> str:
    group_select = ""
    if group_value is not None:
        group_select = (
            '<select name="ctl00$grp" id="ddlMaliTabloGroup">'
            f'<option selected="selected" value="{group_value}">{group_value}</option>'
            '<option value="UFRS_K">UFRS_K</option>'
            '<option value="XI_29">XI_29</option>'
            "</select>"
        )
    return (
        "<html><body><form>"
        '<input type="hidden" name="__VIEWSTATE" value="viewstate" />'
        '<input type="hidden" name="__EVENTVALIDATION" value="eventvalidation" />'
        f"{group_select}{table_html}"
        "</form></body></html>"
    )


class TestFetchPath:
    def setup_method(self):
        self.provider = IsYatirimProvider()

    def test_network_failure_raises_api_error(self):
        class _BoomClient:
            def get(self, *args, **kwargs):
                raise OSError("network down")

        self.provider._client = _BoomClient()
        with pytest.raises(APIError):
            self.provider._fetch_financial_table(
                "THYAO", None, [(2026, 6)], quarterly=True, statement_type="income_stmt"
            )

    @patch.object(IsYatirimProvider, "_cache_set")
    @patch.object(IsYatirimProvider, "_cache_get", return_value=None)
    def test_missing_table_returns_empty_dataframe(self, _cache_get, _cache_set):
        self.provider._client = _FakeClient("<html><body>no table</body></html>")
        with patch.object(
            self.provider,
            "_get_statement_html",
            return_value="<html><body>no table</body></html>",
        ):
            result = self.provider.get_financial_statements(
                "THYAO", "income_stmt", quarterly=True, last_n=4
            )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_fetch_financial_table_parses_fixture(self):
        table = _read(THYAO_FIXTURE)
        with patch.object(self.provider, "_get_statement_html", return_value=table):
            df = self.provider._fetch_financial_table(
                "THYAO",
                None,
                [(2026, 6), (2026, 3), (2025, 12), (2025, 9)],
                quarterly=True,
                statement_type="income_stmt",
            )
        assert len(df) == 51
        assert list(df.columns) == THYAO_PERIODS

    def test_group_none_uses_page_selection(self):
        page = _page_with_form(_read(THYAO_FIXTURE), group_value="XI_29")
        self.provider._client = _FakeClient(page)
        html_text = self.provider._get_statement_html("THYAO")
        assert self.provider._selected_group(html_text) == ("ctl00$grp", "XI_29")

    def test_explicit_group_triggers_postback(self):
        page = _page_with_form(_read(THYAO_FIXTURE), group_value="XI_29")
        self.provider._client = _FakeClient(page)
        self.provider._get_statement_html("THYAO", financial_group="UFRS_K")
        assert self.provider._client.post_calls
        assert self.provider._client.post_calls[0]["ctl00$grp"] == "UFRS_K"

    def test_group_switch_impossible_raises(self):
        self.provider._client = _FakeClient(_read(THYAO_FIXTURE))
        with pytest.raises(APIError, match="ddlMaliTabloGroup"):
            self.provider._get_statement_html("THYAO", financial_group="UFRS_K")

    def test_period_shift_triggers_postback(self):
        page = _page_with_form(_read(THYAO_FIXTURE), group_value="XI_29")
        self.provider._client = _FakeClient(page)
        self.provider._get_statement_html(
            "THYAO", periods=[(2025, 6), (2025, 3), (2024, 12), (2024, 9)]
        )
        assert self.provider._client.post_calls
        posted = self.provider._client.post_calls[0]
        values = [
            posted[key]
            for key in posted
            if key.endswith("ddlMaliTabloDonem1")
            or key.endswith("ddlMaliTabloDonem2")
            or key.endswith("ddlMaliTabloDonem3")
            or key.endswith("ddlMaliTabloDonem4")
        ]
        assert values == ["2025/6", "2025/3", "2024/12", "2024/9"]
        assert posted["__EVENTTARGET"].endswith("ddlMaliTabloDonem1")

    def test_postback_without_state_fields_raises(self):
        # Table alone (no form state) but a period shift is required.
        self.provider._client = _FakeClient(_read(THYAO_FIXTURE))
        with pytest.raises(APIError, match="postback"):
            self.provider._get_statement_html("THYAO", periods=[(2020, 6)])

    def test_unavailable_period_returns_empty(self):
        page = _page_with_form(_read(THYAO_FIXTURE), group_value="XI_29")
        self.provider._client = _FakeClient(page)
        result = self.provider._fetch_financial_table(
            "THYAO", None, [(1990, 12)], quarterly=True, statement_type="income_stmt"
        )
        assert result.empty

    def test_unknown_arabaslik_section_is_skipped(self):
        synthetic = (
            '<table class="excelexport" data-csvname="malitablo"><thead><tr>'
            '<th><select id="ddlMaliTabloDonem1" name="d1">'
            '<option selected="selected" value="2025/12">2025/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem2" name="d2">'
            '<option selected="selected" value="2024/12">2024/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem3" name="d3">'
            '<option selected="selected" value="2023/12">2023/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem4" name="d4">'
            '<option selected="selected" value="2022/12">2022/12</option></select></th>'
            "</tr></thead><tbody>"
            "<tr class='arabaslik'><td colspan='5' class='sectorAverage'>Gelecek Bölüm</td></tr>"
            "<tr><td>Gizemli Satır</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
            "<tr class='arabaslik'><td colspan='5' class='sectorAverage'>Bilanço</td></tr>"
            "<tr><td>Nakit</td><td>10</td><td>20</td><td>30</td><td>40</td></tr>"
            "</tbody></table>"
        )
        df = self.provider._parse_embedded_statement(
            synthetic, quarterly=False, statement_type=None
        )
        assert list(df.index) == ["Nakit"]
        assert df.loc["Nakit", "2025"] == 10.0

    def test_unparseable_value_is_source_faithful(self):
        synthetic = (
            '<table class="excelexport" data-csvname="malitablo"><thead><tr>'
            '<th><select id="ddlMaliTabloDonem1" name="d1">'
            '<option selected="selected" value="2025/12">2025/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem2" name="d2">'
            '<option selected="selected" value="2024/12">2024/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem3" name="d3">'
            '<option selected="selected" value="2023/12">2023/12</option></select></th>'
            '<th><select id="ddlMaliTabloDonem4" name="d4">'
            '<option selected="selected" value="2022/12">2022/12</option></select></th>'
            "</tr></thead><tbody>"
            "<tr class='arabaslik'><td colspan='5' class='sectorAverage'>Bilanço</td></tr>"
            "<tr><td>Metin</td><td>n/a</td><td></td><td>-5</td><td>0</td></tr>"
            "</tbody></table>"
        )
        df = self.provider._parse_embedded_statement(
            synthetic, quarterly=False, statement_type=None
        )
        assert df.loc["Metin", "2025"] == "n/a"
        assert pd.isna(df.loc["Metin", "2024"])
        assert df.loc["Metin", "2023"] == -5.0
        assert df.loc["Metin", "2022"] == 0.0


# =============================================================================
# C7 — integration smoke (network, opt-in via --run-integration)
# =============================================================================


@pytest.mark.integration
class TestLiveSmoke:
    def test_thyao_three_sections_and_cumulative_sales(self):
        provider = IsYatirimProvider()
        try:
            balance = provider.get_financial_statements(
                "THYAO", "balance_sheet", quarterly=True, last_n=4
            )
            income = provider.get_financial_statements(
                "THYAO", "income_stmt", quarterly=True, last_n=4
            )
            cashflow = provider.get_financial_statements(
                "THYAO", "cashflow", quarterly=True, last_n=4
            )
        finally:
            provider.close()

        assert not balance.empty
        assert not income.empty
        assert not cashflow.empty
        assert len(income.columns) == 4
        assert "Satış Gelirleri" in income.index

        sales = pd.to_numeric(income.loc["Satış Gelirleri"], errors="coerce")
        assert sales.dtype.kind == "f"
        assert sales.notna().all()

        # Cumulative (year-to-date) sales must not decrease within a year:
        # 9M <= FY.  Compare columns of the same year when both are present.
        for year in {str(col)[:4] for col in income.columns}:
            year_cols = [c for c in income.columns if str(c).startswith(year)]
            if f"{year}Q3" in year_cols and f"{year}Q4" in year_cols:
                assert sales[f"{year}Q3"] <= sales[f"{year}Q4"]

    def test_akbnk_sections_and_empty_cashflow(self):
        provider = IsYatirimProvider()
        try:
            balance = provider.get_financial_statements(
                "AKBNK", "balance_sheet", quarterly=True, last_n=4
            )
            cashflow = provider.get_financial_statements(
                "AKBNK", "cashflow", quarterly=True, last_n=4
            )
        finally:
            provider.close()

        assert not balance.empty
        assert isinstance(cashflow, pd.DataFrame)
        assert cashflow.empty
