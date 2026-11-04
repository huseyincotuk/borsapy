"""Offline fixture tests for the İş Yatırım absolute-figures parser (FA-11-C1).

The fixtures under ``tests/fixtures/isyatirim/`` are byte-identical copies of
the sealed FA-10a live captures and must never be edited by hand.  All tests
here are fully offline; the only network-shaped flow is exercised with a fake
client.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from borsapy._providers.isyatirim import IsYatirimProvider
from borsapy.cache import TTL
from borsapy.exceptions import APIError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "isyatirim"
THYAO_FIXTURE = FIXTURE_DIR / "fa10a_thyao_sirketkart.html"
EREGL_FIXTURE = FIXTURE_DIR / "fa10a_eregl_sirketkart.html"

THYAO_SHA256 = "a987ebba01a61812eacbcdb05cd75401f890ed1361b9114deb1beaa659509ad1"
EREGL_SHA256 = "28226a134d64c0bf6e929cb5985281ce10c6df803ff59714e2be434bcaece20b"

THYAO_EXPECTED = {
    "symbol": "THYAO",
    "unit": "mnTL",
    "estimate_year": 2026,
    "rows": [
        {
            "year": 2028,
            "kind": "tahmin",
            "sales": 2161190.8,
            "net_profit": 217602.9,
            "ebitda": 356655.2,
        },
        {
            "year": 2027,
            "kind": "tahmin",
            "sales": 1702087.3,
            "net_profit": 148711.8,
            "ebitda": 253066.0,
        },
        {
            "year": 2026,
            "kind": "tahmin",
            "sales": 1328093.1,
            "net_profit": 92138.0,
            "ebitda": 163657.4,
        },
        {
            "year": 2025,
            "kind": "aciklanan",
            "sales": 1187816.2,
            "net_profit": 146952.9,
            "ebitda": 155827.6,
        },
    ],
    "meta": {"source": "isyatirim-sirket-karti", "missing": []},
}

EREGL_EXPECTED = {
    "symbol": "EREGL",
    "unit": "mnTL",
    "estimate_year": 2026,
    "rows": [
        {
            "year": 2028,
            "kind": "tahmin",
            "sales": 407753.0,
            "net_profit": 29550.3,
            "ebitda": 70571.0,
        },
        {
            "year": 2027,
            "kind": "tahmin",
            "sales": 334430.8,
            "net_profit": 11645.1,
            "ebitda": 44137.3,
        },
        {
            "year": 2026,
            "kind": "tahmin",
            "sales": 262631.1,
            "net_profit": 10469.6,
            "ebitda": 27881.4,
        },
        {
            "year": 2025,
            "kind": "aciklanan",
            "sales": 259711.0,
            "net_profit": 636.3,
            "ebitda": 20452.0,
        },
    ],
    "meta": {"source": "isyatirim-sirket-karti", "missing": []},
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> dict:
    return IsYatirimProvider._parse_absolute_figures(_read(path))


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.get_calls: list[str] = []

    def get(self, url, timeout=None, **kwargs):
        self.get_calls.append(url)
        return _FakeResponse(self.text)


class _BoomClient:
    def get(self, *args, **kwargs):
        raise OSError("network down")


# =============================================================================
# Fixture integrity + full pinned dictionaries
# =============================================================================


class TestFixtures:
    def test_thyao_fixture_sha256(self):
        assert hashlib.sha256(THYAO_FIXTURE.read_bytes()).hexdigest() == THYAO_SHA256

    def test_eregl_fixture_sha256(self):
        assert hashlib.sha256(EREGL_FIXTURE.read_bytes()).hexdigest() == EREGL_SHA256


class TestPinnedParses:
    def test_thyao_full_dict(self):
        assert _parse(THYAO_FIXTURE) == THYAO_EXPECTED

    def test_eregl_full_dict(self):
        assert _parse(EREGL_FIXTURE) == EREGL_EXPECTED

    def test_thyao_shape_and_row_order(self):
        result = _parse(THYAO_FIXTURE)
        assert len(result["rows"]) == 4
        assert result["estimate_year"] == 2026
        assert [row["year"] for row in result["rows"]] == [2028, 2027, 2026, 2025]
        assert [row["kind"] for row in result["rows"]] == [
            "tahmin",
            "tahmin",
            "tahmin",
            "aciklanan",
        ]
        for row in result["rows"]:
            for field in ("sales", "net_profit", "ebitda"):
                assert isinstance(row[field], float)

    def test_eregl_shape_and_row_order(self):
        result = _parse(EREGL_FIXTURE)
        assert len(result["rows"]) == 4
        assert result["estimate_year"] == 2026
        assert [row["year"] for row in result["rows"]] == [2028, 2027, 2026, 2025]
        assert [row["kind"] for row in result["rows"]] == [
            "tahmin",
            "tahmin",
            "tahmin",
            "aciklanan",
        ]
        for row in result["rows"]:
            for field in ("sales", "net_profit", "ebitda"):
                assert isinstance(row[field], float)

    def test_no_caveats_key_in_output(self):
        assert "caveats" not in _parse(THYAO_FIXTURE)["meta"]


# =============================================================================
# Network flow (fake client) — normalization, cache key, cache hit, APIError
# =============================================================================


class TestFetchFlow:
    @patch.object(IsYatirimProvider, "_cache_set")
    @patch.object(IsYatirimProvider, "_cache_get", return_value=None)
    def test_symbol_normalization_and_cache_key(self, _cache_get, _cache_set):
        provider = IsYatirimProvider()
        provider._client = _FakeClient(_read(THYAO_FIXTURE))

        result = provider.get_absolute_figures("thyao.is")

        assert result == THYAO_EXPECTED
        assert result["symbol"] == "THYAO"
        assert provider._client.get_calls
        assert "hisse=THYAO" in provider._client.get_calls[0]
        _cache_set.assert_called_once()
        assert _cache_set.call_args.args[0] == "isyatirim:absolute_figures:THYAO"
        assert _cache_set.call_args.args[2] == TTL.FINANCIAL_STATEMENTS

    @patch.object(IsYatirimProvider, "_cache_get", return_value=THYAO_EXPECTED)
    def test_cache_hit_skips_network(self, _cache_get):
        provider = IsYatirimProvider()
        provider._client = _BoomClient()

        result = provider.get_absolute_figures("THYAO")

        assert result == THYAO_EXPECTED
        _cache_get.assert_called_once_with("isyatirim:absolute_figures:THYAO")

    @patch.object(IsYatirimProvider, "_cache_get", return_value=None)
    def test_network_failure_raises_api_error(self, _cache_get):
        provider = IsYatirimProvider()
        provider._client = _BoomClient()

        with pytest.raises(
            APIError, match="Failed to fetch absolute figures for THYAO:"
        ):
            provider.get_absolute_figures("THYAO")


# =============================================================================
# Negative regression: FA-10a consensus output is unchanged
# =============================================================================


class TestConsensusRegression:
    def test_thyao_consensus_net_profit_unchanged(self):
        result = IsYatirimProvider._parse_consensus_estimates(_read(THYAO_FIXTURE))
        assert result["estimates"]["net_profit"] == 92138.0


# =============================================================================
# Synthetic error paths (fail-loud RuntimeError messages)
# =============================================================================


def _option(year: str, selected: bool = False) -> str:
    selected_attr = ' selected="selected"' if selected else ""
    return f'<option{selected_attr} value="1">{year}</option>'


def _row(
    symbol: str = "THYAO",
    sales: str = "1.000,0",
    net_profit: str = "100,0",
    ebitda: str = "50,0",
    kind: str = "Tahmin",
) -> str:
    return (
        "<tr>"
        f"<td>{symbol}</td><td>{sales}</td><td>{net_profit}</td>"
        f"<td>{ebitda}</td><td>{kind}</td>"
        "</tr>"
    )


def _page(
    options: str | None = None,
    rows: str = "",
    marker: bool = True,
) -> str:
    select = ""
    if options is not None:
        select = (
            '<div class="accordion-content">'
            '<select id="ctl00_ctl00_ddlMutlakRakamlarYil" class="select">'
            f"{options}</select></div>"
        )
    marker_attr = ' data-csvname="mutlakrakamlar"' if marker else ""
    return (
        "<html><body>"
        f"{select}"
        f"<table class=\"excelexport\"{marker_attr}>"
        "<thead><tr><th>Kod</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</body></html>"
    )


class TestErrorPaths:
    def test_missing_table_marker_raises(self):
        with pytest.raises(
            RuntimeError, match=r"^mutlakrakamlar tablosu yok \(schema\)$"
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(options=_option("2026", selected=True), marker=False)
            )

    def test_missing_year_select_raises(self):
        with pytest.raises(
            RuntimeError, match=r"^mutlakrakamlar yıl select'i yok \(schema\)$"
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(options=None, rows=_row())
            )

    def test_empty_year_options_raise(self):
        with pytest.raises(
            RuntimeError, match=r"^mutlakrakamlar yıl select'i yok \(schema\)$"
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(options="", rows=_row())
            )

    def test_non_four_digit_year_label_raises(self):
        with pytest.raises(
            RuntimeError, match=r"^yıl etiketi 4 hane değil: '26'$"
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(options=_option("26", selected=True), rows=_row())
            )

    def test_multiple_selected_options_raise(self):
        options = _option("2027", selected=True) + _option("2026", selected=True)
        with pytest.raises(RuntimeError, match=r"^birden fazla selected option$"):
            IsYatirimProvider._parse_absolute_figures(
                _page(options=options, rows=_row() + _row())
            )

    def test_row_year_count_mismatch_raises(self):
        with pytest.raises(
            RuntimeError, match=r"^satır\(2\) yıl\(1\) sayısı eşit değil$"
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(options=_option("2026", selected=True), rows=_row() + _row())
            )

    def test_unknown_kind_raises(self):
        with pytest.raises(
            RuntimeError,
            match=r"^bilinmeyen Tahmin/Açıklanan değeri: 'Gerçekleşen'$",
        ):
            IsYatirimProvider._parse_absolute_figures(
                _page(
                    options=_option("2026", selected=True),
                    rows=_row(kind="Gerçekleşen"),
                )
            )

    def test_blank_cells_become_none_with_ordered_missing(self):
        result = IsYatirimProvider._parse_absolute_figures(
            _page(
                options=_option("2026", selected=True),
                rows=_row(sales="", net_profit="-"),
            )
        )
        assert result["rows"][0]["sales"] is None
        assert result["rows"][0]["net_profit"] is None
        assert result["rows"][0]["ebitda"] == 50.0
        assert result["meta"]["missing"] == [
            "rows[0].sales",
            "rows[0].net_profit",
        ]
