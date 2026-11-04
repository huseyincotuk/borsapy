"""Offline fixture tests for the İş Yatırım consensus parser (FA-10a-C1).

The fixtures under ``tests/fixtures/isyatirim/`` are byte-identical copies of
the sealed live captures and must never be edited by hand.  All tests here are
fully offline; the only network-shaped flow is exercised with a fake client.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from borsapy._providers.isyatirim import IsYatirimProvider, _parse_tr_number
from borsapy.exceptions import APIError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "isyatirim"
THYAO_FIXTURE = FIXTURE_DIR / "fa10a_thyao_sirketkart.html"
EREGL_FIXTURE = FIXTURE_DIR / "fa10a_eregl_sirketkart.html"

THYAO_SHA256 = "a987ebba01a61812eacbcdb05cd75401f890ed1361b9114deb1beaa659509ad1"
EREGL_SHA256 = "28226a134d64c0bf6e929cb5985281ce10c6df803ff59714e2be434bcaece20b"

CAVEATS = [
    "Cari yıl konsensüsüdür; kurum listesi yoktur.",
    "1 Hafta/1 Ay/3 Ay kolonları semantik belirsizdir; "
    "v1 bilinçli olarak kapsam dışıdır.",
]

THYAO_EXPECTED = {
    "symbol": "THYAO",
    "close_tl": 265.0,
    "estimate_year": 2026,
    "unit": "mnTL",
    "estimates": {
        "net_profit": 92138.0,
        "revenue": 1328093.1,
        "ebitda": 163657.4,
        "equity": 1125403.3,
        "total_assets": 2648739.4,
        "premiums_earned": None,
        "net_interest_income": None,
        "deposits": None,
    },
    "update_dates": {
        "net_profit": "07.08.2026",
        "revenue": "07.08.2026",
        "ebitda": "07.08.2026",
        "equity": "07.08.2026",
        "total_assets": "07.08.2026",
        "premiums_earned": None,
        "net_interest_income": None,
        "deposits": None,
    },
    "forward_multiples": {"pe": 4.0, "ev_ebitda": 5.5, "pb": 0.3},
    "forward_premium_discount_pct": {
        "pe": -27.4,
        "ev_ebitda": -22.1,
        "pb": -70.5,
    },
    "meta": {
        "source": "isyatirim-sirket-karti",
        "missing": ["premiums_earned", "net_interest_income", "deposits"],
        "caveats": CAVEATS,
    },
}

EREGL_EXPECTED = {
    "symbol": "EREGL",
    "close_tl": 30.9,
    "estimate_year": 2026,
    "unit": "mnTL",
    "estimates": {
        "net_profit": 10469.7,
        "revenue": 262631.1,
        "ebitda": 27881.4,
        "equity": 296241.7,
        "total_assets": 605171.9,
        "premiums_earned": None,
        "net_interest_income": None,
        "deposits": None,
    },
    "update_dates": {
        "net_profit": "13.10.2026",
        "revenue": "12.10.2026",
        "ebitda": "13.10.2026",
        "equity": "13.10.2026",
        "total_assets": "13.10.2026",
        "premiums_earned": None,
        "net_interest_income": None,
        "deposits": None,
    },
    "forward_multiples": {"pe": 20.7, "ev_ebitda": 9.6, "pb": 0.7},
    "forward_premium_discount_pct": {
        "pe": 77.8,
        "ev_ebitda": 43.5,
        "pb": -46.7,
    },
    "meta": {
        "source": "isyatirim-sirket-karti",
        "missing": ["premiums_earned", "net_interest_income", "deposits"],
        "caveats": CAVEATS,
    },
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> dict:
    return IsYatirimProvider._parse_consensus_estimates(_read(path))


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

    def test_dates_are_raw_strings(self):
        result = _parse(THYAO_FIXTURE)
        assert result["update_dates"]["net_profit"] == "07.08.2026"
        assert isinstance(result["update_dates"]["net_profit"], str)

    def test_eregl_per_item_dates(self):
        result = _parse(EREGL_FIXTURE)
        assert result["update_dates"]["revenue"] == "12.10.2026"
        for key in ("net_profit", "ebitda", "equity", "total_assets"):
            assert result["update_dates"][key] == "13.10.2026"

    def test_meta_caveats_documented(self):
        meta = _parse(THYAO_FIXTURE)["meta"]
        assert meta["source"] == "isyatirim-sirket-karti"
        assert len(meta["caveats"]) >= 2
        assert "konsensüs" in meta["caveats"][0]
        assert "semantik" in meta["caveats"][1]


# =============================================================================
# _parse_tr_number edge cases
# =============================================================================


class TestParseTrNumber:
    def test_empty_is_none(self):
        assert _parse_tr_number("") is None

    def test_plain_integer(self):
        assert _parse_tr_number("265") == 265.0

    def test_thousands_and_decimal(self):
        assert _parse_tr_number("92.138,0") == 92138.0

    def test_negative_decimal(self):
        assert _parse_tr_number("-27,4") == -27.4

    def test_unparseable_is_none(self):
        assert _parse_tr_number("abc") is None


# =============================================================================
# Network flow (fake client) — normalization, cache key, APIError
# =============================================================================


class TestFetchFlow:
    @patch.object(IsYatirimProvider, "_cache_set")
    @patch.object(IsYatirimProvider, "_cache_get", return_value=None)
    def test_symbol_normalization_and_cache_key(self, _cache_get, _cache_set):
        provider = IsYatirimProvider()
        provider._client = _FakeClient(_read(THYAO_FIXTURE))

        result = provider.get_consensus_estimates("thyao.is")

        assert result == THYAO_EXPECTED
        assert result["symbol"] == "THYAO"
        assert provider._client.get_calls
        assert "hisse=THYAO" in provider._client.get_calls[0]
        _cache_set.assert_called_once()
        assert _cache_set.call_args.args[0] == "isyatirim:consensus:THYAO"

    @patch.object(IsYatirimProvider, "_cache_get", return_value=None)
    def test_network_failure_raises_api_error(self, _cache_get):
        provider = IsYatirimProvider()
        provider._client = _BoomClient()

        with pytest.raises(APIError, match="Failed to fetch consensus estimates for THYAO:"):
            provider.get_consensus_estimates("THYAO")


# =============================================================================
# Synthetic error paths
# =============================================================================


def _minimal_page(
    symbol: str = "THYAO",
    cell_count: int = 7,
    forward_symbol: str = "THYAO",
    years: tuple[str, ...] = ("2026",),
) -> str:
    estimate_cells = "".join(
        f'<div class="tahminvarcol NetKar">{value}</div>'
        for value in ("1.000,0",)
    )
    date_cells = '<div class="tahminvarcol NetKar">01.01.2026</div>'
    cells = [
        f"<td>{symbol}</td>",
        "<td>10,0</td>",
        f"<td>{estimate_cells}</td>",
        f"<td>{date_cells}</td>",
        "<td></td>",
        "<td></td>",
        "<td></td>",
    ]
    cells = cells[:cell_count]
    notes = "".join(
        f'<div class="table-note">{year} yılının tahminleri ile hesaplanmıştır</div>'
        for year in years
    )
    return (
        "<html><body>"
        '<table class="excelexport" data-csvname="tahminguncellemeleri">'
        "<thead><tr><th>Kod</th></tr></thead>"
        "<tbody><tr>" + "".join(cells) + "</tr></tbody>"
        "</table>"
        '<table class="excelexport" data-csvname="tarihselortalamalar">'
        "<thead>"
        "<tr><th>Kod</th><th>F/K Tahmin</th><th>Prim/İsk. (%)</th>"
        "<th>FD/FAVÖK Tahmin</th><th>Prim/İsk. (%)</th>"
        "<th>PD/DD Tahmin</th><th>Prim/İsk. (%)</th></tr>"
        "</thead>"
        "<tbody><tr>"
        f"<td>{forward_symbol}</td><td>4,0</td><td>-27,4</td>"
        "<td>5,5</td><td>-22,1</td><td>0,3</td><td>-70,5</td>"
        "</tr></tbody></table>"
        f"{notes}"
        "</body></html>"
    )


class TestErrorPaths:
    def test_missing_table_raises(self):
        with pytest.raises(RuntimeError, match="tahminguncellemeleri tablosu yok"):
            IsYatirimProvider._parse_consensus_estimates("<html><body></body></html>")

    def test_six_cell_row_raises_with_found_count(self):
        with pytest.raises(RuntimeError, match="found 6"):
            IsYatirimProvider._parse_consensus_estimates(_minimal_page(cell_count=6))

    def test_forward_symbol_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="F/K Tahmin tablosunda THYAO"):
            IsYatirimProvider._parse_consensus_estimates(
                _minimal_page(symbol="THYAO", forward_symbol="GARAN")
            )

    def test_conflicting_years_raise(self):
        with pytest.raises(RuntimeError, match="çelişkili tahmin yılı"):
            IsYatirimProvider._parse_consensus_estimates(
                _minimal_page(years=("2026", "2025"))
            )

    def test_minimal_page_parses(self):
        result = IsYatirimProvider._parse_consensus_estimates(_minimal_page())
        assert result["symbol"] == "THYAO"
        assert result["estimates"]["net_profit"] == 1000.0
        assert result["estimate_year"] == 2026
        assert result["meta"]["missing"] == [
            "revenue",
            "ebitda",
            "equity",
            "total_assets",
            "premiums_earned",
            "net_interest_income",
            "deposits",
        ]
