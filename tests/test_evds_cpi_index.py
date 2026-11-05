"""Offline tests for the FA-14a EVDS TÜFE index capture layer.

The sealed fixture under ``tests/fixtures/evds/`` is a byte-identical copy of
the live EVDS3 capture (262 items, 2005-1 .. 2026-10) and must never be
edited by hand. Every test here is offline: the fixture is read from disk and
the only network-shaped flow uses an injected stub provider.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from borsapy._providers.evds import (
    _CPI_DATAGROUP,
    _CPI_SERI_KODU,
    EVDSProvider,
    _normalize_cpi_items,
    clear_evds_key,
    get_cpi_index,
)
from borsapy.exceptions import APIError, DataNotAvailableError

FIXTURE = Path(__file__).parent / "fixtures" / "evds" / "fa14_ham_yanit.json"
FIXTURE_SHA256 = "615f54407ae652a56bee751c831cb2a88717cfd2bd35f3d64e421dc60db1fe17"


class _StubProvider:
    """Minimal provider exposing only ``get_series_data``."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get_series_data(self, series_codes, **kwargs):
        self.calls.append({"series_codes": series_codes, **kwargs})
        return self.payload


def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestFixtureIntegrity:
    def test_fixture_sha256(self):
        assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256

    def test_item_count_and_bounds(self):
        items = _fixture_payload()["items"]
        assert len(items) == 262
        assert items[0]["Tarih"] == "2005-1"
        assert items[-1]["Tarih"] == "2026-10"


class TestNormalizeFixture:
    def test_levels_count_and_padding(self):
        levels = _normalize_cpi_items(_fixture_payload()["items"])
        assert len(levels) == 262
        assert levels["2005-01"] == 3.596523

    def test_first_and_last_level(self):
        levels = _normalize_cpi_items(_fixture_payload()["items"])
        assert levels["2005-01"] == 3.596523
        assert levels["2026-10"] == 140.77

    def test_as_of_and_ordering(self):
        levels = _normalize_cpi_items(_fixture_payload()["items"])
        assert max(levels) == "2026-10"
        keys = list(levels)
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)

    def test_all_levels_positive(self):
        levels = _normalize_cpi_items(_fixture_payload()["items"])
        assert all(value > 0 for value in levels.values())


class TestNormalizeEdgeCases:
    def test_none_level_skipped(self):
        items = [
            {"Tarih": "2020-1", "TP_TUKFIY2025_GENEL": None},
            {"Tarih": "2020-2", "TP_TUKFIY2025_GENEL": "1.5"},
        ]
        assert _normalize_cpi_items(items) == {"2020-02": 1.5}

    def test_comma_decimal_defense(self):
        assert _normalize_cpi_items(
            [{"Tarih": "2020-1", "TP_TUKFIY2025_GENEL": "1.234,56"}]
        ) == {"2020-01": 1234.56}

    def test_malformed_tarih_raises(self):
        with pytest.raises(ValueError, match="row 0"):
            _normalize_cpi_items(
                [{"Tarih": "2020/01", "TP_TUKFIY2025_GENEL": "1.0"}]
            )

    def test_non_numeric_level_raises(self):
        with pytest.raises(ValueError, match="row 0"):
            _normalize_cpi_items(
                [{"Tarih": "2020-1", "TP_TUKFIY2025_GENEL": "abc"}]
            )

    def test_duplicate_month_raises(self):
        items = [
            {"Tarih": "2020-1", "TP_TUKFIY2025_GENEL": "1.0"},
            {"Tarih": "2020-01", "TP_TUKFIY2025_GENEL": "2.0"},
        ]
        with pytest.raises(ValueError, match="duplicate"):
            _normalize_cpi_items(items)


class TestGetCpiIndex:
    def test_contract_tree(self):
        payload = _fixture_payload()
        provider = _StubProvider(payload)
        result = get_cpi_index(
            start="01-01-2005", end="31-10-2026", provider=provider
        )
        assert set(result) == {
            "levels",
            "as_of",
            "seri_kodu",
            "veri_grubu",
            "anchor",
            "kaynak",
        }
        assert result["levels"] == _normalize_cpi_items(payload["items"])
        assert result["as_of"] == "2026-10"
        assert result["seri_kodu"] == _CPI_SERI_KODU == "TP.TUKFIY2025.GENEL"
        assert result["veri_grubu"] == _CPI_DATAGROUP == "bie_tukfiy2025"
        assert result["anchor"] == "2025=100"
        assert result["kaynak"] == "EVDS3 REST / TUIK"

    def test_parameter_propagation(self):
        provider = _StubProvider(_fixture_payload())
        get_cpi_index(
            start="01-02-2010", end="31-12-2020", provider=provider
        )
        call = provider.calls[0]
        assert call["series_codes"] == _CPI_SERI_KODU
        assert call["start"] == "01-02-2010"
        assert call["end"] == "31-12-2020"
        assert call["frequency"] == "monthly"
        assert call["decimals"] == 6

    def test_end_defaults_to_today(self):
        provider = _StubProvider(_fixture_payload())
        get_cpi_index(provider=provider)
        assert provider.calls[0]["end"] == date.today().strftime("%d-%m-%Y")
        assert provider.calls[0]["start"] == "01-01-2005"

    def test_empty_levels_raise_data_not_available(self):
        provider = _StubProvider({"totalCount": 0, "items": []})
        with pytest.raises(DataNotAvailableError):
            get_cpi_index(provider=provider)

    def test_keyless_provider_raises_api_error(self, monkeypatch):
        monkeypatch.delenv("EVDS_API_KEY", raising=False)
        clear_evds_key()
        provider = EVDSProvider()
        assert provider.has_api_key is False
        with pytest.raises(APIError):
            get_cpi_index(provider=provider)
