"""Offline tests for the HedeFiyat median defect fix (FA-10a-C2).

The fixture under ``tests/fixtures/hedeffiyat/`` is a sealed live capture and
must never be edited by hand.  No test here performs network access.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from borsapy._providers.hedeffiyat import HedeFiyatProvider

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "hedeffiyat"
    / "fa10a_thyao_hedef.html"
)
SHA256 = "7221a31bc00c3c22c945b7d4d15f8c3bf776905dab32a38c5db5722956d8e8e8"


def _read() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _parse(html: str) -> dict:
    return HedeFiyatProvider()._parse_price_targets(html)


def test_fixture_sha256():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256


def test_full_parsed_dict_with_median_none():
    assert _parse(_read()) == {
        "current": 0.0,
        "low": 300.0,
        "high": 508.5,
        "mean": 437.99,
        "median": None,
        "numberOfAnalysts": 17,
    }


def test_median_stays_none_when_low_and_high_populated():
    result = _parse(_read())
    assert result["low"] is not None
    assert result["high"] is not None
    assert result["median"] is None


def test_median_is_not_fabricated_on_synthetic_page():
    html = (
        "<html><body>"
        '<div>En Yüksek Tahmin</div><div class="badge">500,00 ₺</div>'
        '<div>En Düşük Tahmin</div><div class="badge">200,00 ₺</div>'
        "</body></html>"
    )
    result = _parse(html)
    assert result["low"] == 200.0
    assert result["high"] == 500.0
    assert result["median"] is None


def test_malformed_html_keeps_silent_dict_behavior():
    result = _parse("<html><body>broken")
    assert result == {
        "current": None,
        "low": None,
        "high": None,
        "mean": None,
        "median": None,
        "numberOfAnalysts": None,
    }
