"""Tests for the Doviz.com tahvil (bond) provider.

The provider scrapes https://www.doviz.com/tahvil, which was restructured to
a multi-country table with ``id="bonds"``, anchor text names (no ``class``)
and a ``data-name`` attribute per row. These tests use a deterministic,
offline fixture HTML replicating that shape and an isolated cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from borsapy._providers.dovizcom_tahvil import DovizcomTahvilProvider
from borsapy.cache import Cache
from borsapy.exceptions import DataNotAvailableError

FIXTURE_HTML = """<html><body>
<table id="bonds">
<thead><tr><th>Son</th><th>Değişim</th></tr></thead>
<tbody>
<tr data-name="TR 2 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/tr-2-yillik-tahvil"><img alt="TR 2 Yıllık Tahvil Faizi" class="currency-icon" src="https://cdn.doviz.com/images/country_flags/tr.png"/><div class="currency-details"><div>TR 2 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">39,44</td>
<td class="text-bold color-down">%-0,65</td>
</tr>
<tr data-name="TR 5 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/tr-5-yillik-tahvil"><img alt="TR 5 Yıllık Tahvil Faizi" class="currency-icon" src="https://cdn.doviz.com/images/country_flags/tr.png"/><div class="currency-details"><div>TR 5 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">38,71</td>
<td class="text-bold color-down">%-0,28</td>
</tr>
<tr data-name="TR 10 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/tr-10-yillik-tahvil"><img alt="TR 10 Yıllık Tahvil Faizi" class="currency-icon" src="https://cdn.doviz.com/images/country_flags/tr.png"/><div class="currency-details"><div>TR 10 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">35,44</td>
<td class="text-bold color-down">%-0,23</td>
</tr>
<tr data-name="ABD 2 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/abd-2-yillik-tahvil"><div class="currency-details"><div>ABD 2 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">4,73</td>
<td class="text-bold color-down">%-1,54</td>
</tr>
<tr data-name="ABD 10 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/abd-10-yillik-tahvil"><div class="currency-details"><div>ABD 10 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">5,06</td>
<td class="text-bold color-down">%-2,47</td>
</tr>
<tr data-name="Test 5 Yıllık Tahvil Faizi">
<td><a href="https://www.doviz.com/tahvil/test-5-yillik-tahvil"><div class="currency-details"><div>Test 5 Yıllık Tahvil Faizi</div></div></a></td>
<td class="text-bold">0,00</td>
<td class="text-bold color-up">%1,00</td>
</tr>
<tr data-name="Short Row">
<td>only</td>
<td>two</td>
</tr>
</tbody>
</table>
</body></html>
"""

NO_TABLE_HTML = """<html><body>
<table id="commodities">
<tbody><tr><td>foo</td><td>1,00</td><td>%1,00</td></tr></tbody>
</table>
</body></html>
"""

GOLDEN = [
    {
        "name": "TR 2 Yıllık Tahvil Faizi",
        "country": "TR",
        "maturity": "2Y",
        "yield": 39.44,
        "yield_decimal": 0.3944,
        "change_pct": -0.65,
        "change": -0.25636000000000003,
        "url": "https://www.doviz.com/tahvil/tr-2-yillik-tahvil",
    },
    {
        "name": "TR 5 Yıllık Tahvil Faizi",
        "country": "TR",
        "maturity": "5Y",
        "yield": 38.71,
        "yield_decimal": 0.3871,
        "change_pct": -0.28,
        "change": -0.10838800000000001,
        "url": "https://www.doviz.com/tahvil/tr-5-yillik-tahvil",
    },
    {
        "name": "TR 10 Yıllık Tahvil Faizi",
        "country": "TR",
        "maturity": "10Y",
        "yield": 35.44,
        "yield_decimal": 0.3544,
        "change_pct": -0.23,
        "change": -0.08151199999999999,
        "url": "https://www.doviz.com/tahvil/tr-10-yillik-tahvil",
    },
    {
        "name": "ABD 2 Yıllık Tahvil Faizi",
        "country": "ABD",
        "maturity": "2Y",
        "yield": 4.73,
        "yield_decimal": 0.0473,
        "change_pct": -1.54,
        "change": -0.072842,
        "url": "https://www.doviz.com/tahvil/abd-2-yillik-tahvil",
    },
    {
        "name": "ABD 10 Yıllık Tahvil Faizi",
        "country": "ABD",
        "maturity": "10Y",
        "yield": 5.06,
        "yield_decimal": 0.0506,
        "change_pct": -2.47,
        "change": -0.12498200000000001,
        "url": "https://www.doviz.com/tahvil/abd-10-yillik-tahvil",
    },
    {
        "name": "Test 5 Yıllık Tahvil Faizi",
        "country": "Test",
        "maturity": "5Y",
        "yield": 0.0,
        "yield_decimal": 0.0,
        "change_pct": 1.0,
        "change": 0.0,
        "url": "https://www.doviz.com/tahvil/test-5-yillik-tahvil",
    },
]


def _make_provider(html: str) -> DovizcomTahvilProvider:
    """Build an offline provider with an isolated cache and stubbed _get."""
    provider = DovizcomTahvilProvider(cache=Cache())
    response = MagicMock()
    response.content = html.encode("utf-8")
    provider._get = MagicMock(return_value=response)
    return provider


def test_get_bond_yields_parses_current_table_shape():
    provider = _make_provider(FIXTURE_HTML)
    bonds = provider.get_bond_yields()

    assert len(bonds) == 6

    for i, expected in enumerate(GOLDEN):
        bond = bonds[i]
        assert bond["name"] == expected["name"]
        assert bond["country"] == expected["country"]
        assert bond["maturity"] == expected["maturity"]
        assert bond["yield"] == pytest.approx(expected["yield"])
        assert bond["yield_decimal"] == pytest.approx(expected["yield_decimal"])
        assert bond["change_pct"] == pytest.approx(expected["change_pct"])
        assert bond["change"] == pytest.approx(expected["change"])
        assert bond["url"] == expected["url"]


def test_get_bond_yields_skips_malformed_rows():
    provider = _make_provider(FIXTURE_HTML)
    bonds = provider.get_bond_yields()

    names = [b["name"] for b in bonds]
    assert "Short Row" not in names
    assert len(bonds) == 6


def test_get_bond_yields_turkish_comma_and_percent_parsing():
    provider = _make_provider(FIXTURE_HTML)
    bonds = provider.get_bond_yields()

    tr_2y = bonds[0]
    assert tr_2y["yield"] == pytest.approx(39.44)
    assert tr_2y["change_pct"] == pytest.approx(-0.65)


def test_get_bond_yields_zero_yield_is_valid_data():
    provider = _make_provider(FIXTURE_HTML)
    bonds = provider.get_bond_yields()

    test_bond = next(b for b in bonds if b["name"].startswith("Test"))
    assert test_bond["yield"] == 0.0
    assert test_bond["yield_decimal"] == 0.0
    assert test_bond["yield_decimal"] is not None
    assert test_bond["change"] == 0.0


def test_get_bond_returns_turkish_bond_only():
    provider = _make_provider(FIXTURE_HTML)
    bond = provider.get_bond("10Y")

    assert bond["name"].startswith("TR")
    assert bond["country"] == "TR"
    assert bond["yield"] == pytest.approx(35.44)


def test_get_bond_unknown_maturity_raises():
    provider = _make_provider(FIXTURE_HTML)

    with pytest.raises(DataNotAvailableError):
        provider.get_bond("30Y")


def test_get_10y_yield_returns_tr_decimal():
    provider = _make_provider(FIXTURE_HTML)

    assert provider.get_10y_yield() == pytest.approx(0.3544)


def test_get_10y_yield_none_when_source_unavailable():
    provider = _make_provider(NO_TABLE_HTML)

    assert provider.get_10y_yield() is None


def test_bonds_dataframe_turkish_only(monkeypatch):
    provider = _make_provider(FIXTURE_HTML)
    monkeypatch.setattr(
        "borsapy.bond.get_tahvil_provider", lambda: provider
    )

    import borsapy.bond as bond_module

    df = bond_module.bonds()

    assert len(df) == 3
    assert list(df.columns) == [
        "name",
        "maturity",
        "yield",
        "change",
        "change_pct",
    ]
    assert list(df["maturity"]) == ["2Y", "5Y", "10Y"]
    assert all(name.startswith("TR") for name in df["name"])


def test_get_bond_yields_uses_cache():
    provider = _make_provider(FIXTURE_HTML)

    first = provider.get_bond_yields()
    second = provider.get_bond_yields()

    assert first == second
    assert provider._get.call_count == 1
