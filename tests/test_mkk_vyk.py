"""Tests for the MKK VYK provider and XBRL presentation parser (FA-01b).

The fixtures under ``tests/fixtures/mkk/`` are byte-identical copies of the
sealed ``fa01b_prep_2026_10_28`` captures and must never be edited by hand.
All parser/synthetic/client tests run offline; the live smoke tests are marked
``integration`` and are skipped unless ``--run-integration`` is passed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from borsapy._providers.mkk import (
    MkkVykProvider,
    clear_mkk_credentials,
    parse_disclosure_detail,
    parse_presentation_block,
    resolve_contexts,
    set_mkk_credentials,
)
from borsapy.exceptions import APIError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mkk"
ULAS_FIXTURE = FIXTURE_DIR / "fa01b_ulas_fr_1211007.json"
UFAK_FIXTURE = FIXTURE_DIR / "fa01b_ufak_fr_1211013.json"

# Full sha256 values recorded in fa01b_prep_goldens.json.
ULAS_SHA256 = "bd74c1fe1c374850f9c6060cff816c3888119c8181e84ceff44dbb90faa50b9a"
UFAK_SHA256 = "7d1f1c83187f1b1bf326ef223a9031fb670d572959ac5ee85eb1244422ca13b4"

ULAS_ROW_COUNTS = {
    "balance_sheet": 54,
    "income_stmt": 20,
    "comprehensive_income": 10,
    "cashflow": 45,
    "equity": 4,
}
UFAK_ROW_COUNTS = {
    "finance_role_210014": 48,
    "finance_role_210502": 9,
    "finance_role_310020": 50,
    "finance_role_420005": 5,
    "finance_role_510010": 27,
    "finance_role_610006": 8,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _clean_explicit_credentials():
    clear_mkk_credentials()
    yield
    clear_mkk_credentials()


@pytest.fixture
def mock_credentials(monkeypatch):
    monkeypatch.setenv("MKK_API_KEY", "mock-key")
    monkeypatch.setenv("MKK_API_SECRET", "mock-secret")
    yield


def _row(df: pd.DataFrame, label: str) -> pd.Series:
    matches = df[df.index == label]
    assert len(matches) == 1, f"expected exactly one row {label!r}, got {len(matches)}"
    return matches.iloc[0]


def _context(ctx_id: str, key: str, *, instant: str | None = None, start: str | None = None, end: str | None = None):
    period = {"instant": instant} if instant else {"startDate": start, "endDate": end}
    return {"id": ctx_id, "key": key, "Period": period}


def _block(role: str, contexts: list[dict], items: dict | list) -> dict:
    return {
        "id": f"general_role_{role}",
        "content": {
            "ContextList": {"Context": contexts},
            "ReportItem": items,
        },
    }


def _leaf(label_tr: str | None, label_en: str | None, name: str, values: list[dict], abstract: bool = False) -> dict:
    langs = []
    if label_tr is not None:
        langs.append({"code": "tr", "value": label_tr})
    if label_en is not None:
        langs.append({"code": "en", "value": label_en})
    item: dict = {"name": name, "abstract": "true" if abstract else "false"}
    if langs:
        item["langs"] = {"lang": langs}
    if values:
        item["Values"] = {"Value": values if len(values) > 1 else values[0]}
    return item


# =============================================================================
# C0 — fixture placement and integrity
# =============================================================================


class TestFixtures:
    def test_ulas_fixture_sha256(self):
        assert hashlib.sha256(ULAS_FIXTURE.read_bytes()).hexdigest() == ULAS_SHA256

    def test_ufak_fixture_sha256(self):
        assert hashlib.sha256(UFAK_FIXTURE.read_bytes()).hexdigest() == UFAK_SHA256

    def test_fixtures_are_valid_json(self):
        assert isinstance(_load(ULAS_FIXTURE), dict)
        assert isinstance(_load(UFAK_FIXTURE), dict)


# =============================================================================
# C4 — offline ULAS (general taxonomy) parsing
# =============================================================================


class TestUlasParsing:
    @pytest.fixture
    def parsed(self) -> dict[str, pd.DataFrame]:
        return parse_disclosure_detail(_load(ULAS_FIXTURE))

    def test_all_five_blocks_present(self, parsed):
        assert set(parsed) == set(ULAS_ROW_COUNTS)

    @pytest.mark.parametrize("statement,count", ULAS_ROW_COUNTS.items())
    def test_row_counts(self, parsed, statement, count):
        assert len(parsed[statement]) == count

    def test_income_columns_and_order(self, parsed):
        assert list(parsed["income_stmt"].columns) == [
            "2023Q3 (YTD)",
            "2023Q3 (3M)",
            "2022Q3 (YTD)",
            "2022Q3 (3M)",
        ]

    def test_balance_columns_are_bare_keys(self, parsed):
        assert list(parsed["balance_sheet"].columns) == ["CURR", "PREV"]

    def test_pinned_esas_faaliyet_kari(self, parsed):
        row = _row(parsed["income_stmt"], "ESAS FAALİYET KARI (ZARARI)")
        assert row["2023Q3 (YTD)"] == 2251210.0
        assert row["2023Q3 (3M)"] == -2170390.0
        assert row["2022Q3 (YTD)"] == 3787575.0
        assert row["2022Q3 (3M)"] == 1218176.0

    def test_pinned_genel_yonetim_giderleri(self, parsed):
        row = _row(parsed["income_stmt"], "Genel Yönetim Giderleri")
        assert row["2023Q3 (YTD)"] == 5955127.0
        assert row["2023Q3 (3M)"] == 2298633.0
        assert row["2022Q3 (YTD)"] == 2493722.0
        assert row["2022Q3 (3M)"] == 934105.0

    def test_pinned_toplam_donen_varliklar(self, parsed):
        row = _row(parsed["balance_sheet"], "TOPLAM DÖNEN VARLIKLAR")
        assert row["CURR"] == 36757158.0
        assert row["PREV"] == 25881422.0

    def test_pinned_donem_kari(self, parsed):
        row = _row(parsed["income_stmt"], "DÖNEM KARI (ZARARI)")
        assert row["2023Q3 (YTD)"] == 57449770.0
        assert row["2023Q3 (3M)"] == -4903129.0
        assert row["2022Q3 (YTD)"] == 3110514.0
        assert row["2022Q3 (3M)"] == 1635955.0

    def test_pinned_cashflow_row(self, parsed):
        row = _row(parsed["cashflow"], "İŞLETME FAALİYETLERİNDEN NAKİT AKIŞLARI")
        assert row["2023Q3 (YTD)"] == 2228969.0
        assert row["2022Q3 (YTD)"] == 5556599.0

    def test_context_id_to_key_map(self):
        detail = _load(ULAS_FIXTURE)
        income_block = detail["presentation"][1]
        contexts = resolve_contexts(income_block)
        assert contexts["2023-07-012023-09-30"] == ("CURR3", "2023-07-012023-09-30")
        assert contexts["2022-01-012022-09-30"] == ("PREV", "2022-01-012022-09-30")
        assert contexts["2022-07-012022-09-30"] == ("PREV3", "2022-07-012022-09-30")
        assert contexts["2023-01-012023-09-30"] == ("CURR", "2023-01-012023-09-30")

    def test_value_string_parsing_zero_and_negative(self, parsed):
        zero_row = _row(parsed["income_stmt"], "TİCARİ FAALİYETLERDEN BRÜT KAR (ZARAR)")
        assert zero_row["2023Q3 (YTD)"] == 0.0
        negative_row = _row(parsed["income_stmt"], "ESAS FAALİYET KARI (ZARARI)")
        assert negative_row["2023Q3 (3M)"] < 0

    def test_currency_metadata(self, parsed):
        assert parsed["income_stmt"].attrs["currency"] == "TRY"


# =============================================================================
# C4 — offline UFAK (finance taxonomy) parsing + dimensional contexts
# =============================================================================


class TestUfakParsing:
    @pytest.fixture
    def detail(self) -> dict:
        return _load(UFAK_FIXTURE)

    @pytest.fixture
    def parsed(self, detail) -> dict[str, pd.DataFrame]:
        return parse_disclosure_detail(detail)

    def test_finance_role_family_recognized(self, parsed):
        assert set(parsed) == {
            "balance_sheet",
            "income_stmt",
            "comprehensive_income",
            "cashflow",
            "equity",
        }

    def test_off_balance_sheet_block_is_skipped_from_statement_map(self, parsed):
        # finance_role_210502 is a real block but not a canonical statement.
        assert parsed["balance_sheet"].attrs["role"] == "210014"

    @pytest.mark.parametrize("block_id,count", UFAK_ROW_COUNTS.items())
    def test_block_row_counts(self, detail, block_id, count):
        block = next(b for b in detail["presentation"] if b["id"] == block_id)
        assert len(parse_presentation_block(block)) == count

    def test_mapped_statement_row_counts(self, parsed):
        assert len(parsed["balance_sheet"]) == 48
        assert len(parsed["income_stmt"]) == 50
        assert len(parsed["comprehensive_income"]) == 5
        assert len(parsed["cashflow"]) == 27
        assert len(parsed["equity"]) == 8

    def test_dimensional_first_value_and_dim_cells(self, parsed):
        label = "NAKİT, NAKİT BENZERLERİ ve MERKEZ BANKASI"
        row = _row(parsed["balance_sheet"], label)
        assert row["CURR"] == 152141000.0
        assert row["PREV"] == 24931000.0
        dim_cells = parsed["balance_sheet"].attrs["dim_cells"]
        assert dim_cells[label]["CURR"] == [152141000.0, 3463000.0, 155604000.0]

    def test_context_resolution_for_instant_block(self, detail):
        block = next(b for b in detail["presentation"] if b["id"] == "finance_role_210014")
        contexts = resolve_contexts(block)
        assert contexts["2023-09-30"] == ("CURR", "2023-09-30")
        assert contexts["2022-12-31"] == ("PREV", "2022-12-31")


# =============================================================================
# C4 — synthetic edge cases
# =============================================================================


class TestSyntheticParsing:
    def test_unknown_role_is_skipped(self):
        block = _block(
            "999999",
            [_context("2023-09-30", "CURR", instant="2023-09-30")],
            _leaf("Satır", "Row", "SomeItem", [{"contextId": "2023-09-30", "value": "1"}]),
        )
        assert parse_disclosure_detail({"presentation": [block]}) == {}

    def test_abstract_rows_are_skipped(self):
        root = {
            "name": "RootAbstract",
            "abstract": "true",
            "langs": {"lang": [{"code": "tr", "value": "Kök"}]},
            "ReportItem": _leaf(
                "Somut",
                "Concrete",
                "ConcreteItem",
                [{"contextId": "2023-09-30", "value": "5"}],
            ),
        }
        block = _block(
            "210015",
            [_context("2023-09-30", "CURR", instant="2023-09-30")],
            root,
        )
        frame = parse_disclosure_detail({"presentation": [block]})["balance_sheet"]
        assert list(frame.index) == ["Somut"]

    def test_label_fallback_tr_en_name(self):
        contexts = [_context("2023-09-30", "CURR", instant="2023-09-30")]
        items = [
            _leaf("Türkçe", "English", "TrItem", [{"contextId": "2023-09-30", "value": "1"}]),
            _leaf(None, "English only", "EnItem", [{"contextId": "2023-09-30", "value": "2"}]),
            _leaf(None, None, "TaxonomyName", [{"contextId": "2023-09-30", "value": "3"}]),
        ]
        frame = parse_presentation_block(_block("210015", contexts, items))
        assert list(frame.index) == ["Türkçe", "English only", "TaxonomyName"]

    def test_empty_values_yields_no_row(self):
        contexts = [_context("2023-09-30", "CURR", instant="2023-09-30")]
        item = {"name": "Empty", "abstract": "false", "Values": {"Value": []}}
        frame = parse_presentation_block(_block("210015", contexts, item))
        assert len(frame) == 0

    def test_unparseable_value_is_nan(self):
        contexts = [_context("2023-09-30", "CURR", instant="2023-09-30")]
        item = _leaf("Bozuk", "Broken", "BrokenItem", [{"contextId": "2023-09-30", "value": "abc"}])
        frame = parse_presentation_block(_block("210015", contexts, item))
        assert pd.isna(frame.iloc[0]["CURR"])

    def test_empty_presentation_yields_empty_map(self):
        assert parse_disclosure_detail({"presentation": []}) == {}
        assert parse_disclosure_detail({}) == {}

    def test_duplicate_row_labels_are_preserved(self):
        contexts = [_context("2023-09-30", "CURR", instant="2023-09-30")]
        items = [
            _leaf("Aynı", "Same", "First", [{"contextId": "2023-09-30", "value": "1"}]),
            _leaf("Aynı", "Same", "Second", [{"contextId": "2023-09-30", "value": "2"}]),
        ]
        frame = parse_presentation_block(_block("210015", contexts, items))
        assert list(frame.index) == ["Aynı", "Aynı"]
        assert list(frame["CURR"]) == [1.0, 2.0]


# =============================================================================
# C4 — client unit tests with mocked httpx
# =============================================================================


class _FakeResponse:
    def __init__(self, payload=None, status: int = 200, *, invalid_json: bool = False):
        self.status_code = status
        self._payload = payload
        self._invalid = invalid_json
        self.headers: dict[str, str] = {}
        if invalid_json:
            self.text = "not-json"
        else:
            self.text = json.dumps(payload) if payload is not None else ""
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._invalid:
            raise ValueError("invalid json")
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _provider(responses, **kwargs) -> MkkVykProvider:
    kwargs.setdefault("min_interval", 0)
    provider = MkkVykProvider(**kwargs)
    provider._client = _FakeClient(responses)
    return provider


class TestClient:
    def test_basic_auth_header(self, mock_credentials):
        set_mkk_credentials("my-key", "my-secret")
        provider = _provider([_FakeResponse({"lastDisclosureIndex": "1231017"})])
        assert provider.last_disclosure_index() == 1231017
        headers = provider._client.calls[0]["headers"]
        expected = base64.b64encode(b"my-key:my-secret").decode()
        assert headers["Authorization"] == f"Basic {expected}"
        assert "my-secret" not in json.dumps(headers).replace(expected, "")

    def test_missing_credentials_raises_and_names_env_vars(self, monkeypatch):
        monkeypatch.delenv("MKK_API_KEY", raising=False)
        monkeypatch.delenv("MKK_API_SECRET", raising=False)
        with pytest.raises(APIError) as excinfo:
            MkkVykProvider()
        message = str(excinfo.value)
        assert "MKK_API_KEY" in message
        assert "MKK_API_SECRET" in message

    def test_retry_on_429_stops_after_three_attempts(self, mock_credentials):
        set_mkk_credentials("k", "s")
        fault = {"fault": {"faultCode": "ER002", "faultString": "too many"}}
        clock = _FakeClock()
        provider = _provider(
            [_FakeResponse(fault, 429), _FakeResponse(fault, 429), _FakeResponse(fault, 429)],
            sleeper=clock.sleep,
        )
        with pytest.raises(APIError):
            provider.last_disclosure_index()
        assert len(provider._client.calls) == 3
        assert clock.sleeps == [1, 2]

    def test_fault_code_is_embedded_in_error(self, mock_credentials):
        set_mkk_credentials("k", "s")
        fault = {"fault": {"faultCode": "ER003", "faultString": "IP not allowed"}}
        provider = _provider([_FakeResponse(fault, 403)])
        with pytest.raises(APIError) as excinfo:
            provider.disclosures(1211007)
        assert "ER003" in str(excinfo.value)

    def test_empty_window_400_returns_empty_list(self, mock_credentials):
        set_mkk_credentials("k", "s")
        fault = {"fault": {"faultCode": "ER005", "faultString": "Bildirim bulunamadı"}}
        provider = _provider([_FakeResponse(fault, 400)])
        assert provider.disclosures(1211007) == []

    def test_empty_window_400_returns_empty_detail(self, mock_credentials):
        set_mkk_credentials("k", "s")
        fault = {"fault": {"faultCode": "ER008", "faultString": "Bildirim bulunamadı"}}
        provider = _provider([_FakeResponse(fault, 400)])
        assert provider.disclosure_detail(1211007) == {}

    def test_invalid_json_raises(self, mock_credentials):
        set_mkk_credentials("k", "s")
        provider = _provider([_FakeResponse(invalid_json=True)])
        with pytest.raises(APIError):
            provider.members()

    def test_rate_interval_spacing_with_mocked_clock(self, mock_credentials):
        set_mkk_credentials("k", "s")
        clock = _FakeClock()
        provider = _provider(
            [
                _FakeResponse({"lastDisclosureIndex": "1"}),
                _FakeResponse([{"id": "1"}]),
            ],
            min_interval=10.5,
            clock=clock,
            sleeper=clock.sleep,
        )
        provider.last_disclosure_index()
        provider.members()
        assert clock.now >= 10.5
        assert clock.sleeps == [10.5]

    def test_get_financial_statements_matches_sender_exch_codes(self, mock_credentials):
        set_mkk_credentials("k", "s")
        detail = _load(ULAS_FIXTURE)
        provider = _provider(
            [
                _FakeResponse({"lastDisclosureIndex": "1211007"}),
                _FakeResponse(
                    [
                        {"disclosureIndex": "1211007", "disclosureClass": "FR"},
                        {"disclosureIndex": "1211006", "disclosureClass": "FR"},
                    ]
                ),
                _FakeResponse(detail),
            ]
        )
        frame = provider.get_financial_statements("ULAS", "income_stmt")
        assert len(frame) == 20
        assert "ESAS FAALİYET KARI (ZARARI)" in frame.index

    def test_get_financial_statements_rejects_unknown_statement(self, mock_credentials):
        set_mkk_credentials("k", "s")
        provider = _provider([])
        with pytest.raises(ValueError):
            provider.get_financial_statements("ULAS", "not_a_statement")


# =============================================================================
# C4 — live integration smoke (structural only, no value pins)
# =============================================================================


def _has_live_credentials() -> bool:
    return bool(os.environ.get("MKK_API_KEY")) and bool(os.environ.get("MKK_API_SECRET"))


@pytest.mark.integration
def test_live_last_disclosure_index_is_positive():
    if not _has_live_credentials():
        pytest.skip("MKK_API_KEY / MKK_API_SECRET not configured")
    with MkkVykProvider(gateway="test") as provider:
        index = provider.last_disclosure_index()
    assert isinstance(index, int)
    assert index > 0


@pytest.mark.integration
def test_live_ulas_detail_structure():
    if not _has_live_credentials():
        pytest.skip("MKK_API_KEY / MKK_API_SECRET not configured")
    with MkkVykProvider(gateway="test") as provider:
        detail = provider.disclosure_detail(1211007)
    presentation = detail.get("presentation") or []
    assert len(presentation) == 5
    keys = set()
    for block in presentation:
        keys.update(key for key, _period in resolve_contexts(block).values())
    assert "CURR3" in keys
