"""Tests for the 2026-04 TEFAS v2 API migration (v0.9.0).

Covers:
- ``fonFiyatBilgiGetir`` history with the new ``periyod`` enum
- Client-side ``start``/``end`` filtering when no native bucket fits
- ``fonProfilBilgiGetir`` profile fields merged into ``get_fund_detail``
- JSON-backed ``get_allocation`` (parsing, universe probe, window splitting)
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from borsapy._providers.tefas import TEFASProvider
from borsapy.cache import Cache
from borsapy.exceptions import APIError, DataNotAvailableError


def _envelope(result_list: list[dict] | None) -> dict:
    """Wrap a payload in the v2 ``{errorCode, errorMessage, resultList}`` envelope."""
    return {"errorCode": None, "errorMessage": None, "resultList": result_list or []}


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.content = body
    resp.headers = {"content-type": "application/json; charset=utf-8"}
    resp.raise_for_status = MagicMock()
    resp.json = lambda: json.loads(body)
    return resp


def _make_provider(post_side_effect):
    """Build a TEFASProvider whose ``_client.post`` returns the given mocks."""
    provider = TEFASProvider.__new__(TEFASProvider)
    provider._client = MagicMock()
    provider._client.post = MagicMock(side_effect=post_side_effect)
    provider._cache = Cache()
    return provider


# =============================================================================
# get_history with fonFiyatBilgiGetir
# =============================================================================


HISTORY_ROWS = [
    {"fonKodu": "AAK", "fonUnvan": "AK PORTFOY", "tarih": "2026-04-01", "fiyat": 35.10},
    {"fonKodu": "AAK", "fonUnvan": "AK PORTFOY", "tarih": "2026-04-15", "fiyat": 35.30},
    {"fonKodu": "AAK", "fonUnvan": "AK PORTFOY", "tarih": "2026-04-30", "fiyat": 35.21},
]


class TestPeriodToPeriyod:
    """``_resolve_periyod`` maps borsapy period strings to API codes."""

    def test_default_mapping(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        for label, code in [
            ("1d", 13), ("5d", 13), ("1mo", 1), ("3mo", 3), ("6mo", 6),
            ("ytd", 0), ("1y", 12), ("3y", 36), ("5y", 60), ("max", 60),
        ]:
            assert provider._resolve_periyod(label, None, None) == code, label

    def test_unknown_period_falls_back_to_1mo(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        assert provider._resolve_periyod("eternity", None, None) == 1

    def test_explicit_start_picks_smallest_covering_bucket(self):
        from datetime import datetime, timedelta
        provider = TEFASProvider.__new__(TEFASProvider)
        now = datetime.now()
        # 5 days back -> weekly (13)
        assert provider._resolve_periyod("ignored", now - timedelta(days=5), None) == 13
        # 25 days back -> 1 month bucket
        assert provider._resolve_periyod("ignored", now - timedelta(days=25), None) == 1
        # 100 days back -> 6 month bucket
        assert provider._resolve_periyod("ignored", now - timedelta(days=100), None) == 6
        # 4 years back -> 5 year bucket
        assert provider._resolve_periyod("ignored", now - timedelta(days=365 * 4), None) == 60


class TestGetHistoryMocked:
    def test_returns_dataframe_with_price(self):
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        df = provider.get_history("AAK", period="1mo")
        assert isinstance(df, pd.DataFrame)
        assert "Price" in df.columns
        assert df["Price"].iloc[0] == 35.10
        assert len(df) == 3
        # Index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_legacy_columns_present_for_compat(self):
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        df = provider.get_history("AAK", period="1mo")
        # FundSize and Investors no longer come from the API but kept for compat
        assert "FundSize" in df.columns
        assert "Investors" in df.columns

    def test_calls_correct_endpoint_with_periyod(self):
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        provider.get_history("AAK", period="3y")
        called_url, called_kwargs = (
            provider._client.post.call_args.args,
            provider._client.post.call_args.kwargs,
        )
        # URL is positional first arg
        assert "fonFiyatBilgiGetir" in called_url[0]
        # JSON body has periyod=36 for 3y
        assert called_kwargs["json"]["periyod"] == 36
        assert called_kwargs["json"]["fonKodu"] == "AAK"

    def test_empty_result_raises_data_not_available(self):
        provider = _make_provider([_mock_response(_envelope([]))])
        with pytest.raises(DataNotAvailableError):
            provider.get_history("ZZZ", period="1mo")

    def test_client_side_date_filtering(self):
        from datetime import datetime
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        df = provider.get_history(
            "AAK",
            start=datetime(2026, 4, 10),
            end=datetime(2026, 4, 20),
        )
        # Only the 2026-04-15 row should remain
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2026-04-15")
        assert df["Price"].iloc[0] == 35.30

    def test_filtered_to_empty_raises(self):
        from datetime import datetime
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        with pytest.raises(DataNotAvailableError):
            provider.get_history(
                "AAK",
                start=datetime(2030, 1, 1),
                end=datetime(2030, 12, 31),
            )

    def test_caches_by_periyod(self):
        provider = _make_provider([_mock_response(_envelope(HISTORY_ROWS))])
        provider.get_history("AAK", period="1mo")
        provider.get_history("AAK", period="1mo")
        # Only one underlying API call thanks to cache
        assert provider._client.post.call_count == 1

    def test_returns_propagated_api_error(self):
        err_resp = _mock_response(
            {"errorCode": None, "errorMessage": "Sistem Hatası!!", "resultList": None}
        )
        provider = _make_provider([err_resp])
        with pytest.raises(APIError, match="Sistem Hatası"):
            provider.get_history("AAK", period="1mo")


# =============================================================================
# get_fund_detail merges fonProfilBilgiGetir
# =============================================================================


FUND_INFO_ROW = {
    "fonKodu": "AAK",
    "fonUnvan": "AK PORTFOY",
    "sonFiyat": 35.21,
    "portBuyukluk": 1_500_000_000.0,
    "yatirimciSayi": 12345,
    "fonKategori": "Karma Fon",
    "gunlukGetiri": 0.42,
    "kategoriDerece": 30,
    "kategoriFonSay": 100,
    "pazarPayi": 0.5,
}

FUND_RETURN_ROW = {
    "fonKodu": "AAK",
    "fonTurAciklama": "Karma Şemsiye Fonu",
    "riskDegeri": "4",
    "getiri1a": 4.78,
    "getiri3a": 2.29,
    "getiri6a": 16.27,
    "getiriyb": 8.88,
    "getiri1y": 40.20,
    "getiri3y": 255.65,
    "getiri5y": 692.75,
}

FUND_PROFILE_ROW = {
    "fonKodu": "AAK",
    "fonUnvan": "AK PORTFOY",
    "isinKodu": "TRMAF1WWWWW4",
    "sonIsSaat": "17:30",
    "basIsSaat": "09:00",
    "minAlis": 1,
    "minSatis": 1,
    "maxAlis": None,
    "maxSatis": None,
    "fonGeriAlisValor": 1,
    "fonSatisValor": 2,
    "girisKomisyonu": None,
    "cikisKomisyonu": None,
    "kapLink": "https://www.kap.org.tr/tr/fon-bilgileri/genel/aak",
    "tefasDurum": "TEFAS'ta işlem görüyor",
    "riskDegeri": "4",
}


class TestGetFundDetailV2:
    def test_includes_isin_and_kap_link(self):
        # 3 sequential API calls: fonBilgiGetir, fonGetiriBazliBilgiGetir, fonProfilBilgiGetir
        provider = _make_provider([
            _mock_response(_envelope([FUND_INFO_ROW])),
            _mock_response(_envelope([FUND_RETURN_ROW])),
            _mock_response(_envelope([FUND_PROFILE_ROW])),
        ])
        detail = provider.get_fund_detail("AAK")
        assert detail["isin"] == "TRMAF1WWWWW4"
        assert detail["kap_link"].startswith("https://www.kap.org.tr/")
        assert detail["last_trading_time"] == "17:30"
        assert detail["first_trading_time"] == "09:00"
        assert detail["buy_valor"] == 1
        assert detail["sell_valor"] == 2
        assert detail["tefas_status"] == "TEFAS'ta işlem görüyor"

    def test_includes_returns_from_returns_endpoint(self):
        provider = _make_provider([
            _mock_response(_envelope([FUND_INFO_ROW])),
            _mock_response(_envelope([FUND_RETURN_ROW])),
            _mock_response(_envelope([FUND_PROFILE_ROW])),
        ])
        detail = provider.get_fund_detail("AAK")
        assert detail["return_1y"] == 40.20
        assert detail["return_5y"] == 692.75
        assert detail["fund_type"] == "Karma Şemsiye Fonu"

    def test_fund_class_detected_yat(self):
        provider = _make_provider([
            _mock_response(_envelope([FUND_INFO_ROW])),
            _mock_response(_envelope([FUND_RETURN_ROW])),
            _mock_response(_envelope([FUND_PROFILE_ROW])),
        ])
        detail = provider.get_fund_detail("AAK", fund_type="YAT")
        assert detail["fund_class"] == "YAT"

    def test_fund_class_falls_back_to_emk_on_yat_miss(self):
        # YAT list doesn't contain AAK; EMK does.
        provider = _make_provider([
            _mock_response(_envelope([FUND_INFO_ROW])),  # fonBilgiGetir
            _mock_response(_envelope([{"fonKodu": "OTHER"}])),  # YAT returns — miss
            _mock_response(_envelope([FUND_RETURN_ROW])),  # EMK returns — hit
            _mock_response(_envelope([FUND_PROFILE_ROW])),
        ])
        detail = provider.get_fund_detail("AAK", fund_type="YAT")
        assert detail["fund_class"] == "EMK"

    def test_profile_fetch_failure_does_not_break_detail(self):
        # fonProfilBilgiGetir returns error — should still return detail with None ISIN
        bad = _mock_response(
            {"errorCode": None, "errorMessage": "Sistem Hatası!!", "resultList": None}
        )
        provider = _make_provider([
            _mock_response(_envelope([FUND_INFO_ROW])),
            _mock_response(_envelope([FUND_RETURN_ROW])),
            bad,
        ])
        detail = provider.get_fund_detail("AAK")
        assert detail["fund_code"] == "AAK"
        assert detail["isin"] is None
        assert detail["kap_link"] is None

    def test_no_data_raises(self):
        provider = _make_provider([_mock_response(_envelope([]))])
        with pytest.raises(DataNotAvailableError):
            provider.get_fund_detail("UNKNOWN")


# =============================================================================
# get_allocation (JSON endpoint, 0.11.0+)
# =============================================================================


# A verbatim row from POST /api/funds/dagilimSiraliGetirT for TPC on
# 2026-08-07, trimmed of its zero columns. Built from the real producer's
# output rather than from what the parser's signature suggests — the shape is
# the whole point of these tests.
TPC_ROW = {
    "fonKodu": "TPC",
    "fonUnvan": "TEB PORTFÖY KIYMETLİ MADENLER FON SEPETİ FONU",
    "tarih": "2026-08-07",
    "bb": 0,
    "byf": 15.86,
    "d": 0,
    "hs": 1.45,
    "tpp": 3.27,
    "vmtl": 1.69,
    "ybyf": 47.33,
    "yyf": 30.4,
    "bilFiyat": "1786114823099",
}


def _provider_with_rows(rows):
    """A provider whose only network call returns ``rows``."""
    provider = TEFASProvider.__new__(TEFASProvider)
    provider._cache = Cache()
    return provider, patch.object(provider, "_post_json_v2", return_value=rows)


class TestGetAllocationParsing:
    def test_parses_weights_and_drops_zero_and_meta_columns(self):
        provider, mocked = _provider_with_rows([TPC_ROW])
        with mocked:
            df = provider.get_allocation("TPC")

        assert isinstance(df, pd.DataFrame)
        assert set(df["code"]) == {"byf", "hs", "tpp", "vmtl", "ybyf", "yyf"}
        # bilFiyat is a millisecond timestamp, not a weight.
        assert "bilFiyat" not in set(df["code"])
        assert df["weight"].sum() == pytest.approx(100.0, abs=0.01)

    def test_sorted_by_magnitude_with_labels_attached(self):
        provider, mocked = _provider_with_rows([TPC_ROW])
        with mocked:
            df = provider.get_allocation("TPC")

        assert df["code"].iloc[0] == "ybyf"
        assert df["asset_type"].iloc[0] == "Yabancı Borsa Yatırım Fonları"
        assert df["asset_name"].iloc[0] == "Foreign ETFs"
        assert df["Date"].iloc[0] == datetime(2026, 8, 7)

    def test_negative_weight_is_kept(self):
        """ABG holds Hisse Senedi 114.14 against Repo -14.14.

        Filtering on truthiness drops zeros, which is right. Filtering on sign
        would erase the borrowing that makes the 114% possible.
        """
        row = {"fonKodu": "ABG", "tarih": "2026-08-07", "hs": 114.14, "r": -14.14}
        provider, mocked = _provider_with_rows([row])
        with mocked:
            df = provider.get_allocation("ABG")

        assert dict(zip(df["code"], df["weight"], strict=True)) == {"hs": 114.14, "r": -14.14}
        assert df["code"].iloc[0] == "hs", "sorted by magnitude, not by value"

    def test_unverified_code_gets_none_not_a_guessed_label(self):
        row = {"fonKodu": "ZJB", "tarih": "2026-08-07", "gas": 4.95}
        provider, mocked = _provider_with_rows([row])
        with mocked:
            df = provider.get_allocation("ZJB")

        assert df["code"].iloc[0] == "gas"
        assert df["asset_type"].iloc[0] is None
        assert df["asset_name"].iloc[0] is None

    def test_no_rows_raises_data_not_available(self):
        provider, mocked = _provider_with_rows([])
        with mocked:
            with pytest.raises(DataNotAvailableError):
                provider.get_allocation("AAK")

    def test_caches_result(self):
        provider, mocked = _provider_with_rows([TPC_ROW])
        with mocked as mock_post:
            provider.get_allocation("TPC")
            provider.get_allocation("TPC")
        assert mock_post.call_count == 1

    def test_snapshot_keeps_only_the_newest_date(self):
        older = dict(TPC_ROW, tarih="2026-08-05", hs=9.0)
        provider, mocked = _provider_with_rows([older, TPC_ROW])
        with mocked:
            df = provider.get_allocation("TPC")
        assert set(df["Date"]) == {datetime(2026, 8, 7)}


class TestAllocationUniverseProbe:
    def test_falls_through_to_the_next_fund_type(self):
        """A fund absent from a universe must not abort the probe.

        TEFAS leaks 'Index 0 out of bounds for length 0' instead of an empty
        list, so treating every errorMessage as fatal stopped the YAT probe
        from ever reaching EMK.
        """
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        emk_row = {"fonKodu": "AAJ", "tarih": "2026-08-07", "dt": 56.17}
        calls = []

        def fake_post(_endpoint, payload, _label, **_kwargs):
            calls.append(payload["fonTipi"])
            return [emk_row] if payload["fonTipi"] == "EMK" else []

        with patch.object(provider, "_post_json_v2", side_effect=fake_post):
            df = provider.get_allocation("AAJ")

        assert calls == ["YAT", "EMK"]
        assert df["code"].iloc[0] == "dt"

    def test_explicit_fund_type_skips_the_probe(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        calls = []

        def fake_post(_endpoint, payload, _label, **_kwargs):
            calls.append(payload["fonTipi"])
            return [TPC_ROW]

        with patch.object(provider, "_post_json_v2", side_effect=fake_post):
            provider.get_allocation("TPC", fund_type="EMK")

        assert calls == ["EMK"]

    def test_nothing_anywhere_raises_data_not_available(self):
        provider, mocked = _provider_with_rows([])
        with mocked:
            with pytest.raises(DataNotAvailableError, match="ZZZZ"):
                provider.get_allocation("ZZZZ")


class TestAllocationWindows:
    def test_single_day_is_one_window(self):
        day = datetime(2026, 8, 7)
        assert TEFASProvider._allocation_windows(day, day) == [(day, day)]

    def test_month_long_window_is_not_split(self):
        windows = TEFASProvider._allocation_windows(
            datetime(2026, 7, 11), datetime(2026, 8, 7)
        )
        assert len(windows) == 1

    def test_wide_range_is_split_without_gaps_or_overlap(self):
        first, last = datetime(2026, 5, 1), datetime(2026, 8, 7)
        windows = TEFASProvider._allocation_windows(first, last)

        assert len(windows) > 1
        assert windows[0][0] == first and windows[-1][1] == last
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
            assert (next_start - prev_end).days == 1
        for w_start, w_end in windows:
            assert (w_end - w_start).days < 31, "wider than TEFAS accepts"

    def test_range_query_hits_every_window(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        seen = []

        def fake_post(_endpoint, payload, _label, **_kwargs):
            seen.append((payload["basTarih"], payload["bitTarih"]))
            return [dict(TPC_ROW, tarih=f"2026-0{len(seen) + 5}-01")]

        with patch.object(provider, "_post_json_v2", side_effect=fake_post):
            df = provider.get_allocation(
                "TPC", start=datetime(2026, 5, 1), end=datetime(2026, 8, 7)
            )

        assert len(seen) == 4, "one request per 28-day window"
        assert seen[0][0] == "20260501" and seen[-1][1] == "20260807"
        assert df["Date"].nunique() == 4, "every window's rows are kept"


class TestPostJsonV2RateLimit:
    def test_429_is_retried_then_reported_as_an_api_error(self):
        """A 429 body is ~240 bytes, so an unguarded .json() would fail as a
        decode error rather than as 'you asked too fast'."""
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        provider._client = MagicMock()
        provider._client.post.return_value = MagicMock(status_code=429)

        with pytest.raises(APIError, match="429"):
            provider._post_json_v2(
                "dagilimSiraliGetirT", {}, "dagilimSiraliGetirT",
                max_retries=2, rate_limit_backoff=0,
            )
        assert provider._client.post.call_count == 2

    def test_errors_as_empty_translates_a_no_match_into_no_rows(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        provider._client = MagicMock()
        provider._client.post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"errorMessage": "Index 0 out of bounds for length 0"}',
        )
        provider._client.post.return_value.json.return_value = {
            "errorMessage": "Index 0 out of bounds for length 0"
        }

        rows = provider._post_json_v2(
            "dagilimSiraliGetirT", {}, "dagilimSiraliGetirT",
            errors_as_empty=("Index 0 out of bounds",),
        )
        assert rows == []

    def test_other_errors_still_raise(self):
        provider = TEFASProvider.__new__(TEFASProvider)
        provider._cache = Cache()
        provider._client = MagicMock()
        payload = {"errorMessage": "Geçersiz veri: Tarih aralığı 1 ayı aşamaz"}
        provider._client.post.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"errorMessage": "x"}',
        )
        provider._client.post.return_value.json.return_value = payload

        with pytest.raises(APIError, match="1 ayı aşamaz"):
            provider._post_json_v2(
                "dagilimSiraliGetirT", {}, "dagilimSiraliGetirT",
                errors_as_empty=("Index 0 out of bounds",),
            )
