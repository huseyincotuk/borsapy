"""İş Yatırım provider for real-time prices and financial statements."""

import html as html_module
import re
import warnings
from datetime import datetime
from typing import Any

import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.cache import TTL
from borsapy.exceptions import APIError, DataNotAvailableError, TickerNotFoundError


class IsYatirimProvider(BaseProvider):
    """
    Provider for real-time stock data and financial statements from İş Yatırım.

    APIs:
        - OneEndeks: Real-time OHLCV data
        - CompanyInfoAjax: Dividends and capital increases
        - sirket-karti.aspx: Embedded financial statement table (HTML)

    Financial statements (source notes)
    ------------------------------------
    The legacy ``Data.aspx/MaliTablo`` JSON endpoint that this provider used to
    call is dead: as of the 28.10.2026 site renewal it answers HTTP 200 with
    ``{"value": []}`` for every symbol/group/period combination.  Financial
    statements now arrive **server-side embedded in HTML** on
    ``sirket-karti.aspx?hisse=<SYMBOL>`` (the same page already scraped by
    :meth:`get_major_holders` and :meth:`get_company_metrics`).  The provider
    extracts ``<table class="excelexport" data-csvname="malitablo">`` from that
    page instead of calling the JSON endpoint.  There is no MaliTablo XHR
    traffic any more.

    Section structure
    -----------------
    The embedded table is split by ``<tr class='arabaslik'>`` header rows into
    four known sections, in source order:

    * ``Bilanço`` — balance sheet (stock table, not used for TTM)
    * ``Gelir Tablosu`` — income statement (flow table)
    * ``Dipnot`` — income-statement supplementary notes (Amortisman Giderleri,
      Kıdem Tazminatı, Finansman Giderleri, Yurtiçi/Yurtdışı Satışlar, net FX
      position).  They are EBITDA inputs, so ``income_stmt`` returns the union
      of ``Gelir Tablosu`` and ``Dipnot`` in source order.
    * ``Nakit Akım Tablosu`` — cash flow statement (flow table); it ends at the
      table ``</table>`` close.

    Unknown ``arabaslik`` sections are skipped for forward compatibility.
    Row names are carried verbatim (only surrounding whitespace is stripped);
    duplicate row names are preserved in source order — no deduplication, no
    renaming.  The UFRS/BDDK family (e.g. AKBNK) uses Roman-numeral BDDK chart
    of accounts row names, which are also carried verbatim.  Bank pages carry
    no ``Nakit Akım Tablosu`` section, so ``get_cashflow`` honestly returns an
    empty DataFrame for them (bank cash flow is expected from MKK in FA-01b).

    Cumulative reporting (critical)
    -------------------------------
    Turkish quarterly flow statements are **year-to-date cumulative**, not
    standalone quarters: THYAO 2026/6 revenue is 9M-to-date while 2026/3 is
    3M-to-date and 2025/12 is the full year.  Never add four quarterly columns
    as if they were standalone.  :meth:`derive_standalone_quarters` derives
    additive quarters (``Qn = cum(Qn) - cum(Qn-1)`` within a fiscal year, Q1 =
    cum(Q1), no clamping — negative values are legitimate) and
    :meth:`borsapy.ticker.Ticker.get_ttm_income_stmt` uses the canonical bridge
    ``TTM(P) = cum(P) + cum(FY_{y-1}) - cum(P_{y-1})``.

    Restatement policy (İş Yatırım TMS-29)
    --------------------------------------
    The table header carries İş Yatırım's official note: "Sanayi
    şirketlerinde en son açıklanmış yıl sonu rakamları enflasyon düzeltmesiyle
    son finansal döneme çekilmektedir."  Some columns are therefore
    inflation-restated and can be fractional (e.g. ``513.912.099.203``) rather
    than as-reported integers.  borsapy transfers these values source-faithfully
    and does not restate, scale or substitute them.  The canonical
    as-reported values come from MKK (FA-01b); İş Yatırım restated values must be
    used in a restated context only.  Restated-column flagging is the FA-02
    policy layer's job.

    Data caveats
    ------------
    * The ready-made "Hisse Başına Kazanç" (EPS) row on the page is unreliable
      (THYAO reports 0 for some periods).  borsapy only carries the rows and
      makes no interpretation: compute EPS yourself from net income and share
      count if you need it.
    * Kardemir / İş Bankası pay-group share-class variants are not specially
      handled.  Their rows are passed through exactly as published.
    * A page without a malitablo table yields an empty DataFrame; network or
      HTTP failures raise :class:`APIError` and are never silently swallowed.
    * AGENTS.md discipline applies: preserve the source, never silently
      substitute missing values, and pay attention to units, periods and
      restatement context.
    """

    BASE_URL = "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common"
    STOCK_INFO_URL = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx"
    STATEMENT_PAGE_URL = (
        "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx"
    )

    # Financial statement groups
    FINANCIAL_GROUP_INDUSTRIAL = "XI_29"  # Sanayi şirketleri
    FINANCIAL_GROUP_BANK = "UFRS"  # Bankalar

    # itemCode prefixes from the dead MaliTablo JSON endpoint.  Kept only for
    # the legacy ``_parse_financial_response`` helper (and its regression
    # tests); the embedded HTML table has no itemCode, so the new path never
    # uses this filter.
    ITEM_CODE_PREFIXES = {
        "balance_sheet": ("1", "2"),  # 1xxx=Aktif, 2xxx=Pasif
        "income_stmt": ("3",),       # 3xxx=Gelir Tablosu
        "cashflow": ("4",),          # 4xxx=Nakit Akış
    }

    # Normalised arabaslik section title -> statement_type membership
    _STATEMENT_SECTIONS = {
        "balance_sheet": ("bilanço",),
        "income_stmt": ("gelir tablosu", "dipnot"),
        "cashflow": ("nakit akım tablosu",),
    }

    # Known market indices
    INDICES = {
        "XU100": "BIST 100",
        "XU050": "BIST 50",
        "XU030": "BIST 30",
        "XBANK": "BIST Banka",
        "XUSIN": "BIST Sınai",
        "XHOLD": "BIST Holding ve Yatırım",
        "XUTEK": "BIST Teknoloji",
        "XGIDA": "BIST Gıda",
        "XTRZM": "BIST Turizm",
        "XULAS": "BIST Ulaştırma",
        "XSGRT": "BIST Sigorta",
        "XMANA": "BIST Metal Ana",
        "XKMYA": "BIST Kimya",
        "XMADN": "BIST Maden",
        "XELKT": "BIST Elektrik",
        "XTEKS": "BIST Tekstil",
        "XILTM": "BIST İletişim",
        "XUMAL": "BIST Mali",
        "XUTUM": "BIST Tüm",
    }

    def __init__(self):
        super().__init__(verify=False)

    def get_realtime_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get real-time quote for a symbol using OneEndeks API.

        .. deprecated::
            This method is deprecated since v0.5.1. Use TradingView provider
            via ``bp.Ticker(symbol).info`` or ``bp.TradingViewStream()`` instead.

        Args:
            symbol: Stock symbol (e.g., "THYAO", "GARAN").

        Returns:
            Dictionary with quote data:
            - symbol: Stock symbol
            - last: Last price
            - open: Opening price
            - high: High price
            - low: Low price
            - close: Previous day close
            - volume: Trading volume
            - bid: Bid price
            - ask: Ask price
            - change: Price change
            - change_percent: Price change percentage
            - update_time: Last update time
        """
        warnings.warn(
            "get_realtime_quote is deprecated since v0.5.1. "
            "Use TradingView provider via bp.Ticker(symbol).info or bp.TradingViewStream() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:quote:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/Data.aspx/OneEndeks"
        params = {"endeks": symbol}

        headers = {
            "Referer": "https://www.isyatirim.com.tr/tr-tr/analiz/Sayfalar/default.aspx"
        }

        try:
            response = self._get(url, params=params, headers=headers)
            data = response.json()
        except Exception as e:
            raise APIError(f"Failed to fetch quote for {symbol}: {e}") from e

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if not data or "symbol" not in data:
            raise TickerNotFoundError(symbol)

        result = self._parse_quote(data)
        self._cache_set(cache_key, result, TTL.REALTIME_PRICE)

        return result

    def get_index_history(
        self,
        index_code: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Get historical data for an index.

        .. deprecated::
            This method is deprecated since v0.5.1. Use TradingView provider
            via ``bp.Index(index_code).history()`` instead.

        Args:
            index_code: Index code (e.g., "XU100", "XU030", "XBANK").
            start: Start date.
            end: End date.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.

        Note:
            This endpoint returns only a single price value per timestamp,
            so Open, High, Low, and Close are all the same value.
        """
        warnings.warn(
            "get_index_history is deprecated since v0.5.1. "
            "Use TradingView provider via bp.Index(index_code).history() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        index_code = index_code.upper()

        # Default date range
        if end is None:
            end = datetime.now()
        if start is None:
            from datetime import timedelta

            start = end - timedelta(days=365)

        start_str = start.strftime("%Y%m%d%H%M%S")
        end_str = end.strftime("%Y%m%d%H%M%S")

        cache_key = f"isyatirim:index_history:{index_code}:{start_str}:{end_str}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/ChartData.aspx/IndexHistoricalAll"

        params = {
            "endeks": index_code,
            "period": "1440",
            "from": start_str,
            "to": end_str,
        }

        headers = {
            "Referer": "https://www.isyatirim.com.tr/tr-tr/analiz/Sayfalar/default.aspx"
        }

        try:
            response = self._get(url, params=params, headers=headers)
            data = response.json()
        except Exception as e:
            raise APIError(f"Failed to fetch index history for {index_code}: {e}") from e

        if not data:
            raise DataNotAvailableError(f"No data for index: {index_code}")

        df = self._parse_index_history(data)
        self._cache_set(cache_key, df, TTL.OHLCV_HISTORY)

        return df

    def _parse_index_history(self, data: dict[str, Any] | list) -> pd.DataFrame:
        """Parse index history response into DataFrame."""
        records = []

        history_data = data
        if isinstance(data, dict):
             history_data = data.get("data", [])

        if not history_data or not isinstance(history_data, list):
             return pd.DataFrame()

        for item in history_data:
            try:
                if isinstance(item, list) and len(item) >= 2:
                    timestamp = float(item[0])
                    value = float(item[1])

                    if not timestamp:
                        continue

                    dt = datetime.fromtimestamp(timestamp / 1000)

                    records.append({
                        "Date": dt,
                        "Open": value,
                        "High": value,
                        "Low": value,
                        "Close": value,
                        "Volume": 0
                    })
                    continue

                date_str = item.get("date", "")
                if date_str:
                    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
                else:
                    continue

                records.append(
                    {
                        "Date": dt,
                        "Open": float(item.get("open", 0)),
                        "High": float(item.get("high", 0)),
                        "Low": float(item.get("low", 0)),
                        "Close": float(item.get("close", 0)),
                        "Volume": int(item.get("volume", 0)),
                    }
                )
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

        return df

    def get_index_info(self, index_code: str) -> dict[str, Any]:
        """
        Get current information for an index.

        .. deprecated::
            This method is deprecated since v0.5.1. Use TradingView provider
            via ``bp.Index(index_code).info`` instead.

        Args:
            index_code: Index code (e.g., "XU100").

        Returns:
            Dictionary with index information.
        """
        warnings.warn(
            "get_index_info is deprecated since v0.5.1. "
            "Use TradingView provider via bp.Index(index_code).info instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        index_code = index_code.upper()

        if index_code not in self.INDICES:
            raise TickerNotFoundError(f"Unknown index: {index_code}")

        # Use the same quote endpoint for indices
        quote = self.get_realtime_quote(index_code)
        quote["name"] = self.INDICES.get(index_code, index_code)
        quote["type"] = "index"

        return quote

    def _get_session_for_stock(self, symbol: str) -> None:
        """Initialize session by visiting stock page to get cookies."""
        stock_page_url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"
        try:
            # Just make a GET request to establish session cookies
            self._client.get(stock_page_url, timeout=10)
        except Exception:
            pass  # Ignore errors, we'll try the API anyway

    def _fetch_sermaye_data(self, symbol: str) -> dict:
        """
        Fetch dividend and capital increase data from İş Yatırım API.

        Returns combined data with both temettuList and sermayeList.
        """
        import json

        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # First establish session to get ASP.NET_SessionId cookie
        self._get_session_for_stock(symbol)

        url = f"{self.STOCK_INFO_URL}/GetSermayeArttirimlari"

        # ASP.NET WebMethod expects specific format
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}",
            "Origin": "https://www.isyatirim.com.tr",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        }

        # Correct payload format with hisseKodu
        payload = json.dumps({
            "hisseKodu": symbol,
            "hisseTanimKodu": "",
            "yil": 0,
            "zaman": "HEPSI",
            "endeksKodu": "09",
            "sektorKodu": "",
        })

        try:
            response = self._client.post(url, content=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise APIError(f"Failed to fetch sermaye data for {symbol}: {e}") from e

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """
        Get dividend history for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            DataFrame with columns: Date, Amount, GrossRate, TotalDividend.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:dividends:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._fetch_sermaye_data(symbol)
        except APIError:
            # Return empty DataFrame if API fails
            return pd.DataFrame(columns=["Amount", "GrossRate", "NetRate", "TotalDividend"])

        # Parse dividends from response
        df = self._parse_dividends(data)
        self._cache_set(cache_key, df, TTL.FINANCIAL_STATEMENTS)

        return df

    def _parse_sermaye_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse GetSermayeArttirimlari response into list of records."""
        import json

        # Response structure: {"d": "[{...}, {...}]"} - JSON string inside JSON
        d_value = data.get("d", "[]")

        if isinstance(d_value, str):
            try:
                return json.loads(d_value)
            except json.JSONDecodeError:
                return []
        elif isinstance(d_value, list):
            return d_value
        return []

    def _parse_dividends(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse dividend data from GetSermayeArttirimlari response."""
        records = []
        items = self._parse_sermaye_response(data)

        for item in items:
            try:
                # Filter for cash dividend type only: 04 (Nakit Temettü)
                tip = item.get("SHT_KODU", "")
                if tip != "04":
                    continue

                # Parse date from timestamp (milliseconds) - strip time
                timestamp = item.get("SHHE_TARIH", 0)
                if timestamp:
                    dt = datetime.fromtimestamp(timestamp / 1000).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    continue

                # Cash dividend rate and amount
                gross_rate = float(item.get("SHHE_NAKIT_TM_ORAN", 0) or 0)
                net_rate = float(item.get("SHHE_NAKIT_TM_ORAN_NET", 0) or 0)
                total_dividend = float(item.get("SHHE_NAKIT_TM_TUTAR", 0) or 0)

                # Calculate per-share amount (rate / 100 since it's percentage of nominal)
                amount = gross_rate / 100 if gross_rate else 0

                records.append(
                    {
                        "Date": dt,
                        "Amount": round(amount, 4),
                        "GrossRate": round(gross_rate, 2),
                        "NetRate": round(net_rate, 2),
                        "TotalDividend": total_dividend,
                    }
                )
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("Date", inplace=True)
            df.sort_index(ascending=False, inplace=True)

        return df

    def get_capital_increases(self, symbol: str) -> pd.DataFrame:
        """
        Get capital increase (split) history for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            DataFrame with columns: Date, Capital, RightsIssue, BonusFromCapital, BonusFromDividend.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:splits:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._fetch_sermaye_data(symbol)
        except APIError:
            # Return empty DataFrame if API fails
            return pd.DataFrame(columns=["Capital", "RightsIssue", "BonusFromCapital", "BonusFromDividend"])

        # Parse capital increases from response
        df = self._parse_capital_increases(data)
        self._cache_set(cache_key, df, TTL.FINANCIAL_STATEMENTS)

        return df

    def _parse_capital_increases(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse capital increase data from GetSermayeArttirimlari response."""
        records = []
        items = self._parse_sermaye_response(data)

        for item in items:
            try:
                # Filter for valid capital increase types:
                # - Type 01: Bedelli Sermaye Artırımı (rights issue only)
                # - Type 02: Bedelsiz Sermaye Artırımı (bonus issue from capital reserves)
                # - Type 03: Bedelli ve Bedelsiz Sermaye Artırımı (rights + bonus issue)
                # - Type 09: Bedelsiz Temettü (stock dividend / bonus from dividend)
                # Exclude: 04, 07, 79, 90, 99 (administrative/technical records)
                tip = item.get("SHT_KODU", "")
                if tip not in ("01", "02", "03", "09"):
                    continue

                # Parse date from timestamp (milliseconds) - strip time
                timestamp = item.get("SHHE_TARIH", 0)
                if timestamp:
                    dt = datetime.fromtimestamp(timestamp / 1000).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    continue

                # Get capital after increase
                capital = float(item.get("HSP_BOLUNME_SONRASI_SERMAYE", 0) or 0)

                # Rights issue rate (Bedelli)
                rights_issue = float(item.get("SHHE_BDLI_ORAN", 0) or 0)

                # Bonus from capital reserves (Bedelsiz İç Kaynak)
                bonus_capital = float(item.get("SHHE_BDSZ_IK_ORAN", 0) or 0)

                # Bonus from dividend (Bedelsiz Temettüden)
                bonus_dividend = float(item.get("SHHE_BDSZ_TM_ORAN", 0) or 0)

                records.append(
                    {
                        "Date": dt,
                        "Capital": capital,
                        "RightsIssue": round(rights_issue, 2),
                        "BonusFromCapital": round(bonus_capital, 2),
                        "BonusFromDividend": round(bonus_dividend, 2),
                    }
                )
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("Date", inplace=True)
            df.sort_index(ascending=False, inplace=True)

        return df

    def _parse_quote(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse OneEndeks response into standardized format."""
        last = float(data.get("last", 0))
        prev_close = float(data.get("dayClose", 0))
        change = last - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        update_str = data.get("updateDate", "")
        try:
            update_time = datetime.fromisoformat(update_str.replace("+03", "+03:00"))
        except (ValueError, AttributeError):
            update_time = datetime.now()

        return {
            "symbol": data.get("symbol", ""),
            "last": last,
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": prev_close,
            "volume": int(data.get("volume", 0)),
            "quantity": int(data.get("quantity", 0)),
            "bid": float(data.get("bid", 0)),
            "ask": float(data.get("ask", 0)),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "week_close": float(data.get("weekClose", 0)),
            "month_close": float(data.get("monthClose", 0)),
            "year_close": float(data.get("yearClose", 0)),
            "update_time": update_time,
        }

    # Maximum periods per API call (İş Yatırım limit)
    _MAX_PERIODS_PER_CALL = 4

    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str = "balance_sheet",
        quarterly: bool = False,
        financial_group: str | None = None,
        last_n: int | str | None = None,
    ) -> pd.DataFrame:
        """
        Get financial statements for a company.

        Args:
            symbol: Stock symbol.
            statement_type: Type of statement ("balance_sheet", "income_stmt", "cashflow").
            quarterly: If True, return quarterly data. If False, return annual data.
            financial_group: Financial group code (XI_29 for industrial, UFRS for banks).
            last_n: Number of periods to fetch. None for default (5), int for exact count,
                    "all" for maximum available (~15 annual, ~40 quarterly).

        Returns:
            DataFrame with financial data.

        Raises:
            ValueError: If last_n is invalid (0, negative, or unsupported string).
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Resolve count
        count = self._resolve_last_n(last_n, quarterly)

        # financial_group=None means "use the group the page has selected"
        # (THYAO -> XI_29, AKBNK -> UFRS_K), which preserves backward
        # compatibility for industrial companies.
        group_key = financial_group if financial_group is not None else "page"

        cache_key = (
            f"isyatirim:financial:{symbol}:{statement_type}:{quarterly}:"
            f"{count}:{group_key}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Generate all periods needed.
        #
        # NOTE: the page's own period dropdown is authoritative.  If a
        # requested period is not offered by the dropdown (young company, 2008
        # boundary) _fetch_financial_table simply returns fewer/empty columns
        # instead of inventing calendar periods.
        current_year = datetime.now().year
        periods = self._get_periods(current_year, quarterly, count=count)

        # The page shows at most 4 period columns; fetch in batches of 4.
        batches = [
            periods[i : i + self._MAX_PERIODS_PER_CALL]
            for i in range(0, len(periods), self._MAX_PERIODS_PER_CALL)
        ]

        # Network/HTTP failures propagate as APIError (fail-loud).  A batch
        # that parses to an empty DataFrame is a legitimate "no data in this
        # range" answer and is skipped.
        all_dfs = []
        for batch_periods in batches:
            df = self._fetch_financial_table(
                symbol=symbol,
                financial_group=financial_group,
                periods=batch_periods,
                quarterly=quarterly,
                statement_type=statement_type,
            )
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            # No malitablo table / no matching section: empty DataFrame.
            return pd.DataFrame()

        # Merge batches horizontally (same rows, different period columns).
        # Row names may legitimately repeat (e.g. "Ticari Alacaklar" appears
        # under current and non-current assets), so an index-based join would
        # multiply rows.  When the row index is identical, concatenate by
        # position; otherwise fall back to a label join (legacy behaviour).
        result = all_dfs[0]
        for extra_df in all_dfs[1:]:
            new_cols = [c for c in extra_df.columns if c not in result.columns]
            if not new_cols:
                continue
            if result.index.equals(extra_df.index):
                result = pd.concat([result, extra_df[new_cols]], axis=1)
            else:
                result = result.join(extra_df[new_cols], how="outer")

        # Sort columns: most recent first
        result = result[sorted(result.columns, key=self._period_sort_key, reverse=True)]

        self._cache_set(cache_key, result, TTL.FINANCIAL_STATEMENTS)

        return result

    @staticmethod
    def _resolve_last_n(last_n: int | str | None, quarterly: bool) -> int:
        """Resolve last_n parameter to an integer count."""
        if last_n is None:
            return 5
        if isinstance(last_n, str):
            if last_n.lower() == "all":
                return 40 if quarterly else 15
            raise ValueError(f"Invalid last_n value: {last_n!r}. Use an integer or 'all'.")
        if not isinstance(last_n, int) or last_n < 1:
            raise ValueError(f"last_n must be a positive integer or 'all', got {last_n!r}")
        return last_n

    @staticmethod
    def derive_standalone_quarters(quarterly_cum_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive standalone (additive) quarters from cumulative flow data.

        Turkish quarterly flow statements are year-to-date cumulative.  This
        helper lappings them into single-quarter values:

        * ``Q1 = cum(Q1)``
        * ``Qn = cum(Qn) - cum(Qn-1)`` within the same fiscal year
        * the series resets at the fiscal-year boundary (a new year's Q1 is
          that year's cumulative Q1)

        If either operand is NaN the result is NaN.  Negative standalone values
        are legitimate (provisions, seasonality) and are preserved — there is
        no clamping, abs or zeroing.  Column order and duplicate row labels of
        the input are preserved.

        Placement: this is an ``IsYatirimProvider`` static method because the
        cumulative convention is a provider/source property.  Independent
        verification against MKK CURR3 data is deferred to FA-01b.

        Args:
            quarterly_cum_df: DataFrame whose columns are ``YYYYQN`` period
                labels (any order) and whose index is the item names.

        Returns:
            DataFrame with the same index and columns, holding standalone
            quarter values.
        """
        if quarterly_cum_df is None or quarterly_cum_df.empty:
            return pd.DataFrame()

        parsed: dict[str, tuple[int, int] | None] = {}
        for column in quarterly_cum_df.columns:
            try:
                year_str, quarter_str = str(column).split("Q")
                parsed[column] = (int(year_str), int(quarter_str))
            except (ValueError, AttributeError):
                parsed[column] = None

        result = pd.DataFrame(index=quarterly_cum_df.index)
        for column in quarterly_cum_df.columns:
            period = parsed[column]
            if period is None:
                result[column] = float("nan")
                continue
            year, quarter = period
            current = pd.to_numeric(
                quarterly_cum_df[column], errors="coerce"
            ).to_numpy(dtype=float)
            if quarter == 1:
                result[column] = current
                continue
            previous_column = f"{year}Q{quarter - 1}"
            if previous_column not in quarterly_cum_df.columns:
                result[column] = float("nan")
                continue
            previous = pd.to_numeric(
                quarterly_cum_df[previous_column], errors="coerce"
            ).to_numpy(dtype=float)
            result[column] = current - previous

        return result[list(quarterly_cum_df.columns)]

    @staticmethod
    def _period_sort_key(col_name: str) -> tuple[int, int]:
        """Sort key for period column names. Returns (year, quarter) tuple.

        Handles both annual ('2024') and quarterly ('2024Q3') formats.
        """
        try:
            if "Q" in col_name:
                year_str, q_str = col_name.split("Q")
                return (int(year_str), int(q_str))
            return (int(col_name), 0)
        except (ValueError, IndexError):
            return (0, 0)

    def _get_periods(
        self,
        current_year: int,
        quarterly: bool,
        count: int = 5,
    ) -> list[tuple[int, int]]:
        """Generate period tuples (year, period) for financial queries.

        For quarterly data, starts from the last completed quarter.
        For annual data, starts from the previous year (current year data
        typically not available until Q1 of next year).

        Returns exactly ``count`` tuples.
        """
        periods = []
        if quarterly:
            # Determine last AVAILABLE quarter based on current month
            # Financial data has ~45-60 day publication delay after quarter end
            current_month = datetime.now().month
            #
            # Publication timeline (approximate):
            #   Q1 (Jan-Mar) data → published May/June
            #   Q2 (Apr-Jun) data → published Aug/Sep
            #   Q3 (Jul-Sep) data → published Nov/Dec
            #   Q4 (Oct-Dec) data → published Feb/Mar of next year
            #
            # So available data by month:
            #   Jan-Feb: Q3 of previous year is latest
            #   Mar-May: Q4 of previous year is latest
            #   Jun-Aug: Q1 of current year is latest
            #   Sep-Nov: Q2 of current year is latest
            #   Dec: Q3 of current year is latest
            #
            if current_month <= 2:
                start_year = current_year - 1
                start_period = 9
            elif current_month <= 5:
                start_year = current_year - 1
                start_period = 12
            elif current_month <= 8:
                start_year = current_year
                start_period = 3
            elif current_month <= 11:
                start_year = current_year
                start_period = 6
            else:
                start_year = current_year
                start_period = 9

            # Generate exactly `count` quarters going backward
            year = start_year
            period = start_period
            for _ in range(count):
                periods.append((year, period))
                period -= 3
                if period <= 0:
                    period = 12
                    year -= 1
        else:
            # Annual: start from previous year (current year data not ready)
            for i in range(count):
                periods.append((current_year - 1 - i, 12))
        return periods

    def _fetch_financial_table(
        self,
        symbol: str,
        financial_group: str | None,
        periods: list[tuple[int, int]],
        quarterly: bool = False,
        statement_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch up to 4 periods of the embedded malitablo table.

        The embedded table carries no itemCode; section headers are used to
        select rows instead (see :meth:`_parse_embedded_statement`).
        """
        html_text = self._get_statement_html(
            symbol=symbol,
            financial_group=financial_group,
            periods=periods,
        )
        if not html_text:
            return pd.DataFrame()

        return self._parse_embedded_statement(
            html_text, quarterly=quarterly, statement_type=statement_type
        )

    # ------------------------------------------------------------------
    # Embedded malitablo HTML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_malitablo_table(html_text: str) -> str | None:
        """Return the ``<table ... data-csvname="malitablo">...</table>`` slice."""
        for tag_match in re.finditer(r"<table\b[^>]*>", html_text, re.IGNORECASE):
            tag = tag_match.group(0)
            if "malitablo" not in tag.lower() or "excelexport" not in tag.lower():
                continue
            end = html_text.lower().find("</table>", tag_match.end())
            if end == -1:
                return None
            return html_text[tag_match.start() : end + len("</table>")]
        return None

    @staticmethod
    def _strip_cell(cell_html: str) -> str:
        """Strip nested tags/entities/whitespace from a table cell."""
        text = re.sub(r"<[^>]+>", "", cell_html)
        text = html_module.unescape(text)
        return text.strip()

    @staticmethod
    def _parse_number(text: str) -> Any:
        """
        Parse a source value cell.

        Dotted thousand separators are removed (``"512.191.000.000"`` ->
        ``512191000000.0``), ``"0"`` -> ``0.0``, empty -> ``NaN``, negatives
        and TMS-29 restated decimals are preserved.  A cell that still cannot
        be parsed is returned verbatim (source-faithful transfer) rather than
        being coerced to NaN.
        """
        if text is None:
            return float("nan")
        raw = str(text).strip()
        if not raw:
            return float("nan")
        compact = raw.replace("\u00a0", " ").strip()
        try:
            return float(compact.replace(".", "").replace(",", "."))
        except ValueError:
            pass
        try:
            return float(compact)
        except ValueError:
            return raw

    @staticmethod
    def _select_elements(html_text: str) -> list[dict[str, Any]]:
        """Parse ``<select>`` elements into id/name/options/selected dicts."""
        selects: list[dict[str, Any]] = []
        for tag, body in re.findall(
            r"<select\b([^>]*)>(.*?)</select>", html_text, re.IGNORECASE | re.DOTALL
        ):
            id_match = re.search(r'id=["\']([^"\']+)["\']', tag)
            name_match = re.search(r'name=["\']([^"\']+)["\']', tag)
            options = re.findall(
                r"<option\b([^>]*)>(.*?)</option>", body, re.IGNORECASE | re.DOTALL
            )
            values: list[str] = []
            selected: str | None = None
            for opt_tag, opt_body in options:
                value_match = re.search(r'value=["\']([^"\']*)["\']', opt_tag)
                value = (
                    html_module.unescape(value_match.group(1))
                    if value_match
                    else IsYatirimProvider._strip_cell(opt_body)
                )
                values.append(value)
                if selected is None and re.search(
                    r"\bselected\b", opt_tag, re.IGNORECASE
                ):
                    selected = value
            if selected is None and values:
                selected = values[0]
            selects.append(
                {
                    "id": id_match.group(1) if id_match else "",
                    "name": name_match.group(1) if name_match else "",
                    "values": values,
                    "selected": selected,
                }
            )
        return selects

    @classmethod
    def _period_selects(cls, html_text: str) -> list[dict[str, Any]]:
        """Return ddlMaliTabloDonem1..N selects ordered by their index."""
        found = []
        for select in cls._select_elements(html_text):
            match = re.fullmatch(r"ddlMaliTabloDonem(\d+)", select["id"])
            if match:
                found.append((int(match.group(1)), select))
        found.sort(key=lambda item: item[0])
        return [select for _, select in found]

    @classmethod
    def _selected_group(cls, html_text: str) -> tuple[str, str | None] | None:
        """Return ``(field_name, selected_value)`` for ddlMaliTabloGroup."""
        for select in cls._select_elements(html_text):
            if select["id"] == "ddlMaliTabloGroup":
                return select["name"], select["selected"]
        return None

    @staticmethod
    def _raw_period_label(year: int, period: int) -> str:
        """Map a (year, period) tuple to the page's ``YYYY/P`` label."""
        return f"{int(year)}/{int(period)}"

    @staticmethod
    def _period_column_label(raw_label: str, quarterly: bool) -> str | None:
        """Map a page ``YYYY/P`` label to a DataFrame column label.

        Quarterly mode: ``2026/6`` -> ``2026Q2`` (``period // 3``).  Annual
        mode emits only full-year (``/12``) columns, labelled ``2025``.
        Non-quarter period values are not representable and return None.
        """
        try:
            year_str, period_str = str(raw_label).split("/")
            year = int(year_str)
            period = int(period_str)
        except (ValueError, AttributeError):
            return None
        if quarterly:
            if period < 3:
                return None
            return f"{year}Q{period // 3}"
        if period == 12:
            return str(year)
        return None

    def _get_statement_html(
        self,
        symbol: str,
        financial_group: str | None = None,
        periods: list[tuple[int, int]] | None = None,
    ) -> str:
        """
        GET the company card page and, when needed, postback to its selects.

        Returns the page HTML.  Raises :class:`APIError` on network/HTTP
        failures and on structurally impossible postbacks; never returns
        silently reduced data.
        """
        url = f"{self.STATEMENT_PAGE_URL}?hisse={symbol}"
        try:
            response = self._client.get(url, timeout=15)
            response.raise_for_status()
            html_text = response.text
        except Exception as e:
            raise APIError(
                f"Failed to fetch financial statement page for {symbol}: {e}"
            ) from e

        updates: dict[str, str] = {}

        page_group = self._selected_group(html_text)
        if page_group is not None:
            group_name, current_group = page_group
            if financial_group is not None and financial_group != current_group:
                if not group_name:
                    raise APIError(
                        f"Cannot switch financial group for {symbol}: "
                        "ddlMaliTabloGroup has no form field name"
                    )
                updates[group_name] = financial_group
        elif financial_group is not None:
            raise APIError(
                f"Cannot select financial group {financial_group!r} for {symbol}: "
                "the page has no ddlMaliTabloGroup selector"
            )

        if periods:
            table_html = self._extract_malitablo_table(html_text) or html_text
            selects = self._period_selects(table_html)
            if selects:
                available = [value for s in selects for value in s["values"]]
                requested = [
                    self._raw_period_label(year, period)
                    for year, period in periods[: self._MAX_PERIODS_PER_CALL]
                ]
                target = [label for label in requested if label in available]
                if not target:
                    return ""
                used: list[str] = []
                for i, select in enumerate(selects):
                    if i < len(target):
                        value = target[i]
                    else:
                        value = select["selected"] or ""
                    if value in used:
                        replacement = next(
                            (v for v in select["values"] if v not in used), None
                        )
                        if replacement is not None:
                            value = replacement
                    if value:
                        used.append(value)
                    if not select["name"]:
                        raise APIError(
                            f"Cannot shift statement periods for {symbol}: "
                            f"{select['id']} has no form field name"
                        )
                    if value and value != select["selected"]:
                        updates[select["name"]] = value

        if updates:
            html_text = self._postback(url, html_text, updates, symbol)

        return html_text

    def _postback(
        self,
        url: str,
        html_text: str,
        updates: dict[str, str],
        symbol: str,
    ) -> str:
        """ASP.NET postback with the page's own form fields plus updates."""
        fields: dict[str, str] = {}
        for tag in re.findall(r"<input\b[^>]*>", html_text, re.IGNORECASE):
            name_match = re.search(r'name=["\']([^"\']+)["\']', tag)
            if not name_match:
                continue
            value_match = re.search(r'value=["\']([^"\']*)["\']', tag)
            fields[name_match.group(1)] = (
                html_module.unescape(value_match.group(1)) if value_match else ""
            )
        for select in self._select_elements(html_text):
            if select["name"]:
                fields.setdefault(select["name"], select["selected"] or "")

        if not fields:
            raise APIError(
                f"Financial statement postback failed for {symbol}: "
                "page carries no form fields"
            )
        if "__VIEWSTATE" not in fields and "__VIEWSTATEGENERATOR" not in fields:
            raise APIError(
                f"Financial statement postback failed for {symbol}: "
                "ASP.NET state fields (__VIEWSTATE/__EVENTVALIDATION) missing"
            )

        event_target = next(iter(updates))
        fields.update(updates)
        fields["__EVENTTARGET"] = event_target
        fields["__EVENTARGUMENT"] = ""

        try:
            response = self._client.post(url, data=fields, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise APIError(
                f"Financial statement postback failed for {symbol}: {e}"
            ) from e

    def _parse_embedded_statement(
        self,
        html_text: str,
        quarterly: bool = False,
        statement_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Parse the embedded malitablo table into a DataFrame.

        ``html_text`` may be the full page or the already-extracted table.
        Column labels come from the ``selected`` option of
        ``ddlMaliTabloDonem1..4`` (newest first).  Section titles are matched
        strip/case-tolerantly; ``arabaslik`` rows themselves are never emitted.
        """
        table_html = self._extract_malitablo_table(html_text)
        if table_html is None:
            if "arabaslik" not in html_text and "tbodyMTablo" not in html_text:
                return pd.DataFrame()
            table_html = html_text

        period_selects = self._period_selects(table_html)
        raw_periods = [
            s["selected"] for s in period_selects if s["selected"] is not None
        ]
        if not raw_periods:
            return pd.DataFrame()

        # Column label per source position (None = column not representable in
        # the requested mode, e.g. interim columns in annual mode).
        column_labels = [
            self._period_column_label(raw, quarterly) for raw in raw_periods
        ]
        emitted_columns = [
            index for index, label in enumerate(column_labels) if label is not None
        ]
        if not emitted_columns:
            return pd.DataFrame()

        if statement_type is None:
            selected_sections: set[str] | None = None
        else:
            sections = self._STATEMENT_SECTIONS.get(statement_type)
            selected_sections = set(sections) if sections is not None else None
        known_sections = {
            section
            for section_list in self._STATEMENT_SECTIONS.values()
            for section in section_list
        }

        row_names: list[str] = []
        row_values: list[list[Any]] = []
        current_section: str | None = None
        body_match = re.search(
            r"<tbody\b[^>]*>(.*?)</tbody>", table_html, re.IGNORECASE | re.DOTALL
        )
        body = body_match.group(1) if body_match else table_html

        for row_tag, row_body in re.findall(
            r"<tr\b([^>]*)>(.*?)</tr>", body, re.IGNORECASE | re.DOTALL
        ):
            cells = re.findall(
                r"<td\b[^>]*>(.*?)</td>", row_body, re.IGNORECASE | re.DOTALL
            )
            if not cells:
                continue
            if re.search(r"\barabaslik\b", row_tag, re.IGNORECASE):
                current_section = self._strip_cell(cells[0]).casefold()
                continue
            if current_section is None:
                continue
            if current_section not in known_sections:
                # Unknown arabaslik section: skipped for forward compatibility.
                continue
            if selected_sections is not None and current_section not in selected_sections:
                continue

            row_name = self._strip_cell(cells[0])
            values = [self._parse_number(self._strip_cell(c)) for c in cells[1:]]
            while len(values) < len(raw_periods):
                values.append(float("nan"))
            row_names.append(row_name)
            row_values.append([values[i] for i in emitted_columns])

        if not row_names:
            return pd.DataFrame()

        columns = [column_labels[i] for i in emitted_columns]
        df = pd.DataFrame(row_values, columns=columns)
        df.insert(0, "Item", row_names)
        df.set_index("Item", inplace=True)
        return df

    def _parse_financial_response(
        self,
        data: Any,
        periods: list[tuple[int, int]],
        quarterly: bool = False,
        statement_type: str | None = None,
    ) -> pd.DataFrame:
        """Parse MaliTablo API response into DataFrame.

        Args:
            data: Raw API response dict.
            periods: List of (year, period) tuples sent in this batch.
            quarterly: Whether these are quarterly periods. Passed explicitly
                       to avoid misdetection when a single-quarter batch has
                       period=12 (which looks like annual).
            statement_type: If given, filter items by itemCode prefix using
                            ITEM_CODE_PREFIXES mapping.
        """
        if not data or not isinstance(data, dict):
            return pd.DataFrame()

        # API returns: {"value": [{"itemDescTr": "...", "value1": ..., "value2": ...}, ...]}
        items = data.get("value", [])
        if not items:
            return pd.DataFrame()

        # Filter by itemCode prefix when statement_type is specified
        prefixes = self.ITEM_CODE_PREFIXES.get(statement_type) if statement_type else None
        if prefixes:
            items = [
                item for item in items
                if str(item.get("itemCode", "")).startswith(prefixes)
            ]
            if not items:
                return pd.DataFrame()

        records = []
        for item in items:
            row_name = item.get("itemDescTr", item.get("itemDescEng", "Unknown"))
            row_data = {"Item": row_name}

            for i, (year, period) in enumerate(periods[:self._MAX_PERIODS_PER_CALL], 1):
                if quarterly:
                    col_name = f"{year}Q{period // 3}"
                else:
                    col_name = str(year)
                value = item.get(f"value{i}")
                if value is not None:
                    try:
                        row_data[col_name] = float(value)
                    except (ValueError, TypeError):
                        row_data[col_name] = value

            records.append(row_data)

        df = pd.DataFrame(records)
        if not df.empty and "Item" in df.columns:
            df.set_index("Item", inplace=True)

        return df

    def get_major_holders(self, symbol: str) -> pd.DataFrame:
        """
        Get major shareholders (ortaklık yapısı) for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            DataFrame with columns: Holder, Percentage.
        """
        import json
        import re

        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:major_holders:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Fetch the stock page HTML
        stock_page_url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"

        try:
            response = self._client.get(stock_page_url, timeout=15)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            raise APIError(f"Failed to fetch major holders for {symbol}: {e}") from e

        # Parse JavaScript variable: var OrtaklikYapisidata = [{name: 'xxx', y: 50.88}, ...]
        pattern = r"var OrtaklikYapisidata = \[(.*?)\];"
        match = re.search(pattern, html, re.DOTALL)

        if not match:
            return pd.DataFrame(columns=["Holder", "Percentage"])

        js_array = match.group(1).strip()
        if not js_array:
            return pd.DataFrame(columns=["Holder", "Percentage"])

        # Convert JS object to valid JSON: {name: 'x'} -> {"name": "x"}
        json_str = re.sub(r"([{,])(\w+):", r'\1"\2":', js_array)
        json_str = json_str.replace("'", '"')

        try:
            data = json.loads(f"[{json_str}]")
        except json.JSONDecodeError:
            return pd.DataFrame(columns=["Holder", "Percentage"])

        records = []
        for item in data:
            holder = item.get("name", "Unknown")
            percentage = float(item.get("y", 0))
            records.append({"Holder": holder, "Percentage": round(percentage, 2)})

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("Holder", inplace=True)

        self._cache_set(cache_key, df, TTL.FINANCIAL_STATEMENTS)
        return df

    def get_recommendations(self, symbol: str) -> dict[str, Any]:
        """
        Get analyst recommendations and target price for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dictionary with:
            - recommendation: Buy/Hold/Sell (AL/TUT/SAT)
            - target_price: Analyst target price
            - upside_potential: Expected upside (%)
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:recommendations:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._fetch_sermaye_data(symbol)
        except APIError:
            return {
                "recommendation": None,
                "target_price": None,
                "upside_potential": None,
            }

        # Parse recommendations from sermaye response
        items = self._parse_sermaye_response(data)

        result = {
            "recommendation": None,
            "target_price": None,
            "upside_potential": None,
        }

        # Get the most recent entry with recommendation data
        for item in items:
            oneri = item.get("ONERI")
            hedef_fiyat = item.get("HEDEF_FIYAT")
            getiri_pot = item.get("GETIRI_POT")

            if oneri:
                result["recommendation"] = oneri
            if hedef_fiyat:
                result["target_price"] = round(float(hedef_fiyat), 2)
            if getiri_pot:
                result["upside_potential"] = round(float(getiri_pot) * 100, 2)

            # Break on first item with data
            if oneri or hedef_fiyat:
                break

        self._cache_set(cache_key, result, TTL.REALTIME_PRICE)
        return result

    def get_company_metrics(self, symbol: str) -> dict[str, Any]:
        """
        Get company metrics from şirket kartı page (Cari Değerler).

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dictionary with:
            - market_cap: Market capitalization (TL)
            - pe_ratio: Price/Earnings ratio (F/K)
            - pb_ratio: Price/Book ratio (PD/DD)
            - ev_ebitda: Enterprise Value/EBITDA (FD/FAVÖK)
            - free_float: Free float percentage
            - foreign_ratio: Foreign ownership percentage
            - net_debt: Net debt (TL)
        """
        import re

        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:metrics:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        stock_page_url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"

        # Retry logic for unstable İş Yatırım connection
        import time
        max_retries = 3
        last_error = None
        html = None

        for attempt in range(max_retries):
            try:
                response = self._client.get(stock_page_url, timeout=15)
                response.raise_for_status()
                html = response.text
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 1s, 2s backoff
                continue

        if html is None:
            raise APIError(f"Failed to fetch company metrics for {symbol}: {last_error}") from last_error

        result: dict[str, Any] = {
            "market_cap": None,
            "pe_ratio": None,
            "pb_ratio": None,
            "ev_ebitda": None,
            "free_float": None,
            "foreign_ratio": None,
            "net_debt": None,
        }

        # Find Cari Değerler section
        idx = html.find("Cari Değerler")
        if idx > 0:
            snippet = html[idx : idx + 3000]

            # Parse th/td pairs
            pattern = r"<th[^>]*>([^<]+)</th>\s*<td[^>]*>([^<]+)</td>"
            matches = re.findall(pattern, snippet)

            for label, value in matches:
                label = label.strip()
                value = value.strip().replace(".", "").replace(",", ".")

                try:
                    if "F/K" in label and "FD" not in label:
                        result["pe_ratio"] = float(value)
                    elif "PD/DD" in label:
                        result["pb_ratio"] = float(value)
                    elif "FD/FAVÖK" in label:
                        result["ev_ebitda"] = float(value)
                    elif "Piyasa Değeri" in label:
                        # Value is in mnTL, convert to TL
                        num = float(re.sub(r"[^\d.]", "", value))
                        result["market_cap"] = int(num * 1_000_000)
                    elif "Net Borç" in label:
                        num = float(re.sub(r"[^\d.]", "", value))
                        result["net_debt"] = int(num * 1_000_000)
                    elif "Halka Açıklık" in label:
                        result["free_float"] = float(re.sub(r"[^\d.]", "", value))
                    elif "Yabancı Oranı" in label:
                        result["foreign_ratio"] = float(re.sub(r"[^\d.]", "", value))
                except (ValueError, TypeError):
                    continue

        self._cache_set(cache_key, result, TTL.REALTIME_PRICE)
        return result

    def get_business_summary(self, symbol: str) -> str | None:
        """
        Get business summary (Faal Alanı) for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Business summary text or None if not available.
        """
        import re

        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        cache_key = f"isyatirim:business_summary:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        stock_page_url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={symbol}"

        try:
            response = self._client.get(stock_page_url, timeout=15)
            response.raise_for_status()
            html = response.text
        except Exception:
            return None

        # Find "Faal Alanı" table row
        pattern = r'<th[^>]*>Faal Alanı</th>\s*<td[^>]*>([^<]+)</td>'
        match = re.search(pattern, html)

        if not match:
            return None

        summary = match.group(1).strip()
        if not summary:
            return None

        self._cache_set(cache_key, summary, TTL.FINANCIAL_STATEMENTS)
        return summary


# Singleton instance
_provider: IsYatirimProvider | None = None


def get_isyatirim_provider() -> IsYatirimProvider:
    """Get the singleton İş Yatırım provider instance."""
    global _provider
    if _provider is None:
        _provider = IsYatirimProvider()
    return _provider
