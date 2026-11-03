"""KAP (Kamuyu Aydınlatma Platformu) provider for disclosures and calendar."""

import json
import re
import time
from datetime import datetime, timedelta

import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.exceptions import APIError

# Next.js SSR flight data embeds the BIST company list as pushed chunks:
#   self.__next_f.push([1,"..."])
# Each chunk is a JSON string literal that must be unescaped before parsing.
_NEXT_F_PUSH_RE = re.compile(
    r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)',
    re.DOTALL,
)

# Member blocks live inside the decoded payload with normal quotes.
# Only blocks that carry a ``stockCode`` field are company records; raw
# ``mkkMemberOid`` blocks without a stock code are other MKK members.
_MEMBER_BLOCK_RE = re.compile(
    r'\{"mkkMemberOid":"([^"]+)","kapMemberTitle":"([^"]*)",'
    r'"relatedMemberTitle":(?:"[^"]*"|null),"stockCode":"([^"]+)"'
    r'(?:,"cityName":(?:"([^"]*)"|null))?',
)


def _extract_next_f_payload(html: str) -> str:
    """Collect and unescape all ``__next_f.push`` chunks in document order."""
    chunks = _NEXT_F_PUSH_RE.findall(html)
    if not chunks:
        raise RuntimeError(
            "KAP SSR schema changed: no __next_f.push chunks found in page"
        )
    return "".join(json.loads(chunk) for chunk in chunks)


def _decode_member_blocks(html: str) -> list[dict]:
    """Decode BIST company member blocks from the SSR page HTML.

    Returns records shaped as ``{"oid", "name", "tickers", "city"}``. Only
    blocks that contain a ``stockCode`` field count as company records.
    """
    payload = _extract_next_f_payload(html)
    matches = _MEMBER_BLOCK_RE.findall(payload)
    if not matches:
        raise RuntimeError(
            "KAP SSR schema changed: no company member blocks found in payload"
        )

    records: list[dict] = []
    seen_oids: set[str] = set()
    for oid, name, codes, city in matches:
        if oid in seen_oids:
            # Deterministic: first record wins (duplicates absent in fixture).
            continue
        seen_oids.add(oid)
        tickers = [code.strip() for code in codes.split(",") if code.strip()]
        records.append(
            {
                "oid": oid,
                "name": name,
                "tickers": tickers,
                "city": city or "",
            }
        )
    return records


class KAPProvider(BaseProvider):
    """
    Provider for KAP (Kamuyu Aydınlatma Platformu) data.

    KAP is the official disclosure platform for publicly traded
    companies in Turkey, similar to SEC EDGAR in the US.

    Provides:
    - List of all BIST companies with ticker codes
    - Company search functionality
    - Company disclosures (bildirimler)
    - Expected disclosure calendar (beklenen bildirimler)
    """

    BIST_COMPANIES_URL = "https://www.kap.org.tr/tr/bist-sirketler"
    DISCLOSURE_URL = "https://www.kap.org.tr/tr/bildirim-sorgu-sonuc"
    CALENDAR_API_URL = "https://kap.org.tr/tr/api/expected-disclosure-inquiry/company"
    COMPANY_INFO_URL = "https://kap.org.tr/tr/sirket-bilgileri/ozet"
    COMPANY_GENERAL_URL = "https://kap.org.tr/tr/sirket-bilgileri/genel"
    CACHE_DURATION = 86400  # 24 hours

    def __init__(self):
        super().__init__()
        self._company_cache: pd.DataFrame | None = None
        self._cache_time: float = 0
        self._oid_map: dict[str, str] | None = None
        self._oid_cache_time: float = 0
        self._company_details_cache: dict[str, dict] = {}
        self._company_details_cache_time: dict[str, float] = {}

    def get_companies(self) -> pd.DataFrame:
        """
        Get list of all BIST companies.

        Returns:
            DataFrame with columns: ticker, name, city
        """
        current_time = time.time()

        # Check cache
        if (
            self._company_cache is not None
            and (current_time - self._cache_time) < self.CACHE_DURATION
        ):
            return self._company_cache

        try:
            response = self._client.get(self.BIST_COMPANIES_URL, timeout=20)
            response.raise_for_status()
        except Exception as e:
            raise APIError(f"Failed to fetch company list: {e}") from e

        records = _decode_member_blocks(response.text)

        rows = []
        for record in records:
            for ticker in record["tickers"]:
                rows.append(
                    {
                        "ticker": ticker,
                        "name": record["name"],
                        "city": record["city"],
                    }
                )

        if not rows:
            raise RuntimeError(
                "KAP company decode produced no rows; refusing empty company list"
            )

        result = pd.DataFrame(rows, columns=["ticker", "name", "city"])
        self._company_cache = result
        self._cache_time = current_time
        return result

    def search(self, query: str) -> pd.DataFrame:
        """
        Search companies by name or ticker.

        Args:
            query: Search query (ticker code or company name)

        Returns:
            DataFrame with matching companies
        """
        if not query:
            return pd.DataFrame(columns=["ticker", "name", "city"])

        companies = self.get_companies()
        if companies.empty:
            return companies

        query_normalized = self._normalize_text(query)
        query_upper = query.upper()

        # Score and filter results
        results = []
        for _, row in companies.iterrows():
            score = 0
            ticker = row["ticker"]
            name = row["name"]

            # Exact ticker match
            if ticker.upper() == query_upper:
                score = 1000
            # Ticker starts with query
            elif ticker.upper().startswith(query_upper):
                score = 500
            # Name contains query
            elif query_normalized in self._normalize_text(name):
                score = 100

            if score > 0:
                results.append((score, row))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        if not results:
            return pd.DataFrame(columns=["ticker", "name", "city"])

        return pd.DataFrame([r[1] for r in results])

    def _normalize_text(self, text: str) -> str:
        """Normalize Turkish text for comparison."""
        tr_map = str.maketrans("İıÖöÜüŞşÇçĞğ", "iioouussccgg")
        normalized = text.translate(tr_map).lower()
        # Remove common suffixes
        normalized = re.sub(r"[\.,']|\s+a\.s\.?|\s+anonim sirketi", "", normalized)
        return normalized.strip()

    def get_member_oid(self, symbol: str) -> str | None:
        """
        Get KAP member OID (mkkMemberOid) for a stock symbol.

        The member OID is required to query disclosures from KAP.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Member OID string or None if not found.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")
        current_time = time.time()

        # Check cache
        if (
            self._oid_map is not None
            and (current_time - self._oid_cache_time) < self.CACHE_DURATION
        ):
            return self._oid_map.get(symbol)

        # Fetch BIST companies list from KAP. Only network errors are swallowed
        # (frozen behaviour); parse errors must surface loudly.
        try:
            response = self._client.get(self.BIST_COMPANIES_URL, timeout=20)
            response.raise_for_status()
        except Exception:
            return None

        oid_map: dict[str, str] = {}
        for record in _decode_member_blocks(response.text):
            for ticker in record["tickers"]:
                oid_map[ticker] = record["oid"]

        self._oid_map = oid_map
        self._oid_cache_time = current_time
        return oid_map.get(symbol)

    def get_disclosures(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """
        Get KAP disclosures (bildirimler) for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").
            limit: Maximum number of disclosures to return (default: 20).

        Returns:
            DataFrame with columns: Date, Title, URL.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return pd.DataFrame(columns=["Date", "Title", "URL"])

        # Fetch disclosures from KAP
        disc_url = f"{self.DISCLOSURE_URL}?member={member_oid}"

        try:
            response = self._client.get(disc_url, timeout=15)
            response.raise_for_status()

            # Parse disclosures from Next.js embedded data
            # Format: publishDate\\":\\"29.12.2025 19:21:18\\",\\"disclosureIndex\\":1530826...
            pattern = (
                r'publishDate\\":\\"([^\\"]+)\\".*?'
                r'disclosureIndex\\":(\d+).*?'
                r'title\\":\\"([^\\"]+)\\"'
            )
            matches = re.findall(pattern, response.text, re.DOTALL)

            records = []
            for date, idx, title in matches[:limit]:
                url = f"https://www.kap.org.tr/tr/Bildirim/{idx}"
                records.append({
                    "Date": date,
                    "Title": title,
                    "URL": url,
                })

            return pd.DataFrame(records)

        except Exception as e:
            raise APIError(f"Failed to fetch disclosures for {symbol}: {e}") from e

    def get_calendar(self, symbol: str) -> pd.DataFrame:
        """
        Get expected disclosure calendar for a stock from KAP.

        Returns upcoming expected disclosures like financial reports,
        annual reports, sustainability reports, and corporate governance reports.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            DataFrame with columns:
            - StartDate: Expected disclosure window start
            - EndDate: Expected disclosure window end
            - Subject: Type of disclosure (e.g., "Finansal Rapor")
            - Period: Report period (e.g., "Yıllık", "3 Aylık")
            - Year: Fiscal year
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return pd.DataFrame(columns=["StartDate", "EndDate", "Subject", "Period", "Year"])

        # Calculate date range: today to 6 months from now
        now = datetime.now()
        start_date = now.strftime("%Y-%m-%d")
        end_date = (now + timedelta(days=180)).strftime("%Y-%m-%d")

        # Fetch expected disclosures from KAP API
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": "https://kap.org.tr",
            "Referer": "https://kap.org.tr/tr/beklenen-bildirim-sorgu",
        }
        payload = {
            "startDate": start_date,
            "endDate": end_date,
            "memberTypes": ["IGS"],
            "mkkMemberOidList": [member_oid],
            "disclosureClass": "",
            "subjects": [],
            "mainSector": "",
            "sector": "",
            "subSector": "",
            "market": "",
            "index": "",
            "year": "",
            "term": "",
            "ruleType": "",
        }

        try:
            response = self._client.post(
                self.CALENDAR_API_URL,
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            records = []
            for item in data:
                records.append({
                    "StartDate": item.get("startDate", ""),
                    "EndDate": item.get("endDate", ""),
                    "Subject": item.get("subject", ""),
                    "Period": item.get("ruleTypeTerm", ""),
                    "Year": item.get("year", ""),
                })

            return pd.DataFrame(records)

        except Exception as e:
            raise APIError(f"Failed to fetch calendar for {symbol}: {e}") from e

    def get_company_details(self, symbol: str) -> dict:
        """
        Get company details from KAP company info page.

        Scrapes the KAP company page for sector, market, and website information.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dict with keys:
            - sector: Company sector (e.g., "ULAŞTIRMA VE DEPOLAMA")
            - market: Stock market (e.g., "YILDIZ PAZAR")
            - website: Company website URL(s)
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")
        current_time = time.time()

        # Check cache
        if symbol in self._company_details_cache:
            cache_time = self._company_details_cache_time.get(symbol, 0)
            if (current_time - cache_time) < self.CACHE_DURATION:
                return self._company_details_cache[symbol]

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return {}

        # Fetch company info page
        url = f"{self.COMPANY_INFO_URL}/{member_oid}"

        try:
            response = self._client.get(url, timeout=15)
            response.raise_for_status()
            html = response.text

            result = {}

            # Extract sector: href="/tr/Sektorler?sector=...">SECTOR_NAME</a>
            sector_match = re.search(
                r'href="/tr/Sektorler\?sector=[^"]*">([^<]+)</a>',
                html
            )
            if sector_match:
                result["sector"] = sector_match.group(1).strip()

            # Extract market: href="/tr/Pazarlar?market=...">MARKET_NAME</a>
            market_match = re.search(
                r'href="/tr/Pazarlar\?market=[^"]*">([^<]+)</a>',
                html
            )
            if market_match:
                result["market"] = market_match.group(1).strip()

            # Extract website: after "İnternet Adresi" label
            # Pattern: <h3...>İnternet Adresi</h3><p class="...">WEBSITE</p>
            website_match = re.search(
                r'İnternet Adresi</h3><p[^>]*>([^<]+)</p>',
                html
            )
            if website_match:
                result["website"] = website_match.group(1).strip()

            # Get business summary from genel page
            business_summary = self._get_business_summary(member_oid)
            if business_summary:
                result["businessSummary"] = business_summary

            # Cache result
            self._company_details_cache[symbol] = result
            self._company_details_cache_time[symbol] = current_time

            return result

        except Exception:
            return {}

    def _get_business_summary(self, member_oid: str) -> str | None:
        """Get business summary (Faaliyet Konusu) from KAP genel page."""
        url = f"{self.COMPANY_GENERAL_URL}/{member_oid}"

        try:
            response = self._client.get(url, timeout=15)
            if response.status_code != 200:
                return None

            text = response.text

            # Find the faaliyet_konu section
            key_idx = text.find("kpy41_acc2_faaliyet_konu")
            if key_idx < 0:
                return None

            # Extract chunk - stop at next section (kpy41_acc2_sure or similar)
            remaining = text[key_idx:]
            next_section = re.search(r'kpy41_acc2_(?!faaliyet)', remaining)
            if next_section:
                chunk = remaining[: next_section.start()]
            else:
                chunk = remaining[:5000]

            # Format 1: HTML-encoded value field (e.g., AKBNK, THYAO)
            # Pattern: value\":\"...\",\"disclosureIndex
            value_match = re.search(
                r'value\\":\\"(.*?)\\",\\"disclosureIndex', chunk, re.DOTALL
            )
            if value_match:
                value = value_match.group(1)
                if "u003e" in value:
                    # Decode HTML entities
                    decoded = value.replace("\\u003e", ">").replace("\\u003c", "<")
                    decoded = decoded.replace("u003e", ">").replace("u003c", "<")
                    # Extract text between > and <
                    text_parts = re.findall(r">([^<>]+)<", decoded)
                    meaningful = [
                        t.replace("\\n", " ").replace("\\\\n", " ").strip()
                        for t in text_parts
                        if len(t.strip()) > 10
                    ]
                    if meaningful:
                        summary = " ".join(meaningful)
                        # Clean up escape sequences
                        summary = summary.replace("\\ ", " ").replace("\\", "")
                        summary = re.sub(r"\s+", " ", summary).strip()
                        if len(summary) > 10:
                            return summary

            # Format 2: React children patterns
            texts = []
            skip_texts = ("Bulunmamaktadır.", "-", "Şirketin Süresi")

            # Children array: \"children\":[\"text\"]
            array_matches = re.findall(r'\\"children\\":\[\\"([^\\"]{10,})\\"', chunk)
            for m in array_matches:
                if not m.startswith("$") and m not in skip_texts:
                    texts.append(m)

            # Children string: \"children\":\"text\"
            string_matches = re.findall(r'\\"children\\":\\"([^\\"]{10,})\\"', chunk)
            for m in string_matches:
                if not m.startswith("$") and m not in skip_texts:
                    texts.append(m)

            if texts:
                summary = " ".join(texts)
                summary = summary.replace("\\n", " ").strip()
                summary = re.sub(r"\s+", " ", summary)
                if len(summary) > 10:
                    return summary

            return None
        except Exception:
            return None

    def get_disclosure_content(self, disclosure_id: int | str) -> str | None:
        """
        Get disclosure HTML content by ID.

        Args:
            disclosure_id: KAP disclosure ID (e.g., 1530826).

        Returns:
            Raw HTML body content or None if failed.
        """
        url = f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"

        try:
            response = self._client.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception:
            return None


# Singleton
_provider: KAPProvider | None = None


def get_kap_provider() -> KAPProvider:
    """Get singleton provider instance."""
    global _provider
    if _provider is None:
        _provider = KAPProvider()
    return _provider
