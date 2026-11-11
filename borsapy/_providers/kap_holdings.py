"""KAP Holdings provider for fund portfolio holdings from disclosure PDFs.

This module extracts detailed portfolio holdings (individual stocks, ETFs, etc.)
from KAP "Portföy Dağılım Raporu" (Portfolio Distribution Report) disclosures.

Two parsing paths exist:

* ``parse_pdr_pdf`` — deterministic, offline, coordinate-based parsing of the
  PDF table (default). No network, no API key and no LLM.
* ``_parse_holdings_with_llm`` — legacy opt-in path using pymupdf4llm and the
  OpenRouter LLM (``parser="llm"``).
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.exceptions import APIError, DataNotAvailableError

# Portfolio Distribution Report disclosure type ID
PORTFOLIO_REPORT_DISCLOSURE_TYPE = "8aca490d502e34b801502e380044002b"

# Cache duration
CACHE_DURATION = 3600  # 1 hour

# OpenRouter default model
OPENROUTER_MODEL = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Deterministic PDR parser constants (BHE-tipi Ak Portföy şablonu)
#
# Source: 075g evidence (seal `gecmis baglam/fon_hatti_smoke_2026_11_10/`),
# measured on BHE donem=44 (idx 1696209) and donem=40 (idx 1680169).
# These x-bands are the column boundaries of that template as printed on the
# page. Other templates (ROTA/ONE/Deniz, bank/insurance, VIOP funds) were NOT
# measured; behaviour there is UNVERIFIED.
# ---------------------------------------------------------------------------
PDR_ISIN_RE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
PDR_TICKER_RE = re.compile(r"[A-Z]{3,6}")
PDR_PERIOD_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})")

#: y grouping tolerance (points), same value used by the 075g measurement.
PDR_ROW_TOLERANCE = 2.5
#: a holding's fields may sit on neighbouring y-lines (multi-line blocks and
#: small y drift), so tokens within this window of the ISIN anchor are merged.
PDR_ROW_WINDOW = 9.0

PDR_BAND_TICKER = (0.0, 60.0)       # MENKUL KIYMET CİNSİ karşılığı ticker
PDR_BAND_CURRENCY = (60.0, 115.0)   # DÖVİZ CİNSİ (TL)
PDR_BAND_NAME = (115.0, 205.0)      # İHRAÇCI KURUM (kurum adı)
PDR_BAND_ISIN = (200.0, 365.0)      # ISIN KODU
PDR_BAND_NOMINAL = (360.0, 416.0)   # NOMİNAL DEĞER
PDR_BAND_PRICE = (418.0, 460.0)     # BİRİM FİYATI
PDR_BAND_MARKET_VALUE = (748.0, 806.0)  # TOPLAM DEĞER (TL)
PDR_BAND_WEIGHT = (838.0, 882.0)    # TOPLAM (FPD GÖRE) — ağırlık %

PDR_CURRENCY_TOKENS = frozenset({"TL", "USD", "EUR", "GBP", "CHF", "JPY"})


class PdrParseError(ValueError):
    """Raised when a PDR PDF payload cannot be parsed deterministically."""


def _parse_turkish_number(text: str) -> float | None:
    """
    Parse a number printed in Turkish format (thousands '.', decimals ',').

    Explicit replace rule, no locale library:
        "23.294,00"  -> 23294.0
        "376,401100" -> 376.4011

    Returns None when the token is not a number (never guesses a value).
    """
    cleaned = text.strip().replace("%", "").replace(" ", "")
    if not cleaned or not re.fullmatch(r"[+-]?\d[\d.,]*", cleaned):
        return None
    try:
        if "," in cleaned:
            # Turkish: '.' is a thousands separator, ',' is the decimal mark.
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", cleaned):
            # Integer printed with thousands separators only.
            cleaned = cleaned.replace(".", "")
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class Holding:
    """Represents a single holding in a fund portfolio."""

    symbol: str
    isin: str | None
    name: str | None
    weight: float
    holding_type: str  # 'stock', 'etf', 'fund', 'viop', 'viop_cash', 'reverse_repo', 'term_deposit'
    country: str | None = None
    nominal_value: float | None = None
    market_value: float | None = None
    unit_price: float | None = None


@dataclass(frozen=True)
class PdrParseResult:
    """Deterministic result of parsing one PDR PDF."""

    holdings: tuple[Holding, ...]
    unparsed_rows: tuple[str, ...]
    page_count: int
    period_range: tuple[str, str] | None


class KAPHoldingsProvider(BaseProvider):
    """
    Provider for extracting fund portfolio holdings from KAP disclosures.

    Uses OpenRouter LLM to parse portfolio distribution PDFs.

    Example:
        >>> provider = KAPHoldingsProvider()
        >>> holdings = provider.get_holdings("YAY", api_key="sk-or-v1-...")
        >>> for h in holdings:
        ...     print(f"{h.symbol}: {h.weight:.2f}%")
    """

    # KAP API endpoints
    DISCLOSURE_FILTER_URL = "https://kap.org.tr/tr/api/disclosure/filter/FILTERYFBF"
    DISCLOSURE_PAGE_URL = "https://kap.org.tr/tr/Bildirim"
    FILE_DOWNLOAD_URL = "https://kap.org.tr/tr/api/file/download"
    FUND_INFO_URL = "https://www.kap.org.tr/tr/fon-bilgileri/genel"

    def __init__(self):
        super().__init__()
        self._fund_id_cache: dict[str, str] = {}
        self._holdings_cache: dict[str, tuple[list[Holding], datetime]] = {}
        self._holdings_cache_time: dict[str, float] = {}

    def get_fund_id(self, fund_code: str) -> str | None:
        """
        Get KAP fund ID for a fund code.

        Uses the KAP fund info page linked from TEFAS to extract the fund's objId.

        Args:
            fund_code: TEFAS fund code (e.g., "YAY", "AAK")

        Returns:
            KAP fund ID or None if not found.
        """
        fund_code = fund_code.upper()

        # Check cache
        if fund_code in self._fund_id_cache:
            return self._fund_id_cache[fund_code]

        try:
            # Get fund info from TEFAS which includes KAP link
            from borsapy._providers.tefas import get_tefas_provider

            tefas = get_tefas_provider()
            info = tefas.get_fund_detail(fund_code)

            kap_link = info.get("kap_link")
            if not kap_link:
                return None

            # Fetch KAP fund info page
            response = self._client.get(kap_link, timeout=30)
            response.raise_for_status()

            # Extract objId from page data. KAP may emit the hex id in either
            # case, and the source casing must be preserved as-is: callers use
            # the id verbatim in the disclosure filter URL path, which is
            # case-sensitive.
            # Patterns seen in Next.js payloads (escaped quotes):
            #   objId\":\"4028328c8b2f3d4e5f60718293a4b5c6\"  (lowercase)
            #   objId\":\"33E5FED7E50300EAE0530A4A622B2AEA\"  (uppercase)
            match = re.search(
                r'objId[\\"\':]+([A-Fa-f0-9]{32})(?![A-Fa-f0-9])', response.text
            )
            if match:
                fund_id = match.group(1)
                self._fund_id_cache[fund_code] = fund_id
                return fund_id

            return None

        except Exception:
            return None

    def get_disclosures(
        self,
        fund_code: str,
        days: int = 365,
    ) -> list[dict[str, Any]]:
        """
        Get portfolio distribution report disclosures for a fund.

        Args:
            fund_code: TEFAS fund code (e.g., "YAY")
            days: Number of days to look back (default: 365)

        Returns:
            List of disclosures with disclosure_id, disclosure_index, publish_date, etc.
        """
        fund_id = self.get_fund_id(fund_code)
        if not fund_id:
            raise DataNotAvailableError(f"Fund not found in KAP: {fund_code}")

        url = f"{self.DISCLOSURE_FILTER_URL}/{fund_id}/{PORTFOLIO_REPORT_DISCLOSURE_TYPE}/{days}"

        try:
            response = self._client.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            disclosures = []
            for item in data:
                basic = item.get("disclosureBasic", {})
                disclosures.append({
                    "disclosure_id": basic.get("disclosureId"),
                    "disclosure_index": basic.get("disclosureIndex"),
                    "publish_date": basic.get("publishDate"),
                    "title": basic.get("title"),
                    "summary": basic.get("summary"),
                    "year": basic.get("year"),
                    "period": basic.get("donem"),  # Month number
                    "attachment_count": basic.get("attachmentCount", 0),
                })

            return disclosures

        except Exception as e:
            raise APIError(f"Failed to fetch disclosures for {fund_code}: {e}") from e

    def get_latest_disclosure(self, fund_code: str) -> dict[str, Any] | None:
        """Get the most recent portfolio distribution report disclosure."""
        disclosures = self.get_disclosures(fund_code, days=365)
        if disclosures:
            return disclosures[0]  # Most recent first
        return None

    def _get_file_id(self, disclosure_index: int) -> str | None:
        """
        Extract file ID from disclosure page HTML.

        KAP emits the 32-char hex id in either casing and the id is used
        verbatim in the file download URL path, which is case-sensitive, so
        the source casing is preserved (no normalization).

        Patterns seen: ``file/download/4028328ca17e4f5e01a1c10c9cfc2f3b``
        (lowercase) and ``file/download/33E5FED7E50300EAE0530A4A622B2AEA``
        (uppercase).

        Args:
            disclosure_index: KAP disclosure index number

        Returns:
            File ID (32 hex chars, source casing kept) or None if not found.
        """
        url = f"{self.DISCLOSURE_PAGE_URL}/{disclosure_index}"

        try:
            response = self._client.get(url, timeout=30)
            response.raise_for_status()

            match = re.search(
                r"file/download/([A-Fa-f0-9]{32})(?![A-Fa-f0-9])", response.text
            )
            if match:
                return match.group(1)

            return None

        except Exception:
            return None

    def _download_pdf(self, file_id: str) -> bytes | None:
        """
        Download and extract PDF from Java-serialized wrapper.

        KAP file downloads return data wrapped in Java serialization format.
        The actual PDF starts at byte offset ~27 with the %PDF- marker.

        Args:
            file_id: KAP file ID (32 hex chars)

        Returns:
            Raw PDF bytes or None if failed.
        """
        url = f"{self.FILE_DOWNLOAD_URL}/{file_id}"

        try:
            response = self._client.get(url, timeout=60)
            response.raise_for_status()

            data = response.content

            # Find %PDF- marker and extract PDF
            pdf_start = data.find(b"%PDF-")
            if pdf_start != -1:
                return data[pdf_start:]

            return None

        except Exception:
            return None

    def _pdr_rows(self, page) -> list[list[tuple[float, float, str]]]:
        """Group page words into visual rows: y-bucket, then ordered by x."""
        buckets: dict[int, list[tuple[float, float, str]]] = {}
        for x0, y0, _x1, _y1, text, *_ in page.get_text("words"):
            key = int(round(y0 / PDR_ROW_TOLERANCE))
            buckets.setdefault(key, []).append((x0, y0, text))
        return [
            sorted(buckets[key], key=lambda token: (token[0], token[1], token[2]))
            for key in sorted(buckets)
        ]

    def _rows_by_y(self, page) -> list[list[str]]:
        """Visual rows of a page as plain word lists (evidence-compatible)."""
        return [[token[2] for token in row] for row in self._pdr_rows(page)]

    @staticmethod
    def _pdr_in_band(token: tuple[float, float, str], band: tuple[float, float]) -> bool:
        x0 = token[0]
        return band[0] <= x0 < band[1]

    @classmethod
    def _pdr_band_number(
        cls,
        tokens: list[tuple[float, float, str]],
        band: tuple[float, float],
    ) -> float | None:
        """First Turkish-format number printed inside an x-band (or None)."""
        for token in sorted(tokens, key=lambda t: (t[1], t[0], t[2])):
            if not cls._pdr_in_band(token, band):
                continue
            text = token[2]
            if not any(char.isdigit() for char in text):
                continue
            value = _parse_turkish_number(text)
            if value is not None:
                return value
        return None

    def _pdr_holding_from_tokens(
        self,
        tokens: list[tuple[float, float, str]],
        isin: str,
        symbol: str,
    ) -> Holding | None:
        """
        Build one Holding from the token window anchored on an ISIN.

        Returns None when the row cannot be classified honestly as an equity
        holding (e.g. a repo line whose code is not a BIST ticker, or a row
        with an unreadable nominal/weight). Callers then keep the raw row in
        ``unparsed_rows`` — no values are invented.
        """
        nominal = self._pdr_band_number(tokens, PDR_BAND_NOMINAL)
        weight = self._pdr_band_number(tokens, PDR_BAND_WEIGHT)
        if nominal is None or weight is None:
            return None

        market_value = self._pdr_band_number(tokens, PDR_BAND_MARKET_VALUE)
        unit_price = self._pdr_band_number(tokens, PDR_BAND_PRICE)

        name_tokens = [
            token for token in tokens
            if self._pdr_in_band(token, PDR_BAND_NAME)
            and any(char.isalpha() for char in token[2])
        ]
        name_tokens.sort(key=lambda t: (t[1], t[0], t[2]))

        # The name is only reported when every alphabetic token of the window
        # belongs to it. Long wrapped names (e.g. fund names crossing into the
        # ticker band) cannot be assembled reliably, so they stay None.
        extra_alpha = False
        for token in tokens:
            text = token[2]
            if not any(char.isalpha() for char in text):
                continue
            if PDR_ISIN_RE.fullmatch(text):
                continue
            if text in PDR_CURRENCY_TOKENS:
                continue
            if self._pdr_in_band(token, PDR_BAND_NAME):
                continue
            if text == symbol and self._pdr_in_band(token, PDR_BAND_TICKER):
                continue
            extra_alpha = True
            break

        name = None
        if name_tokens and not extra_alpha:
            name = " ".join(token[2] for token in name_tokens)

        return Holding(
            symbol=symbol,
            isin=isin,
            name=name,
            weight=weight,
            holding_type="stock",
            country=None,
            nominal_value=nominal,
            market_value=market_value,
            unit_price=unit_price,
        )

    def parse_pdr_pdf(self, pdf_data: bytes) -> PdrParseResult:
        """
        Deterministically parse portfolio rows from a PDR PDF (offline).

        Input are clean ``%PDF-`` bytes as returned by ``_download_pdf``. All
        pages are read (no page skipping); rows are rebuilt from word
        coordinates (y-bucket + x-band) so multi-line blocks and small y drift
        do not split a holding.

        Args:
            pdf_data: Raw PDF bytes (must start with/contain ``%PDF-``).

        Returns:
            PdrParseResult with holdings, raw unparsed rows, page count and the
            page-1 period range (raw dd/mm/yyyy strings, or None if absent).

        Raises:
            PdrParseError: Empty payload, missing ``%PDF-`` marker or an
                unparseable page (fail-closed; never a silent empty result).
        """
        if not pdf_data:
            raise PdrParseError("Empty PDF payload")
        if b"%PDF-" not in pdf_data[:1024]:
            raise PdrParseError("Payload is not a PDF: %PDF- marker missing")

        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for deterministic PDF parsing. "
                "Install with: pip install pymupdf"
            ) from None

        try:
            doc = pymupdf.open(stream=pdf_data, filetype="pdf")
        except Exception as exc:
            raise PdrParseError(f"Could not open PDF: {exc}") from exc

        try:
            page_count = doc.page_count
            if page_count < 1:
                raise PdrParseError("PDF has no pages")

            period_range: tuple[str, str] | None = None
            try:
                first_text = doc[0].get_text()
            except Exception as exc:
                raise PdrParseError(f"Could not read page 1: {exc}") from exc
            period_match = PDR_PERIOD_RE.search(first_text)
            if period_match:
                period_range = (period_match.group(1), period_match.group(2))

            holdings: list[Holding] = []
            unparsed_rows: list[str] = []

            for page_index in range(page_count):
                page = doc[page_index]
                try:
                    rows = self._pdr_rows(page)
                except Exception as exc:
                    raise PdrParseError(
                        f"Could not parse page {page_index + 1}: {exc}"
                    ) from exc

                row_y = [
                    sum(token[1] for token in row) / len(row) if row else 0.0
                    for row in rows
                ]
                consumed: set[int] = set()

                for row_index, row in enumerate(rows):
                    if row_index in consumed:
                        continue
                    for token in row:
                        if not self._pdr_in_band(token, PDR_BAND_ISIN):
                            continue
                        if not PDR_ISIN_RE.fullmatch(token[2]):
                            continue

                        window = [
                            index for index, y in enumerate(row_y)
                            if abs(y - row_y[row_index]) <= PDR_ROW_WINDOW
                        ]
                        if consumed.intersection(window):
                            continue

                        window_tokens = [
                            window_token
                            for index in window
                            for window_token in rows[index]
                        ]
                        tickers = sorted(
                            (
                                candidate for candidate in window_tokens
                                if self._pdr_in_band(candidate, PDR_BAND_TICKER)
                                and PDR_TICKER_RE.fullmatch(candidate[2])
                            ),
                            key=lambda t: (t[1], t[0], t[2]),
                        )
                        if not tickers:
                            continue

                        holding = self._pdr_holding_from_tokens(
                            window_tokens, token[2], tickers[0][2]
                        )
                        if holding is None:
                            continue

                        holdings.append(holding)
                        consumed.update(window)

                for row_index, row in enumerate(rows):
                    if row_index in consumed:
                        continue
                    raw = " ".join(token[2] for token in row)
                    # Group/page totals never become holdings; keep them out of
                    # the unparsed bucket as well (they are not failed rows).
                    if "TOPLAM" in raw.upper():
                        continue
                    # A row is noted as unparsed only when it prints into the
                    # table's TOPLAM DEĞER / weight columns; other page
                    # furniture (headers, performance summary) is not a failed
                    # row.
                    has_value_column = any(
                        self._pdr_in_band(token, band)
                        for token in row
                        for band in (PDR_BAND_MARKET_VALUE, PDR_BAND_WEIGHT)
                    )
                    if not has_value_column:
                        continue
                    if not any(char.isdigit() for char in raw):
                        continue
                    unparsed_rows.append(raw)

            return PdrParseResult(
                holdings=tuple(holdings),
                unparsed_rows=tuple(unparsed_rows),
                page_count=page_count,
                period_range=period_range,
            )

        finally:
            doc.close()

    def _parse_holdings_with_llm(self, pdf_data: bytes, api_key: str) -> list[Holding]:
        """
        Parse holdings from PDF using OpenRouter LLM.

        Args:
            pdf_data: Raw PDF bytes
            api_key: OpenRouter API key

        Returns:
            List of Holding objects.
        """
        try:
            import pymupdf
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for PDF parsing. "
                "Install with: pip install pymupdf4llm"
            ) from None

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenRouter API. "
                "Install with: pip install openai"
            ) from None

        # Convert PDF to markdown using pymupdf4llm (better table handling)
        doc = pymupdf.open(stream=pdf_data, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)

        # Truncate if too long (balance between completeness and speed)
        MAX_CHARS = 80000  # noqa: N806  ~20K tokens - enough for most fund portfolios
        if len(md_text) > MAX_CHARS:
            md_text = md_text[:MAX_CHARS]

        # Prepare OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Simplified prompt focusing on key columns
        prompt = '''Extract stock holdings from this Turkish investment fund portfolio report.

For each stock row, extract:
- symbol: BIST stock ticker (4-5 letters like "ASELS", "THYAO", "GARAN") - NOT the ISIN code!
- name: Company name
- weight: Portfolio weight percentage from "Toplam%" or "Fon Toplam Değerine Oranı %" column (e.g. 5.25)
- type: "stock"
- market_value: Value in TL (number)

IMPORTANT:
- symbol must be BIST ticker (ASELS, THYAO), NOT ISIN (TRAASELS...)
- weight is percentage (usually between 0.01 and 10)
- Turkish number format: "5,25" means 5.25

Return JSON array only:
[{"symbol": "ASELS", "name": "ASELSAN", "weight": 5.25, "type": "stock", "market_value": 228500000}]

DATA:
'''

        try:
            completion = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt + md_text}],
            )

            response_text = completion.choices[0].message.content

            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0]

            # Try to find JSON array in response
            json_text = json_text.strip()

            # Find the start of JSON array
            start_idx = json_text.find("[")
            if start_idx == -1:
                raise APIError("No JSON array found in LLM response")

            # Find matching end bracket
            bracket_count = 0
            end_idx = -1
            for i, char in enumerate(json_text[start_idx:], start_idx):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break

            if end_idx == -1:
                # Try to find any closing bracket
                end_idx = json_text.rfind("]") + 1

            if end_idx <= start_idx:
                raise APIError("Could not find valid JSON array bounds")

            json_text = json_text[start_idx:end_idx]

            # Parse JSON
            data = json.loads(json_text)

            # Convert to Holding objects
            holdings = []
            for item in data:
                if not item or not isinstance(item, dict):
                    continue

                # Clean symbol (remove .E suffix if present)
                raw_symbol = item.get("symbol")
                if not raw_symbol:
                    continue
                symbol = str(raw_symbol).replace(".E", "")

                # Parse weight (handle string or number)
                weight = item.get("weight")
                if weight is None:
                    weight = 0.0
                elif isinstance(weight, str):
                    weight = float(weight.replace(",", ".").replace("%", ""))

                # Sanity check: weight should be 0-100%
                # If weight > 100, it's likely market_value was used by mistake
                if weight and weight > 100:
                    continue

                # Parse market_value
                market_value = item.get("market_value")
                if isinstance(market_value, str):
                    market_value = float(market_value.replace(".", "").replace(",", "."))

                holdings.append(Holding(
                    symbol=symbol,
                    isin=item.get("isin"),
                    name=item.get("name", symbol),
                    weight=weight,
                    holding_type=item.get("type", "stock"),
                    country=item.get("country"),
                    market_value=market_value,
                ))

            return self._deduplicate_holdings(holdings)

        except json.JSONDecodeError as e:
            raise APIError(f"Failed to parse LLM response as JSON: {e}") from e
        except Exception as e:
            raise APIError(f"LLM parsing failed: {e}") from e

    def _deduplicate_holdings(self, holdings: list[Holding]) -> list[Holding]:
        """Remove duplicate holdings based on ISIN or symbol."""
        seen = set()
        unique = []

        for h in holdings:
            # Use ISIN as primary key, fall back to symbol
            key = h.isin if h.isin else h.symbol
            if key not in seen:
                seen.add(key)
                unique.append(h)

        return unique

    def get_holdings(
        self,
        fund_code: str,
        api_key: str | None = None,
        period: str | None = None,
        parser: str = "deterministic",
    ) -> list[Holding]:
        """
        Get detailed portfolio holdings for a fund.

        Args:
            fund_code: TEFAS fund code (e.g., "YAY", "AAK")
            api_key: OpenRouter API key. Only required for ``parser="llm"``.
            period: Optional period in format "YYYY-MM" (e.g., "2025-12")
                   If None, returns the most recent holdings.
            parser: "deterministic" (default) parses the PDF offline with
                   no network/LLM and no API key. "llm" keeps the legacy
                   pymupdf4llm + OpenRouter path.

        Returns:
            List of Holding objects with symbol, weight, type, etc.

        Raises:
            ValueError: Unknown parser, or ``parser="llm"`` without api_key.
            DataNotAvailableError: If fund or holdings not found.
            APIError: If API request fails.

        Example:
            >>> provider = KAPHoldingsProvider()
            >>> holdings = provider.get_holdings("YAY")  # deterministic, offline
            >>> holdings = provider.get_holdings("YAY", api_key="sk-or-v1-...", parser="llm")
        """
        if parser not in ("deterministic", "llm"):
            raise ValueError(
                f"Unknown parser: {parser!r} (expected 'deterministic' or 'llm')"
            )
        if parser == "llm" and api_key is None:
            raise ValueError("parser='llm' requires an OpenRouter api_key")

        fund_code = fund_code.upper()
        cache_key = f"{fund_code}:{period or 'latest'}:{parser}"
        current_time = time.time()

        # Check cache
        if cache_key in self._holdings_cache:
            cache_time = self._holdings_cache_time.get(cache_key, 0)
            if (current_time - cache_time) < CACHE_DURATION:
                return self._holdings_cache[cache_key][0]

        # Get disclosures
        disclosures = self.get_disclosures(fund_code)
        if not disclosures:
            raise DataNotAvailableError(
                f"No portfolio distribution reports found for {fund_code}"
            )

        # Find matching disclosure
        disclosure = None
        if period:
            # Parse period (YYYY-MM)
            match = re.match(r'(\d{4})-(\d{2})', period)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                for d in disclosures:
                    if d.get("year") == year and d.get("period") == month:
                        disclosure = d
                        break

        if disclosure is None:
            disclosure = disclosures[0]  # Most recent

        # Get file ID from disclosure page
        disclosure_index = disclosure.get("disclosure_index")
        if not disclosure_index:
            raise DataNotAvailableError("Disclosure has no index")

        file_id = self._get_file_id(disclosure_index)
        if not file_id:
            raise DataNotAvailableError(
                f"Could not find PDF attachment for disclosure {disclosure_index}"
            )

        # Download and parse PDF
        pdf_data = self._download_pdf(file_id)
        if not pdf_data:
            raise DataNotAvailableError(
                f"Failed to download PDF for file {file_id}"
            )

        if parser == "deterministic":
            holdings = list(self.parse_pdr_pdf(pdf_data).holdings)
        else:
            holdings = self._parse_holdings_with_llm(pdf_data, api_key)

        # Get report date from disclosure
        report_date = None
        if disclosure.get("year") and disclosure.get("period"):
            try:
                report_date = datetime(
                    disclosure["year"],
                    disclosure["period"],
                    1
                )
            except ValueError:
                pass

        # Cache result
        self._holdings_cache[cache_key] = (holdings, report_date)
        self._holdings_cache_time[cache_key] = current_time

        return holdings

    def get_holdings_df(
        self,
        fund_code: str,
        api_key: str | None = None,
        period: str | None = None,
        parser: str = "deterministic",
    ) -> pd.DataFrame:
        """
        Get holdings as a pandas DataFrame.

        Args:
            fund_code: TEFAS fund code
            api_key: OpenRouter API key (only used by ``parser="llm"``)
            period: Optional period in format "YYYY-MM"
            parser: "deterministic" (default, offline) or "llm" (legacy)

        Returns:
            DataFrame with columns: symbol, isin, name, weight, type, country, value
        """
        holdings = self.get_holdings(fund_code, api_key, period, parser=parser)

        if not holdings:
            return pd.DataFrame(columns=[
                'symbol', 'isin', 'name', 'weight', 'type', 'country', 'value'
            ])

        records = []
        for h in holdings:
            records.append({
                'symbol': h.symbol,
                'isin': h.isin,
                'name': h.name,
                'weight': h.weight,
                'type': h.holding_type,
                'country': h.country,
                'value': h.market_value,
            })

        df = pd.DataFrame(records)
        df = df.sort_values('weight', ascending=False).reset_index(drop=True)
        return df


# Singleton
_provider: KAPHoldingsProvider | None = None


def get_kap_holdings_provider() -> KAPHoldingsProvider:
    """Get singleton provider instance."""
    global _provider
    if _provider is None:
        _provider = KAPHoldingsProvider()
    return _provider
