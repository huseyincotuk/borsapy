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
import statistics
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
# Deterministic PDR parser (template-independent).
#
# The 075g fixed x-bands below are kept only for the legacy helper methods;
# ``parse_pdr_pdf`` no longer uses them. Column geometry is now discovered per
# PDF from the printed header labels (see ``_discover_pdr_columns``), because
# the 075l discovery seal (`gecmis baglam/pw15b_discovery_2026_11_11/`)
# measured three templates whose columns sit fully outside those bands.
# ---------------------------------------------------------------------------
PDR_ISIN_RE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
#: BIST ticker, optionally carrying the PDR template's `.E` lot suffix
#: (one or two letters only; `.PAY`-style header tokens are not tickers).
PDR_TICKER_RE = re.compile(r"[A-Z]{3,6}(?:\.[A-Z]{1,2})?")
PDR_PERIOD_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})")

#: y grouping tolerance (points), same value used by the 075g measurement.
PDR_ROW_TOLERANCE = 2.5
#: a holding's fields may sit on neighbouring y-lines (multi-line blocks and
#: small y drift), so tokens within this window of the ISIN anchor are merged.
PDR_ROW_WINDOW = 9.0
#: how far a ticker row / numeric block may sit from its owner row (measured
#: Deniz-type maximum is 22 pt; a larger window steals neighbouring ISINs).
PDR_OWNER_WINDOW = 24.0
#: numeric tokens closer than this on the x axis belong to one column.
PDR_COLUMN_GAP = 3.0
#: default half-width of a discovered column band.
PDR_COLUMN_TOLERANCE = 6.0
#: a header label may be split over at most this horizontal span.
PDR_HEADER_SPAN = 60.0

#: 075g BHE bands — retained for the legacy helper methods only.
PDR_BAND_TICKER = (0.0, 60.0)       # MENKUL KIYMET CİNSİ karşılığı ticker
PDR_BAND_CURRENCY = (60.0, 115.0)   # DÖVİZ CİNSİ (TL)
PDR_BAND_NAME = (115.0, 205.0)      # İHRAÇCI KURUM (kurum adı)
PDR_BAND_ISIN = (200.0, 365.0)      # ISIN KODU
PDR_BAND_NOMINAL = (360.0, 416.0)   # NOMİNAL DEĞER
PDR_BAND_PRICE = (418.0, 460.0)     # BİRİM FİYATI
PDR_BAND_MARKET_VALUE = (748.0, 806.0)  # TOPLAM DEĞER (TL)
PDR_BAND_WEIGHT = (838.0, 882.0)    # TOPLAM (FPD GÖRE) — ağırlık %

PDR_CURRENCY_TOKENS = frozenset({"TL", "USD", "EUR", "GBP", "CHF", "JPY"})

_PDR_NUMBER_TOKEN_RE = re.compile(r"%?[+-]?\d[\d.,]*%?")
_PDR_NUMBER_BODY_RE = re.compile(r"[+-]?\d[\d.,]*")
_PDR_FOLD_TABLE = str.maketrans(
    "İIŞĞÜÖÇıişğüöç", "IISGUOCIisguoc"
)


class PdrParseError(ValueError):
    """Raised when a PDR PDF payload cannot be parsed deterministically."""


class PdrSilentZeroError(PdrParseError):
    """Raised when ISIN rows exist but no holding and no unparsed row came out."""


def _pdr_fold(text: str) -> str:
    """Turkish-aware upper-case fold for header labels."""
    return text.translate(_PDR_FOLD_TABLE).upper()


def _pdr_is_number_token(text: str) -> bool:
    """True when the token is a plain printed number (no date, no ISIN)."""
    stripped = text.strip()
    if re.fullmatch(r"\d{2}[./]\d{2}[./]\d{2,4}", stripped):
        return False
    return bool(_PDR_NUMBER_TOKEN_RE.fullmatch(stripped)) and any(
        char.isdigit() for char in stripped
    )


def _pdr_number_class(text: str) -> str | None:
    """
    Classify a printed number as ``"tr"``, ``"us"`` or ``None`` (ambiguous).

    The classification only carries evidence from the separators themselves;
    it never guesses. A token with no separators yields None (any format can
    read an integer), and a token that could be read either way (e.g.
    ``"100,000"`` or ``"1.234"``) also yields None.
    """
    body = text.strip().replace("%", "").replace(" ", "")
    if not _PDR_NUMBER_BODY_RE.fullmatch(body):
        return None
    body = body.lstrip("+-")
    if not body or not body[0].isdigit():
        return None
    has_dot = "." in body
    has_comma = "," in body
    if has_dot and has_comma:
        return "us" if body.rfind(".") > body.rfind(",") else "tr"
    if has_comma:
        parts = body.split(",")
        if len(parts) > 2:
            # Multiple commas can only be US thousands separators.
            return "us"
        left, right = parts
        if len(left) <= 3 and len(right) == 3:
            return None
        return "tr"
    if has_dot:
        parts = body.split(".")
        if len(parts) > 2:
            # Multiple dots can only be Turkish thousands separators.
            return "tr"
        left, right = parts
        if len(left) <= 3 and len(right) == 3:
            return None
        return "us"
    return None


def _pdr_parse_number(text: str, fmt: str) -> float | None:
    """Read one cell with an already-discovered column format (no guessing)."""
    body = text.strip().replace("%", "").replace(" ", "")
    if not _PDR_NUMBER_BODY_RE.fullmatch(body):
        return None
    try:
        if fmt == "tr":
            if "," in body:
                body = body.replace(".", "").replace(",", ".")
            else:
                body = body.replace(".", "")
        else:
            body = body.replace(",", "")
        return float(body)
    except ValueError:
        return None


def _pdr_cell_number(text: str, fmt: str | None) -> float | None:
    """
    Parse one numeric cell under the discovered column format.

    Returns None (never an invented value) when the column format is unknown
    or when the cell's own format contradicts it (Turkish/US mix in one row).
    """
    if fmt is None:
        return None
    cell_class = _pdr_number_class(text)
    if cell_class is not None and cell_class != fmt:
        return None
    return _pdr_parse_number(text, fmt)


@dataclass(frozen=True)
class _PdrCluster:
    """One x-column discovered from the page's printed numbers."""

    center: float
    low: float
    high: float
    texts: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.texts)

    @property
    def tolerance(self) -> float:
        return max(PDR_COLUMN_TOLERANCE, (self.high - self.low) / 2 + 2.0)


@dataclass(frozen=True)
class _PdrColumns:
    """Per-PDF column geometry, derived from the data (no fixed positions)."""

    ticker_center: float
    ticker_tolerance: float
    isin_center: float
    name_band: tuple[float, float]
    nominal: _PdrCluster | None
    price: _PdrCluster | None
    value: _PdrCluster | None
    weight: _PdrCluster | None
    formats: dict[str, str | None]

    @property
    def discovered(self) -> bool:
        return self.nominal is not None and self.value is not None and self.weight is not None


def _pdr_cluster_tokens(
    tokens: list[tuple[float, float, float, str]],
    gap: float = PDR_COLUMN_GAP,
) -> list[_PdrCluster]:
    """Group tokens into x-columns by their right edge (mod/median centre)."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: token[2])
    groups: list[list[tuple[float, float, float, str]]] = [[ordered[0]]]
    for token in ordered[1:]:
        if token[2] - groups[-1][-1][2] <= gap:
            groups[-1].append(token)
        else:
            groups.append([token])
    clusters: list[_PdrCluster] = []
    for group in groups:
        edges = [token[2] for token in group]
        clusters.append(
            _PdrCluster(
                center=statistics.median(edges),
                low=min(edges),
                high=max(edges),
                texts=tuple(token[3] for token in group),
            )
        )
    return clusters


def _pdr_ticker_parts(raw: str) -> tuple[str, str | None]:
    """``AKBNK.E`` -> ``("AKBNK", "AKBNK.E")``; plain ticker -> ``(raw, None)``."""
    if "." in raw:
        return raw.split(".", 1)[0], raw
    return raw, None


def _pdr_nearest_owner(
    owners: list[Any], y: float, window: float
) -> tuple[Any | None, bool]:
    """
    Pick the nearest owner row on the y axis.

    Returns ``(owner, ambiguous)``. An exact tie between the two closest rows
    yields ``(None, True)``: neighbouring theft is never resolved by guessing,
    the caller must route the row to ``unparsed_rows``.
    """
    candidates = [owner for owner in owners if abs(owner["y"] - y) <= window]
    if not candidates:
        return None, False
    candidates.sort(key=lambda owner: abs(owner["y"] - y))
    if (
        len(candidates) > 1
        and abs(abs(candidates[1]["y"] - y) - abs(candidates[0]["y"] - y)) < 0.01
    ):
        return None, True
    return candidates[0], False


def _pdr_column_format(cluster: _PdrCluster | None) -> str | None:
    """Discover one column's number format; conflicting evidence -> None."""
    if cluster is None:
        return None
    votes = {_pdr_number_class(text) for text in cluster.texts}
    votes.discard(None)
    if len(votes) == 1:
        return votes.pop()
    return None


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
    notes: str | None = None


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

    # ------------------------------------------------------------------
    # 075l: per-PDF column discovery (data-derived, no fixed x positions)
    # ------------------------------------------------------------------
    def _pdr_word_rows(self, page) -> list[list[tuple[float, float, float, str]]]:
        """Visual rows of a page as ``(x0, y0, x1, text)`` tokens."""
        buckets: dict[int, list[tuple[float, float, float, str]]] = {}
        for x0, y0, x1, _y1, text, *_ in page.get_text("words"):
            key = int(round(y0 / PDR_ROW_TOLERANCE))
            buckets.setdefault(key, []).append((x0, y0, x1, text))
        return [
            sorted(buckets[key], key=lambda token: (token[0], token[1], token[3]))
            for key in sorted(buckets)
        ]

    @staticmethod
    def _pdr_row_y(row: list[tuple[float, float, float, str]]) -> float:
        return sum(token[1] for token in row) / len(row) if row else 0.0

    @staticmethod
    def _pdr_in_cluster(
        token: tuple[float, float, float, str], cluster: _PdrCluster | None
    ) -> bool:
        if cluster is None:
            return False
        return abs(token[2] - cluster.center) <= cluster.tolerance

    @staticmethod
    def _pdr_ticker_parts(raw: str) -> tuple[str, str | None]:
        """``AKBNK.E`` -> ``("AKBNK", "AKBNK.E")``; plain ticker -> ``(raw, None)``."""
        return _pdr_ticker_parts(raw)

    def _discover_pdr_columns(
        self,
        pages: list[list[list[tuple[float, float, float, str]]]],
    ) -> _PdrColumns | None:
        """
        Derive the table's column geometry from the page itself.

        Numeric tokens are clustered by their right edge (the printed columns
        are right-aligned), the header labels are located on the page and each
        label is then mapped to its cluster. Every band is therefore measured
        per PDF; nothing is assumed about where a template prints a column.
        """
        all_tokens = [token for rows in pages for row in rows for token in row]
        page_ys = [[self._pdr_row_y(row) for row in rows] for rows in pages]
        numeric = [token for token in all_tokens if _pdr_is_number_token(token[3])]
        clusters = _pdr_cluster_tokens(numeric)

        isin_tokens = [
            token for token in all_tokens if PDR_ISIN_RE.fullmatch(token[3])
        ]
        if not isin_tokens:
            return None
        isin_clusters = _pdr_cluster_tokens(
            [(token[0], token[1], token[0], token[3]) for token in isin_tokens]
        )
        isin_cluster = max(
            isin_clusters, key=lambda cluster: (cluster.count, cluster.center)
        )
        isin_center = isin_cluster.center

        # The ticker column is the leftmost dense column of ticker-shaped words.
        ticker_candidates = [
            token
            for token in all_tokens
            if token[0] < isin_center - 40.0 and PDR_TICKER_RE.fullmatch(token[3])
        ]
        if not ticker_candidates:
            return None
        ticker_clusters = _pdr_cluster_tokens(
            [(token[0], token[1], token[0], token[3]) for token in ticker_candidates]
        )
        # The ticker column is the leftmost dense block of ticker-shaped words
        # (the printed table always starts with MENKUL KIYMET CİNSİ).
        ticker_cluster = min(
            (cluster for cluster in ticker_clusters if cluster.count >= 3),
            key=lambda cluster: cluster.center,
        )
        ticker_center = ticker_cluster.center
        ticker_tolerance = max(PDR_COLUMN_TOLERANCE, ticker_cluster.tolerance)

        def block_anchors(
            tokens: list[tuple[float, float, float, str]],
        ) -> dict[str, list[tuple[int, float]]]:
            """Locate labels inside a header block (labels may wrap in y)."""
            found: dict[str, list[tuple[int, float]]] = {
                "nominal": [],
                "price": [],
                "value": [],
                "weight": [],
            }
            ordered = sorted(tokens, key=lambda token: (token[0], token[1]))
            for index, token in enumerate(ordered):
                fold = _pdr_fold(token[3])
                if fold.startswith("NOMINAL"):
                    for follower in ordered[index + 1:]:
                        if follower[0] - token[0] > PDR_HEADER_SPAN:
                            break
                        if abs(follower[1] - token[1]) > 15.0:
                            continue
                        if _pdr_fold(follower[3]).startswith("DEGER"):
                            found["nominal"].append((10, follower[2]))
                            break
                if fold.startswith("BIRIM"):
                    for follower in ordered[index + 1:]:
                        if follower[0] - token[0] > PDR_HEADER_SPAN + 30:
                            break
                        if abs(follower[1] - token[1]) > 15.0:
                            continue
                        if _pdr_fold(follower[3]).startswith(("FIYAT", "FYT")):
                            found["price"].append((10, follower[2]))
                            break
                if fold.startswith("TOPLAM"):
                    for follower in ordered[index + 1:]:
                        if follower[0] - token[0] > PDR_HEADER_SPAN:
                            break
                        if abs(follower[1] - token[1]) > 15.0:
                            continue
                        follower_fold = _pdr_fold(follower[3])
                        if follower_fold.startswith("DEGER"):
                            found["value"].append((10, follower[2]))
                            break
                        if follower[3].startswith("(") and (
                            "FPD" in follower_fold or "FTD" in follower_fold
                        ):
                            priority = 1 if "FPD" in follower_fold else 5
                            found["weight"].append((priority, follower[2]))
                if fold.startswith("TOPLAM%"):
                    found["weight"].append((15, token[2]))
                elif fold.startswith("GRUP%"):
                    found["weight"].append((20, token[2]))
            return found

        # The holding table's header is the block that prints a NOMİNAL label
        # to the right of the ISIN column; labels may wrap over a few lines
        # (e.g. TOPLAM / (FPD GÖRE)), so a small y-window is the unit. The
        # block that resolves the most roles wins, which keeps secondary
        # repo/VIOP summary tables out.
        header_anchors: dict[str, list[tuple[int, float]]] = {
            "nominal": [],
            "price": [],
            "value": [],
            "weight": [],
        }
        best_score = -1
        for page_index, rows in enumerate(pages):
            for row_index, row in enumerate(rows):
                if not any(
                    token[0] > isin_center + 15.0
                    and _pdr_fold(token[3]).startswith("NOMINAL")
                    for token in row
                ):
                    continue
                anchor_y = page_ys[page_index][row_index]
                block_tokens = [
                    token
                    for other_row, other_y in zip(rows, page_ys[page_index], strict=False)
                    if abs(other_y - anchor_y) <= 15.0
                    for token in other_row
                ]
                block = block_anchors(block_tokens)
                score = sum(
                    1 for role in ("nominal", "value", "weight") if block[role]
                ) + (1 if block["price"] else 0)
                if score > best_score:
                    best_score = score
                    header_anchors = block
                if best_score >= 4:
                    break
            if best_score >= 4:
                break
        anchors = header_anchors
        if not anchors["nominal"]:
            for rows in pages:
                for row in rows:
                    found = block_anchors(row)
                    for role in anchors:
                        anchors[role].extend(found[role])

        def map_anchor(anchor: float) -> _PdrCluster | None:
            dense = [cluster for cluster in clusters if cluster.count >= 2]
            if not dense:
                return None
            best = min(dense, key=lambda cluster: (abs(cluster.center - anchor), -cluster.count))
            return best if abs(best.center - anchor) <= 40.0 else None

        def pick(role: str) -> _PdrCluster | None:
            mapped = []
            for priority, anchor in anchors[role]:
                cluster = map_anchor(anchor)
                if cluster is not None:
                    mapped.append((priority, abs(cluster.center - anchor), cluster))
            if mapped:
                return min(mapped, key=lambda item: (item[0], item[1]))[2]
            return None

        nominal = pick("nominal")
        value = pick("value")
        weight = pick("weight")
        price = pick("price")

        # Fallback: derive the bands from the numeric clusters themselves when
        # a header label is absent. Still measured from this PDF's data.
        dense = [cluster for cluster in clusters if cluster.count >= 2]
        non_percent = [
            cluster
            for cluster in dense
            if not any("%" in text for text in cluster.texts)
        ]

        def magnitude(cluster: _PdrCluster) -> float:
            values = []
            for text in cluster.texts:
                parsed = _pdr_parse_number(text, _pdr_number_class(text) or "tr")
                if parsed is not None:
                    values.append(abs(parsed))
            return statistics.median(values) if values else 0.0

        right_of_isin = [
            cluster for cluster in non_percent if cluster.center > isin_center + 15.0
        ]
        if nominal is None and right_of_isin:
            nominal = min(right_of_isin, key=lambda cluster: cluster.center)
        if value is None and right_of_isin:
            value = max(right_of_isin, key=magnitude)
        if weight is None:
            percent = [
                cluster
                for cluster in dense
                if any("%" in text for text in cluster.texts)
            ]
            if percent:
                weight = max(percent, key=lambda cluster: cluster.center)
        if price is None and nominal is not None and value is not None:
            between = [
                cluster
                for cluster in non_percent
                if nominal.center < cluster.center < value.center
            ]
            if between:
                price = min(between, key=lambda cluster: cluster.center)

        formats = {
            role: _pdr_column_format(cluster)
            for role, cluster in (
                ("nominal", nominal),
                ("price", price),
                ("value", value),
                ("weight", weight),
            )
        }
        name_band = (ticker_center + 25.0, isin_center - 8.0)
        return _PdrColumns(
            ticker_center=ticker_center,
            ticker_tolerance=ticker_tolerance,
            isin_center=isin_center,
            name_band=name_band,
            nominal=nominal,
            price=price,
            value=value,
            weight=weight,
            formats=formats,
        )

    @staticmethod
    def _pdr_owner_name(
        rows: list[list[tuple[float, float, float, str]]],
        row_ys: list[float],
        owner_y: float,
        next_owner_y: float | None,
        name_band: tuple[float, float],
        isin_y: float,
        ticker_band: tuple[float, float],
    ) -> str | None:
        """
        Assemble the issuer name that belongs to one owner row.

        Only rows between the previous and the next ticker row are used, so a
        neighbour's name is never merged in. Tokens printed in the ticker
        column are skipped (they belong to another holding), while the issuer
        name's own first word — which may repeat the ticker — is kept.
        """
        lower = owner_y - 4.0
        upper = (next_owner_y - 4.0) if next_owner_y is not None else isin_y + 15.0
        name_tokens: list[tuple[float, float, float, str]] = []
        extra_alpha = False
        for row, y in zip(rows, row_ys, strict=False):
            if not (lower <= y <= upper):
                continue
            if y > isin_y + 3.0 and any(
                PDR_ISIN_RE.fullmatch(token[3]) for token in row
            ):
                # Another ISIN block starts here (e.g. the repo table); its
                # text must never be merged into this name.
                break
            if any(
                "TOPLAM" in _pdr_fold(token[3])
                or "GRUP" in _pdr_fold(token[3])
                for token in row
            ):
                continue
            for token in row:
                text = token[3]
                if not any(char.isalpha() for char in text):
                    continue
                if ticker_band[0] <= token[0] <= ticker_band[1]:
                    continue
                if PDR_ISIN_RE.fullmatch(text) or text in PDR_CURRENCY_TOKENS:
                    continue
                if name_band[0] <= token[0] <= name_band[1]:
                    name_tokens.append(token)
                else:
                    extra_alpha = True
        if not name_tokens or extra_alpha:
            return None
        name_tokens.sort(key=lambda token: (token[1], token[0], token[3]))
        return " ".join(token[3] for token in name_tokens)

    @staticmethod
    def _pdr_check_silent_zero(
        isin_rows: int,
        holdings: list[Holding],
        unparsed_rows: list[str],
    ) -> None:
        """
        Fail loud on the silent-zero class (075l discovery evidence).

        ISIN-bearing rows that produced neither a holding nor an unparsed entry
        mean the geometry was misread; that contradiction must never be
        returned as a successful empty result.
        """
        if isin_rows > 0 and not holdings and not unparsed_rows:
            raise PdrSilentZeroError(
                f"PDR contradiction: {isin_rows} ISIN row(s) produced "
                "0 holdings and 0 unparsed rows"
            )

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
            pages: list[list[list[tuple[float, float, float, str]]]] = []
            page_ys: list[list[float]] = []
            for page_index in range(page_count):
                page = doc[page_index]
                try:
                    page_rows = self._pdr_word_rows(page)
                except Exception as exc:
                    raise PdrParseError(
                        f"Could not parse page {page_index + 1}: {exc}"
                    ) from exc
                pages.append(page_rows)
                page_ys.append([self._pdr_row_y(row) for row in page_rows])

            isin_rows = 0
            for rows in pages:
                for row in rows:
                    if any(PDR_ISIN_RE.fullmatch(token[3]) for token in row):
                        isin_rows += 1

            columns = self._discover_pdr_columns(pages)
            if columns is None or not columns.discovered:
                # Band discovery failed: every ISIN-bearing row is carried in
                # unparsed_rows (never silently dropped) plus a row-count note.
                for rows in pages:
                    for row in rows:
                        raw = " ".join(token[3] for token in row)
                        if "TOPLAM" in raw.upper():
                            continue
                        if not any(
                            PDR_ISIN_RE.fullmatch(token[3]) for token in row
                        ):
                            continue
                        unparsed_rows.append(raw)
                if isin_rows:
                    unparsed_rows.append(
                        f"[pdr] {isin_rows} ISIN row(s): column bands could not "
                        "be discovered"
                    )
                self._pdr_check_silent_zero(isin_rows, holdings, unparsed_rows)
                return PdrParseResult(
                    holdings=tuple(holdings),
                    unparsed_rows=tuple(unparsed_rows),
                    page_count=page_count,
                    period_range=period_range,
                )

            roles = {
                "nominal": columns.nominal,
                "price": columns.price,
                "value": columns.value,
                "weight": columns.weight,
            }

            def token_roles(
                token: tuple[float, float, float, str]
            ) -> list[str]:
                return [
                    role
                    for role, cluster in roles.items()
                    if self._pdr_in_cluster(token, cluster)
                ]

            # --- owner rows: a ticker with its numeric block (same row or the
            # couple of points below it, as printed by wrapped templates) -----
            owners: list[dict[str, Any]] = []
            page_owners: list[list[dict[str, Any]]] = [[] for _ in pages]
            for page_index, rows in enumerate(pages):
                for row_index, row in enumerate(rows):
                    ticker_token = None
                    for token in row:
                        if abs(token[0] - columns.ticker_center) > columns.ticker_tolerance:
                            continue
                        if not PDR_TICKER_RE.fullmatch(token[3]):
                            continue
                        if _pdr_fold(token[3]) in {"TOPLAM", "GRUP"}:
                            continue
                        ticker_token = token
                        break
                    if ticker_token is None:
                        continue
                    owner_y = page_ys[page_index][row_index]
                    has_cell = any(
                        (owner_y <= y <= owner_y + 4.0)
                        and _pdr_is_number_token(token[3])
                        and token_roles(token)
                        for other_row, y in zip(rows, page_ys[page_index], strict=False)
                        for token in other_row
                    )
                    if not has_cell:
                        continue
                    symbol, suffix = self._pdr_ticker_parts(ticker_token[3])
                    owner = {
                        "page": page_index,
                        "row": row_index,
                        "y": owner_y,
                        "symbol": symbol,
                        "raw_ticker": ticker_token[3],
                        "suffix": suffix,
                        "cells": {role: [] for role in roles},
                    }
                    owners.append(owner)
                    page_owners[page_index].append(owner)

            # --- assign numeric blocks to their nearest owner -----------------
            ambiguous_owners: set[int] = set()
            for page_index, rows in enumerate(pages):
                for row in rows:
                    for token in row:
                        if not _pdr_is_number_token(token[3]):
                            continue
                        matched_roles = token_roles(token)
                        if not matched_roles:
                            continue
                        best, ambiguous = _pdr_nearest_owner(
                            page_owners[page_index],
                            token[1],
                            PDR_OWNER_WINDOW,
                        )
                        if ambiguous:
                            # Equal distance: no guessing, both rows are unparsed.
                            tied = [
                                owner
                                for owner in page_owners[page_index]
                                if abs(owner["y"] - token[1]) <= PDR_OWNER_WINDOW
                            ]
                            tied.sort(key=lambda owner: abs(owner["y"] - token[1]))
                            ambiguous_owners.update(id(owner) for owner in tied[:2])
                            continue
                        if best is None:
                            continue
                        distance = abs(best["y"] - token[1])
                        for role in matched_roles:
                            best["cells"][role].append((distance, token))

            # --- assign ISIN rows to their nearest owner ----------------------
            isin_owner: dict[int, list[tuple[int, int, tuple]]] = {}
            for page_index, rows in enumerate(pages):
                for row_index, row in enumerate(rows):
                    for token in row:
                        if not PDR_ISIN_RE.fullmatch(token[3]):
                            continue
                        best, ambiguous = _pdr_nearest_owner(
                            page_owners[page_index],
                            token[1],
                            PDR_OWNER_WINDOW,
                        )
                        if ambiguous:
                            tied = [
                                owner
                                for owner in page_owners[page_index]
                                if abs(owner["y"] - token[1]) <= PDR_OWNER_WINDOW
                            ]
                            tied.sort(key=lambda owner: abs(owner["y"] - token[1]))
                            ambiguous_owners.update(id(owner) for owner in tied[:2])
                            continue
                        if best is None:
                            continue
                        isin_owner.setdefault(id(best), []).append(
                            (page_index, row_index, token)
                        )

            added_unparsed: set[tuple[int, int]] = set()

            def add_unparsed(page_index: int, row_index: int) -> None:
                key = (page_index, row_index)
                if key in added_unparsed:
                    return
                added_unparsed.add(key)
                raw = " ".join(
                    token[3] for token in pages[page_index][row_index]
                )
                if not raw or "TOPLAM" in raw.upper():
                    return
                unparsed_rows.append(raw)

            def owner_cell(
                owner: dict[str, Any], role: str
            ) -> tuple[tuple | None, bool]:
                """Nearest cell of one column; ``(token, ambiguous)``."""
                entries = owner["cells"][role]
                if not entries:
                    return None, False
                entries = sorted(entries, key=lambda item: item[0])
                if (
                    len(entries) > 1
                    and abs(entries[1][0] - entries[0][0]) < 0.01
                ):
                    return None, True
                return entries[0][1], False

            next_owner_y = {
                id(owner): None for owner in owners
            }
            for _page_index, owner_list in enumerate(page_owners):
                for index, owner in enumerate(owner_list):
                    if index + 1 < len(owner_list):
                        next_owner_y[id(owner)] = owner_list[index + 1]["y"]

            consumed: list[set[int]] = [set() for _ in pages]
            for owner in owners:
                page_index = owner["page"]
                assigned = isin_owner.get(id(owner), [])
                if len(assigned) != 1 or id(owner) in ambiguous_owners:
                    add_unparsed(page_index, owner["row"])
                    for assigned_page, assigned_row, _token in assigned:
                        add_unparsed(assigned_page, assigned_row)
                    continue

                isin_page, isin_row, isin_token = assigned[0]
                cells: dict[str, float | None] = {}
                row_failed = False
                for role in ("nominal", "value", "weight", "price"):
                    token, ambiguous = owner_cell(owner, role)
                    if ambiguous:
                        row_failed = True
                        break
                    if token is None:
                        cells[role] = None
                        continue
                    value = _pdr_cell_number(token[3], columns.formats.get(role))
                    if value is None:
                        if role in ("nominal", "value", "weight"):
                            row_failed = True
                            break
                        cells[role] = None
                        continue
                    cells[role] = value

                if row_failed or cells.get("nominal") is None or cells.get("weight") is None:
                    add_unparsed(page_index, owner["row"])
                    add_unparsed(isin_page, isin_row)
                    continue

                name = self._pdr_owner_name(
                    pages[page_index],
                    page_ys[page_index],
                    owner["y"],
                    next_owner_y[id(owner)],
                    columns.name_band,
                    isin_token[1],
                    (
                        columns.ticker_center - columns.ticker_tolerance,
                        columns.ticker_center + columns.ticker_tolerance,
                    ),
                )
                notes = (
                    f"source_ticker={owner['raw_ticker']}"
                    if owner["suffix"] is not None
                    else None
                )
                holdings.append(
                    Holding(
                        symbol=owner["symbol"],
                        isin=isin_token[3],
                        name=name,
                        weight=cells["weight"],
                        holding_type="stock",
                        country=None,
                        nominal_value=cells["nominal"],
                        market_value=cells.get("value"),
                        unit_price=cells.get("price"),
                        notes=notes,
                    )
                )
                consumed[page_index].add(owner["row"])
                consumed[isin_page].add(isin_row)

            # --- leftover rows that could not become holdings -----------------
            for page_index, rows in enumerate(pages):
                for row_index, row in enumerate(rows):
                    if row_index in consumed[page_index]:
                        continue
                    if (page_index, row_index) in added_unparsed:
                        continue
                    raw = " ".join(token[3] for token in row)
                    if "TOPLAM" in raw.upper():
                        continue
                    has_isin = any(
                        PDR_ISIN_RE.fullmatch(token[3]) for token in row
                    )
                    has_value_column = any(
                        _pdr_is_number_token(token[3])
                        and any(
                            role in ("value", "weight")
                            for role in token_roles(token)
                        )
                        for token in row
                    )
                    if not (has_isin or has_value_column):
                        continue
                    if not any(char.isdigit() for char in raw):
                        continue
                    unparsed_rows.append(raw)
                    added_unparsed.add((page_index, row_index))

            self._pdr_check_silent_zero(isin_rows, holdings, unparsed_rows)
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
