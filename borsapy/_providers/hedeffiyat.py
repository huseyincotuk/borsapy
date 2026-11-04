"""HedeFiyat provider for analyst price targets."""

import re

from borsapy._providers.base import BaseProvider


class HedeFiyatProvider(BaseProvider):
    """
    Provider for analyst price targets from hedeffiyat.com.tr.

    HedeFiyat aggregates analyst price targets from 30+ Turkish
    financial institutions.
    """

    BASE_URL = "https://www.hedeffiyat.com.tr"
    SEARCH_URL = "https://www.hedeffiyat.com.tr/arama"
    CACHE_DURATION = 86400  # 24 hours

    def __init__(self):
        super().__init__()
        self._url_cache: dict[str, str] = {}

    def get_price_targets(self, symbol: str) -> dict[str, float | int | None]:
        """
        Get analyst price targets for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dictionary with:
            - current: Current stock price
            - low: Lowest analyst target
            - high: Highest analyst target
            - mean: Average target
            - median: Median target
            - numberOfAnalysts: Number of analysts
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Check cache
        cache_key = f"hedeffiyat_{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        result = {
            "current": None,
            "low": None,
            "high": None,
            "mean": None,
            "median": None,
            "numberOfAnalysts": None,
        }

        try:
            # Get stock page URL
            page_url = self._get_stock_url(symbol)
            if not page_url:
                return result

            # Fetch stock page
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Upgrade-Insecure-Requests": "1",
            }
            response = self._client.get(page_url, headers=headers, timeout=15)
            # Server may return non-200 status but still serve content
            html = response.text
            if not html:
                return result

            # Parse price data from HTML
            result = self._parse_price_targets(html)

            # Cache the result
            if result.get("numberOfAnalysts"):
                self._cache_set(cache_key, result, self.CACHE_DURATION)

            return result

        except Exception:
            return result

    def get_recommendations_summary(self, symbol: str) -> dict[str, int]:
        """
        Get analyst recommendation summary (buy/hold/sell counts).

        Parses individual analyst recommendations from hedeffiyat.com.tr
        and aggregates them into strongBuy, buy, hold, sell, strongSell counts.

        Recommendation mapping:
        - strongBuy: "Güçlü Al"
        - buy: "Al", "Endeks Üstü Getiri", "Endeks Üstü Get."
        - hold: "Tut", "Nötr", "Endekse Paralel"
        - sell: "Sat", "Endeks Altı Getiri", "Endeks Altı Get."
        - strongSell: "Güçlü Sat"

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dictionary with counts for each recommendation category.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Check cache
        cache_key = f"hedeffiyat_recsummary_{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        result = {
            "strongBuy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strongSell": 0,
        }

        try:
            # Get stock page URL
            page_url = self._get_stock_url(symbol)
            if not page_url:
                return result

            # Fetch stock page
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            }
            response = self._client.get(page_url, headers=headers, timeout=15)
            html = response.text
            if not html:
                return result

            # Parse recommendation buttons
            # Pattern: btn-sm btn-(success|warning|danger|primary)...>RECOMMENDATION</a
            # Note: btn-primary is used for "Tut" (Hold) recommendations
            pattern = r'btn-sm\s+btn-(success|warning|danger|primary)[^>]*>([^<]+)</a'
            matches = re.findall(pattern, html, re.IGNORECASE)

            for btn_class, rec_text in matches:
                rec_text = rec_text.strip().lower()
                btn_class = btn_class.lower()

                # Map recommendation text to category
                if rec_text in ("güçlü al", "güçlü alım"):
                    result["strongBuy"] += 1
                elif rec_text in ("al", "alım", "endeks üstü get.", "endeks üstü getiri"):
                    result["buy"] += 1
                elif rec_text in ("tut", "tutma", "nötr", "endekse paralel"):
                    result["hold"] += 1
                elif rec_text in ("sat", "satım", "endeks altı get.", "endeks altı getiri"):
                    result["sell"] += 1
                elif rec_text in ("güçlü sat", "güçlü satım"):
                    result["strongSell"] += 1
                # Fallback to button color if text doesn't match
                elif btn_class == "success":
                    result["buy"] += 1
                elif btn_class in ("warning", "primary"):
                    result["hold"] += 1
                elif btn_class == "danger":
                    result["sell"] += 1

            # Cache if we found any recommendations
            if sum(result.values()) > 0:
                self._cache_set(cache_key, result, self.CACHE_DURATION)

            return result

        except Exception:
            return result

    def _get_stock_url(self, symbol: str) -> str | None:
        """
        Get the hedeffiyat.com.tr URL for a stock symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Full URL or None if not found.
        """
        # Check URL cache
        if symbol in self._url_cache:
            return self._url_cache[symbol]

        try:
            # Fetch the stock list page to find URL mapping
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Upgrade-Insecure-Requests": "1",
            }

            response = self._client.get(
                f"{self.BASE_URL}/senetler",
                headers=headers,
                timeout=15,
            )
            # Note: Server may return 404 but still serve content
            # Only check if we got a response with content
            if not response.text:
                return None

            # Search for the stock link in option values
            # Pattern: value="/senet/thyao-turk-hava-yollari-a.o.-410"
            pattern = rf'value="(/senet/{symbol.lower()}-[^"]+)"'
            match = re.search(pattern, response.text, re.IGNORECASE)

            if match:
                url = f"{self.BASE_URL}{match.group(1)}"
                self._url_cache[symbol] = url
                return url

            return None

        except Exception:
            return None

    def _search_stock_url(self, symbol: str) -> str | None:
        """
        Search for stock URL using the search functionality.

        Args:
            symbol: Stock symbol.

        Returns:
            Stock page URL or None.
        """
        try:
            # Use search page
            headers = {
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "tr-TR,tr;q=0.9",
            }

            response = self._client.get(
                self.SEARCH_URL,
                params={"q": symbol},
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()

            # Look for stock link in search results
            pattern = rf'href="(/senet/{symbol.lower()}-[^"]+)"'
            match = re.search(pattern, response.text, re.IGNORECASE)

            if match:
                url = f"{self.BASE_URL}{match.group(1)}"
                self._url_cache[symbol] = url
                return url

            return None

        except Exception:
            return None

    def _parse_price_targets(self, html: str) -> dict[str, float | int | None]:
        """
        Parse price target data from HTML.

        The page carries no analyst distribution, so the previously fabricated
        median (``(low + high) / 2``) was removed (FA-10a-B) and ``median`` is
        always ``None``.

        Args:
            html: Page HTML content.

        Returns:
            Dictionary with price target data.
        """
        result = {
            "current": None,
            "low": None,
            "high": None,
            "mean": None,
            "median": None,
            "numberOfAnalysts": None,
        }

        try:
            # Current price: "Güncel Fiyat" followed by <strong...>268,50 ₺</strong>
            current_match = re.search(
                r'Güncel\s*Fiyat.*?<strong[^>]*>\s*([\d.,]+)\s*₺',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if current_match:
                result["current"] = self._parse_number(current_match.group(1))

            # Highest target: "En Yüksek Tahmin" followed by badge with price
            high_match = re.search(
                r'En\s*Yüksek\s*Tahmin</div>\s*<div[^>]*>\s*([\d.,]+)\s*₺',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if high_match:
                result["high"] = self._parse_number(high_match.group(1))

            # Lowest target: "En Düşük Tahmin" followed by badge with price
            low_match = re.search(
                r'En\s*Düşük\s*Tahmin</div>\s*<div[^>]*>\s*([\d.,]+)\s*₺',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if low_match:
                result["low"] = self._parse_number(low_match.group(1))

            # Average price: "Ortalama Fiyat Tahmini" followed by badge
            avg_match = re.search(
                r'Ortalama\s*Fiyat\s*Tahmini</div>\s*<div[^>]*>\s*([\d.,]+)\s*₺',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if avg_match:
                result["mean"] = self._parse_number(avg_match.group(1))

            # Analyst count: "Kurum Sayısı" followed by <strong>19</strong>
            count_match = re.search(
                r'Kurum\s*Sayısı.*?<strong[^>]*>\s*(\d+)\s*</strong>',
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if count_match:
                result["numberOfAnalysts"] = int(count_match.group(1))

            return result

        except Exception:
            return result

    def _parse_number(self, text: str) -> float | None:
        """
        Parse a Turkish-formatted number.

        Args:
            text: Number string (e.g., "1.234,56" or "1234.56").

        Returns:
            Float value or None.
        """
        if not text:
            return None

        try:
            # Remove spaces
            text = text.strip()

            # Handle Turkish format: 1.234,56 -> 1234.56
            if "," in text and "." in text:
                # Turkish format: dots are thousands, comma is decimal
                text = text.replace(".", "").replace(",", ".")
            elif "," in text:
                # Comma might be decimal separator
                text = text.replace(",", ".")

            return float(text)
        except (ValueError, TypeError):
            return None


# Singleton
_provider: HedeFiyatProvider | None = None


def get_hedeffiyat_provider() -> HedeFiyatProvider:
    """Get singleton provider instance."""
    global _provider
    if _provider is None:
        _provider = HedeFiyatProvider()
    return _provider
