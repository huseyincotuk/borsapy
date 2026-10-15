"""Doviz.com Bond/Tahvil provider for borsapy."""

from typing import Any

from bs4 import BeautifulSoup

from borsapy._providers.base import BaseProvider
from borsapy.cache import TTL
from borsapy.exceptions import APIError, DataNotAvailableError


class DovizcomTahvilProvider(BaseProvider):
    """
    Provider for Turkish government bond yields from doviz.com.

    URL: https://www.doviz.com/tahvil
    """

    BASE_URL = "https://www.doviz.com/tahvil"

    # Maturity mapping
    MATURITY_MAP = {
        "2Y": ["2 Yıllık", "2 yıllık", "2-yillik"],
        "5Y": ["5 Yıllık", "5 yıllık", "5-yillik"],
        "10Y": ["10 Yıllık", "10 yıllık", "10-yillik"],
    }

    def _parse_float(self, text: str) -> float | None:
        """Parse float from Turkish-formatted text."""
        try:
            cleaned = text.strip().replace(",", ".").replace("%", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def _get_maturity(self, name: str) -> str | None:
        """Get maturity code from bond name."""
        for maturity, patterns in self.MATURITY_MAP.items():
            if any(p in name for p in patterns):
                return maturity
        return None

    def get_bond_yields(self) -> list[dict[str, Any]]:
        """
        Get bond yields for the countries listed on the page.

        Each returned row carries a ``country`` field holding the first
        whitespace-separated token of the bond name (e.g. ``"TR"`` for
        ``"TR 2 Yıllık Tahvil Faizi"``, ``"ABD"`` for the US rows).

        Returns:
            List of bond data with name, country, maturity, yield, change, etc.

        Raises:
            APIError: If API request fails.
            DataNotAvailableError: If bond data not found.
        """
        cache_key = "dovizcom:tahvil:all"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            response = self._get(self.BASE_URL)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the bonds table
            table = soup.find("table", {"id": "bonds"})
            if not table:
                raise DataNotAvailableError("Bond table not found on page")

            tbody = table.find("tbody")
            if not tbody:
                raise DataNotAvailableError("Bond data not found")

            bonds = []

            for row in tbody.find_all("tr"):
                try:
                    cells = row.find_all("td")
                    if len(cells) < 3:
                        continue

                    # Parse bond name and URL. The anchor no longer carries a
                    # class; fall back to the row's data-name attribute.
                    name_anchor = cells[0].find("a")
                    name = name_anchor.text.strip() if name_anchor else ""
                    if not name:
                        name = (row.get("data-name") or "").strip()
                    if not name:
                        continue
                    url = name_anchor.get("href", "") if name_anchor else ""

                    # Parse current yield
                    yield_text = cells[1].text.strip()
                    yield_rate = self._parse_float(yield_text)

                    # Parse change percentage
                    change_text = cells[2].text.strip()
                    change_pct = self._parse_float(change_text)

                    # Get maturity
                    maturity = self._get_maturity(name)

                    # Country is the first whitespace-separated token of the name
                    name_tokens = name.split()
                    country = name_tokens[0] if name_tokens else None

                    bond_data = {
                        "name": name,
                        "country": country,
                        "maturity": maturity,
                        "yield": yield_rate,
                        "yield_decimal": (
                            yield_rate / 100 if yield_rate is not None else None
                        ),
                        "change": (
                            yield_rate * (change_pct / 100)
                            if yield_rate is not None and change_pct is not None
                            else None
                        ),
                        "change_pct": change_pct,
                        "url": url,
                    }

                    bonds.append(bond_data)

                except Exception:
                    continue

            if not bonds:
                raise DataNotAvailableError("No bond data found")

            self._cache_set(cache_key, bonds, TTL.FX_RATES)
            return bonds

        except (DataNotAvailableError, APIError):
            raise
        except Exception as e:
            raise APIError(f"Failed to fetch bond yields: {e}") from e

    def get_bond(self, maturity: str) -> dict[str, Any]:
        """
        Get a specific bond by maturity.

        Args:
            maturity: Bond maturity (2Y, 5Y, 10Y).

        Returns:
            Bond data dict.

        Raises:
            DataNotAvailableError: If bond not found.
        """
        maturity = maturity.upper()
        bonds = self.get_bond_yields()

        for bond in bonds:
            if bond["maturity"] == maturity and bond.get("country") == "TR":
                return bond

        raise DataNotAvailableError(f"Bond with maturity {maturity} not found")

    def get_10y_yield(self) -> float | None:
        """
        Get current 10-year Turkish government bond yield as decimal.

        Useful for DCF calculations.

        Returns:
            10Y bond yield as decimal (e.g., 0.28 for 28%).
        """
        try:
            bond = self.get_bond("10Y")
            return bond.get("yield_decimal")
        except DataNotAvailableError:
            return None


# Singleton instance
_provider: DovizcomTahvilProvider | None = None


def get_tahvil_provider() -> DovizcomTahvilProvider:
    """Get the singleton tahvil provider instance."""
    global _provider
    if _provider is None:
        _provider = DovizcomTahvilProvider()
    return _provider
