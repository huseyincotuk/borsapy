"""Tests for TCMB interest rates."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from borsapy.tcmb import TCMB, policy_rate

# =============================================================================
# TCMB Class Tests
# =============================================================================


class TestTCMB:
    """Tests for TCMB class."""

    @pytest.fixture
    def tcmb(self):
        """TCMB instance for testing."""
        return TCMB()

    def test_policy_rate_type(self, tcmb):
        """Test policy_rate returns float or None."""
        rate = tcmb.policy_rate
        assert rate is None or isinstance(rate, float)

    def test_policy_rate_range(self, tcmb):
        """Test policy_rate is in reasonable range if available."""
        rate = tcmb.policy_rate
        if rate is not None:
            assert 0 < rate < 100, f"Policy rate {rate}% seems unreasonable"

    def test_overnight_structure(self, tcmb):
        """Test overnight returns dict with borrowing and lending."""
        overnight = tcmb.overnight
        assert isinstance(overnight, dict)
        assert "borrowing" in overnight
        assert "lending" in overnight

    def test_overnight_values(self, tcmb):
        """Test overnight values are float or None."""
        overnight = tcmb.overnight
        assert overnight["borrowing"] is None or isinstance(overnight["borrowing"], float)
        assert overnight["lending"] is None or isinstance(overnight["lending"], float)

    def test_late_liquidity_structure(self, tcmb):
        """Test late_liquidity returns dict with borrowing and lending."""
        late_liquidity = tcmb.late_liquidity
        assert isinstance(late_liquidity, dict)
        assert "borrowing" in late_liquidity
        assert "lending" in late_liquidity

    def test_late_liquidity_values(self, tcmb):
        """Test late_liquidity values are float or None."""
        late_liquidity = tcmb.late_liquidity
        assert late_liquidity["borrowing"] is None or isinstance(late_liquidity["borrowing"], float)
        assert late_liquidity["lending"] is None or isinstance(late_liquidity["lending"], float)

    def test_rates_dataframe(self, tcmb):
        """Test rates returns DataFrame with expected columns."""
        rates = tcmb.rates
        assert isinstance(rates, pd.DataFrame)
        assert "type" in rates.columns
        assert "borrowing" in rates.columns
        assert "lending" in rates.columns

    def test_rates_row_count(self, tcmb):
        """Test rates has 3 rows (policy, overnight, late_liquidity)."""
        rates = tcmb.rates
        assert len(rates) == 3

    def test_history_returns_dataframe(self, tcmb):
        """Test history returns DataFrame."""
        hist = tcmb.history("policy")
        assert isinstance(hist, pd.DataFrame)

    def test_history_columns(self, tcmb):
        """Test history DataFrame has expected columns."""
        hist = tcmb.history("policy")
        if not hist.empty:
            assert "borrowing" in hist.columns or "lending" in hist.columns

    def test_history_period_filter(self, tcmb):
        """Test history with period filter."""
        hist_all = tcmb.history("policy")
        hist_1y = tcmb.history("policy", period="1y")
        # 1y should have equal or fewer rows than all
        assert len(hist_1y) <= len(hist_all)

    def test_history_invalid_rate_type(self, tcmb):
        """Test history raises error for invalid rate_type."""
        with pytest.raises(ValueError):
            tcmb.history("invalid_type")

    def test_repr(self, tcmb):
        """Test string representation."""
        repr_str = repr(tcmb)
        assert "TCMB" in repr_str


# =============================================================================
# Shortcut Function Tests
# =============================================================================


class TestPolicyRateFunction:
    """Tests for policy_rate() shortcut function."""

    def test_returns_float_or_none(self):
        """Test policy_rate() returns float or None."""
        rate = policy_rate()
        assert rate is None or isinstance(rate, float)

    def test_matches_class_property(self):
        """Test shortcut function matches class property."""
        func_rate = policy_rate()
        class_rate = TCMB().policy_rate
        assert func_rate == class_rate


# =============================================================================
# get_inflation Convenience Function Tests (offline, mocked provider)
# =============================================================================


class TestGetInflation:
    """Offline tests for the get_inflation() module-level function."""

    def test_exported_from_package(self):
        """get_inflation is part of the public API surface."""
        import borsapy

        assert "get_inflation" in borsapy.__all__
        assert callable(borsapy.get_inflation)

    def test_delegates_tufe_defaults(self):
        from borsapy.inflation import get_inflation

        fake_provider = MagicMock()
        fake_provider.get_data.return_value = pd.DataFrame()
        with patch("borsapy.inflation.get_tcmb_provider", return_value=fake_provider):
            get_inflation()
        fake_provider.get_data.assert_called_once_with("tufe", None, None, None)

    def test_delegates_ufe_with_filters(self):
        from borsapy.inflation import get_inflation

        fake_provider = MagicMock()
        fake_provider.get_data.return_value = pd.DataFrame()
        with patch("borsapy.inflation.get_tcmb_provider", return_value=fake_provider):
            frame = get_inflation("ufe", start="2023-01-01", end="2023-12-31", limit=6)
        fake_provider.get_data.assert_called_once_with(
            "ufe", "2023-01-01", "2023-12-31", 6
        )
        assert isinstance(frame, pd.DataFrame)

    def test_invalid_type_raises_value_error(self):
        """Validation lives in the provider and runs before any network call."""
        from borsapy._providers.tcmb import TCMBProvider
        from borsapy.inflation import get_inflation

        provider = TCMBProvider()
        with patch("borsapy.inflation.get_tcmb_provider", return_value=provider):
            with pytest.raises(ValueError, match="Invalid type"):
                get_inflation("cpi")

    def test_calculate_inflation_docstring_flags_flakiness(self):
        """The intermittent-forbidden availability note stays in the docstring."""
        from borsapy._providers.tcmb import TCMBProvider

        doc = TCMBProvider.calculate_inflation.__doc__ or ""
        assert "intermittently" in doc
        assert "APIError" in doc
