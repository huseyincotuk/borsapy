"""Tests for the deterministic BIST data-readiness capability.

All tests are offline and deterministic: the readiness evaluator is static and
must never perform network calls.
"""

import dataclasses
import json
import socket

import pytest

import borsapy as bp
from borsapy.readiness import (
    CAPABILITY_IDS,
    CapabilityReadiness,
    ReadinessState,
    StockAnalysisReadiness,
    evaluate_stock_readiness,
)

EXPECTED_CAPABILITY_IDS = (
    "recent_price",
    "historical_ohlcv",
    "company_profile",
    "fundamental_data",
    "financial_statements",
    "financial_ratios",
    "dividend_history",
    "corporate_actions",
    "ownership_fund_holdings",
    "kap_disclosures_news",
)


class TestReadinessState:
    """REQ-001: three-member enum that serializes deterministically."""

    def test_has_exactly_three_members(self):
        assert {member.name for member in ReadinessState} == {
            "AVAILABLE",
            "PARTIAL",
            "NOT_AVAILABLE",
        }

    def test_members_serialize_to_their_names(self):
        assert ReadinessState.AVAILABLE.value == "AVAILABLE"
        assert ReadinessState.PARTIAL.value == "PARTIAL"
        assert ReadinessState.NOT_AVAILABLE.value == "NOT_AVAILABLE"
        assert str(ReadinessState.PARTIAL) == "PARTIAL"

    def test_is_publicly_exported(self):
        assert bp.ReadinessState is ReadinessState
        assert "ReadinessState" in bp.__all__


class TestCapabilityReadiness:
    """REQ-002: immutable per-capability record with validation."""

    def _make(self, readiness, provider=None, missing=None):
        return CapabilityReadiness(
            capability_id="demo",
            readiness=readiness,
            rationale="demo rationale",
            provider=provider,
            missing_capabilities=list(missing or []),
        )

    def test_available_record_may_have_empty_missing(self):
        record = self._make(ReadinessState.AVAILABLE)
        assert record.missing_capabilities == []

    def test_partial_requires_missing_descriptors(self):
        with pytest.raises(ValueError):
            self._make(ReadinessState.PARTIAL)

    def test_not_available_requires_missing_descriptors(self):
        with pytest.raises(ValueError):
            self._make(ReadinessState.NOT_AVAILABLE)

    def test_not_available_must_have_no_provider(self):
        with pytest.raises(ValueError):
            self._make(
                ReadinessState.NOT_AVAILABLE,
                provider="borsapy.Ticker",
                missing=["something"],
            )

    def test_available_rejects_missing_descriptors(self):
        with pytest.raises(ValueError):
            self._make(ReadinessState.AVAILABLE, missing=["something"])

    def test_record_is_frozen(self):
        record = self._make(ReadinessState.AVAILABLE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.rationale = "changed"

    def test_to_dict_is_json_compatible(self):
        record = self._make(
            ReadinessState.PARTIAL,
            provider="borsapy.Ticker.info",
            missing=["more_data"],
        )
        assert record.to_dict() == {
            "capability_id": "demo",
            "readiness": "PARTIAL",
            "rationale": "demo rationale",
            "provider": "borsapy.Ticker.info",
            "missing_capabilities": ["more_data"],
        }


class TestEvaluateStockReadiness:
    """REQ-003/004/005: aggregate static evaluation."""

    def test_thyao_covers_all_ten_categories(self):
        result = evaluate_stock_readiness("THYAO")
        assert isinstance(result, StockAnalysisReadiness)
        assert result.symbol == "THYAO"
        assert len(result.capabilities) == 10
        assert tuple(c.capability_id for c in result.capabilities) == EXPECTED_CAPABILITY_IDS
        assert CAPABILITY_IDS == EXPECTED_CAPABILITY_IDS
        for capability in result.capabilities:
            assert isinstance(capability.readiness, ReadinessState)

    def test_result_exercises_all_three_states(self):
        result = evaluate_stock_readiness("THYAO")
        states = {c.readiness for c in result.capabilities}
        assert states == {
            ReadinessState.AVAILABLE,
            ReadinessState.PARTIAL,
            ReadinessState.NOT_AVAILABLE,
        }
        assert result.get("financial_ratios").readiness is ReadinessState.NOT_AVAILABLE
        assert (
            result.get("ownership_fund_holdings").readiness is ReadinessState.PARTIAL
        )
        assert result.get("historical_ohlcv").readiness is ReadinessState.AVAILABLE

    def test_overall_is_most_restrictive_state(self):
        result = evaluate_stock_readiness("THYAO")
        assert result.overall_readiness is ReadinessState.NOT_AVAILABLE

    def test_determinism_identical_serialized_output(self):
        first = evaluate_stock_readiness("THYAO")
        second = evaluate_stock_readiness("THYAO")
        assert first.to_dict() == second.to_dict()
        assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
            second.to_dict(), sort_keys=True
        )

    def test_missing_capabilities_follow_readiness(self):
        result = evaluate_stock_readiness("THYAO")
        for capability in result.capabilities:
            if capability.readiness is ReadinessState.AVAILABLE:
                assert capability.missing_capabilities == []
                assert capability.provider is not None
            else:
                assert capability.missing_capabilities
                assert all(
                    isinstance(item, str) and item
                    for item in capability.missing_capabilities
                )
            if capability.readiness is ReadinessState.NOT_AVAILABLE:
                assert capability.provider is None

    def test_frozen_aggregate(self):
        result = evaluate_stock_readiness("THYAO")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.symbol = "GARAN"

    def test_serialization_roundtrip(self):
        result = evaluate_stock_readiness("THYAO")
        payload = result.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["mode"] == "static"
        assert payload["evaluator_version"]
        assert len(payload["capabilities"]) == 10
        assert payload["overall_readiness"] == "NOT_AVAILABLE"

    def test_no_network_access_during_evaluation(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("network access attempted during evaluation")

        monkeypatch.setattr(socket, "socket", _boom)
        monkeypatch.setattr(socket, "create_connection", _boom)
        result = evaluate_stock_readiness("THYAO")
        assert result.symbol == "THYAO"

    def test_probe_opt_in_is_rejected(self):
        with pytest.raises(NotImplementedError):
            evaluate_stock_readiness("THYAO", probe=True)


class TestSymbolHandling:
    """REQ-006: invalid-symbol handling consistent with borsapy conventions."""

    def test_normalization_strips_suffix_and_uppercases(self):
        assert evaluate_stock_readiness("thyao.is").symbol == "THYAO"
        assert evaluate_stock_readiness("garan.e").symbol == "GARAN"

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n", ".E"])
    def test_empty_symbols_raise_value_error(self, bad):
        with pytest.raises(ValueError):
            evaluate_stock_readiness(bad)

    @pytest.mark.parametrize("bad", ["THY AO", "123", "@@@", "THYAO-IS", "A!"])
    def test_malformed_symbols_raise_value_error(self, bad):
        with pytest.raises(ValueError):
            evaluate_stock_readiness(bad)

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError):
            evaluate_stock_readiness(None)

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            evaluate_stock_readiness(123)


class TestBackwardCompatibility:
    """REQ-007: new API is additive; existing public names still resolve."""

    def test_existing_public_names_still_resolve(self):
        for name in ("Ticker", "Tickers", "search", "screen_stocks", "companies"):
            assert hasattr(bp, name)
            assert name in bp.__all__

    def test_new_names_are_exported(self):
        for name in (
            "ReadinessState",
            "CapabilityReadiness",
            "StockAnalysisReadiness",
            "evaluate_stock_readiness",
        ):
            assert hasattr(bp, name)
            assert name in bp.__all__
