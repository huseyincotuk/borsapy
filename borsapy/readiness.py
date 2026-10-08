"""Deterministic BIST stock-analysis data-readiness evaluation.

This module adds an additive, offline capability that answers one question for
a BIST equity symbol: which of the analysis inputs required by a stock-research
workflow can currently be obtained through borsapy, and which cannot.

The evaluation is *static*. It is derived only from borsapy's existing public
API and provider structure; it never performs network calls. Identical inputs
therefore always produce identical outputs, so the result can be consumed by a
later deterministic analysis/orchestration layer.

Nothing in this module produces investment advice: there are no buy/sell/hold
signals, price targets, rankings or numeric scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "CAPABILITY_IDS",
    "EVALUATOR_VERSION",
    "CapabilityReadiness",
    "ReadinessState",
    "StockAnalysisReadiness",
    "evaluate_stock_readiness",
]

#: Version of the static capability mapping and evaluation contract.
EVALUATOR_VERSION = "1.0.0"

#: Modes an evaluation can run in. Only the deterministic static mode exists.
EVALUATION_MODE = "static"


class ReadinessState(str, Enum):
    """Availability of a required BIST analysis capability in borsapy.

    The three members serialize deterministically to their own names so the
    result is stable across JSON boundaries.
    """

    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"

    def __str__(self) -> str:
        return self.value


#: Restrictiveness ordering used to derive the aggregate status without scoring.
_RESTRICTIVENESS: dict[ReadinessState, int] = {
    ReadinessState.AVAILABLE: 0,
    ReadinessState.PARTIAL: 1,
    ReadinessState.NOT_AVAILABLE: 2,
}


@dataclass(frozen=True)
class CapabilityReadiness:
    """Immutable readiness result for a single analysis capability.

    Attributes:
        capability_id: Stable identifier of the capability category.
        readiness: One of :class:`ReadinessState`.
        rationale: Human-readable justification for the classification.
        provider: Concrete borsapy provider/interface path, or ``None`` when the
            capability is not available.
        missing_capabilities: Machine-readable descriptors of what is missing.
            Always non-empty for ``PARTIAL`` and ``NOT_AVAILABLE``; empty for
            ``AVAILABLE``.
    """

    capability_id: str
    readiness: ReadinessState
    rationale: str
    provider: str | None
    missing_capabilities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id:
            raise ValueError("capability_id must be a non-empty string")
        if not isinstance(self.readiness, ReadinessState):
            raise TypeError("readiness must be a ReadinessState member")
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be a non-empty string")
        if self.provider is not None and not isinstance(self.provider, str):
            raise TypeError("provider must be a string or None")

        missing = list(self.missing_capabilities)
        if self.readiness is ReadinessState.AVAILABLE:
            if missing:
                raise ValueError(
                    "AVAILABLE capabilities must not declare missing_capabilities"
                )
        else:
            if not missing:
                raise ValueError(
                    "PARTIAL/NOT_AVAILABLE capabilities must declare at least one "
                    "missing capability"
                )
            if self.readiness is ReadinessState.NOT_AVAILABLE and self.provider is not None:
                raise ValueError("NOT_AVAILABLE capabilities must have provider=None")

        # Detach the input list so the frozen record owns an independent copy.
        object.__setattr__(self, "missing_capabilities", missing)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return {
            "capability_id": self.capability_id,
            "readiness": self.readiness.value,
            "rationale": self.rationale,
            "provider": self.provider,
            "missing_capabilities": list(self.missing_capabilities),
        }


@dataclass(frozen=True)
class StockAnalysisReadiness:
    """Immutable aggregate data-readiness result for one BIST equity symbol.

    Attributes:
        symbol: Normalized BIST symbol the result describes.
        capabilities: All per-capability results, covering every required
            analysis category.
        overall_readiness: Most restrictive state present across capabilities.
        evaluator_version: Version of the static mapping used.
        mode: Evaluation mode (always ``"static"``).
    """

    symbol: str
    capabilities: tuple[CapabilityReadiness, ...]
    overall_readiness: ReadinessState
    evaluator_version: str = EVALUATOR_VERSION
    mode: str = EVALUATION_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.capabilities, tuple):
            raise TypeError("capabilities must be a tuple")
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        if not all(isinstance(c, CapabilityReadiness) for c in self.capabilities):
            raise TypeError("capabilities must contain CapabilityReadiness records")
        if not isinstance(self.overall_readiness, ReadinessState):
            raise TypeError("overall_readiness must be a ReadinessState member")

    def get(self, capability_id: str) -> CapabilityReadiness | None:
        """Return the per-capability result for ``capability_id``, if present."""
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return {
            "symbol": self.symbol,
            "overall_readiness": self.overall_readiness.value,
            "evaluator_version": self.evaluator_version,
            "mode": self.mode,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }


@dataclass(frozen=True)
class _CapabilitySpec:
    """Static, immutable definition of one capability category."""

    capability_id: str
    readiness: ReadinessState
    rationale: str
    provider: str | None
    missing_capabilities: tuple[str, ...]

    def to_record(self) -> CapabilityReadiness:
        return CapabilityReadiness(
            capability_id=self.capability_id,
            readiness=self.readiness,
            rationale=self.rationale,
            provider=self.provider,
            missing_capabilities=list(self.missing_capabilities),
        )


#: Ordered static mapping from the ten required analysis categories to the
#: capabilities currently exposed by borsapy. Evidence paths refer to the
#: existing implementation inspected in this repository.
_CAPABILITY_SPECS: tuple[_CapabilitySpec, ...] = (
    _CapabilitySpec(
        capability_id="recent_price",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.get_live_quote() and Ticker.info expose current quote fields "
            "(last/bid/ask/volume) through the TradingView provider."
        ),
        provider="borsapy._providers.tradingview.TradingViewProvider.get_quote",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="historical_ohlcv",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.history(period, interval, ...) returns Open/High/Low/Close/"
            "Volume DataFrames via the TradingView provider."
        ),
        provider="borsapy._providers.tradingview.TradingViewProvider.get_history",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="company_profile",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.info exposes sector, industry, website and longBusinessSummary "
            "sourced from KAP company details; bp.companies()/search_companies() "
            "list BIST companies."
        ),
        provider="borsapy._providers.kap.KAPProvider.get_company_details",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="fundamental_data",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.info/fast_info expose valuation and balance fundamentals "
            "(marketCap, trailingPE, priceToBook, enterpriseToEbitda, netDebt, "
            "sharesOutstanding) via İş Yatırım company metrics."
        ),
        provider="borsapy._providers.isyatirim.IsYatirimProvider.get_company_metrics",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="financial_statements",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.get_balance_sheet(), get_income_stmt() and get_cashflow() "
            "(annual/quarterly plus TTM helpers) return statements via the "
            "İş Yatırım provider."
        ),
        provider="borsapy._providers.isyatirim.IsYatirimProvider.get_financial_statements",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="financial_ratios",
        readiness=ReadinessState.NOT_AVAILABLE,
        rationale=(
            "borsapy has no financial-ratio capability for a given symbol. Only a "
            "few valuation multiples (PE, PB, EV/EBITDA) are surfaced as raw "
            "fundamental fields (see fundamental_data); no ratio API or ratio "
            "computation exists. The İş Yatırım screener only uses ratio fields as "
            "cohort filter bounds and returns matching rows, not on-demand "
            "per-symbol ratios."
        ),
        provider=None,
        missing_capabilities=(
            "profitability_ratios",
            "liquidity_ratios",
            "leverage_ratios",
            "efficiency_ratios",
        ),
    ),
    _CapabilitySpec(
        capability_id="dividend_history",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.dividends returns cash dividend history (Amount, GrossRate, "
            "NetRate, TotalDividend) and Ticker.info exposes dividend yield/ex-date "
            "via İş Yatırım."
        ),
        provider="borsapy._providers.isyatirim.IsYatirimProvider.get_dividends",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="corporate_actions",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.splits returns capital-increase history (rights issues, bonus "
            "shares) and Ticker.actions combines dividends and splits; "
            "Ticker.history(actions=True) merges them into OHLCV."
        ),
        provider="borsapy._providers.isyatirim.IsYatirimProvider.get_capital_increases",
        missing_capabilities=(),
    ),
    _CapabilitySpec(
        capability_id="ownership_fund_holdings",
        readiness=ReadinessState.PARTIAL,
        rationale=(
            "Ticker.major_holders exposes the ownership breakdown from İş Yatırım "
            "and Ticker.etf_holders lists international ETFs holding the stock via "
            "TradingView; licensed MKK custody/takas data and Turkish fund "
            "portfolio holdings of a stock are not supported."
        ),
        provider="borsapy.Ticker.major_holders/etf_holders",
        missing_capabilities=(
            "mkk_takas_custody",
            "fund_portfolio_holdings",
        ),
    ),
    _CapabilitySpec(
        capability_id="kap_disclosures_news",
        readiness=ReadinessState.AVAILABLE,
        rationale=(
            "Ticker.news returns recent KAP disclosures, get_news_content() returns "
            "full disclosure HTML, and Ticker.calendar lists expected disclosure "
            "dates via the KAP provider."
        ),
        provider="borsapy._providers.kap.KAPProvider.get_disclosures",
        missing_capabilities=(),
    ),
)

#: Immutable, ordered list of every required capability identifier.
CAPABILITY_IDS: tuple[str, ...] = tuple(spec.capability_id for spec in _CAPABILITY_SPECS)

# BIST equity symbols start with a letter and contain only letters/digits.
_SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9]{0,11}")


def _normalize_symbol(symbol: str) -> str:
    """Normalize and validate a BIST symbol following borsapy conventions.

    Mirrors the normalization used across borsapy providers
    (``upper().replace(".IS", "").replace(".E", "")``) and rejects malformed
    input with a ``ValueError`` since no matching borsapy exception exists.
    """
    if symbol is None:
        raise ValueError("symbol must be a non-empty string, got None")
    if not isinstance(symbol, str):
        raise TypeError(f"symbol must be a str, got {type(symbol).__name__}")

    normalized = symbol.strip().upper().replace(".IS", "").replace(".E", "")
    if not normalized:
        raise ValueError("symbol must be a non-empty string")
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError(f"invalid BIST symbol: {symbol!r}")
    return normalized


def _derive_overall(capabilities: tuple[CapabilityReadiness, ...]) -> ReadinessState:
    """Return the most restrictive readiness state across ``capabilities``."""
    return max(
        (capability.readiness for capability in capabilities),
        key=lambda state: _RESTRICTIVENESS[state],
    )


def evaluate_stock_readiness(
    symbol: str,
    *,
    probe: bool = False,
) -> StockAnalysisReadiness:
    """Evaluate borsapy data readiness for a BIST equity symbol.

    The evaluation is fully static and deterministic: it reads the immutable
    capability mapping and never performs network calls, so identical inputs
    yield structurally identical outputs regardless of time or connectivity.

    Args:
        symbol: BIST equity symbol, e.g. ``"THYAO"``. A ``.IS``/``.E`` suffix is
            accepted and stripped, matching existing borsapy conventions.
        probe: Reserved for a future opt-in runtime probe. Runtime probing is not
            an established deterministic borsapy pattern, so ``True`` raises
            ``NotImplementedError`` rather than performing hidden network calls.

    Returns:
        A :class:`StockAnalysisReadiness` aggregate covering all ten required
        analysis categories.

    Raises:
        ValueError: If ``symbol`` is empty, ``None``, whitespace-only or
            malformed.
        TypeError: If ``symbol`` is not a string.
        NotImplementedError: If ``probe=True``.
    """
    if probe:
        raise NotImplementedError(
            "Runtime probing is not supported: readiness is derived statically "
            "from borsapy's provider structure and never performs network calls."
        )

    normalized = _normalize_symbol(symbol)
    capabilities = tuple(spec.to_record() for spec in _CAPABILITY_SPECS)
    return StockAnalysisReadiness(
        symbol=normalized,
        capabilities=capabilities,
        overall_readiness=_derive_overall(capabilities),
    )
