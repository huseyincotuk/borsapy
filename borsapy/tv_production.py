"""Deterministic production gate composition for TradingView live quotes.

This module adds an additive, offline capability that turns a single live quote
snapshot (in the shape produced by the existing public
``Ticker.get_live_quote()`` API) into ONE immutable, machine-readable governed
record. It composes the frozen reliability assessment layer
(:mod:`borsapy.tv_reliability`) into an exact verdict classification:

- freshness of the quote against a caller-supplied evaluation instant and a
  caller-supplied freshness policy;
- bid/ask and price sanity of the quote;
- (optionally) the timestamp-sequence quality of the new quote relative to a
  caller-supplied sequence of prior quotes.

Everything here is deterministic, pure and offline. The layer never performs
network calls, never touches the filesystem, never reads a wall clock and never
uses randomness. Acquisition belongs entirely to the caller-supplied ``source``
callable; evaluation instants, freshness thresholds, session windows and prior
quotes are all explicit caller-supplied data. There is **no baked-in market
calendar**: BIST (or any exchange) session times are caller data and are never
invented, defaulted or embedded in this module.

The source mapping is consumed verbatim. The layer never mutates, normalizes,
backfills, coerces or fabricates any quote field, and it never interprets
``auth_transport`` (or any other key) as entitlement or realtime proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from borsapy.tv_reliability import (
    FreshnessAssessment,
    FreshnessPolicy,
    QuoteSanityIssue,
    QuoteSequenceIssue,
    TradingSessionWindow,
    assess_live_quote_freshness,
    assess_live_quote_sanity,
    assess_quote_sequence,
)

__all__ = [
    "GovernedLiveQuote",
    "govern_live_quote",
]

#: The exact set of verdict literals emitted by this module.
_VERDICTS: tuple[str, ...] = (
    "USABLE",
    "DEGRADED",
    "UNUSABLE",
)


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is a real integer, else raise ``TypeError``.

    ``bool`` is explicitly rejected even though it subclasses ``int``: booleans
    are never accepted as integers anywhere in this module.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class GovernedLiveQuote:
    """Immutable, machine-readable governed representation of one live quote.

    The record is a pure composition of the frozen reliability assessments and
    the verbatim source quote metadata. It carries no recommendation, score or
    presentation text.

    Attributes:
        symbol: The source quote's ``symbol`` value carried verbatim when the
            key is present (including an explicit ``None``), otherwise ``None``.
            Never invented or coerced.
        auth_transport: The source quote's ``auth_transport`` value carried
            verbatim when the key is present (including an explicit ``None``),
            otherwise ``None``. Never interpreted as entitlement or realtime
            proof.
        quote_timestamp_ms: The raw quote timestamp, carried from the freshness
            assessment.
        evaluated_at_ms: The caller-supplied evaluation instant.
        freshness: The :class:`~borsapy.tv_reliability.FreshnessAssessment`
            returned verbatim by the frozen freshness assessment.
        sanity_issues: Tuple of
            :class:`~borsapy.tv_reliability.QuoteSanityIssue` records, verbatim
            from the frozen sanity assessment, in its exact emission order.
        sequence_issues: Tuple of
            :class:`~borsapy.tv_reliability.QuoteSequenceIssue` records caused
            by the NEW quote's timestamp relative to the previous quote. Empty
            when no prior quotes were supplied.
        verdict: Exactly one of ``"USABLE"``, ``"DEGRADED"``, ``"UNUSABLE"``.
    """

    symbol: str | None
    auth_transport: str | None
    quote_timestamp_ms: int
    evaluated_at_ms: int
    freshness: FreshnessAssessment
    sanity_issues: tuple[QuoteSanityIssue, ...]
    sequence_issues: tuple[QuoteSequenceIssue, ...]
    verdict: str

    def __post_init__(self) -> None:
        _require_int(self.quote_timestamp_ms, "quote_timestamp_ms")
        _require_int(self.evaluated_at_ms, "evaluated_at_ms")
        if not isinstance(self.freshness, FreshnessAssessment):
            raise TypeError("freshness must be a FreshnessAssessment instance")
        if self.verdict not in _VERDICTS:
            raise ValueError("verdict must be one of " + ", ".join(_VERDICTS))

        sanity_issues = tuple(self.sanity_issues)
        if not all(isinstance(issue, QuoteSanityIssue) for issue in sanity_issues):
            raise TypeError(
                "sanity_issues must contain only QuoteSanityIssue instances"
            )
        sequence_issues = tuple(self.sequence_issues)
        if not all(isinstance(issue, QuoteSequenceIssue) for issue in sequence_issues):
            raise TypeError(
                "sequence_issues must contain only QuoteSequenceIssue instances"
            )
        # Detach the inputs so the frozen record owns immutable tuples.
        object.__setattr__(self, "sanity_issues", sanity_issues)
        object.__setattr__(self, "sequence_issues", sequence_issues)


def govern_live_quote(
    source: Callable[[], Mapping[str, Any]],
    *,
    evaluated_at_ms: int,
    policy: FreshnessPolicy,
    sessions: Sequence[TradingSessionWindow] | None = None,
    prior_quotes: Sequence[Mapping[str, Any]] = (),
    max_gap_ms: int | None = None,
) -> GovernedLiveQuote:
    """Assess one live quote and emit a single governed verdict record.

    The layer calls ``source`` exactly once, consumes the returned mapping
    verbatim, assesses it with the frozen reliability layer and returns one
    immutable :class:`GovernedLiveQuote`. It performs no network access of its
    own and never reads a wall clock: ``evaluated_at_ms`` is caller-supplied and
    all acquisition belongs to ``source``.

    Verdict lattice (exact):

    * ``UNUSABLE`` iff any of: ``freshness.classification == "EXPIRED"``;
      ``freshness.classification == "OUT_OF_SESSION"``; ANY sanity issue
      (``CROSSED_MARKET`` or ``NON_POSITIVE_PRICE``); ANY sequence issue of kind
      ``OUT_OF_ORDER``.
    * ``DEGRADED`` iff not ``UNUSABLE`` and any of:
      ``freshness.classification == "STALE"``; any sequence issue of kind
      ``DUPLICATE_TIMESTAMP`` or ``TIMESTAMP_GAP``.
    * ``USABLE`` otherwise (freshness ``FRESH``, zero sanity issues, zero
      sequence issues).

    ``UNUSABLE`` takes precedence over ``DEGRADED`` when both would apply.

    Session gating semantics are identical to the frozen
    :func:`~borsapy.tv_reliability.assess_live_quote_freshness`: ``None``
    disables gating, while a provided (possibly empty) sequence enables it and a
    provided empty sequence classifies every evaluation ``OUT_OF_SESSION``.

    Sequence assessment is performed only when ``prior_quotes`` is non-empty:
    the layer assesses ``tuple(prior_quotes) + (quote,)`` and reports ONLY the
    issues whose ``index`` equals ``len(prior_quotes)`` (the issues caused by the
    new quote's timestamp relative to the previous quote). Earlier-index issues
    belong to earlier ticks and are not re-reported. ``max_gap_ms`` is required
    to be a positive int when ``prior_quotes`` is non-empty and must be ``None``
    otherwise; violations fail closed.

    Args:
        source: A no-argument callable returning a quote mapping in the shape of
            ``Ticker.get_live_quote()``. Called exactly once. Exceptions raised
            by ``source`` propagate unchanged: there is no retry and no
            fabricated failure record.
        evaluated_at_ms: Caller-supplied evaluation instant, in milliseconds.
        policy: A :class:`~borsapy.tv_reliability.FreshnessPolicy`.
        sessions: Optional caller-supplied sequence of
            :class:`~borsapy.tv_reliability.TradingSessionWindow`;
            ``None`` disables session gating.
        prior_quotes: Optional ordered sequence of earlier quote mappings.
        max_gap_ms: Positive integer maximum tolerated consecutive gap, required
            exactly when ``prior_quotes`` is non-empty.

    Returns:
        A :class:`GovernedLiveQuote`.

    Raises:
        TypeError: If ``source`` is not callable, ``prior_quotes`` is not a
            sequence, ``max_gap_ms`` is required but missing or not an integer,
            or the frozen assessment layer rejects a non-mapping quote or an
            invalid ``policy``.
        ValueError: If ``max_gap_ms`` is not positive, is supplied while
            ``prior_quotes`` is empty, or the frozen assessment layer rejects
            an out-of-domain value.
    """
    if not callable(source):
        raise TypeError(
            f"source must be a callable returning a mapping, got {type(source).__name__}"
        )
    if isinstance(prior_quotes, (str, bytes)) or not isinstance(prior_quotes, Sequence):
        raise TypeError(
            f"prior_quotes must be a sequence of mappings, got "
            f"{type(prior_quotes).__name__}"
        )
    prior = tuple(prior_quotes)

    if prior:
        if max_gap_ms is None:
            raise TypeError("max_gap_ms is required when prior_quotes is non-empty")
        gap = _require_int(max_gap_ms, "max_gap_ms")
        if gap <= 0:
            raise ValueError("max_gap_ms must be a positive integer")
    else:
        if max_gap_ms is not None:
            raise ValueError("max_gap_ms must be None when prior_quotes is empty")
        gap = None

    # Call the caller-supplied source exactly once; let any exception propagate.
    quote = source()

    freshness = assess_live_quote_freshness(
        quote,
        evaluated_at_ms=evaluated_at_ms,
        policy=policy,
        sessions=sessions,
    )
    sanity_issues = assess_live_quote_sanity(quote)

    symbol = quote["symbol"] if "symbol" in quote else None
    auth_transport = quote["auth_transport"] if "auth_transport" in quote else None

    if prior:
        assessment = assess_quote_sequence(prior + (quote,), max_gap_ms=gap)
        new_index = len(prior)
        sequence_issues = tuple(
            issue for issue in assessment.issues if issue.index == new_index
        )
    else:
        sequence_issues = ()

    freshness_unusable = freshness.classification in ("EXPIRED", "OUT_OF_SESSION")
    sanity_unusable = any(
        issue.kind in ("CROSSED_MARKET", "NON_POSITIVE_PRICE")
        for issue in sanity_issues
    )
    sequence_unusable = any(
        issue.kind == "OUT_OF_ORDER" for issue in sequence_issues
    )

    if freshness_unusable or sanity_unusable or sequence_unusable:
        verdict = "UNUSABLE"
    else:
        sequence_degraded = any(
            issue.kind in ("DUPLICATE_TIMESTAMP", "TIMESTAMP_GAP")
            for issue in sequence_issues
        )
        if freshness.classification == "STALE" or sequence_degraded:
            verdict = "DEGRADED"
        else:
            verdict = "USABLE"

    return GovernedLiveQuote(
        symbol=symbol,
        auth_transport=auth_transport,
        quote_timestamp_ms=freshness.quote_timestamp_ms,
        evaluated_at_ms=freshness.evaluated_at_ms,
        freshness=freshness,
        sanity_issues=sanity_issues,
        sequence_issues=sequence_issues,
        verdict=verdict,
    )
