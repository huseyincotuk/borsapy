"""Deterministic live-quote reliability assessment for TradingView snapshots.

This module adds an additive, offline capability that assesses the reliability
of live quote snapshots in the shape produced by the existing public
``Ticker.get_live_quote()`` API. The acquisition/transport layer already exists
in borsapy; this module adds *pure assessment* only:

- freshness of a quote against a caller-supplied evaluation instant and a
  caller-supplied freshness policy;
- quality of a timestamp sequence over an ordered sequence of quote dicts;
- bid/ask and price sanity for a single quote.

Everything here is deterministic, pure and offline. The layer never performs
network calls, never touches the filesystem, never reads a wall clock and never
uses randomness. Evaluation instants, freshness thresholds and trading-session
windows are all explicit caller-supplied data. There is **no baked-in market
calendar**: BIST (or any exchange) session times are caller data and are never
invented, defaulted or embedded in this module.

Quote inputs are consumed verbatim. The assessment layer never mutates,
normalizes, backfills, coerces or fabricates any quote field, and it never
reads or interprets ``auth_transport`` or any other non-specified key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FreshnessAssessment",
    "FreshnessPolicy",
    "QuoteSanityIssue",
    "QuoteSequenceAssessment",
    "QuoteSequenceIssue",
    "TradingSessionWindow",
    "assess_live_quote_freshness",
    "assess_live_quote_sanity",
    "assess_quote_sequence",
]

#: The exact set of freshness classification literals.
_FRESHNESS_CLASSIFICATIONS: tuple[str, ...] = (
    "FRESH",
    "STALE",
    "EXPIRED",
    "OUT_OF_SESSION",
)

#: The exact set of sequence-issue kind literals.
_SEQUENCE_ISSUE_KINDS: tuple[str, ...] = (
    "DUPLICATE_TIMESTAMP",
    "OUT_OF_ORDER",
    "TIMESTAMP_GAP",
)

#: The exact set of sanity-issue kind literals.
_SANITY_ISSUE_KINDS: tuple[str, ...] = (
    "CROSSED_MARKET",
    "NON_POSITIVE_PRICE",
)


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is a real integer, else raise ``TypeError``.

    ``bool`` is explicitly rejected even though it subclasses ``int``: booleans
    are never accepted as integers anywhere in this module.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a mapping, else raise ``TypeError``."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class TradingSessionWindow:
    """An immutable, caller-supplied trading-session window.

    Attributes:
        start_ms: Inclusive start of the window, in milliseconds. Required
            integer with ``start_ms < end_ms``.
        end_ms: Exclusive end of the window, in milliseconds. Required integer
            with ``start_ms < end_ms``.
        label: Optional non-empty human-readable label; ``None`` by default.

    There is no market calendar in this module. A window is pure caller data:
    it is never defaulted, invented or embedded.
    """

    start_ms: int
    end_ms: int
    label: str | None = None

    def __post_init__(self) -> None:
        start_ms = _require_int(self.start_ms, "start_ms")
        end_ms = _require_int(self.end_ms, "end_ms")
        if start_ms >= end_ms:
            raise ValueError("start_ms must be strictly less than end_ms")
        if self.label is not None:
            if not isinstance(self.label, str):
                raise TypeError("label must be a str or None")
            if not self.label:
                raise ValueError("label must be a non-empty string or None")


@dataclass(frozen=True)
class FreshnessPolicy:
    """An immutable freshness policy with exact age boundaries.

    Attributes:
        fresh_max_age_ms: Maximum age, in milliseconds, that still classifies
            ``FRESH``. Positive integer.
        stale_max_age_ms: Maximum age, in milliseconds, that still classifies
            ``STALE``. Positive integer with
            ``fresh_max_age_ms <= stale_max_age_ms``.
    """

    fresh_max_age_ms: int
    stale_max_age_ms: int

    def __post_init__(self) -> None:
        fresh = _require_int(self.fresh_max_age_ms, "fresh_max_age_ms")
        stale = _require_int(self.stale_max_age_ms, "stale_max_age_ms")
        if fresh <= 0 or stale <= 0:
            raise ValueError("freshness thresholds must be positive integers")
        if fresh > stale:
            raise ValueError(
                "fresh_max_age_ms must be less than or equal to stale_max_age_ms"
            )


@dataclass(frozen=True)
class FreshnessAssessment:
    """Immutable result of a single quote freshness assessment.

    Attributes:
        classification: Exactly one of ``"FRESH"``, ``"STALE"``, ``"EXPIRED"``,
            ``"OUT_OF_SESSION"``.
        quote_timestamp_ms: The raw quote timestamp, verbatim, without
            normalization.
        evaluated_at_ms: The caller-supplied evaluation instant.
        age_ms: ``evaluated_at_ms - quote_timestamp_ms`` when an age was
            computed, otherwise ``None`` (e.g. ``OUT_OF_SESSION``).
    """

    classification: str
    quote_timestamp_ms: int
    evaluated_at_ms: int
    age_ms: int | None

    def __post_init__(self) -> None:
        if self.classification not in _FRESHNESS_CLASSIFICATIONS:
            raise ValueError(
                "classification must be one of "
                + ", ".join(_FRESHNESS_CLASSIFICATIONS)
            )
        _require_int(self.quote_timestamp_ms, "quote_timestamp_ms")
        _require_int(self.evaluated_at_ms, "evaluated_at_ms")
        if self.age_ms is not None:
            _require_int(self.age_ms, "age_ms")


@dataclass(frozen=True)
class QuoteSequenceIssue:
    """Immutable record of a single timestamp-sequence issue.

    Attributes:
        kind: Exactly one of ``"DUPLICATE_TIMESTAMP"``, ``"OUT_OF_ORDER"``,
            ``"TIMESTAMP_GAP"``.
        index: Index of the later quote in the offending consecutive pair.
        delta_ms: Raw consecutive delta ``timestamp[i] - timestamp[i-1]``.
    """

    kind: str
    index: int
    delta_ms: int

    def __post_init__(self) -> None:
        if self.kind not in _SEQUENCE_ISSUE_KINDS:
            raise ValueError("kind must be one of " + ", ".join(_SEQUENCE_ISSUE_KINDS))
        _require_int(self.index, "index")
        _require_int(self.delta_ms, "delta_ms")


@dataclass(frozen=True)
class QuoteSequenceAssessment:
    """Immutable result of a timestamp-sequence assessment.

    Attributes:
        quote_count: Number of quotes assessed.
        issues: Ordered tuple of :class:`QuoteSequenceIssue` records, in
            ascending index order.
    """

    quote_count: int
    issues: tuple[QuoteSequenceIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_int(self.quote_count, "quote_count")
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True)
class QuoteSanityIssue:
    """Immutable record of a single bid/ask or price sanity issue.

    Attributes:
        kind: Exactly one of ``"CROSSED_MARKET"``, ``"NON_POSITIVE_PRICE"``.
        field: The offending field name (``"bid/ask"`` for a crossed market,
            otherwise one of ``"last"``, ``"bid"``, ``"ask"``).
    """

    kind: str
    field: str

    def __post_init__(self) -> None:
        if self.kind not in _SANITY_ISSUE_KINDS:
            raise ValueError("kind must be one of " + ", ".join(_SANITY_ISSUE_KINDS))
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("field must be a non-empty string")


def _validate_session_windows(sessions: Sequence[Any]) -> tuple[TradingSessionWindow, ...]:
    """Validate and sort a caller-supplied session sequence.

    Raises:
        TypeError: If ``sessions`` is not a sequence or contains a non-window
            element.
        ValueError: If any two windows overlap once sorted by ``start_ms``.
    """
    if isinstance(sessions, (str, bytes)) or not isinstance(sessions, Sequence):
        raise TypeError(
            f"sessions must be a sequence of TradingSessionWindow, got "
            f"{type(sessions).__name__}"
        )
    windows: list[TradingSessionWindow] = []
    for window in sessions:
        if not isinstance(window, TradingSessionWindow):
            raise TypeError(
                "sessions must contain only TradingSessionWindow instances"
            )
        windows.append(window)
    windows.sort(key=lambda window: window.start_ms)
    for previous, current in zip(windows, windows[1:], strict=False):
        if current.start_ms < previous.end_ms:
            raise ValueError("session windows must not overlap")
    return tuple(windows)


def _quote_timestamp_ms(quote: Mapping[str, Any]) -> int:
    """Extract and validate the ``timestamp`` field of a quote mapping.

    Raises:
        ValueError: If the ``timestamp`` key is missing.
        TypeError: If the timestamp is not a real integer (booleans rejected).
    """
    if "timestamp" not in quote:
        raise ValueError("quote is missing the 'timestamp' key")
    return _require_int(quote["timestamp"], "quote timestamp")


def assess_live_quote_freshness(
    quote: Mapping[str, Any],
    *,
    evaluated_at_ms: int,
    policy: FreshnessPolicy,
    sessions: Sequence[TradingSessionWindow] | None = None,
) -> FreshnessAssessment:
    """Assess the freshness of a live quote snapshot.

    The evaluation instant ``evaluated_at_ms`` is caller-supplied: this layer
    never reads a wall clock.

    Session gating semantics:

    1. ``sessions=None`` disables session gating entirely.
    2. ``sessions`` provided (possibly empty) enables gating; an empty provided
       set means "no trading windows" and classifies every evaluation as
       ``OUT_OF_SESSION``.
    3. With gating enabled, an ``evaluated_at_ms`` not inside any window
       classifies ``OUT_OF_SESSION``.

    Otherwise ``age_ms = evaluated_at_ms - quote_timestamp_ms`` and the
    classification boundaries are exact:

    - ``age_ms <= fresh_max_age_ms`` -> ``FRESH``
    - ``fresh_max_age_ms < age_ms <= stale_max_age_ms`` -> ``STALE``
    - ``age_ms > stale_max_age_ms`` -> ``EXPIRED``
    - ``age_ms < 0`` (a quote timestamp in the future) fails closed with
      ``ValueError``

    Args:
        quote: A quote mapping in the shape of ``Ticker.get_live_quote()``.
        evaluated_at_ms: Caller-supplied evaluation instant, in milliseconds.
        policy: A :class:`FreshnessPolicy`.
        sessions: Optional caller-supplied sequence of
            :class:`TradingSessionWindow`; ``None`` disables session gating.

    Returns:
        A :class:`FreshnessAssessment`. ``age_ms`` is ``None`` for
        ``OUT_OF_SESSION`` results.

    Raises:
        TypeError: If ``quote`` is not a mapping, ``evaluated_at_ms`` is not a
            real integer, ``policy`` is not a :class:`FreshnessPolicy`, or
            ``sessions``/window types are invalid.
        ValueError: If the quote is missing ``timestamp``, the timestamp is in
            the future relative to a gated-in evaluation instant, or session
            windows overlap.
    """
    quote = _require_mapping(quote, "quote")
    quote_timestamp_ms = _quote_timestamp_ms(quote)
    evaluated = _require_int(evaluated_at_ms, "evaluated_at_ms")
    if not isinstance(policy, FreshnessPolicy):
        raise TypeError("policy must be a FreshnessPolicy instance")

    if sessions is not None:
        windows = _validate_session_windows(sessions)
        gated_in = any(
            window.start_ms <= evaluated < window.end_ms for window in windows
        )
        if not gated_in:
            return FreshnessAssessment(
                classification="OUT_OF_SESSION",
                quote_timestamp_ms=quote_timestamp_ms,
                evaluated_at_ms=evaluated,
                age_ms=None,
            )

    age_ms = evaluated - quote_timestamp_ms
    if age_ms < 0:
        raise ValueError("quote timestamp is in the future relative to evaluated_at_ms")

    if age_ms <= policy.fresh_max_age_ms:
        classification = "FRESH"
    elif age_ms <= policy.stale_max_age_ms:
        classification = "STALE"
    else:
        classification = "EXPIRED"

    return FreshnessAssessment(
        classification=classification,
        quote_timestamp_ms=quote_timestamp_ms,
        evaluated_at_ms=evaluated,
        age_ms=age_ms,
    )


def assess_quote_sequence(
    quotes: Sequence[Mapping[str, Any]],
    *,
    max_gap_ms: int,
) -> QuoteSequenceAssessment:
    """Assess the timestamp sequence quality of an ordered quote sequence.

    For each consecutive pair, comparing quote ``i`` against quote ``i - 1``:

    - equal timestamps -> ``DUPLICATE_TIMESTAMP`` at ``i`` (delta ``0``);
    - strictly earlier -> ``OUT_OF_ORDER`` at ``i`` (negative delta preserved);
    - strictly later by more than ``max_gap_ms`` -> ``TIMESTAMP_GAP`` at ``i``
      carrying the raw consecutive delta.

    The three kinds are mutually exclusive per pair. Issues are listed in
    ascending index order. An empty sequence and a single-quote sequence are
    valid with zero issues. Gaps are raw consecutive deltas; session awareness
    is not applied here.

    Args:
        quotes: An ordered sequence of quote mappings.
        max_gap_ms: Positive integer maximum tolerated consecutive gap.

    Returns:
        A :class:`QuoteSequenceAssessment`.

    Raises:
        TypeError: If ``quotes`` is not a sequence, an element is not a
            mapping, ``max_gap_ms`` is not a real integer, or a timestamp is not
            a real integer (booleans rejected).
        ValueError: If ``max_gap_ms`` is not positive or an element is missing
            the ``timestamp`` key.
    """
    max_gap = _require_int(max_gap_ms, "max_gap_ms")
    if max_gap <= 0:
        raise ValueError("max_gap_ms must be a positive integer")
    if isinstance(quotes, (str, bytes)) or not isinstance(quotes, Sequence):
        raise TypeError(
            f"quotes must be a sequence of mappings, got {type(quotes).__name__}"
        )

    timestamps: list[int] = []
    for quote in quotes:
        mapping = _require_mapping(quote, "quote")
        timestamps.append(_quote_timestamp_ms(mapping))

    issues: list[QuoteSequenceIssue] = []
    for index in range(1, len(timestamps)):
        delta_ms = timestamps[index] - timestamps[index - 1]
        if delta_ms == 0:
            issues.append(
                QuoteSequenceIssue(
                    kind="DUPLICATE_TIMESTAMP", index=index, delta_ms=0
                )
            )
        elif delta_ms < 0:
            issues.append(
                QuoteSequenceIssue(
                    kind="OUT_OF_ORDER", index=index, delta_ms=delta_ms
                )
            )
        elif delta_ms > max_gap:
            issues.append(
                QuoteSequenceIssue(
                    kind="TIMESTAMP_GAP", index=index, delta_ms=delta_ms
                )
            )

    return QuoteSequenceAssessment(quote_count=len(timestamps), issues=tuple(issues))


def assess_live_quote_sanity(
    quote: Mapping[str, Any],
) -> tuple[QuoteSanityIssue, ...]:
    """Assess bid/ask and price sanity for a single quote snapshot.

    Emission order is exact. When both ``bid`` and ``ask`` are present and not
    ``None`` and ``bid >= ask``, a ``CROSSED_MARKET`` issue with field
    ``"bid/ask"`` is emitted first. Then, for each of ``"last"``, ``"bid"``,
    ``"ask"`` checked in that exact order, a present non-``None`` value ``<= 0``
    emits ``NON_POSITIVE_PRICE`` with that field name.

    Absent keys and ``None`` values are skipped, never inferred.

    Args:
        quote: A quote mapping in the shape of ``Ticker.get_live_quote()``.

    Returns:
        A tuple of :class:`QuoteSanityIssue` records; empty when no issues
        apply.

    Raises:
        TypeError: If ``quote`` is not a mapping.
    """
    quote = _require_mapping(quote, "quote")

    issues: list[QuoteSanityIssue] = []

    bid = quote.get("bid")
    ask = quote.get("ask")
    if bid is not None and ask is not None and bid >= ask:
        issues.append(QuoteSanityIssue(kind="CROSSED_MARKET", field="bid/ask"))

    for field_name in ("last", "bid", "ask"):
        value = quote.get(field_name)
        if value is not None and value <= 0:
            issues.append(
                QuoteSanityIssue(kind="NON_POSITIVE_PRICE", field=field_name)
            )

    return tuple(issues)
