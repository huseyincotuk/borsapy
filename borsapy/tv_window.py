"""Deterministic tick-window state machine for TradingView live quotes.

This module adds the next pure slice of the TradingView production pipeline on
top of the frozen governed live-quote gate
(:mod:`borsapy.tv_production`): a bounded, chronological window of ingested
quote ticks and their governed verdicts.

Everything here is deterministic, pure and offline. The layer never performs
network calls, never touches the filesystem, never reads a wall clock, never
uses randomness, keeps no global mutable state and uses no default mutable
arguments. Acquisition belongs entirely to the caller-supplied ``source``
callable; evaluation instants, freshness policies, session windows and gaps are
all explicit caller-supplied data. There is **no baked-in market calendar**.

The raw source mappings are retained by reference, verbatim: they are never
deep-copied, normalized, backfilled, coerced or mutated. ALL assessment work
(freshness, sanity, sequence and verdict rules) is delegated to the frozen
:func:`~borsapy.tv_production.govern_live_quote`; this module re-implements none
of it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from borsapy.tv_production import GovernedLiveQuote, govern_live_quote

__all__ = [
    "WindowEntry",
    "QuoteWindow",
    "ingest_quote",
]

#: The exact set of verdict literals counted by :meth:`QuoteWindow.verdict_counts`.
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
class WindowEntry:
    """One ingested tick's paired record.

    Attributes:
        raw_quote: The caller's mapping, stored BY REFERENCE, verbatim, never
            deep-copied and never mutated.
        governed: The :class:`~borsapy.tv_production.GovernedLiveQuote` produced
            by the frozen gate for this tick.
    """

    raw_quote: Mapping[str, Any]
    governed: GovernedLiveQuote

    def __post_init__(self) -> None:
        if not isinstance(self.raw_quote, Mapping):
            raise TypeError(
                f"raw_quote must be a mapping, got {type(self.raw_quote).__name__}"
            )
        if not isinstance(self.governed, GovernedLiveQuote):
            raise TypeError(
                f"governed must be a GovernedLiveQuote, got "
                f"{type(self.governed).__name__}"
            )


@dataclass(frozen=True)
class QuoteWindow:
    """An immutable, bounded chronological window of ingested ticks.

    Attributes:
        entries: Tuple of :class:`WindowEntry` in ingest order. The oldest entry
            is the first element.
        max_size: Positive maximum number of entries retained.
        dropped_count: Number of entries evicted by FIFO overflow; ``>= 0``.

    Construction fails closed: ``max_size`` must be a positive real integer
    (booleans and floats rejected), ``dropped_count`` must be a non-negative
    real integer, every entry must be a :class:`WindowEntry` and
    ``len(entries) <= max_size`` must hold.
    """

    entries: tuple[WindowEntry, ...]
    max_size: int
    dropped_count: int

    def __post_init__(self) -> None:
        if isinstance(self.entries, (str, bytes)) or not isinstance(
            self.entries, Sequence
        ):
            raise TypeError(
                f"entries must be a sequence of WindowEntry, got "
                f"{type(self.entries).__name__}"
            )
        entries = tuple(self.entries)
        if not all(isinstance(entry, WindowEntry) for entry in entries):
            raise TypeError("entries must contain only WindowEntry instances")

        max_size = _require_int(self.max_size, "max_size")
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        dropped_count = _require_int(self.dropped_count, "dropped_count")
        if dropped_count < 0:
            raise ValueError("dropped_count must be a non-negative integer")

        if len(entries) > max_size:
            raise ValueError("len(entries) must be less than or equal to max_size")

        # Detach the input so the frozen record owns an immutable tuple.
        object.__setattr__(self, "entries", entries)

    def __len__(self) -> int:
        return len(self.entries)

    def verdict_counts(self) -> dict[str, int]:
        """Return a NEW dict counting entries by governed verdict.

        Exactly the keys ``"USABLE"``, ``"DEGRADED"`` and ``"UNUSABLE"`` are
        always present, including zeros.
        """
        counts = dict.fromkeys(_VERDICTS, 0)
        for entry in self.entries:
            verdict = entry.governed.verdict
            if verdict in counts:
                counts[verdict] += 1
        return counts


def ingest_quote(
    window: QuoteWindow,
    source: Callable[[], Mapping[str, Any]],
    *,
    evaluated_at_ms: int | None = None,
    policy: Any,
    sessions: Sequence[Any] | None = None,
    max_gap_ms: int | None = None,
    clock: Callable[[], int] | None = None,
) -> QuoteWindow:
    """Ingest one tick into a new bounded window.

    Calls ``source`` EXACTLY ONCE in total, delegates ALL assessment to the
    frozen :func:`~borsapy.tv_production.govern_live_quote` (passing the
    window's raw mappings as ``prior_quotes`` in ingest order and forwarding the
    evaluation instant, ``policy`` / ``sessions`` / ``max_gap_ms`` verbatim),
    appends the new :class:`WindowEntry` and returns a NEW :class:`QuoteWindow`.
    When the result would exceed ``max_size`` the oldest entry is evicted (FIFO)
    and ``dropped_count`` increments by exactly one.

    The input window is never mutated. A source exception propagates verbatim
    and no new window is produced.

    Evaluation instant (``evaluated_at_ms`` is "the instant the quote was
    evaluated — sampled after retrieval, before governing"):

    * when ``clock`` is ``None`` the caller-supplied ``evaluated_at_ms`` governs
      unchanged, exactly as before;
    * when ``clock`` is provided, the governing instant is sampled from ``clock``
      AFTER ``source`` has returned and BEFORE governing, and used both for
      :func:`~borsapy.tv_production.govern_live_quote` and for the new entry's
      ``evaluated_at_ms``. This closes the deterministic clock-sampling race
      where a pre-fetch instant makes a freshly fetched quote look
      "future-dated" to the frozen fail-closed guard.

    Args:
        window: The current :class:`QuoteWindow`.
        source: A no-argument callable returning a quote mapping. Called exactly
            once; exceptions propagate unchanged.
        evaluated_at_ms: Caller-supplied evaluation instant, in milliseconds.
            Required when ``clock`` is ``None``; ignored (the ``clock`` sample
            governs) when ``clock`` is provided.
        policy: A :class:`~borsapy.tv_reliability.FreshnessPolicy`.
        sessions: Optional caller-supplied session windows; ``None`` disables
            gating.
        max_gap_ms: Positive integer maximum tolerated consecutive gap, required
            exactly when the window already holds prior quotes.
        clock: ``None`` or a no-argument callable returning the evaluation
            instant in milliseconds. When provided it is sampled AFTER ``source``
            returns and governs the tick.

    Returns:
        A new :class:`QuoteWindow`.

    Raises:
        TypeError: If ``window`` is not a :class:`QuoteWindow`, ``source`` or
            ``clock`` is not callable, ``evaluated_at_ms`` is ``None`` while
            ``clock`` is ``None``, or the frozen gate rejects an input.
        ValueError: If the frozen gate rejects an out-of-domain value.
    """
    if not isinstance(window, QuoteWindow):
        raise TypeError(
            f"window must be a QuoteWindow, got {type(window).__name__}"
        )
    if not callable(source):
        raise TypeError(
            f"source must be a callable returning a mapping, got "
            f"{type(source).__name__}"
        )
    if clock is not None and not callable(clock):
        raise TypeError(
            f"clock must be a callable returning an int, got {type(clock).__name__}"
        )
    if clock is None and evaluated_at_ms is None:
        raise TypeError("evaluated_at_ms is required when clock is not provided")

    prior_quotes = [entry.raw_quote for entry in window.entries]

    if clock is None:
        captured: list[Mapping[str, Any]] = []

        def _capturing_source() -> Mapping[str, Any]:
            quote = source()
            captured.append(quote)
            return quote

        governed = govern_live_quote(
            _capturing_source,
            evaluated_at_ms=evaluated_at_ms,
            policy=policy,
            sessions=sessions,
            prior_quotes=prior_quotes,
            max_gap_ms=max_gap_ms,
        )

        raw_quote = captured[0]
    else:
        # Fetch FIRST, then sample the governing instant from ``clock``: the
        # evaluation instant reflects the post-retrieval moment and is never a
        # pre-fetch sample. The quote is replayed to the frozen gate so the
        # caller-supplied ``source`` is still invoked exactly once.
        raw_quote = source()
        sampled_evaluated_at_ms = clock()

        def _replay_source() -> Mapping[str, Any]:
            return raw_quote

        governed = govern_live_quote(
            _replay_source,
            evaluated_at_ms=sampled_evaluated_at_ms,
            policy=policy,
            sessions=sessions,
            prior_quotes=prior_quotes,
            max_gap_ms=max_gap_ms,
        )

    entries = window.entries + (WindowEntry(raw_quote=raw_quote, governed=governed),)
    dropped_count = window.dropped_count
    if len(entries) > window.max_size:
        # Ingestion adds exactly one entry, so at most one eviction is needed.
        entries = entries[1:]
        dropped_count += 1

    return QuoteWindow(
        entries=entries,
        max_size=window.max_size,
        dropped_count=dropped_count,
    )
