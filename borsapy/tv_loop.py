"""Deterministic live-loop driver for TradingView quote pulls.

This module adds the final operational composition of the TradingView production
pipeline on top of the frozen bounded tick-window state machine
(:mod:`borsapy.tv_window`): ONE call that folds
:func:`~borsapy.tv_window.ingest_quote` over repeated pulls from a
caller-supplied quote source, with caller-supplied time (a ``clock`` callable),
optional pacing (a ``pause`` callable), a tick budget and an optional stop
predicate.

Everything here is deterministic, pure and offline. The loop never reads a wall
clock, never sleeps by itself, never performs network access, never uses
randomness, keeps no global mutable state and uses no default mutable arguments.
Time, acquisition and pacing are supplied by the caller. In production the
``source`` is the real transport bound through :func:`ticker_quote_source`, but
this module is transport-agnostic and never imports ``borsapy.ticker``.

All assessment, verdict classification, prior-quote joining and FIFO eviction is
delegated to the frozen :func:`~borsapy.tv_window.ingest_quote` (and, beneath it,
:func:`~borsapy.tv_production.govern_live_quote`); this module re-implements
none of it. Any exception raised by ``stop``, ``pause``, ``clock``, ``source`` or
the frozen layers propagates verbatim: there is no retry and no fabricated
failure record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from borsapy.tv_window import QuoteWindow, ingest_quote

__all__ = [
    "LiveLoopResult",
    "run_quote_loop",
    "consecutive_unusable",
    "ticker_quote_source",
]

#: The exact set of terminal stop-reason literals.
_STOP_REASONS: tuple[str, ...] = (
    "TICK_BUDGET_EXHAUSTED",
    "STOP_PREDICATE",
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
class LiveLoopResult:
    """Immutable outcome of one :func:`run_quote_loop` execution.

    Attributes:
        window: The final :class:`~borsapy.tv_window.QuoteWindow`.
        tick_budget: Positive integer maximum number of ingest iterations
            ATTEMPTED.
        iterations: Number of ingest iterations EXECUTED,
            ``0 <= iterations <= tick_budget``.
        stop_reason: Exactly ``"TICK_BUDGET_EXHAUSTED"`` or
            ``"STOP_PREDICATE"``.

    Construction fails closed. ``window`` must be a
    :class:`~borsapy.tv_window.QuoteWindow` (``TypeError``); ``tick_budget`` must
    be a positive real integer; ``iterations`` must be a non-negative real
    integer not greater than ``tick_budget``; ``stop_reason`` must be one of the
    two literals; and the cross-field invariant is enforced exactly:
    ``"TICK_BUDGET_EXHAUSTED"`` holds if and only if ``iterations == tick_budget``
    while ``"STOP_PREDICATE"`` holds if and only if ``iterations < tick_budget``.
    """

    window: QuoteWindow
    tick_budget: int
    iterations: int
    stop_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.window, QuoteWindow):
            raise TypeError(
                f"window must be a QuoteWindow, got {type(self.window).__name__}"
            )
        tick_budget = _require_int(self.tick_budget, "tick_budget")
        if tick_budget <= 0:
            raise ValueError("tick_budget must be a positive integer")
        iterations = _require_int(self.iterations, "iterations")
        if iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if iterations > tick_budget:
            raise ValueError("iterations must be less than or equal to tick_budget")
        if self.stop_reason not in _STOP_REASONS:
            raise ValueError(
                "stop_reason must be one of " + ", ".join(_STOP_REASONS)
            )
        if self.stop_reason == "TICK_BUDGET_EXHAUSTED":
            if iterations != tick_budget:
                raise ValueError(
                    "stop_reason TICK_BUDGET_EXHAUSTED requires "
                    "iterations == tick_budget"
                )
        else:  # "STOP_PREDICATE"
            if iterations >= tick_budget:
                raise ValueError(
                    "stop_reason STOP_PREDICATE requires iterations < tick_budget"
                )


def run_quote_loop(
    window: QuoteWindow,
    source: Callable[[], Mapping[str, Any]],
    *,
    clock: Callable[[], int],
    policy: Any,
    sessions: Sequence[Any] | None = None,
    max_gap_ms: int | None = None,
    tick_budget: int,
    stop: Callable[[QuoteWindow], bool] | None = None,
    pause: Callable[[], None] | None = None,
) -> LiveLoopResult:
    """Fold :func:`~borsapy.tv_window.ingest_quote` over repeated quote pulls.

    The loop itself is deterministic, pure and offline: it never reads a wall
    clock, never sleeps, never performs network access and re-implements none of
    the assessment, verdict, prior-quote joining or FIFO logic. Each executed
    iteration consults the optional ``stop`` predicate first, then (from the
    second iteration onward) calls the optional ``pause``, then performs EXACTLY
    one :func:`~borsapy.tv_window.ingest_quote` call. The governing evaluation
    instant ("the instant the quote was evaluated — sampled after retrieval,
    before governing") is sampled INSIDE that call: ``ingest_quote`` reads the
    caller-supplied ``clock`` exactly once, AFTER its internal ``source()``
    fetch returns and BEFORE governing, so the loop never feeds a pre-fetch
    clock sample into governing.

    Args:
        window: The initial :class:`~borsapy.tv_window.QuoteWindow`. It may be
            pre-seeded (non-empty).
        source: A no-argument callable returning a quote mapping. In production
            this is the real transport bound via :func:`ticker_quote_source`.
        clock: A no-argument callable returning the governing
            ``evaluated_at_ms`` integer. It is threaded into
            :func:`~borsapy.tv_window.ingest_quote`, which samples it after the
            fetch and before governing. The loop never reads a wall clock.
        policy: Forwarded verbatim to the frozen layers; the loop does not
            pre-validate it.
        sessions: Optional caller-supplied session windows forwarded verbatim.
        max_gap_ms: ``None`` or a positive integer. It is threaded to the frozen
            gate per tick as the supplied value if and only if the window is
            non-empty at that tick, and ``None`` otherwise.
        tick_budget: Positive integer maximum number of ingest iterations
            attempted (``bool`` rejected).
        stop: ``None`` or a ``Callable[[QuoteWindow], bool]`` consulted before
            each iteration, including the first.
        pause: ``None`` or a no-argument callable invoked for pacing between
            iterations. The loop itself never sleeps.

    Returns:
        A :class:`LiveLoopResult`.

    Raises:
        TypeError: If ``window`` is not a :class:`QuoteWindow`, ``source``,
            ``clock``, ``stop`` or ``pause`` is not callable, or ``tick_budget``
            / ``max_gap_ms`` is not a real integer.
        ValueError: If ``tick_budget`` is not positive, ``max_gap_ms`` is not
            positive, or ``max_gap_ms`` is ``None`` while the initial window is
            non-empty or ``tick_budget > 1``.
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
    if not callable(clock):
        raise TypeError(
            f"clock must be a callable returning an int, got {type(clock).__name__}"
        )
    if stop is not None and not callable(stop):
        raise TypeError(
            f"stop must be a callable or None, got {type(stop).__name__}"
        )
    if pause is not None and not callable(pause):
        raise TypeError(
            f"pause must be a callable or None, got {type(pause).__name__}"
        )

    budget = _require_int(tick_budget, "tick_budget")
    if budget <= 0:
        raise ValueError("tick_budget must be a positive integer")

    if max_gap_ms is None:
        if len(window) > 0 or budget > 1:
            raise ValueError(
                "max_gap_ms must be a positive int; it cannot be None when the "
                "window is non-empty or tick_budget > 1"
            )
        gap: int | None = None
    else:
        gap = _require_int(max_gap_ms, "max_gap_ms")
        if gap <= 0:
            raise ValueError("max_gap_ms must be a positive integer")

    current = window
    iterations = 0
    for index in range(1, budget + 1):
        if stop is not None and stop(current):
            return LiveLoopResult(
                window=current,
                tick_budget=budget,
                iterations=index - 1,
                stop_reason="STOP_PREDICATE",
            )
        if index > 1 and pause is not None:
            pause()
        tick_gap = gap if len(current) > 0 else None
        current = ingest_quote(
            current,
            source,
            policy=policy,
            sessions=sessions,
            max_gap_ms=tick_gap,
            clock=clock,
        )
        iterations = index

    return LiveLoopResult(
        window=current,
        tick_budget=budget,
        iterations=iterations,
        stop_reason="TICK_BUDGET_EXHAUSTED",
    )


def consecutive_unusable(window: QuoteWindow) -> int:
    """Count the trailing entries whose governed verdict is ``"UNUSABLE"``.

    The count walks the entries from the newest backwards and stops at the first
    entry whose ``governed.verdict`` is not exactly ``"UNUSABLE"``. It is ``0``
    for an empty window or when the newest entry is not ``UNUSABLE``. This is a
    pure helper for writing stop predicates; it carries no policy of its own.

    Raises:
        TypeError: If ``window`` is not a :class:`QuoteWindow`.
    """
    if not isinstance(window, QuoteWindow):
        raise TypeError(
            f"window must be a QuoteWindow, got {type(window).__name__}"
        )
    count = 0
    for entry in reversed(window.entries):
        if entry.governed.verdict == "UNUSABLE":
            count += 1
        else:
            break
    return count


def ticker_quote_source(ticker: Any) -> Callable[[], Mapping[str, Any]]:
    """Bind the real ``Ticker.get_live_quote`` transport into a loop source.

    The binding is duck-typed and fail-closed: ``ticker`` must expose a callable
    ``get_live_quote`` attribute. The returned callable is that bound method, so
    quotes and exceptions pass through unchanged and the loop performs no retry.
    The module never imports ``borsapy.ticker`` and validates structurally.

    Raises:
        TypeError: If ``ticker`` has no callable ``get_live_quote`` attribute.
    """
    getter = getattr(ticker, "get_live_quote", None)
    if not callable(getter):
        raise TypeError(
            "ticker must expose a callable get_live_quote attribute, got "
            f"{type(ticker).__name__}"
        )
    return getter
