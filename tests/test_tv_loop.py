"""Tests for the deterministic TradingView live-loop driver.

All tests are fully offline and deterministic: every quote is a fixed literal,
every clock is a scripted sequence and every source is a fake callable. The
module under test never reads a wall clock, performs network access or
instantiates real TradingView providers.
"""

import dataclasses

import pytest

import borsapy as bp
from borsapy.tv_loop import (
    LiveLoopResult,
    consecutive_unusable,
    run_quote_loop,
    ticker_quote_source,
)
from borsapy.tv_production import govern_live_quote
from borsapy.tv_reliability import FreshnessPolicy
from borsapy.tv_window import QuoteWindow, WindowEntry, ingest_quote

POLICY = FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000)


def _quote(**overrides):
    """Build a quote dict in ``get_live_quote()`` shape with fixed defaults."""
    quote = {
        "symbol": "THYAO",
        "exchange": "BIST",
        "last": 10.0,
        "bid": 9.9,
        "ask": 10.1,
        "volume": 1000,
        "timestamp": 1000,
        "auth_transport": "auth_token",
    }
    quote.update(overrides)
    return quote


def _source(quote):
    """Return a no-argument fake source callable returning ``quote``."""
    return lambda: quote


def _govern(quote, **kwargs):
    """Govern ``quote`` with sensible deterministic defaults."""
    kwargs.setdefault("evaluated_at_ms", 1500)
    kwargs.setdefault("policy", POLICY)
    return govern_live_quote(_source(quote), **kwargs)


def _empty_window(max_size=5):
    """Return an empty :class:`QuoteWindow` with the given bounded size."""
    return QuoteWindow(entries=(), max_size=max_size, dropped_count=0)


def _seeded_window(max_size=5):
    """Return a window pre-seeded with one FRESH/USABLE tick at timestamp 1000."""
    return ingest_quote(
        _empty_window(max_size),
        _source(_quote(timestamp=1000)),
        evaluated_at_ms=1500,
        policy=POLICY,
    )


def _scripted_source(quotes):
    """Return a source callable yielding ``quotes`` in order."""
    state = {"index": 0}

    def source():
        quote = quotes[state["index"]]
        state["index"] += 1
        return quote

    return source


def _scripted_clock(instants):
    """Return a clock callable yielding ``instants`` in order."""
    state = {"index": 0}

    def clock():
        value = instants[state["index"]]
        state["index"] += 1
        return value

    return clock


def test_loop_requires_quote_window():
    with pytest.raises(TypeError):
        run_quote_loop(
            "not-a-window",
            _source(_quote()),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=1,
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            None,
            _source(_quote()),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=1,
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            [],
            _source(_quote()),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=1,
        )


def test_loop_requires_callable_inputs():
    window = _empty_window()
    good_source = _source(_quote(timestamp=1000))

    def good_clock():
        return 1500


    with pytest.raises(TypeError):
        run_quote_loop(window, "nope", clock=good_clock, policy=POLICY, tick_budget=1)
    with pytest.raises(TypeError):
        run_quote_loop(window, good_source, clock=42, policy=POLICY, tick_budget=1)
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            good_source,
            clock=good_clock,
            policy=POLICY,
            tick_budget=1,
            stop="nope",
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            good_source,
            clock=good_clock,
            policy=POLICY,
            tick_budget=1,
            pause="nope",
        )


def test_loop_tick_budget_validation():
    window = _empty_window()

    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            run_quote_loop(
                window,
                _source(_quote(timestamp=1000)),
                clock=lambda: 1500,
                policy=POLICY,
                tick_budget=bad,
            )
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=True,
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget="1",
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=1.5,
        )

    result = run_quote_loop(
        window,
        _source(_quote(timestamp=1000)),
        clock=lambda: 1500,
        policy=POLICY,
        tick_budget=1,
    )
    assert result.iterations == 1
    assert result.tick_budget == 1


def test_loop_max_gap_prevalidation_fails_closed():
    window = _empty_window()

    # None with tick_budget > 1 fails closed before ANY call.
    with pytest.raises(ValueError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            tick_budget=2,
        )

    # None with a pre-seeded non-empty window fails closed.
    with pytest.raises(ValueError):
        run_quote_loop(
            _seeded_window(),
            _source(_quote(timestamp=2000)),
            clock=lambda: 2500,
            policy=POLICY,
            tick_budget=1,
        )

    # None with tick_budget == 1 and an empty window is accepted.
    accepted = run_quote_loop(
        window,
        _source(_quote(timestamp=1000)),
        clock=lambda: 1500,
        policy=POLICY,
        tick_budget=1,
    )
    assert accepted.iterations == 1

    with pytest.raises(ValueError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            max_gap_ms=0,
            tick_budget=1,
        )
    with pytest.raises(ValueError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            max_gap_ms=-5,
            tick_budget=1,
        )
    with pytest.raises(TypeError):
        run_quote_loop(
            window,
            _source(_quote(timestamp=1000)),
            clock=lambda: 1500,
            policy=POLICY,
            max_gap_ms=True,
            tick_budget=1,
        )

    supplied = run_quote_loop(
        window,
        _source(_quote(timestamp=1000)),
        clock=lambda: 1500,
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=1,
    )
    assert supplied.iterations == 1


def test_loop_budget_exhausted_golden():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
    ]
    instants = [1500, 2500, 3500]
    result = run_quote_loop(
        _empty_window(),
        _scripted_source(quotes),
        clock=_scripted_clock(instants),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=3,
    )

    assert result.iterations == 3
    assert result.tick_budget == 3
    assert result.stop_reason == "TICK_BUDGET_EXHAUSTED"
    assert len(result.window) == 3
    assert [entry.raw_quote["timestamp"] for entry in result.window.entries] == [
        1000,
        2000,
        3000,
    ]
    assert [entry.governed.evaluated_at_ms for entry in result.window.entries] == [
        1500,
        2500,
        3500,
    ]
    assert result.window.verdict_counts() == {
        "USABLE": 3,
        "DEGRADED": 0,
        "UNUSABLE": 0,
    }


def test_loop_stop_before_first_tick():
    window = _seeded_window()
    source_calls = []
    clock_calls = []

    def source():
        source_calls.append(1)
        return _quote(timestamp=2000)

    def clock():
        clock_calls.append(1)
        return 2500

    result = run_quote_loop(
        window,
        source,
        clock=clock,
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=3,
        stop=lambda current: True,
    )

    assert result.iterations == 0
    assert result.stop_reason == "STOP_PREDICATE"
    assert result.window is window
    assert result.window.entries == window.entries
    assert source_calls == []
    assert clock_calls == []


def test_loop_stop_midway():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
    ]
    instants = [1500, 2500, 3500]
    base_source = _scripted_source(quotes)
    source_calls = []

    def source():
        source_calls.append(1)
        return base_source()

    result = run_quote_loop(
        _empty_window(),
        source,
        clock=_scripted_clock(instants),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=5,
        stop=lambda current: len(current) >= 2,
    )

    assert result.iterations == 2
    assert result.stop_reason == "STOP_PREDICATE"
    assert len(result.window) == 2
    assert len(source_calls) == 2
    assert [entry.raw_quote["timestamp"] for entry in result.window.entries] == [
        1000,
        2000,
    ]


def test_loop_exact_call_counts():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
        _quote(timestamp=4000),
    ]
    instants = [1500, 2500, 3500, 4500]
    base_source = _scripted_source(quotes)
    base_clock = _scripted_clock(instants)
    source_calls = []
    clock_calls = []
    pause_calls = []

    def source():
        source_calls.append(1)
        return base_source()

    def clock():
        clock_calls.append(1)
        return base_clock()

    def pause():
        pause_calls.append(1)

    result = run_quote_loop(
        _empty_window(),
        source,
        clock=clock,
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=4,
        pause=pause,
    )

    assert result.iterations == 4
    assert result.stop_reason == "TICK_BUDGET_EXHAUSTED"
    assert len(source_calls) == 4
    assert len(clock_calls) == 4
    assert len(pause_calls) == 3

    # With a stop at entry, pause is never called.
    source_calls2 = []
    clock_calls2 = []
    pause_calls2 = []

    def source2():
        source_calls2.append(1)
        return _quote(timestamp=2000)

    def clock2():
        clock_calls2.append(1)
        return 2500

    def pause2():
        pause_calls2.append(1)

    result2 = run_quote_loop(
        _seeded_window(),
        source2,
        clock=clock2,
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=4,
        stop=lambda current: True,
        pause=pause2,
    )

    assert result2.iterations == 0
    assert result2.stop_reason == "STOP_PREDICATE"
    assert pause_calls2 == []
    assert source_calls2 == []
    assert clock_calls2 == []


def test_loop_max_gap_threading(monkeypatch):
    import borsapy.tv_loop as tv_loop

    received = []
    real_ingest = tv_loop.ingest_quote

    def wrapper(
        window,
        source,
        *,
        evaluated_at_ms=None,
        policy,
        sessions=None,
        max_gap_ms=None,
        clock=None,
    ):
        received.append(max_gap_ms)
        return real_ingest(
            window,
            source,
            evaluated_at_ms=evaluated_at_ms,
            policy=policy,
            sessions=sessions,
            max_gap_ms=max_gap_ms,
            clock=clock,
        )

    monkeypatch.setattr(tv_loop, "ingest_quote", wrapper)

    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
    ]
    result = run_quote_loop(
        _empty_window(),
        _scripted_source(quotes),
        clock=_scripted_clock([1500, 2500, 3500]),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=3,
    )

    assert received == [None, 1000, 1000]
    assert len(received) == result.iterations


def test_loop_preseeded_window_forwards_max_gap_first_tick(monkeypatch):
    import borsapy.tv_loop as tv_loop

    received = []
    real_ingest = tv_loop.ingest_quote

    def wrapper(
        window,
        source,
        *,
        evaluated_at_ms=None,
        policy,
        sessions=None,
        max_gap_ms=None,
        clock=None,
    ):
        received.append(max_gap_ms)
        return real_ingest(
            window,
            source,
            evaluated_at_ms=evaluated_at_ms,
            policy=policy,
            sessions=sessions,
            max_gap_ms=max_gap_ms,
            clock=clock,
        )

    monkeypatch.setattr(tv_loop, "ingest_quote", wrapper)

    result = run_quote_loop(
        _seeded_window(),
        _source(_quote(timestamp=2000)),
        clock=lambda: 2500,
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=1,
    )

    assert received == [1000]
    assert result.iterations == 1


def test_loop_evaluated_at_ms_follows_clock_sequence():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
    ]
    instants = [1500, 2500, 3500]
    result = run_quote_loop(
        _empty_window(),
        _scripted_source(quotes),
        clock=_scripted_clock(instants),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=3,
    )

    got = [entry.governed.evaluated_at_ms for entry in result.window.entries]
    assert got == instants


def test_loop_source_exception_propagates_verbatim():
    class BoomError(RuntimeError):
        pass

    boom = BoomError("kaboom")
    clock_calls = []
    state = {"count": 0}

    def source():
        state["count"] += 1
        if state["count"] == 2:
            raise boom
        return _quote(timestamp=1000)

    def clock():
        clock_calls.append(1)
        return 1500

    with pytest.raises(BoomError) as excinfo:
        run_quote_loop(
            _empty_window(),
            source,
            clock=clock,
            policy=POLICY,
            max_gap_ms=1000,
            tick_budget=4,
        )

    assert excinfo.value is boom
    # The governing clock sample is taken AFTER the fetch: iteration 1 fetches
    # and samples once; iteration 2's fetch raises before its sample, so the
    # failed tick contributes no clock() call.
    assert len(clock_calls) == 1


def test_loop_clock_and_pause_exceptions_propagate():
    class ClockError(RuntimeError):
        pass

    class PauseError(RuntimeError):
        pass

    clock_boom = ClockError("clock")
    pause_boom = PauseError("pause")

    clock_calls = []

    def clock():
        clock_calls.append(1)
        if len(clock_calls) == 2:
            raise clock_boom
        return 1500

    with pytest.raises(ClockError) as excinfo:
        run_quote_loop(
            _empty_window(),
            _source(_quote(timestamp=1000)),
            clock=clock,
            policy=POLICY,
            max_gap_ms=1000,
            tick_budget=3,
        )
    assert excinfo.value is clock_boom

    def pause():
        raise pause_boom

    with pytest.raises(PauseError) as excinfo2:
        run_quote_loop(
            _seeded_window(),
            _source(_quote(timestamp=2000)),
            clock=lambda: 2500,
            policy=POLICY,
            max_gap_ms=1000,
            tick_budget=3,
            pause=pause,
        )
    assert excinfo2.value is pause_boom


def test_loop_fifo_overflow_inherited():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000),
        _quote(timestamp=4000),
    ]
    instants = [1500, 2500, 3500, 4500]
    result = run_quote_loop(
        _empty_window(max_size=2),
        _scripted_source(quotes),
        clock=_scripted_clock(instants),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=4,
    )

    assert result.iterations == 4
    assert len(result.window) == 2
    assert result.window.max_size == 2
    assert result.window.dropped_count == 2
    assert [entry.raw_quote["timestamp"] for entry in result.window.entries] == [
        3000,
        4000,
    ]


def test_loop_verdict_pipeline_integration():
    quotes = [
        _quote(timestamp=1000),
        _quote(timestamp=2000),
        _quote(timestamp=3000, bid=10.2, ask=10.0),
    ]
    instants = [1500, 5000, 3500]
    result = run_quote_loop(
        _empty_window(),
        _scripted_source(quotes),
        clock=_scripted_clock(instants),
        policy=POLICY,
        max_gap_ms=1000,
        tick_budget=3,
    )

    got = [entry.governed.verdict for entry in result.window.entries]
    expected = [
        _govern(quotes[0], evaluated_at_ms=instants[0]).verdict,
        _govern(quotes[1], evaluated_at_ms=instants[1]).verdict,
        _govern(quotes[2], evaluated_at_ms=instants[2]).verdict,
    ]
    assert got == expected == ["USABLE", "DEGRADED", "UNUSABLE"]


def test_consecutive_unusable_trailing_run():
    usable = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    unusable = _govern(_quote(timestamp=1000, last=0.0), evaluated_at_ms=1500)
    assert usable.verdict == "USABLE"
    assert unusable.verdict == "UNUSABLE"

    def window_of(*governed_items):
        entries = tuple(
            WindowEntry(raw_quote=_quote(timestamp=1000), governed=governed)
            for governed in governed_items
        )
        return QuoteWindow(
            entries=entries, max_size=max(len(entries), 1), dropped_count=0
        )

    assert consecutive_unusable(_empty_window()) == 0
    assert consecutive_unusable(window_of(usable)) == 0
    assert consecutive_unusable(window_of(unusable, usable)) == 0
    assert consecutive_unusable(window_of(usable, unusable, unusable)) == 2
    assert consecutive_unusable(window_of(unusable, unusable, unusable)) == 3

    with pytest.raises(TypeError):
        consecutive_unusable("not-a-window")
    with pytest.raises(TypeError):
        consecutive_unusable(None)


def test_live_loop_result_invariants_and_frozen():
    window = _empty_window()

    valid_exhausted = LiveLoopResult(
        window=window,
        tick_budget=2,
        iterations=2,
        stop_reason="TICK_BUDGET_EXHAUSTED",
    )
    assert valid_exhausted.iterations == 2

    valid_predicate = LiveLoopResult(
        window=window,
        tick_budget=2,
        iterations=1,
        stop_reason="STOP_PREDICATE",
    )
    assert valid_predicate.stop_reason == "STOP_PREDICATE"

    with pytest.raises(ValueError):
        LiveLoopResult(
            window=window,
            tick_budget=2,
            iterations=3,
            stop_reason="TICK_BUDGET_EXHAUSTED",
        )
    with pytest.raises(ValueError):
        LiveLoopResult(
            window=window,
            tick_budget=2,
            iterations=2,
            stop_reason="WHATEVER",
        )
    with pytest.raises(ValueError):
        LiveLoopResult(
            window=window,
            tick_budget=2,
            iterations=1,
            stop_reason="TICK_BUDGET_EXHAUSTED",
        )
    with pytest.raises(ValueError):
        LiveLoopResult(
            window=window,
            tick_budget=2,
            iterations=2,
            stop_reason="STOP_PREDICATE",
        )
    with pytest.raises(TypeError):
        LiveLoopResult(
            window="not-a-window",
            tick_budget=2,
            iterations=2,
            stop_reason="TICK_BUDGET_EXHAUSTED",
        )

    with pytest.raises(AttributeError):
        valid_exhausted.window = window
    with pytest.raises(AttributeError):
        valid_exhausted.tick_budget = 5
    with pytest.raises(AttributeError):
        valid_exhausted.iterations = 0
    with pytest.raises(AttributeError):
        valid_exhausted.stop_reason = "STOP_PREDICATE"

    assert dataclasses.is_dataclass(valid_exhausted)


def test_ticker_quote_source_binds_and_validates():
    class StubTicker:
        def __init__(self, quote):
            self._quote = quote

        def get_live_quote(self):
            return self._quote

    quote = _quote(timestamp=1000)
    stub = StubTicker(quote)
    bound = ticker_quote_source(stub)
    assert bound == stub.get_live_quote
    assert bound() is quote

    class BoomError(RuntimeError):
        pass

    class BadStub:
        def get_live_quote(self):
            raise BoomError("boom")

    raised = ticker_quote_source(BadStub())
    with pytest.raises(BoomError) as excinfo:
        raised()
    assert str(excinfo.value) == "boom"

    class NoMethod:
        pass

    with pytest.raises(TypeError) as excinfo2:
        ticker_quote_source(NoMethod())
    assert "get_live_quote" in str(excinfo2.value)

    class NonCallable:
        get_live_quote = 5

    with pytest.raises(TypeError) as excinfo3:
        ticker_quote_source(NonCallable())
    assert "get_live_quote" in str(excinfo3.value)


def test_no_third_party_imports():
    import ast
    import pathlib
    import sys

    import borsapy.tv_loop as tv_loop

    source = pathlib.Path(tv_loop.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    modules = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
                imported.add(node.module)

    # ``borsapy`` is first-party; every other import must be stdlib.
    allowed = set(sys.stdlib_module_names) | {"borsapy"}
    assert modules
    assert modules <= allowed

    # The module must not couple to the concrete transport.
    assert "borsapy.ticker" not in imported


def test_public_imports():
    import borsapy.tv_loop as tv_loop

    assert set(tv_loop.__all__) == {
        "LiveLoopResult",
        "run_quote_loop",
        "consecutive_unusable",
        "ticker_quote_source",
    }

    for name in tv_loop.__all__:
        assert getattr(bp, name) is getattr(tv_loop, name)

    # Every pre-existing TV name remains exported at package level.
    for name in (
        "govern_live_quote",
        "GovernedLiveQuote",
        "QuoteWindow",
        "WindowEntry",
        "ingest_quote",
        "FreshnessPolicy",
        "FreshnessAssessment",
        "QuoteSequenceAssessment",
        "QuoteSequenceIssue",
        "QuoteSanityIssue",
        "TradingSessionWindow",
        "assess_live_quote_freshness",
        "assess_quote_sequence",
        "assess_live_quote_sanity",
    ):
        assert name in bp.__all__
        assert hasattr(bp, name)


#: Freshness policy used by the evaluation-clock race regression tests.
_RACE_POLICY = FreshnessPolicy(fresh_max_age_ms=60_000, stale_max_age_ms=300_000)


def test_run_quote_loop_race_clock_advances_during_fetch_does_not_raise():
    """Golden case-1: a fetch that advances the clock must not trip the guard.

    ``clock`` starts at 1500; ``source`` models the ~1.5 s fetch latency by
    advancing it to 1600 and returns a quote stamped 1600. Sampling the governing
    instant AFTER the fetch makes the age ``0`` and the verdict ``USABLE``; the
    pre-fetch sampling would make age ``-1000`` and raise. This is the RED proof
    of the regression.
    """
    state = {"now": 1500}

    def clock():
        return state["now"]

    def source():
        state["now"] = 1600
        return _quote(timestamp=1600)

    result = run_quote_loop(
        _empty_window(),
        source,
        clock=clock,
        policy=_RACE_POLICY,
        sessions=None,
        max_gap_ms=None,
        tick_budget=1,
    )

    governed = result.window.entries[0].governed
    assert governed.evaluated_at_ms == 1600
    assert governed.evaluated_at_ms - governed.quote_timestamp_ms == 0
    assert governed.verdict == "USABLE"


def test_run_quote_loop_future_quote_still_fails_closed():
    """Case-2: a genuinely future timestamp still fails closed after the fix.

    The post-fetch clock reads 1600 while the quote is stamped 1700: the age is
    ``-1000`` and the frozen fail-closed guard must still raise ``ValueError``.
    """
    state = {"now": 1500}

    def clock():
        return state["now"]

    def source():
        state["now"] = 1600
        return _quote(timestamp=1700)

    with pytest.raises(ValueError) as excinfo:
        run_quote_loop(
            _empty_window(),
            source,
            clock=clock,
            policy=_RACE_POLICY,
            sessions=None,
            max_gap_ms=None,
            tick_budget=1,
        )

    message = str(excinfo.value)
    assert "future" in message


def test_run_quote_loop_ordinary_tick_unaffected():
    """Case-3: the ordinary no-latency tick is unchanged by the fix.

    The clock reads 1600 before and after the (instant) fetch and the quote is
    stamped 1500: age ``100`` and verdict ``USABLE``.
    """
    state = {"now": 1600}

    def clock():
        return state["now"]

    def source():
        return _quote(timestamp=1500)

    result = run_quote_loop(
        _empty_window(),
        source,
        clock=clock,
        policy=_RACE_POLICY,
        sessions=None,
        max_gap_ms=None,
        tick_budget=1,
    )

    governed = result.window.entries[0].governed
    assert governed.freshness.age_ms == 100
    assert governed.verdict == "USABLE"


def test_run_quote_loop_clock_sampled_after_source_call():
    """Case-4: the governing clock sample happens AFTER the fetch, once per tick.

    An instrumented ``clock``/``source`` pair records a single shared event log.
    In every iteration ``source`` must be logged before the governing ``clock``
    sample, and the total ``clock()`` call count equals the exact-count pin
    (``N`` ticks => ``N`` governing samples, no extra sample).
    """
    events = []
    instants = [1500, 2500, 3500]
    base_clock = _scripted_clock(instants)
    base_source = _scripted_source(
        [
            _quote(timestamp=1000),
            _quote(timestamp=2000),
            _quote(timestamp=3000),
        ]
    )
    clock_calls = []

    def clock():
        clock_calls.append(1)
        events.append("clock")
        return base_clock()

    def source():
        events.append("source")
        return base_source()

    result = run_quote_loop(
        _empty_window(),
        source,
        clock=clock,
        policy=_RACE_POLICY,
        max_gap_ms=1000,
        tick_budget=3,
    )

    assert result.iterations == 3
    assert events == [
        "source",
        "clock",
        "source",
        "clock",
        "source",
        "clock",
    ]
    assert len(clock_calls) == 3
    assert [entry.governed.evaluated_at_ms for entry in result.window.entries] == (
        instants
    )
