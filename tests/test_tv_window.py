"""Tests for the deterministic TradingView bounded tick-window state machine.

All tests are fully offline and deterministic: every input is a fixed literal
and every quote source is a fake callable. The module under test never reads a
wall clock, performs network access or instantiates real TradingView providers.
"""

import pytest

from borsapy.tv_production import GovernedLiveQuote, govern_live_quote
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


def test_window_entry_construction_validates():
    governed = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    with pytest.raises(TypeError):
        WindowEntry(raw_quote=[("last", 1.0)], governed=governed)
    with pytest.raises(TypeError):
        WindowEntry(raw_quote="not-a-mapping", governed=governed)
    with pytest.raises(TypeError):
        WindowEntry(raw_quote=None, governed=governed)
    with pytest.raises(TypeError):
        WindowEntry(raw_quote=_quote(), governed="not-governed")
    with pytest.raises(TypeError):
        WindowEntry(raw_quote=_quote(), governed=None)


def test_quote_window_construction_validates():
    governed = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    entry = WindowEntry(raw_quote=_quote(timestamp=1000), governed=governed)

    # max_size must be a positive real int; bools and floats are rejected.
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=0, dropped_count=0)
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=-1, dropped_count=0)
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=True, dropped_count=0)
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=False, dropped_count=0)
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=1.5, dropped_count=0)

    # dropped_count must be a non-negative real int.
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=3, dropped_count=-1)
    with pytest.raises((TypeError, ValueError)):
        QuoteWindow(entries=(), max_size=3, dropped_count=True)

    # len(entries) <= max_size is enforced at construction.
    with pytest.raises(ValueError):
        QuoteWindow(entries=(entry, entry), max_size=1, dropped_count=0)

    # Non-WindowEntry elements are rejected.
    with pytest.raises(TypeError):
        QuoteWindow(entries=(entry, "nope"), max_size=3, dropped_count=0)


def test_empty_window_verdict_counts():
    window = QuoteWindow(entries=(), max_size=5, dropped_count=0)
    assert len(window) == 0
    assert window.verdict_counts() == {
        "USABLE": 0,
        "DEGRADED": 0,
        "UNUSABLE": 0,
    }


def test_ingest_calls_source_exactly_once():
    calls = []
    quote = _quote(timestamp=1000)

    def source():
        calls.append(1)
        return quote

    window = QuoteWindow(entries=(), max_size=5, dropped_count=0)
    ingest_quote(window, source, evaluated_at_ms=1500, policy=POLICY)
    assert calls == [1]


def test_ingest_appends_entries_in_order():
    q1 = _quote(timestamp=1000)
    q2 = _quote(timestamp=2000)
    q3 = _quote(timestamp=3000)
    window = QuoteWindow(entries=(), max_size=5, dropped_count=0)

    w1 = ingest_quote(
        window, _source(q1), evaluated_at_ms=3500, policy=POLICY
    )
    w2 = ingest_quote(
        w1, _source(q2), evaluated_at_ms=3500, policy=POLICY, max_gap_ms=2000
    )
    w3 = ingest_quote(
        w2, _source(q3), evaluated_at_ms=3500, policy=POLICY, max_gap_ms=2000
    )

    assert len(w3) == 3
    assert [entry.raw_quote for entry in w3.entries] == [q1, q2, q3]
    assert w3.entries[0].raw_quote is q1
    assert w3.entries[1].raw_quote is q2
    assert w3.entries[2].raw_quote is q3
    assert all(isinstance(entry.governed, GovernedLiveQuote) for entry in w3.entries)
    # The new entry is appended; earlier entries are carried by reference.
    assert w3.entries[1].governed is w2.entries[1].governed


def test_ingest_passes_prior_quotes_in_order():
    window = QuoteWindow(entries=(), max_size=5, dropped_count=0)

    # Empty window: no prior quotes -> no sequence issues.
    first = ingest_quote(
        window, _source(_quote(timestamp=1000)), evaluated_at_ms=1500, policy=POLICY
    )
    assert first.entries[0].governed.sequence_issues == ()
    assert first.entries[0].governed.verdict == "USABLE"

    # Prior tick in the window with an equal timestamp -> DUPLICATE_TIMESTAMP.
    second = ingest_quote(
        first,
        _source(_quote(timestamp=1000)),
        evaluated_at_ms=1500,
        policy=POLICY,
        max_gap_ms=1000,
    )
    kinds = [issue.kind for issue in second.entries[1].governed.sequence_issues]
    assert kinds == ["DUPLICATE_TIMESTAMP"]
    assert second.entries[1].governed.verdict == "DEGRADED"


def test_ingest_evicts_oldest_and_counts_dropped():
    window = QuoteWindow(entries=(), max_size=2, dropped_count=0)
    w1 = ingest_quote(
        window, _source(_quote(timestamp=1000)), evaluated_at_ms=1500, policy=POLICY
    )
    w2 = ingest_quote(
        w1, _source(_quote(timestamp=2000)), evaluated_at_ms=2500,
        policy=POLICY, max_gap_ms=1000,
    )
    assert len(w2) == 2
    assert w2.dropped_count == 0

    # One overflow ingest: the oldest entry is gone and dropped_count is 1.
    w3 = ingest_quote(
        w2, _source(_quote(timestamp=3000)), evaluated_at_ms=3500,
        policy=POLICY, max_gap_ms=1000,
    )
    assert len(w3) == 2
    assert w3.dropped_count == 1
    assert [entry.raw_quote["timestamp"] for entry in w3.entries] == [2000, 3000]

    # A further overflow ingest makes dropped_count 2.
    w4 = ingest_quote(
        w3, _source(_quote(timestamp=4000)), evaluated_at_ms=4500,
        policy=POLICY, max_gap_ms=1000,
    )
    assert len(w4) == 2
    assert w4.dropped_count == 2
    assert [entry.raw_quote["timestamp"] for entry in w4.entries] == [3000, 4000]


def test_ingest_propagates_source_errors():
    class BoomError(RuntimeError):
        pass

    window = QuoteWindow(entries=(), max_size=3, dropped_count=0)

    def source():
        raise BoomError("exploded")

    with pytest.raises(BoomError) as excinfo:
        ingest_quote(window, source, evaluated_at_ms=1500, policy=POLICY)
    assert str(excinfo.value) == "exploded"

    # The original window object is unchanged.
    assert window == QuoteWindow(entries=(), max_size=3, dropped_count=0)
    assert len(window) == 0
    assert window.dropped_count == 0


def test_ingest_forwards_evaluation_parameters():
    window = QuoteWindow(entries=(), max_size=3, dropped_count=0)
    result = ingest_quote(
        window,
        _source(_quote(timestamp=1000)),
        evaluated_at_ms=101000,
        policy=POLICY,
    )
    governed = result.entries[0].governed
    assert governed.evaluated_at_ms == 101000
    assert governed.freshness.classification == "EXPIRED"
    assert governed.verdict == "UNUSABLE"


def test_verdict_counts_goldens():
    usable = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    degraded = _govern(_quote(timestamp=1000), evaluated_at_ms=4000)
    unusable = _govern(_quote(timestamp=1000, last=0.0), evaluated_at_ms=1500)
    assert usable.verdict == "USABLE"
    assert degraded.verdict == "DEGRADED"
    assert unusable.verdict == "UNUSABLE"

    entries = (
        WindowEntry(raw_quote=_quote(timestamp=1000), governed=usable),
        WindowEntry(raw_quote=_quote(timestamp=1000), governed=usable),
        WindowEntry(raw_quote=_quote(timestamp=1000), governed=degraded),
        WindowEntry(raw_quote=_quote(timestamp=1000, last=0.0), governed=unusable),
    )
    window = QuoteWindow(entries=entries, max_size=10, dropped_count=0)
    assert window.verdict_counts() == {
        "USABLE": 2,
        "DEGRADED": 1,
        "UNUSABLE": 1,
    }
    assert set(window.verdict_counts()) == {"USABLE", "DEGRADED", "UNUSABLE"}


def test_window_records_immutable():
    governed = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    raw = _quote(timestamp=1000)
    entry = WindowEntry(raw_quote=raw, governed=governed)

    with pytest.raises(AttributeError):
        entry.raw_quote = {}
    with pytest.raises(AttributeError):
        entry.governed = governed

    window = QuoteWindow(entries=(entry,), max_size=3, dropped_count=0)
    with pytest.raises(AttributeError):
        window.max_size = 10
    with pytest.raises(AttributeError):
        window.dropped_count = 1

    new_window = ingest_quote(
        window, _source(_quote(timestamp=2000)), evaluated_at_ms=2500,
        policy=POLICY, max_gap_ms=1000,
    )
    assert new_window is not window
    assert window == QuoteWindow(entries=(entry,), max_size=3, dropped_count=0)
    assert window.entries == (entry,)
    assert new_window.entries[0] is entry


def test_no_third_party_imports():
    import ast
    import pathlib
    import sys

    import borsapy.tv_window as tv_window

    source = pathlib.Path(tv_window.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])

    # ``borsapy`` is first-party; every other import must be stdlib.
    allowed = set(sys.stdlib_module_names) | {"borsapy"}
    assert modules
    assert modules <= allowed
