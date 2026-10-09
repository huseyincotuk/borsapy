"""Tests for the deterministic TradingView governed live-quote production gate.

All tests are fully offline and deterministic: every input is a fixed literal
and every quote source is a fake callable. The module under test never reads a
wall clock, performs network access or instantiates real TradingView providers.
"""

import dataclasses

import pytest

import borsapy as bp
from borsapy.tv_production import GovernedLiveQuote, govern_live_quote
from borsapy.tv_reliability import FreshnessPolicy

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


def test_verdict_usable_fresh_clean():
    record = _govern(_quote(timestamp=1000), evaluated_at_ms=1500)
    assert isinstance(record, GovernedLiveQuote)
    assert record.freshness.classification == "FRESH"
    assert record.sanity_issues == ()
    assert record.sequence_issues == ()
    assert record.verdict == "USABLE"


def test_verdict_degraded_stale():
    record = _govern(_quote(timestamp=0), evaluated_at_ms=2000)
    assert record.freshness.classification == "STALE"
    assert record.sanity_issues == ()
    assert record.sequence_issues == ()
    assert record.verdict == "DEGRADED"


def test_verdict_degraded_duplicate_timestamp():
    record = _govern(
        _quote(timestamp=1000),
        evaluated_at_ms=1500,
        prior_quotes=(_quote(timestamp=1000),),
        max_gap_ms=10,
    )
    kinds = [issue.kind for issue in record.sequence_issues]
    assert kinds == ["DUPLICATE_TIMESTAMP"]
    assert record.freshness.classification == "FRESH"
    assert record.verdict == "DEGRADED"


def test_verdict_degraded_timestamp_gap():
    record = _govern(
        _quote(timestamp=2000),
        evaluated_at_ms=2500,
        prior_quotes=(_quote(timestamp=1000),),
        max_gap_ms=100,
    )
    kinds = [issue.kind for issue in record.sequence_issues]
    assert kinds == ["TIMESTAMP_GAP"]
    assert record.freshness.classification == "FRESH"
    assert record.verdict == "DEGRADED"


def test_verdict_unusable_expired():
    record = _govern(_quote(timestamp=0), evaluated_at_ms=9000)
    assert record.freshness.classification == "EXPIRED"
    assert record.verdict == "UNUSABLE"


def test_verdict_unusable_out_of_session():
    record = _govern(_quote(timestamp=1000), evaluated_at_ms=1500, sessions=[])
    assert record.freshness.classification == "OUT_OF_SESSION"
    assert record.verdict == "UNUSABLE"


def test_verdict_unusable_any_sanity_issue():
    crossed = _govern(_quote(bid=10.2, ask=10.0))
    assert any(issue.kind == "CROSSED_MARKET" for issue in crossed.sanity_issues)
    assert crossed.verdict == "UNUSABLE"

    non_positive = _govern(_quote(last=0.0))
    assert any(
        issue.kind == "NON_POSITIVE_PRICE" for issue in non_positive.sanity_issues
    )
    assert non_positive.verdict == "UNUSABLE"


def test_verdict_unusable_out_of_order():
    record = _govern(
        _quote(timestamp=1000),
        evaluated_at_ms=1500,
        prior_quotes=(_quote(timestamp=2000),),
        max_gap_ms=10,
    )
    kinds = [issue.kind for issue in record.sequence_issues]
    assert kinds == ["OUT_OF_ORDER"]
    assert record.verdict == "UNUSABLE"


def test_verdict_precedence_unusable_over_degraded():
    # STALE (a DEGRADED condition) combined with OUT_OF_ORDER (an UNUSABLE one).
    record = _govern(
        _quote(timestamp=2000),
        evaluated_at_ms=5000,
        prior_quotes=(_quote(timestamp=3000),),
        max_gap_ms=10,
    )
    assert record.freshness.classification == "STALE"
    assert any(issue.kind == "OUT_OF_ORDER" for issue in record.sequence_issues)
    assert record.verdict == "UNUSABLE"


def test_source_called_exactly_once():
    calls = []
    quote = _quote()

    def source():
        calls.append(1)
        return quote

    govern_live_quote(source, evaluated_at_ms=1500, policy=POLICY)
    assert calls == [1]


def test_source_exception_propagates():
    class BoomError(RuntimeError):
        pass

    def source():
        raise BoomError("exploded")

    with pytest.raises(BoomError) as excinfo:
        govern_live_quote(source, evaluated_at_ms=1500, policy=POLICY)
    assert str(excinfo.value) == "exploded"


def test_sequence_issues_only_for_new_quote():
    # The duplicate at index 1 belongs to earlier ticks and must not reappear.
    record = _govern(
        _quote(timestamp=2000),
        evaluated_at_ms=2500,
        prior_quotes=(_quote(timestamp=1000), _quote(timestamp=1000)),
        max_gap_ms=5000,
    )
    assert record.sequence_issues == ()
    assert record.verdict == "USABLE"


def test_sequence_requires_max_gap_ms_with_prior_quotes():
    prior = (_quote(),)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=prior, max_gap_ms=None)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=prior, max_gap_ms=0)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=prior, max_gap_ms=-5)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=prior, max_gap_ms=True)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=prior, max_gap_ms=1.5)


def test_sequence_rejects_max_gap_ms_without_prior_quotes():
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=(), max_gap_ms=10)
    with pytest.raises((TypeError, ValueError)):
        _govern(_quote(), prior_quotes=[], max_gap_ms=10)


def test_quote_consumed_verbatim():
    quote = _quote(timestamp=1000, extra="kept", missingish=None)
    snapshot = dict(quote)
    record = govern_live_quote(
        _source(quote), evaluated_at_ms=1500, policy=POLICY
    )
    # The returned mapping is never mutated, including None values.
    assert quote == snapshot
    assert quote["missingish"] is None
    assert record.symbol == "THYAO"
    assert record.auth_transport == "auth_token"
    assert record.quote_timestamp_ms == 1000
    assert record.evaluated_at_ms == 1500


def test_symbol_and_auth_transport_missing_are_none():
    quote = {"timestamp": 1000, "last": 10.0, "bid": 9.9, "ask": 10.1}
    record = govern_live_quote(
        _source(quote), evaluated_at_ms=1500, policy=POLICY
    )
    assert record.symbol is None
    assert record.auth_transport is None

    # Explicit None values present are carried verbatim as None.
    explicit = {"timestamp": 1000, "symbol": None, "auth_transport": None}
    record_explicit = govern_live_quote(
        _source(explicit), evaluated_at_ms=1500, policy=POLICY
    )
    assert record_explicit.symbol is None
    assert record_explicit.auth_transport is None


def test_governed_record_immutable():
    record = _govern(_quote())
    assert dataclasses.is_dataclass(record)
    with pytest.raises(AttributeError):
        record.verdict = "USABLE"
    with pytest.raises(AttributeError):
        record.symbol = "OTHER"
    with pytest.raises(AttributeError):
        record.freshness = None
    assert isinstance(record.sanity_issues, tuple)
    assert isinstance(record.sequence_issues, tuple)


def test_deterministic_repeat_calls_equal():
    quote = _quote(timestamp=2000)
    prior = (_quote(timestamp=1500),)
    first = govern_live_quote(
        _source(quote),
        evaluated_at_ms=2500,
        policy=POLICY,
        prior_quotes=prior,
        max_gap_ms=1000,
    )
    second = govern_live_quote(
        _source(quote),
        evaluated_at_ms=2500,
        policy=POLICY,
        prior_quotes=prior,
        max_gap_ms=1000,
    )
    assert first == second
    assert hash(first) == hash(second)


def test_invalid_inputs_fail_closed():
    # Non-callable source rejected before any invocation.
    with pytest.raises(TypeError):
        govern_live_quote(None, evaluated_at_ms=1500, policy=POLICY)
    with pytest.raises(TypeError):
        govern_live_quote(42, evaluated_at_ms=1500, policy=POLICY)

    # Invalid policy rejected by the frozen layer's fail-closed validation.
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(_source(_quote()), evaluated_at_ms=1500, policy="nope")

    # Non-mapping quote rejected through the frozen assessment layer.
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(_source([]), evaluated_at_ms=1500, policy=POLICY)

    # Invalid evaluated_at_ms propagates from the frozen layer.
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(_source(_quote()), evaluated_at_ms=True, policy=POLICY)

    # Non-sequence prior_quotes rejected.
    with pytest.raises(TypeError):
        govern_live_quote(
            _source(_quote()), evaluated_at_ms=1500, policy=POLICY, prior_quotes=5
        )

    # Missing timestamp propagates from the frozen layer.
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(
            _source({"last": 1}), evaluated_at_ms=1500, policy=POLICY
        )


def test_public_imports():
    from borsapy import GovernedLiveQuote as PkgGovernedLiveQuote
    from borsapy import govern_live_quote as pkg_govern_live_quote

    assert PkgGovernedLiveQuote is bp.GovernedLiveQuote
    assert pkg_govern_live_quote is bp.govern_live_quote
    assert PkgGovernedLiveQuote is GovernedLiveQuote
    assert pkg_govern_live_quote is govern_live_quote
    assert "GovernedLiveQuote" in bp.__all__
    assert "govern_live_quote" in bp.__all__

    module_all = set(__import__("borsapy.tv_production", fromlist=["__all__"]).__all__)
    assert module_all == {"GovernedLiveQuote", "govern_live_quote"}


def test_golden_non_callable_source_rejected():
    with pytest.raises(TypeError):
        govern_live_quote(object(), evaluated_at_ms=1500, policy=POLICY)


def test_golden_non_mapping_quote_rejected():
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(_source("not-a-mapping"), evaluated_at_ms=1500, policy=POLICY)


def test_golden_invalid_freshness_policy_rejected():
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(_source(_quote()), evaluated_at_ms=1500, policy=None)
    with pytest.raises((TypeError, ValueError)):
        govern_live_quote(
            _source(_quote()), evaluated_at_ms=1500, policy={"fresh": 1}
        )


def test_golden_package_level_import_of_every_public_name():
    import importlib

    production = importlib.import_module("borsapy.tv_production")
    for name in production.__all__:
        assert hasattr(bp, name)
        assert getattr(bp, name) is getattr(production, name)
