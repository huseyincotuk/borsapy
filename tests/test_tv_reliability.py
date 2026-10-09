"""Tests for the deterministic TradingView live-quote reliability layer.

All tests are fully offline and deterministic: every input is a fixed literal
and the module under test never reads a wall clock or performs network access.
"""

import dataclasses

import pytest

import borsapy as bp
from borsapy.tv_reliability import (
    FreshnessAssessment,
    FreshnessPolicy,
    QuoteSanityIssue,
    QuoteSequenceAssessment,
    QuoteSequenceIssue,
    TradingSessionWindow,
    assess_live_quote_freshness,
    assess_live_quote_sanity,
    assess_quote_sequence,
)

SESSION_A = TradingSessionWindow(start_ms=0, end_ms=1000)
SESSION_B = TradingSessionWindow(start_ms=2000, end_ms=3000)
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
        "timestamp": 1_000,
        "auth_transport": "auth_token",
    }
    quote.update(overrides)
    return quote


def test_freshness_fresh_within_threshold():
    result = assess_live_quote_freshness(
        _quote(timestamp=1000),
        evaluated_at_ms=2000,
        policy=FreshnessPolicy(fresh_max_age_ms=1500, stale_max_age_ms=3000),
    )
    assert isinstance(result, FreshnessAssessment)
    assert result.classification == "FRESH"
    assert result.age_ms == 1000
    assert result.quote_timestamp_ms == 1000
    assert result.evaluated_at_ms == 2000


def test_freshness_stale_between_thresholds():
    result = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=2000,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert result.classification == "STALE"
    assert result.age_ms == 2000


def test_freshness_expired_beyond_threshold():
    result = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=9000,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert result.classification == "EXPIRED"
    assert result.age_ms == 9000


def test_freshness_boundary_exactly_at_thresholds():
    exact_fresh = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=1000,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert exact_fresh.classification == "FRESH"

    exact_stale = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=5000,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert exact_stale.classification == "STALE"

    just_stale = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=1001,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert just_stale.classification == "STALE"

    just_expired = assess_live_quote_freshness(
        _quote(timestamp=0),
        evaluated_at_ms=5001,
        policy=FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=5000),
    )
    assert just_expired.classification == "EXPIRED"


def test_freshness_out_of_session():
    result = assess_live_quote_freshness(
        _quote(timestamp=1000),
        evaluated_at_ms=1500,
        policy=POLICY,
        sessions=(SESSION_A, SESSION_B),
    )
    assert result.classification == "OUT_OF_SESSION"
    assert result.age_ms is None

    in_session = assess_live_quote_freshness(
        _quote(timestamp=1000),
        evaluated_at_ms=1500,
        policy=POLICY,
        sessions=(TradingSessionWindow(start_ms=0, end_ms=2000),),
    )
    assert in_session.classification == "FRESH"


def test_freshness_no_gating_when_sessions_none():
    result = assess_live_quote_freshness(
        _quote(timestamp=1000),
        evaluated_at_ms=10_000_000,
        policy=POLICY,
        sessions=None,
    )
    assert result.classification == "EXPIRED"


def test_freshness_empty_session_set_always_out_of_session():
    result = assess_live_quote_freshness(
        _quote(timestamp=1000),
        evaluated_at_ms=1000,
        policy=POLICY,
        sessions=[],
    )
    assert result.classification == "OUT_OF_SESSION"
    assert result.age_ms is None


def test_freshness_future_quote_fails_closed():
    with pytest.raises(ValueError):
        assess_live_quote_freshness(
            _quote(timestamp=2000),
            evaluated_at_ms=1000,
            policy=POLICY,
        )


def test_sequence_duplicate_timestamp():
    result = assess_quote_sequence(
        [_quote(timestamp=100), _quote(timestamp=100)],
        max_gap_ms=10,
    )
    assert isinstance(result, QuoteSequenceAssessment)
    assert result.quote_count == 2
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert isinstance(issue, QuoteSequenceIssue)
    assert issue.kind == "DUPLICATE_TIMESTAMP"
    assert issue.index == 1
    assert issue.delta_ms == 0


def test_sequence_out_of_order():
    result = assess_quote_sequence(
        [_quote(timestamp=200), _quote(timestamp=100)],
        max_gap_ms=10,
    )
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.kind == "OUT_OF_ORDER"
    assert issue.index == 1
    assert issue.delta_ms == -100


def test_sequence_timestamp_gap():
    result = assess_quote_sequence(
        [_quote(timestamp=100), _quote(timestamp=1000)],
        max_gap_ms=500,
    )
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.kind == "TIMESTAMP_GAP"
    assert issue.index == 1
    assert issue.delta_ms == 900

    exact = assess_quote_sequence(
        [_quote(timestamp=100), _quote(timestamp=600)],
        max_gap_ms=500,
    )
    assert exact.issues == ()


def test_sequence_empty_and_single_quote_valid():
    empty = assess_quote_sequence([], max_gap_ms=5)
    assert empty.quote_count == 0
    assert empty.issues == ()

    single = assess_quote_sequence([_quote(timestamp=42)], max_gap_ms=5)
    assert single.quote_count == 1
    assert single.issues == ()


def test_sanity_crossed_market():
    crossed_equal = assess_live_quote_sanity(_quote(bid=10.0, ask=10.0))
    assert QuoteSanityIssue(kind="CROSSED_MARKET", field="bid/ask") in crossed_equal

    crossed_above = assess_live_quote_sanity(_quote(bid=10.2, ask=10.0))
    assert crossed_above[0].kind == "CROSSED_MARKET"
    assert crossed_above[0].field == "bid/ask"

    normal = assess_live_quote_sanity(_quote(bid=9.9, ask=10.1))
    assert all(issue.kind != "CROSSED_MARKET" for issue in normal)


def test_sanity_non_positive_price():
    issues = assess_live_quote_sanity(
        _quote(last=0.0, bid=-1.0, ask=0.0)
    )
    fields = [issue.field for issue in issues if issue.kind == "NON_POSITIVE_PRICE"]
    assert fields == ["last", "bid", "ask"]
    assert isinstance(issues, tuple)


def test_sanity_missing_and_none_fields_skipped():
    assert assess_live_quote_sanity({"timestamp": 1}) == ()
    assert assess_live_quote_sanity({"last": None, "bid": None, "ask": None}) == ()
    assert assess_live_quote_sanity({"last": None, "bid": 10.0, "ask": 10.0})[0].kind == (
        "CROSSED_MARKET"
    )


def test_deterministic_repeat_calls_equal():
    quote = _quote(timestamp=1000)
    first_fresh = assess_live_quote_freshness(quote, evaluated_at_ms=1500, policy=POLICY)
    second_fresh = assess_live_quote_freshness(quote, evaluated_at_ms=1500, policy=POLICY)
    assert first_fresh == second_fresh

    quotes = [
        _quote(timestamp=100),
        _quote(timestamp=100),
        _quote(timestamp=50),
        _quote(timestamp=5000),
    ]
    first_seq = assess_quote_sequence(quotes, max_gap_ms=10)
    second_seq = assess_quote_sequence(quotes, max_gap_ms=10)
    assert first_seq == second_seq

    sanity_quote = _quote(last=-1.0, bid=5.0, ask=5.0)
    assert assess_live_quote_sanity(sanity_quote) == assess_live_quote_sanity(sanity_quote)

    # Frozen result records support equality and hashing.
    assert hash(first_fresh) == hash(second_fresh)
    assert dataclasses.is_dataclass(first_seq)


def test_invalid_inputs_fail_closed():
    # Invalid session window bounds and types.
    with pytest.raises(ValueError):
        TradingSessionWindow(start_ms=1, end_ms=1)
    with pytest.raises(ValueError):
        TradingSessionWindow(start_ms=2, end_ms=1)
    with pytest.raises(TypeError):
        TradingSessionWindow(start_ms=True, end_ms=2)
    with pytest.raises(TypeError):
        TradingSessionWindow(start_ms=0.0, end_ms=2)
    with pytest.raises(TypeError):
        TradingSessionWindow(start_ms=0, end_ms="2")
    with pytest.raises(ValueError):
        TradingSessionWindow(start_ms=0, end_ms=2, label="")
    with pytest.raises(TypeError):
        TradingSessionWindow(start_ms=0, end_ms=2, label=1)
    assert TradingSessionWindow(start_ms=0, end_ms=2, label="BIST").label == "BIST"

    # Overlapping windows rejected; adjacent windows accepted.
    with pytest.raises(ValueError):
        assess_live_quote_freshness(
            _quote(),
            evaluated_at_ms=100,
            policy=POLICY,
            sessions=(
                TradingSessionWindow(start_ms=0, end_ms=100),
                TradingSessionWindow(start_ms=50, end_ms=150),
            ),
        )
    with pytest.raises(ValueError):
        assess_live_quote_freshness(
            _quote(),
            evaluated_at_ms=100,
            policy=POLICY,
            sessions=(
                TradingSessionWindow(start_ms=50, end_ms=150),
                TradingSessionWindow(start_ms=0, end_ms=100),
            ),
        )
    adjacent = assess_live_quote_freshness(
        _quote(timestamp=100),
        evaluated_at_ms=100,
        policy=POLICY,
        sessions=(
            TradingSessionWindow(start_ms=0, end_ms=100),
            TradingSessionWindow(start_ms=100, end_ms=200),
        ),
    )
    assert adjacent.classification == "FRESH"

    # Non-sequence sessions.
    for bad_sessions in (1, "abc", {"a": 1}):
        with pytest.raises(TypeError):
            assess_live_quote_freshness(
                _quote(), evaluated_at_ms=100, policy=POLICY, sessions=bad_sessions
            )

    # Invalid policy.
    with pytest.raises(ValueError):
        FreshnessPolicy(fresh_max_age_ms=5000, stale_max_age_ms=1000)
    with pytest.raises(ValueError):
        FreshnessPolicy(fresh_max_age_ms=0, stale_max_age_ms=1000)
    with pytest.raises(ValueError):
        FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=-1)
    with pytest.raises(TypeError):
        FreshnessPolicy(fresh_max_age_ms=True, stale_max_age_ms=1000)
    with pytest.raises(TypeError):
        FreshnessPolicy(fresh_max_age_ms=1.0, stale_max_age_ms=1000)
    assert FreshnessPolicy(fresh_max_age_ms=1000, stale_max_age_ms=1000)

    # Non-Mapping quote.
    with pytest.raises(TypeError):
        assess_live_quote_freshness([], evaluated_at_ms=100, policy=POLICY)
    with pytest.raises(TypeError):
        assess_live_quote_sanity("not-a-mapping")

    # Missing timestamp key.
    with pytest.raises(ValueError):
        assess_live_quote_freshness({"last": 1}, evaluated_at_ms=100, policy=POLICY)

    # Non-integer timestamps, booleans explicitly rejected.
    for bad_ts in (1.5, True, "123", None):
        with pytest.raises(TypeError):
            assess_live_quote_freshness(
                _quote(timestamp=bad_ts), evaluated_at_ms=100, policy=POLICY
            )

    # Non-integer evaluated_at_ms, booleans explicitly rejected.
    with pytest.raises(TypeError):
        assess_live_quote_freshness(_quote(), evaluated_at_ms=True, policy=POLICY)

    # Invalid max_gap_ms.
    with pytest.raises(TypeError):
        assess_quote_sequence([_quote()], max_gap_ms=True)
    with pytest.raises(ValueError):
        assess_quote_sequence([_quote()], max_gap_ms=0)

    # Non-Mapping sequence element and non-integer sequence timestamps.
    with pytest.raises(TypeError):
        assess_quote_sequence([_quote(), 5], max_gap_ms=10)
    with pytest.raises(TypeError):
        assess_quote_sequence([_quote(timestamp=True)], max_gap_ms=10)
    with pytest.raises(ValueError):
        assess_quote_sequence([{"last": 1}], max_gap_ms=10)

    # Non-sequence quotes.
    with pytest.raises(TypeError):
        assess_quote_sequence("not-a-sequence", max_gap_ms=10)


def test_public_imports():
    from borsapy import (
        FreshnessAssessment as PkgFreshnessAssessment,
    )
    from borsapy import (
        FreshnessPolicy as PkgFreshnessPolicy,
    )
    from borsapy import (
        QuoteSanityIssue as PkgQuoteSanityIssue,
    )
    from borsapy import (
        QuoteSequenceAssessment as PkgQuoteSequenceAssessment,
    )
    from borsapy import (
        QuoteSequenceIssue as PkgQuoteSequenceIssue,
    )
    from borsapy import (
        TradingSessionWindow as PkgTradingSessionWindow,
    )
    from borsapy import (
        assess_live_quote_freshness as pkg_assess_live_quote_freshness,
    )
    from borsapy import (
        assess_live_quote_sanity as pkg_assess_live_quote_sanity,
    )
    from borsapy import (
        assess_quote_sequence as pkg_assess_quote_sequence,
    )

    assert PkgFreshnessAssessment is bp.FreshnessAssessment
    assert PkgFreshnessPolicy is bp.FreshnessPolicy
    assert PkgQuoteSanityIssue is bp.QuoteSanityIssue
    assert PkgQuoteSequenceAssessment is bp.QuoteSequenceAssessment
    assert PkgQuoteSequenceIssue is bp.QuoteSequenceIssue
    assert PkgTradingSessionWindow is bp.TradingSessionWindow
    assert pkg_assess_live_quote_freshness is bp.assess_live_quote_freshness
    assert pkg_assess_live_quote_sanity is bp.assess_live_quote_sanity
    assert pkg_assess_quote_sequence is bp.assess_quote_sequence

    names = (
        "TradingSessionWindow",
        "FreshnessPolicy",
        "FreshnessAssessment",
        "QuoteSequenceAssessment",
        "QuoteSequenceIssue",
        "QuoteSanityIssue",
        "assess_live_quote_freshness",
        "assess_quote_sequence",
        "assess_live_quote_sanity",
    )
    for name in names:
        assert hasattr(bp, name)
        assert name in bp.__all__
