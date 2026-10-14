"""Live smoke slice for the governed TradingView quote pipeline.

This module is an INTEGRATION test: it drives the REAL network transport
(``Ticker.get_live_quote()``) through the full governed loop
(``ticker_quote_source`` -> ``run_quote_loop`` -> ``QuoteWindow`` /
``GovernedLiveQuote`` records) with a REAL wall clock, and asserts only
STRUCTURAL invariants. Verdict VALUES are live-data-dependent and are never
asserted beyond lattice membership: the smoke contract proves the pipeline
composes end-to-end over the real wire, not that the market is in any
particular state.

Both tests carry ``@pytest.mark.integration`` and are skipped by default
through the frozen ``tests/conftest.py`` gate (``--run-integration`` flag or
``RUN_INTEGRATION=1``). Importing this module performs NO network access, does
NOT construct a provider, does NOT read a wall clock and writes no files. The
anonymous transport is the smoke default: no credentials are read, configured
or printed, and none are required.

There are no retries around any network call: a provider ``APIError`` (or any
other transport exception) fails the smoke test verbatim.

Race caveat: the deterministic evaluation-clock sampling race (a pre-fetch
governing instant making a freshly fetched quote look future-dated) is
structurally CLOSED by post-fetch sampling — ``run_quote_loop`` threads its
``clock`` into ``ingest_quote``, which samples the governing instant AFTER the
fetch returns and BEFORE governing. The assertions below are unchanged.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping

import pytest

from borsapy.ticker import Ticker
from borsapy.tv_loop import LiveLoopResult, run_quote_loop, ticker_quote_source
from borsapy.tv_production import GovernedLiveQuote
from borsapy.tv_reliability import FreshnessPolicy
from borsapy.tv_window import QuoteWindow, WindowEntry

#: The exact 8-key public allowlist returned by ``Ticker.get_live_quote()``.
_PUBLIC_KEYS = (
    "symbol",
    "exchange",
    "last",
    "bid",
    "ask",
    "volume",
    "timestamp",
    "auth_transport",
)

#: The exact ``auth_transport`` categorical values.
_AUTH_TRANSPORTS = ("auth_token", "anonymous_token", "unknown")

#: The exact governed verdict lattice.
_VERDICTS = ("USABLE", "DEGRADED", "UNUSABLE")

#: Closed-left/open-right epoch bands: seconds and milliseconds.
_SECONDS_MIN = 10**9
_SECONDS_MAX = 10**12
_MILLIS_MIN = 10**12
_MILLIS_MAX = 10**14

#: Inclusive tick-budget clamp band for the smoke loop.
_MIN_SMOKE_TICKS = 1
_MAX_SMOKE_TICKS = 10

#: The symbol under test; overridable via ``BORSAPY_TV_SMOKE_SYMBOL``.
SMOKE_SYMBOL = os.environ.get("BORSAPY_TV_SMOKE_SYMBOL", "THYAO")

#: Raw tick-budget configuration from ``BORSAPY_TV_SMOKE_TICKS``. It is PARSED,
#: validated and clamped inside the loop test (never at module import time).
SMOKE_TICKS = os.environ.get("BORSAPY_TV_SMOKE_TICKS", "3")


def _epoch_unit_scale(timestamp: int) -> int:
    """Classify a positive epoch value into its seconds/milliseconds unit.

    Returns ``1`` for the seconds band ``10**9 <= timestamp < 10**12`` and
    ``1000`` for the milliseconds band ``10**12 <= timestamp < 10**14``. Both
    bands are closed on the left and open on the right. Any other magnitude —
    including values ``< 10**9`` and values ``>= 10**14`` (e.g. microseconds
    ``10**15`` or garbage) — fails closed with ``ValueError`` naming the
    observed value and both accepted bands with their exact bounds.
    """
    if not isinstance(timestamp, bool) and isinstance(timestamp, int):
        if _SECONDS_MIN <= timestamp < _SECONDS_MAX:
            return 1
        if _MILLIS_MIN <= timestamp < _MILLIS_MAX:
            return 1000
    raise ValueError(
        f"epoch timestamp {timestamp!r} is outside both accepted bands: "
        f"seconds [{_SECONDS_MIN}, {_SECONDS_MAX}) (10**9 <= t < 10**12) "
        f"or milliseconds [{_MILLIS_MIN}, {_MILLIS_MAX}) (10**12 <= t < 10**14)"
    )


def _now(unit_scale: int) -> int:
    """Return the real wall clock in the calibrated unit ``unit_scale``."""
    return int(time.time() * unit_scale)


def _clamp_tick_budget(raw: str) -> int:
    """Parse ``raw`` as an integer and clamp it into the inclusive 1..10 band.

    Fails closed with ``ValueError`` on non-integer input: the literal is never
    coerced from a non-numeric value.
    """
    try:
        ticks = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"BORSAPY_TV_SMOKE_TICKS must be an integer, got {raw!r}"
        ) from exc
    return max(_MIN_SMOKE_TICKS, min(_MAX_SMOKE_TICKS, ticks))


@pytest.mark.integration
def test_live_quote_public_shape():
    """One real quote is structurally well-formed; the epoch helper is pinned.

    The single ``get_live_quote()`` call exercises the real wire and is asserted
    only against the public shape: a ``dict`` whose keys lie within the exact
    8-key allowlist, with a positive-int ``timestamp`` and a categorical
    ``auth_transport`` when present. Market fields (``last``/``bid``/``ask``/
    ``volume``) are intentionally never asserted. The ``_epoch_unit_scale``
    boundary table and fail-closed cases are pinned offline (no network).
    """
    ticker = Ticker(SMOKE_SYMBOL)
    quote = ticker.get_live_quote()

    assert isinstance(quote, dict)
    assert set(quote) <= set(_PUBLIC_KEYS)

    timestamp = quote.get("timestamp")
    if timestamp is not None:
        assert isinstance(timestamp, int)
        assert not isinstance(timestamp, bool)
        assert timestamp > 0

    auth_transport = quote.get("auth_transport")
    if auth_transport is not None:
        assert auth_transport in _AUTH_TRANSPORTS

    # Environment-gated activation: when the test process has a non-blank
    # ``TRADINGVIEW_SESSION`` loaded, the lazy environment bootstrap MUST have
    # activated the authenticated transport for this real quote. When absent,
    # nothing about the transport is asserted (anonymous remains acceptable).
    # Credential values are never read, printed or persisted here.
    if os.environ.get("TRADINGVIEW_SESSION", "").strip():
        assert auth_transport == "auth_token"

    # Golden boundary table (closed-left/open-right).
    assert _epoch_unit_scale(10**9) == 1
    assert _epoch_unit_scale(10**12 - 1) == 1
    assert _epoch_unit_scale(10**12) == 1000
    assert _epoch_unit_scale(10**14 - 1) == 1000

    # Fail-closed out-of-band magnitudes.
    for bad in (10**8, 0, 10**14, 10**15):
        with pytest.raises(ValueError) as excinfo:
            _epoch_unit_scale(bad)
        message = str(excinfo.value)
        assert str(bad) in message
        assert "seconds" in message
        assert "milliseconds" in message
        assert str(_SECONDS_MIN) in message
        assert str(_SECONDS_MAX) in message
        assert str(_MILLIS_MIN) in message
        assert str(_MILLIS_MAX) in message


@pytest.mark.integration
def test_live_quote_loop_smoke():
    """The real transport composes end-to-end through the governed loop.

    A single calibration pull fixes the epoch unit, then ``run_quote_loop``
    ingests the clamped tick budget from the real transport with a real wall
    clock. Only structural invariants are asserted: the loop exhausts its tick
    budget, retains every tick, emits governed records in the exact verdict
    lattice with positive-int timestamps, never leaks provider-internal keys,
    and sees non-decreasing raw timestamps in ingest order. No verdict value,
    freshness classification or ``age_ms`` is asserted.
    """
    tick_budget = _clamp_tick_budget(SMOKE_TICKS)

    ticker = Ticker(SMOKE_SYMBOL)
    calibration = ticker.get_live_quote()
    timestamp = calibration.get("timestamp")
    if timestamp is None:
        pytest.fail(
            f"{SMOKE_SYMBOL}: live quote carried no timestamp to calibrate "
            f"the loop against"
        )
    unit_scale = _epoch_unit_scale(timestamp)

    policy = FreshnessPolicy(
        fresh_max_age_ms=60 * unit_scale,
        stale_max_age_ms=300 * unit_scale,
    )
    max_gap_ms = 600 * unit_scale
    source = ticker_quote_source(ticker)
    window = QuoteWindow(entries=(), max_size=16, dropped_count=0)

    def clock() -> int:
        return _now(unit_scale)

    result = run_quote_loop(
        window,
        source,
        clock=clock,
        policy=policy,
        sessions=None,
        max_gap_ms=max_gap_ms,
        tick_budget=tick_budget,
        stop=None,
        pause=None,
    )

    assert isinstance(result, LiveLoopResult)
    assert result.iterations == tick_budget
    assert result.stop_reason == "TICK_BUDGET_EXHAUSTED"
    assert len(result.window.entries) == tick_budget
    assert result.window.dropped_count == 0

    observed_timestamps = []
    for entry in result.window.entries:
        assert isinstance(entry, WindowEntry)
        governed = entry.governed
        assert isinstance(governed, GovernedLiveQuote)
        assert governed.verdict in _VERDICTS

        assert isinstance(governed.quote_timestamp_ms, int)
        assert not isinstance(governed.quote_timestamp_ms, bool)
        assert governed.quote_timestamp_ms > 0
        assert isinstance(governed.evaluated_at_ms, int)
        assert not isinstance(governed.evaluated_at_ms, bool)
        assert governed.evaluated_at_ms > 0

        raw_quote = entry.raw_quote
        assert isinstance(raw_quote, Mapping)
        assert set(raw_quote) <= set(_PUBLIC_KEYS)

        quote_timestamp = raw_quote.get("timestamp")
        if quote_timestamp is not None:
            observed_timestamps.append(quote_timestamp)

    for previous, current in zip(
        observed_timestamps, observed_timestamps[1:], strict=False
    ):
        assert current >= previous

    assert set(result.window.verdict_counts()) == set(_VERDICTS)
