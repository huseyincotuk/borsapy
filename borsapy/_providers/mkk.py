"""MKK VYK (Kamuyu Aydınlatma Platformu) official API provider.

Source provenance
-----------------
This provider talks to the **official MKK VYK publication API**, not to KAP
web pages.  The disclosure list, disclosure detail, last-index and members
services are consumed directly from the gateway; nothing is scraped from the
KAP HTML site.  Source metadata (disclosure index, sender codes, year, period,
currency, rounding) is carried through untouched.

Gateways and access
-------------------
=====================  ==========================================  =====================
Gateway                Base URL                                    Status
=====================  ==========================================  =====================
``test`` (default)     ``https://apigwdev.mkk.com.tr``              live, verified
``prod``               ``https://apigw.mkk.com.tr``                 IP allow-list required
=====================  ==========================================  =====================

The production gateway completes TCP+TLS and then silently drops requests for
hosts that are not on the MKK allow-list.  Production access therefore needs a
**user action** (requesting an IP allow-list entry from MKK); it is out of
scope here.  ``gateway="prod"`` selects the production base URL and any error
received from that gateway is surfaced unchanged — it is never rewritten or
masked.

Authentication is HTTP **Basic** (``base64(MKK_API_KEY:MKK_API_SECRET)``), as
verified against the test gateway.  The ``generateToken``/Bearer flow is *not*
implemented (out of scope until production access exists).  Credentials are
read from the ``MKK_API_KEY`` and ``MKK_API_SECRET`` environment variables, or
set explicitly via :func:`set_mkk_credentials`.  Missing credentials raise
:class:`~borsapy.exceptions.APIError` naming the environment variables; the
values themselves are never written to messages or logs.

Rate limiting and retries
-------------------------
The free plan allows **6 calls/minute**.  The client enforces a client-side
minimum interval of 10.5 seconds between HTTP requests, and applies exponential
backoff (1s, 2s, … capped) on HTTP 429/5xx/408/425, with at most 3 attempts.

Fault codes and empty windows
-----------------------------
MKK fault codes (``ER001``–``ER008``, ``OI001``) are embedded in
:class:`APIError` messages.  An HTTP 400 carrying ``ER005`` or ``ER008`` is the
gateway's "no disclosure in this window" answer: it returns an empty result, not
an error.  A malformed/non-JSON success body raises :class:`APIError` — the
client never fails silently with an empty value.

XBRL presentation model
-----------------------
``disclosureDetail`` (``fileType=data``) for a financial report (``FR``) carries
``presentation`` blocks.  A block is ``{id, content: {ContextList, ReportItem,
isMultiDimensional}}``.  The numeric role code in the block id identifies the
statement:

===========================  =====================  ==========================
Role family                  Statement              Taxonomy
===========================  =====================  ==========================
``210015`` / ``210014``      ``balance_sheet``      general / finance
``310000`` / ``310020``      ``income_stmt``        general / finance
``420000`` / ``420005``      ``comprehensive_income``  general / finance
``520003`` / ``510010``      ``cashflow``           general / finance
``610000`` / ``610006``      ``equity``             general / finance
===========================  =====================  ==========================

The ``general_role_`` / ``finance_role_`` prefix is normalized away; the numeric
part is authoritative.  Unknown roles are skipped for forward compatibility
(the finance ``210502`` off-balance-sheet block is not mapped to a canonical
statement).

``ContextList.Context[]`` maps each context id to a key and period.  The keys
are ``CURR`` (year-to-date cumulative), ``CURR3`` (last three months — the pure,
as-reported quarter), ``PREV`` (prior-year YTD) and ``PREV3`` (prior-year same
quarter).  Instant contexts (balance sheet) render ``YYYY-MM-DD``; duration
contexts render the source's compact concatenated form
(``2023-07-012023-09-30``) — i.e. the intra-date dashes are kept and only the
separator between the two dates is removed.

``ReportItem`` is a recursive tree.  Abstract rows are skipped, row labels use
the Turkish label first, then English, then the taxonomy name, and duplicate row
labels are preserved in source order.  Leaf ``Values.Value[]`` entries carry a
string ``value`` (``"2251210"``, ``"-2170390"``) plus ``currency`` and
``rounding``; values are converted to float without scaling.  Unparseable
values become ``NaN`` rather than being silently substituted.

Dimensional contexts
--------------------
When several ``Value`` entries share the same context key (dimensional
contexts, e.g. the finance balance sheet's cash row with three ``CURR`` cells),
the **first** entry in source order deterministically becomes the cell value.
Every value for that key (including the selected first one) is preserved in
``DataFrame.attrs["dim_cells"]`` keyed by row label, so no source information is
lost.  ``DataFrame.attrs["currency"]`` holds the single currency when uniform,
or the first currency when mixed.

CURR3 and the FA-01a lapping bridge
-----------------------------------
``CURR3`` is carried exactly as reported: it is the pure quarter supplied by
MKK, with no derivation or normalization.  It is **not** the İş Yatırım TMS-29
inflation-restated series and the two must never be mixed in one series.  The
FA-01a ``IsYatirimProvider.derive_standalone_quarters`` docstring promises
"independent verification against MKK CURR3 data is deferred to FA-01b"; this
provider supplies the as-reported data, while the actual comparison of the
İş Yatırım derived quarter against the MKK ``CURR3`` value (same quarter) is
deferred to the FA-02 cross-validation task.

NOTICE
------
Retry/backoff, rate-friendly transport and XBRL-walk patterns are adapted from
the MIT-licensed project ``hasancagrigungor/kapmcp`` (hc_client.py and
hc_kap_financials.py).  Copyright (c) the kapmcp authors; MIT License.

AGENTS.md discipline applies throughout: the source is preserved, values are
never scaled, restated or imputed, and missing/unavailable data is never
silently replaced.
"""

from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.exceptions import APIError, DataNotAvailableError

# ----- Gateways ---------------------------------------------------------------

GATEWAY_TEST = "https://apigwdev.mkk.com.tr"
GATEWAY_PROD = "https://apigw.mkk.com.tr"
GATEWAYS = {"test": GATEWAY_TEST, "prod": GATEWAY_PROD}

# Free plan: 6 calls/minute → stay below it with a 10.5 s client-side interval.
MIN_REQUEST_INTERVAL = 10.5

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
EMPTY_WINDOW_FAULT_CODES = frozenset({"ER005", "ER008"})
MAX_ATTEMPTS = 3

# Documented MKK fault codes.  Descriptions are advisory; the code itself is
# always embedded verbatim in the raised APIError.
FAULT_CODES = {
    "ER001": "Service access not authorized (API key not entitled to this service)",
    "ER002": "Unauthorized request",
    "ER003": "IP address not defined — the IP must be allow-listed by MKK",
    "ER004": "Invalid token",
    "ER005": "IP could not be verified; the gateway also reuses this code for 'no disclosure found'",
    "ER006": "Invalid token",
    "ER007": "Token information could not be verified",
    "ER008": "Authorization token is not valid",
    "OI001": "Authentication policy failed (missing/invalid API key, secret or token)",
}

# ----- Statement roles --------------------------------------------------------

ROLE_TO_STATEMENT = {
    "210015": "balance_sheet",
    "210014": "balance_sheet",
    "310000": "income_stmt",
    "310020": "income_stmt",
    "420000": "comprehensive_income",
    "420005": "comprehensive_income",
    "520003": "cashflow",
    "510010": "cashflow",
    "610000": "equity",
    "610006": "equity",
}

STATEMENT_TYPES = tuple(dict.fromkeys(ROLE_TO_STATEMENT.values()))
CONTEXT_KEY_ORDER = ("CURR", "CURR3", "PREV", "PREV3")

# ----- Credentials ------------------------------------------------------------

_api_key: str | None = None
_api_secret: str | None = None


def set_mkk_credentials(api_key: str, api_secret: str) -> None:
    """Set the MKK VYK credentials explicitly (EVDS-style setter).

    Args:
        api_key: The ``MKK_API_KEY`` value.
        api_secret: The ``MKK_API_SECRET`` value.

    Raises:
        ValueError: If either value is empty.
    """
    global _api_key, _api_secret
    if not api_key or not isinstance(api_key, str):
        raise ValueError("MKK API key must be a non-empty string")
    if not api_secret or not isinstance(api_secret, str):
        raise ValueError("MKK API secret must be a non-empty string")
    _api_key = api_key.strip()
    _api_secret = api_secret.strip()


def clear_mkk_credentials() -> None:
    """Forget explicitly configured MKK credentials (fall back to env vars)."""
    global _api_key, _api_secret
    _api_key = None
    _api_secret = None


def get_mkk_credentials() -> tuple[str | None, str | None]:
    """Return ``(api_key, api_secret)`` from the setter or environment.

    Explicitly set values win over ``MKK_API_KEY`` / ``MKK_API_SECRET``.
    """
    key = _api_key or os.environ.get("MKK_API_KEY", "").strip() or None
    secret = _api_secret or os.environ.get("MKK_API_SECRET", "").strip() or None
    return key, secret


_EMPTY = object()


def _as_list(value: Any) -> list[Any]:
    """Normalize the gateway's single-vs-list JSON shape to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_abstract(item: dict[str, Any]) -> bool:
    return str(item.get("abstract", "")).strip().lower() == "true"


def _parse_float(raw: Any) -> float:
    """Convert a source value string to float; unparseable → NaN (never 0)."""
    if raw is None:
        return float("nan")
    if isinstance(raw, bool):
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace("\u00a0", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def normalize_role(block_id: Any) -> str | None:
    """Extract the numeric role code from a presentation block id.

    ``general_role_210015`` and ``finance_role_210014`` both normalize to their
    numeric tail; unknown/non-numeric ids return ``None``.
    """
    if block_id is None:
        return None
    matches = re.findall(r"\d{6}", str(block_id))
    return matches[-1] if matches else None


@dataclass(frozen=True)
class _Context:
    """Resolved context metadata for one presentation block."""

    key: str
    period: str
    instant: bool
    start_date: str | None
    end_date: str | None


def _resolve_context_objects(block: dict[str, Any]) -> dict[str, _Context]:
    content = block.get("content") if isinstance(block, dict) else None
    if not isinstance(content, dict):
        return {}
    context_list = content.get("ContextList") or {}
    contexts = _as_list(context_list.get("Context") if isinstance(context_list, dict) else None)

    resolved: dict[str, _Context] = {}
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        ctx_id = ctx.get("id")
        key = ctx.get("key")
        if ctx_id is None or key is None:
            continue
        period = ctx.get("Period") or {}
        if not isinstance(period, dict):
            period = {}
        start = period.get("startDate")
        end = period.get("endDate")
        instant_value = period.get("instant")
        instant = instant_value is not None or (start is None and end is None)
        if instant:
            period_text = str(instant_value if instant_value is not None else ctx_id)
        else:
            # Source-faithful compact form: the intra-date dashes are kept and
            # only the separator between the two dates is removed (the source
            # context id, e.g. "2023-07-012023-09-30").
            period_text = str(ctx_id)
            if period_text in ("", "None"):
                period_text = f"{start}{end}" if start and end else period_text
        resolved[str(ctx_id)] = _Context(
            key=str(key),
            period=period_text,
            instant=instant,
            start_date=str(start) if start else None,
            end_date=str(end) if end else None,
        )
    return resolved


def resolve_contexts(block: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Resolve a block's ``ContextList`` to ``{contextId: (key, period)}``.

    Instant contexts use ``YYYY-MM-DD``; duration contexts use the compact
    source form (e.g. ``2023-07-012023-09-30``).
    """
    return {ctx_id: (ctx.key, ctx.period) for ctx_id, ctx in _resolve_context_objects(block).items()}


def _column_label(ctx: _Context) -> str:
    if ctx.instant:
        return ctx.key
    end = ctx.end_date or ctx.period
    year = end[:4]
    try:
        month = int(end[5:7])
    except (TypeError, ValueError):
        return ctx.key
    quarter = (month - 1) // 3 + 1
    if ctx.key in ("CURR", "PREV"):
        return f"{year}Q{quarter} (YTD)"
    if ctx.key in ("CURR3", "PREV3"):
        return f"{year}Q{quarter} (3M)"
    return ctx.key


def _item_label(item: dict[str, Any]) -> str:
    langs = item.get("langs") or {}
    lang_list = _as_list(langs.get("lang") if isinstance(langs, dict) else None)
    by_code = {
        str(lang.get("code")): lang.get("value")
        for lang in lang_list
        if isinstance(lang, dict) and lang.get("code") is not None
    }
    for code in ("tr", "en"):
        value = by_code.get(code)
        if value:
            return str(value)
    name = item.get("name")
    return str(name) if name is not None else ""


def _item_values(item: dict[str, Any]) -> list[dict[str, Any]]:
    values = item.get("Values")
    if not isinstance(values, dict):
        return []
    return [v for v in _as_list(values.get("Value")) if isinstance(v, dict)]


def _iter_report_items(node: Any):
    """Depth-first walk over a ``ReportItem`` tree (dict or list shaped)."""
    for item in _as_list(node):
        if not isinstance(item, dict):
            continue
        yield item
        yield from _iter_report_items(item.get("ReportItem"))


def parse_presentation_block(block: dict[str, Any]) -> pd.DataFrame:
    """Parse one XBRL presentation block into a row-per-item DataFrame.

    The frame is indexed by source row labels (duplicates preserved) with one
    column per context key in ``CURR, CURR3, PREV, PREV3`` order.  Currency and
    dimensional-context evidence live in ``DataFrame.attrs``.
    """
    contexts = _resolve_context_objects(block)

    labels: list[str] = []
    rows: list[dict[str, float]] = []
    dim_cells: dict[str, dict[str, list[float]]] = {}
    currencies: list[str] = []

    content = block.get("content") if isinstance(block, dict) else None
    root = content.get("ReportItem") if isinstance(content, dict) else None

    for item in _iter_report_items(root):
        if _is_abstract(item):
            continue
        raw_values = _item_values(item)
        if not raw_values:
            continue

        label = _item_label(item)
        per_key: dict[str, list[float]] = {}
        for raw in raw_values:
            ctx = contexts.get(str(raw.get("contextId")))
            if ctx is None:
                # A value without a resolvable context cannot be placed; keep
                # walking rather than inventing a column.
                continue
            per_key.setdefault(ctx.key, []).append(_parse_float(raw.get("value")))
            currency = raw.get("currency")
            if currency:
                currencies.append(str(currency))

        if not per_key:
            continue

        labels.append(label)
        rows.append({key: values[0] for key, values in per_key.items()})
        multi = {key: values for key, values in per_key.items() if len(values) > 1}
        if multi:
            dim_cells[label] = multi

    present_keys = {key for row in rows for key in row}
    ordered_keys = [key for key in CONTEXT_KEY_ORDER if key in present_keys]
    ordered_keys += sorted(present_keys - set(CONTEXT_KEY_ORDER))

    columns: list[str] = []
    for key in ordered_keys:
        ctx = next((c for c in contexts.values() if c.key == key), None)
        columns.append(_column_label(ctx) if ctx is not None else key)

    frame = pd.DataFrame(
        [[row.get(key, float("nan")) for key in ordered_keys] for row in rows],
        index=pd.Index(labels, name="item"),
        columns=columns,
        dtype=float,
    )
    # Currency is uniform in the captured fixtures; when a block mixes
    # currencies the first one in source order is carried (see module docstring).
    frame.attrs["currency"] = currencies[0] if currencies else None
    frame.attrs["dim_cells"] = dim_cells
    frame.attrs["role"] = normalize_role(block.get("id") if isinstance(block, dict) else None)
    return frame


def parse_disclosure_detail(detail: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Map a ``disclosureDetail`` response to ``{statement: DataFrame}``.

    Recognized role families are converted; unknown roles are skipped so a new
    taxonomy block never breaks the parse.  When a detail carries more than one
    block for the same statement key, the last block in source order wins.
    """
    result: dict[str, pd.DataFrame] = {}
    if not isinstance(detail, dict):
        return result

    for block in _as_list(detail.get("presentation")):
        if not isinstance(block, dict):
            continue
        statement = ROLE_TO_STATEMENT.get(normalize_role(block.get("id")) or "")
        if statement is None:
            continue
        result[statement] = parse_presentation_block(block)
    return result


class MkkVykProvider(BaseProvider):
    """Client for the official MKK VYK API (see the module docstring)."""

    DISCLOSURE_PAGE_SIZE = 50

    def __init__(
        self,
        gateway: str = "test",
        timeout: float = 30.0,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_attempts: int = MAX_ATTEMPTS,
        clock: Any = None,
        sleeper: Any = None,
        **kwargs: Any,
    ):
        """
        Args:
            gateway: ``"test"`` (default) or ``"prod"``.
            timeout: HTTP timeout in seconds.
            min_interval: Minimum seconds between HTTP requests (10.5 default).
            max_attempts: Total attempts for retryable failures (default 3).
            clock: Injectable monotonic clock (tests).
            sleeper: Injectable sleep function (tests).
        """
        key, secret = get_mkk_credentials()
        if not key or not secret:
            raise APIError(
                "MKK VYK credentials are missing: set the MKK_API_KEY and "
                "MKK_API_SECRET environment variables (or call set_mkk_credentials)"
            )
        if gateway not in GATEWAYS:
            raise ValueError(f"gateway must be one of {sorted(GATEWAYS)}, got {gateway!r}")

        super().__init__(timeout=timeout, **kwargs)
        self.gateway = gateway
        self.base_url = GATEWAYS[gateway]
        self.min_interval = min_interval
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._last_request_at: float | None = None

    # ----- transport ---------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        key, secret = get_mkk_credentials()
        if not key or not secret:
            raise APIError(
                "MKK VYK credentials are missing: set the MKK_API_KEY and "
                "MKK_API_SECRET environment variables (or call set_mkk_credentials)"
            )
        token = base64.b64encode(f"{key}:{secret}".encode()).decode()
        return {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        }

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            self._last_request_at = self._clock()
            return
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def _backoff(self, attempt: int, response: httpx.Response | None) -> None:
        delay = min(2**attempt, 20)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after and str(retry_after).isdigit():
                delay = max(delay, min(int(retry_after), 60))
        self._sleep(delay)

    @staticmethod
    def _fault(response: httpx.Response) -> tuple[str | None, str | None]:
        try:
            body = response.json()
        except ValueError:
            return None, None
        fault = body.get("fault") if isinstance(body, dict) and isinstance(body.get("fault"), dict) else body
        if not isinstance(fault, dict):
            return None, None
        code = fault.get("faultCode") or fault.get("code")
        message = fault.get("faultString") or fault.get("message")
        return (str(code) if code is not None else None, str(message) if message is not None else None)

    def _error(self, path: str, response: httpx.Response, code: str | None, message: str | None) -> APIError:
        detail = message or response.text[:300]
        if code in FAULT_CODES:
            detail = f"{code}: {FAULT_CODES[code]}" + (f" [{message}]" if message else "")
        elif code:
            detail = f"{code}: {detail}"
        return APIError(
            f"MKK VYK request failed (GET {path}): {detail}",
            status_code=response.status_code,
        )

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_transport_error: Exception | None = None

        for attempt in range(self.max_attempts):
            self._throttle()
            headers = self._auth_headers()
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise APIError(
                        f"MKK VYK request failed after {attempt + 1} attempts "
                        f"(GET {path}): {type(exc).__name__}"
                    ) from exc
                self._backoff(attempt, None)
                continue

            if response.status_code < 400:
                if not response.content:
                    raise APIError(
                        f"MKK VYK empty response body (GET {path})",
                        status_code=response.status_code,
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    raise APIError(
                        f"MKK VYK invalid JSON response (GET {path})",
                        status_code=response.status_code,
                    ) from exc

            code, message = self._fault(response)
            if response.status_code == 400 and code in EMPTY_WINDOW_FAULT_CODES:
                return _EMPTY

            if response.status_code in RETRY_STATUSES and attempt + 1 < self.max_attempts:
                self._backoff(attempt, response)
                continue

            raise self._error(path, response, code, message)

        # Defensive: the loop always returns or raises.
        raise APIError(
            f"MKK VYK request failed (GET {path}): {type(last_transport_error).__name__}"
        ) from last_transport_error

    # ----- endpoints ---------------------------------------------------------

    def last_disclosure_index(self) -> int:
        """Return the gateway's last published disclosure index as an int."""
        data = self._request("/api/vyk/lastDisclosureIndex")
        if data is _EMPTY:
            raise APIError("MKK VYK lastDisclosureIndex returned an empty window")
        if isinstance(data, dict):
            value = data.get("lastDisclosureIndex")
        else:
            value = data
        if value is None:
            raise APIError("MKK VYK lastDisclosureIndex response is missing 'lastDisclosureIndex'")
        try:
            return int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise APIError(f"MKK VYK lastDisclosureIndex is not an integer: {value!r}") from exc

    def disclosures(
        self,
        cursor: int | str,
        disclosure_class: str = "FR",
        disclosure_type: str | None = None,
        company_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return one raw disclosure page starting at ``cursor`` (source order)."""
        params: dict[str, Any] = {"disclosureIndex": int(cursor)}
        if disclosure_class:
            params["disclosureClass"] = disclosure_class
        if disclosure_type:
            params["disclosureTypes"] = disclosure_type
        if company_id:
            params["companyId"] = company_id
        data = self._request("/api/vyk/disclosures", params)
        if data is _EMPTY:
            return []
        if isinstance(data, dict):
            data = data.get("disclosures") or data.get("data") or []
        return [row for row in _as_list(data) if isinstance(row, dict)]

    def disclosure_detail(self, index: int | str, file_type: str = "data") -> dict[str, Any]:
        """Return the raw disclosure detail dict (parsing is the caller's job)."""
        params = {"fileType": file_type}
        data = self._request(f"/api/vyk/disclosureDetail/{int(index)}", params)
        if data is _EMPTY:
            return {}
        if not isinstance(data, dict):
            raise APIError(f"MKK VYK disclosureDetail/{index} returned a non-object body")
        return data

    def members(self) -> list[dict[str, Any]]:
        """Return the raw member list (1037 records at capture time)."""
        data = self._request("/api/vyk/members")
        if data is _EMPTY:
            return []
        if isinstance(data, dict):
            data = data.get("members") or data.get("data") or []
        return [row for row in _as_list(data) if isinstance(row, dict)]

    # ----- symbol → statement ------------------------------------------------

    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str,
        *,
        max_pages: int = 10,
        max_candidates: int = 50,
    ) -> pd.DataFrame:
        """Return the requested statement for ``symbol``'s latest FR disclosure.

        The symbol is matched against ``senderExchCodes`` in the disclosure
        **detail** (the list endpoint does not carry exchange codes).  Pages are
        walked backwards from the last disclosure index with
        ``disclosureClass=FR``; the newest matching disclosure that actually
        contains the requested block is returned.

        This method deliberately stays inside the provider layer — it does not
        touch ``Ticker`` (orchestration is FA-02).
        """
        if statement_type not in STATEMENT_TYPES:
            raise ValueError(
                f"statement_type must be one of {list(STATEMENT_TYPES)}, got {statement_type!r}"
            )
        wanted = symbol.strip().upper()
        if not wanted:
            raise ValueError("symbol must be a non-empty string")

        last = self.last_disclosure_index()
        cursor = last
        seen: set[int] = set()
        candidates = 0

        for _ in range(max(1, int(max_pages))):
            if cursor < 1:
                break
            page = self.disclosures(cursor, disclosure_class="FR")
            records: list[tuple[int, dict[str, Any]]] = []
            for record in page:
                try:
                    index = int(str(record.get("disclosureIndex")).strip())
                except (TypeError, ValueError):
                    continue
                if index > last or index in seen:
                    continue
                records.append((index, record))

            for index, _record in sorted(records, key=lambda pair: pair[0], reverse=True):
                if candidates >= max_candidates:
                    break
                seen.add(index)
                candidates += 1
                detail = self.disclosure_detail(index)
                codes = {str(code).strip().upper() for code in _as_list(detail.get("senderExchCodes"))}
                if wanted not in codes:
                    continue
                parsed = parse_disclosure_detail(detail)
                if statement_type in parsed:
                    return parsed[statement_type]
            if candidates >= max_candidates or not records:
                break

            next_cursor = min(index for index, _ in records) - self.DISCLOSURE_PAGE_SIZE
            if next_cursor >= cursor:
                break
            cursor = next_cursor

        raise DataNotAvailableError(
            f"No FR disclosure with a {statement_type!r} block found for {wanted}"
        )
