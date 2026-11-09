"""KAP (Kamuyu Aydınlatma Platformu) provider for disclosures and calendar."""

import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.exceptions import APIError, KAPStructureError

# Literal marker used for source-side missing values ("Bilgi Mevcut Değil").
DATA_UNAVAILABLE = "VERİ YOK"

# KAP "Genel Bilgiler" (company general info) grid/item keys parsed by
# ``get_company_structure``. These are server-rendered item keys of the
# capital-structure accordion; see ``KAP_genel_sayfa_FD_dogrulama.md``.
STRUCTURE_ITEM_PAID_IN_CAPITAL = "kpy41_acc5_odenmis_sermaye"
STRUCTURE_ITEM_CAPITAL_CEILING = "kpy41_acc5_kayitli_sermaye_tavani"
STRUCTURE_ITEM_DIRECT_SHAREHOLDERS = "kpy41_acc5_sermayede_dogrudan"
STRUCTURE_ITEM_INDIRECT_SHAREHOLDERS = "kpy41_acc5_son_durum_sermayeye"
STRUCTURE_ITEM_SHARE_GROUPS = "kpy41_acc5_sermayeyi_temsil_eden"
STRUCTURE_ITEM_ACTUAL_FLOAT = "kpy41_acc5_fiili_dolasimdaki_pay"

# Grid carrying the free-float methodology note (SPK 54/1626).
STRUCTURE_GRID_ACTUAL_FLOAT = "actualSharesOutstandingGrid"

# --- VBTS (Volatilite Bazlı Tedbir Sistemi) ---------------------------------
# Borsa İstanbul A.Ş.'s KAP member OID. K0-a provenance: **data derivation**
# (KAP smart search, measured 2026-11-09 19:52; evidence:
# ``vbts_dkb_arastirma_2026_11_09/kanit/kap_bist_member_disclosures_raw.html``).
BIST_MEMBER_OID = "4028e4a14bcf2a06014be4d7e6e256b6"

# VBTS records are matched on THREE fields at once. A title-only match is
# wrong: the same title also carries 448 non-VBTS notices, and the SPK measure
# family ("Sermaye Piyasası Kurulu Tedbir Kararı", e.g. IEYHO 17.10.2026) is not
# VBTS. See research RAPOR.md §2.1/§2.5.
VBTS_DISCLOSURE_CLASS = "DKB"
VBTS_TITLE = "BISTECH Pay Piyasası Alım Satım Sistemi Duyurusu"
VBTS_SUMMARY = "Pay Piyasasında Volatilite Bazlı Tedbir Sistemi"

#: Default lookback window. PARAMETER + DEFAULT stamp (K0-a (c)): the 90-day
#: window is a chosen parameter, not a source fact.
DEFAULT_VBTS_WINDOW_DAYS = 90

#: Escaped-JSON record marker used by both the company and the member pages.
_RECORD_MARKER = 'publishDate\\":\\"'

# Turkish formatted numbers: thousands ".", decimals "," (e.g. 154.873.002,50).
# KAP mixes formats (some rows carry "1379999999.99"), see _parse_tr_decimal.
_TR_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")
_TR_PLAIN_DECIMAL_RE = re.compile(r"\d+\.\d+")
_UNAVAILABLE_TEXTS = {"", "-", "Bilgi Mevcut Değil", "Bilgi mevcut değil", "BİLGİ MEVCUT DEĞİL"}

# Next.js SSR flight data embeds the BIST company list as pushed chunks:
#   self.__next_f.push([1,"..."])
# Each chunk is a JSON string literal that must be unescaped before parsing.
_NEXT_F_PUSH_RE = re.compile(
    r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)',
    re.DOTALL,
)

# Member blocks live inside the decoded payload with normal quotes.
# Only blocks that carry a ``stockCode`` field are company records; raw
# ``mkkMemberOid`` blocks without a stock code are other MKK members.
_MEMBER_BLOCK_RE = re.compile(
    r'\{"mkkMemberOid":"([^"]+)","kapMemberTitle":"([^"]*)",'
    r'"relatedMemberTitle":(?:"[^"]*"|null),"stockCode":"([^"]+)"'
    r'(?:,"cityName":(?:"([^"]*)"|null))?',
)


def _extract_next_f_payload(html: str) -> str:
    """Collect and unescape all ``__next_f.push`` chunks in document order."""
    chunks = _NEXT_F_PUSH_RE.findall(html)
    if not chunks:
        raise RuntimeError(
            "KAP SSR schema changed: no __next_f.push chunks found in page"
        )
    return "".join(json.loads(chunk) for chunk in chunks)


def _decode_member_blocks(html: str) -> list[dict]:
    """Decode BIST company member blocks from the SSR page HTML.

    Returns records shaped as ``{"oid", "name", "tickers", "city"}``. Only
    blocks that contain a ``stockCode`` field count as company records.
    """
    payload = _extract_next_f_payload(html)
    matches = _MEMBER_BLOCK_RE.findall(payload)
    if not matches:
        raise RuntimeError(
            "KAP SSR schema changed: no company member blocks found in payload"
        )

    records: list[dict] = []
    seen_oids: set[str] = set()
    for oid, name, codes, city in matches:
        if oid in seen_oids:
            # Deterministic: first record wins (duplicates absent in fixture).
            continue
        seen_oids.add(oid)
        tickers = [code.strip() for code in codes.split(",") if code.strip()]
        records.append(
            {
                "oid": oid,
                "name": name,
                "tickers": tickers,
                "city": city or "",
            }
        )
    return records


def _match_json_object(text: str, start: int) -> str | None:
    """Return the balanced ``{...}`` slice starting at ``start``.

    Brace counting is string-aware so that braces inside JSON string values
    (HTML snippets, notes) do not terminate the object early.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_named_object(payload: str, marker: str) -> dict | None:
    """Decode the JSON object that follows ``"<marker>":`` in the payload."""
    position = payload.find(f'"{marker}":')
    if position < 0:
        return None
    brace = payload.find("{", position)
    if brace < 0:
        return None
    chunk = _match_json_object(payload, brace)
    if chunk is None:
        return None
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_item_object(payload: str, item_key: str) -> dict | None:
    """Decode the ``itemObject`` block carrying ``item_key``."""
    key_position = payload.find(f'"itemKey":"{item_key}"')
    if key_position < 0:
        return None
    object_marker = payload.rfind('"itemObject":', 0, key_position)
    start = payload.find("{", object_marker) if object_marker >= 0 else -1
    if start < 0 or start > key_position:
        start = payload.rfind('{"no":', 0, key_position)
    if start < 0:
        return None
    chunk = _match_json_object(payload, start)
    if chunk is None:
        return None
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _item_display_text(payload: str, item_key: str) -> str | None:
    """Return the text KAP renders next to an item block (e.g. "Bilgi Mevcut Değil")."""
    key_position = payload.find(f'"itemKey":"{item_key}"')
    if key_position < 0:
        return None
    start = payload.rfind('{"no":', 0, key_position)
    if start < 0:
        return None
    chunk = _match_json_object(payload, start)
    if chunk is None:
        return None
    tail = payload[start + len(chunk) : start + len(chunk) + 1500]
    match = re.search(r'"children":"((?:[^"\\]|\\.)*)"', tail)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"').strip()
    except json.JSONDecodeError:
        return None


def _option_text(raw: object) -> str | None:
    """Flatten KAP's ``{"key": ..., "text": ...}`` option values to their text."""
    if isinstance(raw, dict):
        text = raw.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None
    if isinstance(raw, str):
        return raw.strip() or None
    return None


def _is_unavailable(raw: object) -> bool:
    """True when the source expresses absence instead of a value."""
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip() in _UNAVAILABLE_TEXTS
    if isinstance(raw, (list, tuple, dict)):
        return len(raw) == 0
    return False


def _parse_tr_decimal(raw: object) -> Decimal | None:
    """Parse a KAP number into ``Decimal`` (no guessing).

    Accepted layouts:

    * ``154.873.002,50`` — Turkish (dot = thousands, comma = decimals)
    * ``393516000``      — plain integer
    * ``1379999999.99``  — plain decimal, present in some KAP rows

    A lone dot followed by exactly three digits (``1.234``) is read as a
    Turkish thousands separator; part-whole reconciliations guard the result.
    """
    if _is_unavailable(raw):
        return None
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))

    text = str(raw).strip().replace("\u00a0", "").replace(" ", "")
    if not re.fullmatch(r"-?\d[\d.,]*", text):
        return None

    negative = text.startswith("-")
    body = text[1:] if negative else text

    if "," in body:
        if body.count(",") > 1:
            return None
        integer_part, decimal_part = body.split(",")
        if not re.fullmatch(r"\d+|" + _TR_THOUSANDS_RE.pattern, integer_part):
            return None
        if not re.fullmatch(r"\d+", decimal_part):
            return None
        normalized = integer_part.replace(".", "") + "." + decimal_part
    elif "." in body:
        if _TR_THOUSANDS_RE.fullmatch(body):
            normalized = body.replace(".", "")
        elif _TR_PLAIN_DECIMAL_RE.fullmatch(body):
            normalized = body
        else:
            return None
    else:
        normalized = body

    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    return -value if negative else value


def _parse_iso_date_arg(raw: object) -> date | None:
    """Parse an ``YYYY-MM-DD`` argument into a ``date`` (``None`` if invalid)."""
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


def _iso_to_date(raw: str) -> date:
    """Convert an already-normalized ``YYYY-MM-DD`` string to ``date``."""
    return date.fromisoformat(raw)


def _parse_kap_publish_datetime(raw: object) -> tuple[str | None, str | None]:
    """Parse KAP ``publishDate`` (``dd.mm.yyyy HH:MM:SS``).

    Returns ``(iso_date, raw_text)``; ``(None, raw)`` when unparseable so the
    caller can report the source value instead of guessing.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, None
    text = raw.strip()
    match = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})(?: \d{2}:\d{2}:\d{2})?", text)
    if not match:
        return None, text
    return f"{match.group(3)}-{match.group(2)}-{match.group(1)}", text


def _split_escaped_records(text: str) -> list[str]:
    """Slice KAP's escaped-JSON payload into one chunk per disclosure record.

    Records are keyed by ``publishDate\":\"``. Slicing (instead of one global
    regex) keeps a record missing a field from borrowing the next record's
    value - the failure mode that would silently mislabel disclosure classes.
    """
    starts = [match.start() for match in re.finditer(re.escape(_RECORD_MARKER), text)]
    if not starts:
        return []
    bounds = starts[1:] + [min(starts[-1] + 8000, len(text))]
    return [text[start:end] for start, end in zip(starts, bounds, strict=True)]


def _escaped_string_field(chunk: str, key: str) -> str | None:
    """Read ``\"key\":\"value\"`` from one escaped-JSON record chunk."""
    match = re.search(
        re.escape(key) + r'\\":\\"((?:[^\\]|\\.)*?)\\"', chunk
    )
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _escaped_int_field(chunk: str, key: str) -> int | None:
    """Read an integer ``\"key\":123`` field from an escaped-JSON chunk."""
    match = re.search(re.escape(key) + r'\\":(\d+)', chunk)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_disclosure_records(text: str) -> list[dict]:
    """Parse every disclosure record embedded in a KAP listing page.

    Works for both the company page (``?member=<oid>``) and the BIST member
    page. Fields that are absent stay ``None`` - nothing is inferred. Records
    that carry a ``disclosureIndex`` but no ``publishDate`` cannot be sliced, so
    they are reported as ``orphan`` entries rather than silently dropped.
    """
    records: list[dict] = []
    for chunk in _split_escaped_records(text):
        iso_date, raw_date = _parse_kap_publish_datetime(
            _escaped_string_field(chunk, "publishDate")
        )
        records.append(
            {
                "publish_date": iso_date,
                "publish_date_raw": raw_date,
                "disclosure_index": _escaped_int_field(chunk, "disclosureIndex"),
                "title": _escaped_string_field(chunk, "title"),
                "related_stocks": _escaped_string_field(
                    chunk, "relatedStocks"
                ),
                "disclosure_class": _escaped_string_field(chunk, "disclosureClass"),
                "summary": _escaped_string_field(chunk, "summary"),
            }
        )

    index_markers = len(re.findall(r'disclosureIndex\\":\d+', text))
    for _ in range(max(0, index_markers - len(records))):
        records.append(
            {
                "publish_date": None,
                "publish_date_raw": None,
                "disclosure_index": None,
                "title": None,
                "related_stocks": None,
                "disclosure_class": None,
                "summary": None,
                "orphan": True,
            }
        )
    return records


def _parse_dmy_date(raw: object) -> str | None:
    """Normalize ``dd/mm/yyyy`` or ``yyyymmdd`` to ISO ``yyyy-mm-dd``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})(?: \d{2}:\d{2}:\d{2})?", text)
    if match:
        return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return None


def _require_item_object(
    payload: str, item_key: str, field_name: str, symbol: str
) -> dict:
    """Return a mandatory item block or fail closed."""
    item = _extract_item_object(payload, item_key)
    if item is None:
        raise KAPStructureError(
            f"KAP company page is missing the '{field_name}' block "
            f"({item_key}) for {symbol}; refusing to return a partial structure"
        )
    return item


def _parse_company_structure(
    payload: str,
    *,
    symbol: str,
    member_oid: str,
    source_url: str,
    retrieved_at: str | None = None,
) -> dict:
    """Parse the capital-structure blocks of a KAP "Genel Bilgiler" page.

    Only values literally present in the page are returned. Missing source
    values are reported as :data:`DATA_UNAVAILABLE` with a reason; absent
    mandatory blocks raise :class:`KAPStructureError` (fail-closed).
    """
    missing: list[dict] = []

    # --- Ödenmiş / çıkarılmış sermaye -------------------------------------
    paid_item = _require_item_object(
        payload, STRUCTURE_ITEM_PAID_IN_CAPITAL, "ödenmiş sermaye", symbol
    )
    paid_raw = paid_item.get("value")
    paid_in_capital = _parse_tr_decimal(paid_raw)
    if paid_in_capital is None:
        raise KAPStructureError(
            f"KAP company page has an unparseable paid-in capital value for "
            f"{symbol}: {paid_raw!r}"
        )

    # --- Kayıtlı sermaye tavanı (section must exist; value may be absent) --
    ceiling_item = _require_item_object(
        payload, STRUCTURE_ITEM_CAPITAL_CEILING, "kayıtlı sermaye tavanı", symbol
    )
    ceiling_raw = ceiling_item.get("value")
    capital_ceiling = _parse_tr_decimal(ceiling_raw)
    if capital_ceiling is None:
        missing.append(
            {
                "field": "registered_capital_ceiling",
                "status": DATA_UNAVAILABLE,
                "reason": f"kaynakta değer yok ({ceiling_raw!r})",
            }
        )

    # --- Doğrudan ortaklar (≥ %3) -----------------------------------------
    direct_item = _require_item_object(
        payload, STRUCTURE_ITEM_DIRECT_SHAREHOLDERS, "doğrudan ortaklar", symbol
    )
    direct_raw_rows = direct_item.get("value")
    direct_rows: list[dict] = []
    reported_total: dict | None = None
    if not _is_unavailable(direct_raw_rows) and not isinstance(direct_raw_rows, list):
        raise KAPStructureError(
            f"KAP company page has an unexpected direct-shareholder shape for "
            f"{symbol}: {type(direct_raw_rows).__name__}"
        )
    if _is_unavailable(direct_raw_rows):
        direct_status = DATA_UNAVAILABLE
        missing.append(
            {
                "field": "direct_shareholders",
                "status": DATA_UNAVAILABLE,
                "reason": "kaynakta ortak satırı yok",
            }
        )
    else:
        direct_status = "OK"
        for row in direct_raw_rows:
            name = (row.get("shareholder") or "").strip()
            parsed_row = {
                "name": name,
                "share_tl": _parse_tr_decimal(row.get("shareInCapital")),
                "ratio_percent": _parse_tr_decimal(row.get("ratioInCapital")),
                "voting_ratio_percent": _parse_tr_decimal(
                    row.get("votingRightRatio")
                ),
                "raw": dict(row),
            }
            if name.upper() == "TOPLAM":
                reported_total = parsed_row
            else:
                direct_rows.append(parsed_row)

    direct_block = {
        "status": direct_status,
        "as_of": _parse_dmy_date(direct_item.get("creationDate")),
        "as_of_raw": direct_item.get("creationDate"),
        "section_title": (direct_item.get("itemName") or "").strip() or None,
        "rows": direct_rows,
        "reported_total": reported_total,
        "raw_rows": direct_raw_rows if isinstance(direct_raw_rows, list) else [],
    }

    # --- Dolaylı ortaklar (kaynak "Bilgi Mevcut Değil" diyebilir) ---------
    indirect_item = _extract_item_object(payload, STRUCTURE_ITEM_INDIRECT_SHAREHOLDERS)
    indirect_raw_value = indirect_item.get("value") if indirect_item else None
    indirect_rows = (
        indirect_raw_value
        if isinstance(indirect_raw_value, list)
        and indirect_raw_value
        and all(isinstance(row, dict) for row in indirect_raw_value)
        else None
    )
    if indirect_rows is None:
        # KAP expresses absence either literally ("Bilgi Mevcut Değil") or with
        # a short code such as "DOBY" whose rendered text is shown next to it.
        display_text = _item_display_text(
            payload, STRUCTURE_ITEM_INDIRECT_SHAREHOLDERS
        )
        reason = (
            f"kaynak ifadesi: {display_text}"
            if display_text
            else f"kaynakta satır yok (ham değer: {indirect_raw_value!r})"
        )
        indirect_block = {
            "status": DATA_UNAVAILABLE,
            "reason": reason,
            "as_of": _parse_dmy_date(
                indirect_item.get("creationDate") if indirect_item else None
            ),
            "rows": [],
            "raw_value": indirect_raw_value,
            "raw_rows": [],
        }
        missing.append(
            {
                "field": "indirect_shareholders",
                "status": DATA_UNAVAILABLE,
                "reason": reason,
            }
        )
    else:
        indirect_block = {
            "status": "OK",
            "reason": None,
            "as_of": _parse_dmy_date(indirect_item.get("creationDate")),
            "rows": [dict(row) for row in indirect_rows],
            "raw_value": indirect_raw_value,
            "raw_rows": indirect_rows,
        }

    # --- Sermayeyi temsil eden paylar (A/B grupları) ----------------------
    groups_item = _require_item_object(
        payload, STRUCTURE_ITEM_SHARE_GROUPS, "pay grupları", symbol
    )
    groups_raw_rows = groups_item.get("value")
    if _is_unavailable(groups_raw_rows) or not isinstance(groups_raw_rows, list):
        raise KAPStructureError(
            f"KAP company page has no share-group rows for {symbol}"
        )
    group_rows = [
        {
            "group": (row.get("shareGroup") or "").strip(),
            "nominal_value": _parse_tr_decimal(row.get("nominalValueOfShares")),
            "ratio_to_total_capital": _parse_tr_decimal(
                row.get("ratioToTotalCapital")
            ),
            "exchange_traded": _option_text(row.get("exchangeTradedOrNot")),
            "raw": dict(row),
        }
        for row in groups_raw_rows
    ]
    groups_total = sum(
        (row["nominal_value"] for row in group_rows if row["nominal_value"]), Decimal(0)
    )
    groups_block = {
        "status": "OK",
        "as_of": _parse_dmy_date(groups_item.get("creationDate")),
        "as_of_raw": groups_item.get("creationDate"),
        "rows": group_rows,
        "total_nominal": groups_total,
        "matches_paid_in_capital": groups_total == paid_in_capital,
        "raw_rows": groups_raw_rows,
    }

    # --- Fiili dolaşım ----------------------------------------------------
    float_item = _require_item_object(
        payload, STRUCTURE_ITEM_ACTUAL_FLOAT, "fiili dolaşım", symbol
    )
    float_raw_rows = float_item.get("value")
    if _is_unavailable(float_raw_rows) or not isinstance(float_raw_rows, list):
        raise KAPStructureError(
            f"KAP company page has no free-float row for {symbol}"
        )
    float_row = float_raw_rows[0]
    total_shares = _parse_tr_decimal(float_row.get("totalShares"))
    non_traded_shares = _parse_tr_decimal(float_row.get("nonTradedShares"))
    traded_shares = _parse_tr_decimal(float_row.get("tradedShares"))
    excluded_c = _parse_tr_decimal(float_row.get("exclSharesC"))
    excluded_de = _parse_tr_decimal(float_row.get("exclSharesDE"))
    actual_float = _parse_tr_decimal(float_row.get("actualSharesOutstanding"))
    float_ratio = _parse_tr_decimal(float_row.get("actualOutstandingSharesRatio"))

    required_numbers = {
        "total_shares": total_shares,
        "non_traded_shares": non_traded_shares,
        "traded_shares": traded_shares,
        "actual_float_shares": actual_float,
        "actual_float_ratio_percent": float_ratio,
    }
    for field_name, value in required_numbers.items():
        if value is None:
            raise KAPStructureError(
                f"KAP company page free-float row is missing '{field_name}' for "
                f"{symbol}"
            )

    # --- Metodoloji künyesi (SPK 54/1626) ---------------------------------
    float_grid = _extract_named_object(payload, STRUCTURE_GRID_ACTUAL_FLOAT)
    note = float_grid.get("note1") if float_grid else None
    methodology_note = note.strip() if isinstance(note, str) and note.strip() else None
    if methodology_note is None:
        methodology_note_display = DATA_UNAVAILABLE
        effective_from = None
        rule = None
        missing.append(
            {
                "field": "float_methodology",
                "status": DATA_UNAVAILABLE,
                "reason": "fiili dolaşım tablosunun dipnotu bulunamadı",
            }
        )
    else:
        methodology_note_display = methodology_note
        effective_match = re.search(
            r"(\d{2})\.(\d{2})\.(\d{4}) tarihinden itibaren", methodology_note
        )
        effective_from = (
            f"{effective_match.group(3)}-{effective_match.group(2)}-"
            f"{effective_match.group(1)}"
            if effective_match
            else None
        )
        rule_match = re.search(
            r"(\d{2}\.\d{2}\.\d{4}) tarihli ve (\d+/\d+) sayılı İlke Kararı",
            methodology_note,
        )
        rule = (
            f"SPK {rule_match.group(1)} tarihli ve {rule_match.group(2)} sayılı "
            "İlke Kararı"
            if rule_match
            else None
        )

    # --- Parça-bütün mutabakatları (üç kontrol) ---------------------------
    share_sum_left = traded_shares + non_traded_shares
    share_sum_ok = share_sum_left == total_shares
    float_arith_left = (
        traded_shares - (excluded_c or Decimal(0)) - (excluded_de or Decimal(0))
    )
    float_arith_ok = float_arith_left == actual_float
    computed_ratio = (actual_float / total_shares) * Decimal(100)
    computed_ratio_quantized = computed_ratio.quantize(Decimal("0.0001"))
    ratio_ok = abs(computed_ratio - float_ratio) <= Decimal("0.01")

    reconciliations = [
        {
            "name": "traded_plus_non_traded_equals_total",
            "ok": share_sum_ok,
            "left": share_sum_left,
            "right": total_shares,
            "difference": share_sum_left - total_shares,
        },
        {
            "name": "traded_minus_exclusions_equals_float",
            "ok": float_arith_ok,
            "left": float_arith_left,
            "right": actual_float,
            "difference": float_arith_left - actual_float,
        },
        {
            "name": "float_over_total_equals_reported_ratio",
            "ok": ratio_ok,
            "left": computed_ratio_quantized,
            "right": float_ratio,
            "difference": computed_ratio_quantized - float_ratio,
            "tolerance_pp": Decimal("0.01"),
        },
    ]

    as_of = _parse_dmy_date(float_row.get("creationDate")) or _parse_dmy_date(
        float_item.get("creationDate")
    )
    if as_of is None:
        missing.append(
            {
                "field": "as_of",
                "status": DATA_UNAVAILABLE,
                "reason": "fiili dolaşım satırında tarih yok",
            }
        )

    float_block = {
        "status": "OK",
        "as_of": as_of,
        "as_of_raw": float_row.get("creationDate"),
        "isin": float_row.get("isin"),
        "total_shares": total_shares,
        "non_traded_shares": non_traded_shares,
        "traded_shares": traded_shares,
        "excluded_shares_c": excluded_c,
        "excluded_shares_de": excluded_de,
        "actual_float_shares": actual_float,
        "actual_float_ratio_percent": float_ratio,
        "computed_float_ratio_percent": computed_ratio.quantize(Decimal("0.0001")),
        "raw_row": dict(float_row),
    }

    return {
        "symbol": symbol,
        "member_oid": member_oid,
        "source": {
            "url": source_url,
            "member_oid": member_oid,
            "retrieved_at": retrieved_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "as_of": as_of,
        "methodology": {
            "note": methodology_note_display,
            "effective_from": effective_from,
            "rule": rule,
            "applies_to": STRUCTURE_GRID_ACTUAL_FLOAT,
        },
        "paid_in_capital": paid_in_capital,
        "registered_capital_ceiling": capital_ceiling,
        "direct_shareholders": direct_block,
        "indirect_shareholders": indirect_block,
        "share_groups": groups_block,
        "float": float_block,
        "reconciliations": reconciliations,
        "reconciliations_all_ok": all(item["ok"] for item in reconciliations),
        "missing": missing,
        "raw": {
            "paid_in_capital": paid_raw,
            "registered_capital_ceiling": ceiling_raw,
        },
    }


class KAPProvider(BaseProvider):
    """
    Provider for KAP (Kamuyu Aydınlatma Platformu) data.

    KAP is the official disclosure platform for publicly traded
    companies in Turkey, similar to SEC EDGAR in the US.

    Provides:
    - List of all BIST companies with ticker codes
    - Company search functionality
    - Company disclosures (bildirimler)
    - Expected disclosure calendar (beklenen bildirimler)
    """

    BIST_COMPANIES_URL = "https://www.kap.org.tr/tr/bist-sirketler"
    DISCLOSURE_URL = "https://www.kap.org.tr/tr/bildirim-sorgu-sonuc"
    CALENDAR_API_URL = "https://kap.org.tr/tr/api/expected-disclosure-inquiry/company"
    COMPANY_INFO_URL = "https://kap.org.tr/tr/sirket-bilgileri/ozet"
    COMPANY_GENERAL_URL = "https://kap.org.tr/tr/sirket-bilgileri/genel"
    CACHE_DURATION = 86400  # 24 hours

    def __init__(self):
        super().__init__()
        self._company_cache: pd.DataFrame | None = None
        self._cache_time: float = 0
        self._oid_map: dict[str, str] | None = None
        self._oid_cache_time: float = 0
        self._company_details_cache: dict[str, dict] = {}
        self._company_details_cache_time: dict[str, float] = {}
        self._general_page_cache: dict[str, tuple[float, str]] = {}
        self._company_structure_cache: dict[str, dict] = {}
        self._company_structure_cache_time: dict[str, float] = {}
        self._vbts_cache: tuple[list[dict], dict] | None = None
        self._vbts_cache_time: float = 0

    def get_companies(self) -> pd.DataFrame:
        """
        Get list of all BIST companies.

        Returns:
            DataFrame with columns: ticker, name, city
        """
        current_time = time.time()

        # Check cache
        if (
            self._company_cache is not None
            and (current_time - self._cache_time) < self.CACHE_DURATION
        ):
            return self._company_cache

        try:
            response = self._client.get(self.BIST_COMPANIES_URL, timeout=20)
            response.raise_for_status()
        except Exception as e:
            raise APIError(f"Failed to fetch company list: {e}") from e

        records = _decode_member_blocks(response.text)

        rows = []
        for record in records:
            for ticker in record["tickers"]:
                rows.append(
                    {
                        "ticker": ticker,
                        "name": record["name"],
                        "city": record["city"],
                    }
                )

        if not rows:
            raise RuntimeError(
                "KAP company decode produced no rows; refusing empty company list"
            )

        result = pd.DataFrame(rows, columns=["ticker", "name", "city"])
        self._company_cache = result
        self._cache_time = current_time
        return result

    def search(self, query: str) -> pd.DataFrame:
        """
        Search companies by name or ticker.

        Args:
            query: Search query (ticker code or company name)

        Returns:
            DataFrame with matching companies
        """
        if not query:
            return pd.DataFrame(columns=["ticker", "name", "city"])

        companies = self.get_companies()
        if companies.empty:
            return companies

        query_normalized = self._normalize_text(query)
        query_upper = query.upper()

        # Score and filter results
        results = []
        for _, row in companies.iterrows():
            score = 0
            ticker = row["ticker"]
            name = row["name"]

            # Exact ticker match
            if ticker.upper() == query_upper:
                score = 1000
            # Ticker starts with query
            elif ticker.upper().startswith(query_upper):
                score = 500
            # Name contains query
            elif query_normalized in self._normalize_text(name):
                score = 100

            if score > 0:
                results.append((score, row))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        if not results:
            return pd.DataFrame(columns=["ticker", "name", "city"])

        return pd.DataFrame([r[1] for r in results])

    def _normalize_text(self, text: str) -> str:
        """Normalize Turkish text for comparison."""
        tr_map = str.maketrans("İıÖöÜüŞşÇçĞğ", "iioouussccgg")
        normalized = text.translate(tr_map).lower()
        # Remove common suffixes
        normalized = re.sub(r"[\.,']|\s+a\.s\.?|\s+anonim sirketi", "", normalized)
        return normalized.strip()

    def get_member_oid(self, symbol: str) -> str | None:
        """
        Get KAP member OID (mkkMemberOid) for a stock symbol.

        The member OID is required to query disclosures from KAP.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Member OID string or None if not found.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")
        current_time = time.time()

        # Check cache
        if (
            self._oid_map is not None
            and (current_time - self._oid_cache_time) < self.CACHE_DURATION
        ):
            return self._oid_map.get(symbol)

        # Fetch BIST companies list from KAP. Only network errors are swallowed
        # (frozen behaviour); parse errors must surface loudly.
        try:
            response = self._client.get(self.BIST_COMPANIES_URL, timeout=20)
            response.raise_for_status()
        except Exception:
            return None

        oid_map: dict[str, str] = {}
        for record in _decode_member_blocks(response.text):
            for ticker in record["tickers"]:
                oid_map[ticker] = record["oid"]

        self._oid_map = oid_map
        self._oid_cache_time = current_time
        return oid_map.get(symbol)

    def get_disclosures(
        self, symbol: str, limit: int = 20, disclosure_class: str | None = None
    ) -> pd.DataFrame:
        """
        Get KAP disclosures (bildirimler) for a stock.

        Args:
            symbol: Stock symbol (e.g., "THYAO").
            limit: Maximum number of disclosures to return (default: 20).
            disclosure_class: Optional KAP disclosure class filter (e.g.
                ``"FR"``, ``"DKB"``, ``"ODA"``). Filtering happens client side
                on the already-fetched page - it issues NO extra request.
                ``None`` (default) means no filtering, so the previous
                behaviour is preserved.

        Returns:
            DataFrame with columns: Date, Title, URL, DisclosureClass.

            ``DisclosureClass`` is additive: records whose class could not be
            read carry ``None`` (never a guessed value). Those records are
            also listed in ``df.attrs["unparsed_class_records"]``.
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return pd.DataFrame(columns=["Date", "Title", "URL", "DisclosureClass"])

        wanted_class = disclosure_class.strip().upper() if disclosure_class else None

        # Fetch disclosures from KAP
        disc_url = f"{self.DISCLOSURE_URL}?member={member_oid}"

        try:
            response = self._client.get(disc_url, timeout=15)
            response.raise_for_status()

            parsed = _parse_disclosure_records(response.text)

            records = []
            unparsed: list[dict] = []
            for item in parsed:
                date = item["publish_date_raw"]
                index = item["disclosure_index"]
                title = item["title"]
                if date is None or index is None or title is None:
                    unparsed.append(item)
                    continue
                if wanted_class is not None and item["disclosure_class"] != wanted_class:
                    continue
                records.append(
                    {
                        "Date": date,
                        "Title": title,
                        "URL": f"https://www.kap.org.tr/tr/Bildirim/{index}",
                        "DisclosureClass": item["disclosure_class"],
                    }
                )
                if len(records) >= limit:
                    break

            frame = pd.DataFrame(
                records, columns=["Date", "Title", "URL"]
            )
            # Object dtype keeps a genuinely missing class as ``None`` instead of
            # letting pandas turn it into NaN (no guessing, explicit absence).
            frame["DisclosureClass"] = pd.Series(
                [row["DisclosureClass"] for row in records], dtype=object
            )
            frame = frame[["Date", "Title", "URL", "DisclosureClass"]]
            frame.attrs["unparsed_class_records"] = unparsed
            frame.attrs["disclosure_class_filter"] = wanted_class
            return frame

        except Exception as e:
            raise APIError(f"Failed to fetch disclosures for {symbol}: {e}") from e

    def get_vbts_status(
        self, days: int = DEFAULT_VBTS_WINDOW_DAYS, *, as_of: str | None = None
    ) -> dict:
        """
        Get the VBTS (volatility-based measure) status of the whole BIST market.

        Reads the KAP member page of Borsa İstanbul A.Ş. - ONE request returns
        every exchange notice - and keeps only genuine VBTS records. A record
        counts as VBTS only when ALL THREE fields match::

            disclosureClass == "DKB"
            title            == "BISTECH Pay Piyasası Alım Satım Sistemi Duyurusu"
            summary          == "Pay Piyasasında Volatilite Bazlı Tedbir Sistemi"

        Matching on the title alone is wrong: the same title is also used for
        non-VBTS notices, and the SPK measure family ("Sermaye Piyasası Kurulu
        Tedbir Kararı", e.g. IEYHO 17.10.2026) is NOT VBTS.

        Args:
            days: Lookback window in days. PARAMETER + DEFAULT stamp (K0-a c);
                default :data:`DEFAULT_VBTS_WINDOW_DAYS`. Applied client side.
            as_of: Optional reference date (``YYYY-MM-DD``) for the window.
                Defaults to today; exists so audits can reproduce a window.

        Returns:
            Dict with ``window``, ``source`` (URL, sha256, fetch time), ``counts``
            and ``symbols`` (per symbol: ``vbts_var``, ``kayit_sayisi``,
            ``son_tarih``, ``disclosure_index``, ``within_window``,
            ``ham_related``).

        Measure TYPE is deliberately NOT derived: the research proved keyword
        extraction is unusable (§2.5), so ``tedbir_turu`` is ``VERİ YOK`` and
        ``disclosure_index`` is left for operator on-demand review. Never
        invent a type.

        Raises:
            KAPStructureError: page unreachable, non-200, or unparseable. An
                empty body means VERİ YOK - it is NEVER read as "no measures".

        Known limit: the member page returns ~11.5 months / 2,000 records;
        older history is unverified and would need pagination (not implemented).
        """
        if days is None or int(days) <= 0:
            raise ValueError(f"days must be a positive integer, got {days!r}")
        days = int(days)

        reference = _parse_iso_date_arg(as_of) if as_of else datetime.now().date()
        if reference is None:
            raise ValueError(f"as_of must be YYYY-MM-DD, got {as_of!r}")
        window_from = reference - timedelta(days=days)

        cached = self._vbts_cache
        if (
            cached is not None
            and (time.time() - self._vbts_cache_time) < self.CACHE_DURATION
        ):
            parsed, meta = cached
        else:
            url = f"{self.DISCLOSURE_URL}?member={BIST_MEMBER_OID}"
            try:
                response = self._client.get(url, timeout=60)
            except Exception as exc:
                raise KAPStructureError(
                    f"KAP BIST member page could not be fetched: {exc}"
                ) from exc

            if response.status_code != 200:
                raise KAPStructureError(
                    "KAP BIST member page returned HTTP "
                    f"{response.status_code}; refusing to report VBTS status"
                )

            text = response.text or ""
            parsed = _parse_disclosure_records(text)
            if not parsed:
                # Throttle lesson: KAP can answer 200 with an EMPTY body.
                raise KAPStructureError(
                    "KAP BIST member page returned no disclosure records; an "
                    "empty body means VERİ YOK and must not be read as 'no VBTS "
                    "measures'"
                )

            meta = {
                "url": url,
                "member_oid": BIST_MEMBER_OID,
                "member_oid_provenance": (
                    "veri-türetimi (KAP smart search, ölçüm 2026-11-09 19:52)"
                ),
                "http_status": response.status_code,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bytes": len(text.encode("utf-8")),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._vbts_cache = (parsed, meta)
            self._vbts_cache_time = time.time()

        unparseable = [
            item
            for item in parsed
            if item["publish_date"] is None or item["disclosure_index"] is None
        ]

        vbts_records = [
            item
            for item in parsed
            if item["disclosure_class"] == VBTS_DISCLOSURE_CLASS
            and item["title"] == VBTS_TITLE
            and item["summary"] == VBTS_SUMMARY
        ]

        symbols: dict[str, dict] = {}
        in_window_symbols = 0
        in_window_records = 0

        for item in vbts_records:
            date = item["publish_date"]
            within = date is not None and window_from <= _iso_to_date(date) <= reference
            if within:
                in_window_records += 1

            for raw_symbol in (item["related_stocks"] or "").split(","):
                symbol = raw_symbol.strip().upper()
                if not symbol:
                    continue
                entry = symbols.setdefault(
                    symbol,
                    {
                        "vbts_var": True,
                        "kayit_sayisi": 0,
                        "son_tarih": None,
                        "disclosure_index": [],
                        "within_window": False,
                        "ham_related": item["related_stocks"],
                    },
                )
                entry["kayit_sayisi"] += 1
                if item["disclosure_index"] is not None:
                    entry["disclosure_index"].append(item["disclosure_index"])
                if date is not None and (entry["son_tarih"] is None or date > entry["son_tarih"]):
                    entry["son_tarih"] = date
                if within:
                    entry["within_window"] = True

        in_window_symbols = sum(1 for entry in symbols.values() if entry["within_window"])

        return {
            "window": {
                "days": days,
                "from": window_from.isoformat(),
                "to": reference.isoformat(),
                "as_of_supplied": as_of is not None,
                "provenance": "PARAMETRE+VARSAYILAN (K0-a c)",
            },
            "source": meta,
            "counts": {
                "records_total": len(parsed),
                "dkb_total": sum(
                    1 for item in parsed if item["disclosure_class"] == "DKB"
                ),
                "vbts_matched": len(vbts_records),
                "records_in_window": in_window_records,
                "symbols_total": len(symbols),
                "symbols_in_window": in_window_symbols,
                "unparseable_records": len(unparseable),
            },
            "tedbir_turu": (
                "VERİ YOK - tedbir türü çıkarılmaz (anahtar-kelime yöntemi "
                "çürütüldü, RAPOR.md §2.5); gövde okunmaz, disclosure_index "
                "operatör-on-demand içindir"
            ),
            "symbols": dict(sorted(symbols.items())),
        }

    def get_calendar(self, symbol: str) -> pd.DataFrame:
        """
        Get expected disclosure calendar for a stock from KAP.

        Returns upcoming expected disclosures like financial reports,
        annual reports, sustainability reports, and corporate governance reports.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            DataFrame with columns:
            - StartDate: Expected disclosure window start
            - EndDate: Expected disclosure window end
            - Subject: Type of disclosure (e.g., "Finansal Rapor")
            - Period: Report period (e.g., "Yıllık", "3 Aylık")
            - Year: Fiscal year
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return pd.DataFrame(columns=["StartDate", "EndDate", "Subject", "Period", "Year"])

        # Calculate date range: today to 6 months from now
        now = datetime.now()
        start_date = now.strftime("%Y-%m-%d")
        end_date = (now + timedelta(days=180)).strftime("%Y-%m-%d")

        # Fetch expected disclosures from KAP API
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": "https://kap.org.tr",
            "Referer": "https://kap.org.tr/tr/beklenen-bildirim-sorgu",
        }
        payload = {
            "startDate": start_date,
            "endDate": end_date,
            "memberTypes": ["IGS"],
            "mkkMemberOidList": [member_oid],
            "disclosureClass": "",
            "subjects": [],
            "mainSector": "",
            "sector": "",
            "subSector": "",
            "market": "",
            "index": "",
            "year": "",
            "term": "",
            "ruleType": "",
        }

        try:
            response = self._client.post(
                self.CALENDAR_API_URL,
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            records = []
            for item in data:
                records.append({
                    "StartDate": item.get("startDate", ""),
                    "EndDate": item.get("endDate", ""),
                    "Subject": item.get("subject", ""),
                    "Period": item.get("ruleTypeTerm", ""),
                    "Year": item.get("year", ""),
                })

            return pd.DataFrame(records)

        except Exception as e:
            raise APIError(f"Failed to fetch calendar for {symbol}: {e}") from e

    def get_company_details(self, symbol: str) -> dict:
        """
        Get company details from KAP company info page.

        Scrapes the KAP company page for sector, market, and website information.

        Args:
            symbol: Stock symbol (e.g., "THYAO").

        Returns:
            Dict with keys:
            - sector: Company sector (e.g., "ULAŞTIRMA VE DEPOLAMA")
            - market: Stock market (e.g., "YILDIZ PAZAR")
            - website: Company website URL(s)
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")
        current_time = time.time()

        # Check cache
        if symbol in self._company_details_cache:
            cache_time = self._company_details_cache_time.get(symbol, 0)
            if (current_time - cache_time) < self.CACHE_DURATION:
                return self._company_details_cache[symbol]

        # Get KAP member OID for the symbol
        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            return {}

        # Fetch company info page
        url = f"{self.COMPANY_INFO_URL}/{member_oid}"

        try:
            response = self._client.get(url, timeout=15)
            response.raise_for_status()
            html = response.text

            result = {}

            # Extract sector: href="/tr/Sektorler?sector=...">SECTOR_NAME</a>
            sector_match = re.search(
                r'href="/tr/Sektorler\?sector=[^"]*">([^<]+)</a>',
                html
            )
            if sector_match:
                result["sector"] = sector_match.group(1).strip()

            # Extract market: href="/tr/Pazarlar?market=...">MARKET_NAME</a>
            market_match = re.search(
                r'href="/tr/Pazarlar\?market=[^"]*">([^<]+)</a>',
                html
            )
            if market_match:
                result["market"] = market_match.group(1).strip()

            # Extract website: after "İnternet Adresi" label
            # Pattern: <h3...>İnternet Adresi</h3><p class="...">WEBSITE</p>
            website_match = re.search(
                r'İnternet Adresi</h3><p[^>]*>([^<]+)</p>',
                html
            )
            if website_match:
                result["website"] = website_match.group(1).strip()

            # Get business summary from genel page
            business_summary = self._get_business_summary(member_oid)
            if business_summary:
                result["businessSummary"] = business_summary

            # Cache result
            self._company_details_cache[symbol] = result
            self._company_details_cache_time[symbol] = current_time

            return result

        except Exception:
            return {}

    def _get_general_page_html(self, member_oid: str) -> str | None:
        """Fetch (and cache) the KAP company general-info page HTML once."""
        current_time = time.time()
        cached = self._general_page_cache.get(member_oid)
        if cached is not None and (current_time - cached[0]) < self.CACHE_DURATION:
            return cached[1]

        url = f"{self.COMPANY_GENERAL_URL}/{member_oid}"
        try:
            response = self._client.get(url, timeout=15)
            if response.status_code != 200:
                return None
        except Exception:
            return None

        self._general_page_cache[member_oid] = (current_time, response.text)
        return response.text

    def get_company_structure(self, symbol: str) -> dict:
        """
        Get the capital structure of a company from KAP "Genel Bilgiler".

        Parses the same page ``get_company_details`` already fetches. Returns
        only source values: paid-in capital, registered capital ceiling,
        direct shareholders (>= 3%), share totals, I.C / I.D-E exclusions,
        actual free float with its SPK 54/1626 methodology note, and nominal
        distribution by share group.

        Args:
            symbol: Stock symbol (e.g., "PENTA").

        Returns:
            Dict with ``Decimal`` values, a ``methodology`` credit block, three
            part-whole ``reconciliations`` and a ``missing`` list marking
            ``VERİ YOK`` fields with reasons.

        Raises:
            KAPStructureError: if the member OID or a mandatory page block is
                unavailable/unparseable (fail-closed, never partial data).
        """
        symbol = symbol.upper().replace(".IS", "").replace(".E", "")
        current_time = time.time()

        cached = self._company_structure_cache.get(symbol)
        if cached is not None:
            cached_at = self._company_structure_cache_time.get(symbol, 0)
            if (current_time - cached_at) < self.CACHE_DURATION:
                return cached

        member_oid = self.get_member_oid(symbol)
        if not member_oid:
            raise KAPStructureError(
                f"KAP member OID not found for {symbol}; company structure "
                "cannot be read"
            )

        html = self._get_general_page_html(member_oid)
        if html is None:
            raise KAPStructureError(
                f"KAP company general page could not be fetched for {symbol} "
                f"(member OID {member_oid})"
            )

        try:
            payload = _extract_next_f_payload(html)
        except RuntimeError as exc:
            raise KAPStructureError(
                f"KAP company general page schema changed for {symbol}: {exc}"
            ) from exc

        result = _parse_company_structure(
            payload,
            symbol=symbol,
            member_oid=member_oid,
            source_url=f"{self.COMPANY_GENERAL_URL}/{member_oid}",
        )

        self._company_structure_cache[symbol] = result
        self._company_structure_cache_time[symbol] = current_time
        return result

    def _get_business_summary(self, member_oid: str) -> str | None:
        """Get business summary (Faaliyet Konusu) from KAP genel page."""
        try:
            text = self._get_general_page_html(member_oid)
            if text is None:
                return None

            # Find the faaliyet_konu section
            key_idx = text.find("kpy41_acc2_faaliyet_konu")
            if key_idx < 0:
                return None

            # Extract chunk - stop at next section (kpy41_acc2_sure or similar)
            remaining = text[key_idx:]
            next_section = re.search(r'kpy41_acc2_(?!faaliyet)', remaining)
            if next_section:
                chunk = remaining[: next_section.start()]
            else:
                chunk = remaining[:5000]

            # Format 1: HTML-encoded value field (e.g., AKBNK, THYAO)
            # Pattern: value\":\"...\",\"disclosureIndex
            value_match = re.search(
                r'value\\":\\"(.*?)\\",\\"disclosureIndex', chunk, re.DOTALL
            )
            if value_match:
                value = value_match.group(1)
                if "u003e" in value:
                    # Decode HTML entities
                    decoded = value.replace("\\u003e", ">").replace("\\u003c", "<")
                    decoded = decoded.replace("u003e", ">").replace("u003c", "<")
                    # Extract text between > and <
                    text_parts = re.findall(r">([^<>]+)<", decoded)
                    meaningful = [
                        t.replace("\\n", " ").replace("\\\\n", " ").strip()
                        for t in text_parts
                        if len(t.strip()) > 10
                    ]
                    if meaningful:
                        summary = " ".join(meaningful)
                        # Clean up escape sequences
                        summary = summary.replace("\\ ", " ").replace("\\", "")
                        summary = re.sub(r"\s+", " ", summary).strip()
                        if len(summary) > 10:
                            return summary

            # Format 2: React children patterns
            texts = []
            skip_texts = ("Bulunmamaktadır.", "-", "Şirketin Süresi")

            # Children array: \"children\":[\"text\"]
            array_matches = re.findall(r'\\"children\\":\[\\"([^\\"]{10,})\\"', chunk)
            for m in array_matches:
                if not m.startswith("$") and m not in skip_texts:
                    texts.append(m)

            # Children string: \"children\":\"text\"
            string_matches = re.findall(r'\\"children\\":\\"([^\\"]{10,})\\"', chunk)
            for m in string_matches:
                if not m.startswith("$") and m not in skip_texts:
                    texts.append(m)

            if texts:
                summary = " ".join(texts)
                summary = summary.replace("\\n", " ").strip()
                summary = re.sub(r"\s+", " ", summary)
                if len(summary) > 10:
                    return summary

            return None
        except Exception:
            return None

    def get_disclosure_content(self, disclosure_id: int | str) -> str | None:
        """
        Get disclosure HTML content by ID.

        Args:
            disclosure_id: KAP disclosure ID (e.g., 1530826).

        Returns:
            Raw HTML body content or None if failed.
        """
        url = f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"

        try:
            response = self._client.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception:
            return None


# Singleton
_provider: KAPProvider | None = None


def get_kap_provider() -> KAPProvider:
    """Get singleton provider instance."""
    global _provider
    if _provider is None:
        _provider = KAPProvider()
    return _provider
