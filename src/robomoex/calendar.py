"""MOEX trading calendar adapter.

The stock calendar is read from the public MOEX ISS machine-readable endpoint.
Only dates marked as traded are turned into the one continuous session contract
used by the v1 engine. Session hours remain explicit because a board can expose
several intraday regimes; the default is the main TQBR session.
"""

from datetime import datetime, time
from urllib.parse import urlencode

import pandas as pd

from .data import utc, validate_sessions
from .moex import get_json

CALENDAR_URL = "https://apim.moex.com/iss/calendars/stock.json"


def _block_to_frame(response: dict, name: str = "off_days") -> pd.DataFrame:
    block = response.get(name)
    if not isinstance(block, dict) or not {"columns", "data"}.issubset(block):
        raise ValueError(f"MOEX calendar response has no {name} table")
    return pd.DataFrame(block["data"], columns=block["columns"])


def download_stock_calendar(
    start,
    end,
    *,
    open_at: str = "09:50:00",
    close_at: str = "18:50:00",
    transport=get_json,
) -> pd.DataFrame:
    """Return selected continuous stock sessions in UTC.

    ``start`` and ``end`` are timezone-aware instants. The calendar is queried
    by Moscow dates and includes transfered/working weekend days (``is_traded``
    equals one). No holiday rules are duplicated locally.
    """

    first, last = utc(start), utc(end)
    if first >= last:
        raise ValueError("start must precede end")
    try:
        start_clock = time.fromisoformat(open_at)
        close_clock = time.fromisoformat(close_at)
    except ValueError as error:
        raise ValueError("open_at and close_at must be HH:MM[:SS]") from error
    if start_clock >= close_clock:
        raise ValueError("open_at must precede close_at")
    query = urlencode(
        {
            "from": first.tz_convert("Europe/Moscow").date().isoformat(),
            "till": last.tz_convert("Europe/Moscow").date().isoformat(),
            "show_all_days": 1,
            "iss.only": "off_days",
            "iss.meta": "off",
        }
    )
    frame = _block_to_frame(transport(f"{CALENDAR_URL}?{query}"))
    required = {"tradedate", "is_traded"}
    if required - set(frame.columns):
        raise ValueError("MOEX calendar is missing tradedate/is_traded")
    dates = pd.to_datetime(frame.tradedate, errors="raise").dt.date
    traded = pd.to_numeric(frame.is_traded, errors="coerce") == 1
    rows = []
    for day, is_traded in zip(dates, traded, strict=True):
        if not is_traded:
            continue
        opening = pd.Timestamp(datetime.combine(day, start_clock), tz="Europe/Moscow")
        closing = pd.Timestamp(datetime.combine(day, close_clock), tz="Europe/Moscow")
        if closing <= first or opening >= last:
            continue
        rows.append(
            {"open_time": opening.tz_convert("UTC"), "close_time": closing.tz_convert("UTC")}
        )
    if not rows:
        raise ValueError("MOEX calendar returned no trading sessions in range")
    return validate_sessions(pd.DataFrame(rows))

