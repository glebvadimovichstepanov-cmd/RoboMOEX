"""Read-only public MOEX ISS minute candles; no broker SDK or trading credentials."""

import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .data import BAR_COLUMNS, utc, validate_bars


def get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=15) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
        time.sleep(0.5 * 2**attempt)
    raise RuntimeError("MOEX request failed")


def download_minutes(
    symbol: str, start, end, *, board: str = "TQBR", transport=get_json
) -> pd.DataFrame:
    for value in (symbol, board):
        if not re.fullmatch(r"[A-Z0-9_]{1,20}", value):
            raise ValueError("Invalid symbol or board")
    first, last = utc(start), utc(end)
    if first >= last:
        raise ValueError("start must precede end")
    base = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/boards/"
        f"{board}/securities/{symbol}/candles.json"
    )
    frames, offset, previous_last = [], 0, None
    for _ in range(100_000):
        query = urlencode(
            {
                "from": first.tz_convert("Europe/Moscow").date().isoformat(),
                "till": last.tz_convert("Europe/Moscow").date().isoformat(),
                "interval": 1,
                "start": offset,
                "iss.meta": "off",
                "iss.only": "candles",
            }
        )
        response = transport(f"{base}?{query}")
        block = response["candles"]
        if not block["data"]:
            break
        frame = pd.DataFrame(block["data"], columns=block["columns"])
        begin = pd.to_datetime(frame["begin"])
        if begin.dt.tz is None:
            begin = begin.dt.tz_localize("Europe/Moscow")
        frame["open_time"] = begin.dt.tz_convert("UTC")
        frame["close_time"] = frame.open_time + pd.Timedelta(minutes=1)
        frame = validate_bars(frame[BAR_COLUMNS])
        if previous_last is not None and frame.open_time.iloc[0] <= previous_last:
            raise ValueError("MOEX pagination did not advance")
        previous_last = frame.open_time.iloc[-1]
        frames.append(frame)
        offset += len(frame)
        if frame.close_time.iloc[-1] >= last:
            break
    else:
        raise ValueError("MOEX pagination limit exceeded")
    if not frames:
        raise ValueError("MOEX returned no candles")
    result = pd.concat(frames, ignore_index=True)
    result = result[(result.open_time >= first) & (result.close_time <= last)]
    return validate_bars(result)
