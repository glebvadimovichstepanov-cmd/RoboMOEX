"""Native MOEX daily candles; completed days only, overlapping incremental refresh."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from .cache import load_cache, save_cache
from .data import validate_bars
from .moex import get_json


def download(start, end, transport=get_json):
    rows, offset, previous = [], 0, None
    while True:
        query = urlencode(dict(interval=24, **{"from": start, "till": end}, start=offset))
        url = (
            "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
            "securities/SNGS/candles.json?" + query
        )
        block = transport(url)["candles"]
        page = pd.DataFrame(block["data"], columns=block["columns"])
        if page.empty:
            break
        stamp = pd.to_datetime(page.begin)
        if not stamp.is_monotonic_increasing or stamp.duplicated().any():
            raise ValueError("Unordered ISS candles")
        if previous is not None and stamp.iloc[0] <= previous:
            raise ValueError("ISS pagination did not advance")
        previous = stamp.iloc[-1]
        rows.append(page)
        offset += len(page)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    frame["open_time"] = pd.to_datetime(frame.begin).dt.tz_localize("Europe/Moscow")
    frame["close_time"] = pd.to_datetime(frame.end).dt.tz_localize("Europe/Moscow") + pd.Timedelta(
        microseconds=1
    )
    dates = frame.open_time.dt.date
    if not dates.between(pd.Timestamp(start).date(), pd.Timestamp(end).date()).all():
        raise ValueError("ISS returned candles outside requested dates")
    return validate_bars(frame)


def update(path, start, end, transport=get_json, now=None):
    path = Path(path)
    now = pd.Timestamp.now(tz="Europe/Moscow") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must have timezone")
    end = min(
        pd.Timestamp(end).date(), now.tz_convert("Europe/Moscow").date() - pd.Timedelta(days=1)
    )
    start = pd.Timestamp(start).date()
    if start > end:
        raise ValueError("No completed days requested")
    identity = dict(provider="MOEX ISS", ticker="SNGS", board="TQBR", interval=24)
    old = load_cache(path, identity) if path.exists() else pd.DataFrame()
    ranges = [(start, end)]
    if not old.empty:
        first = old.open_time.iloc[0].tz_convert("Europe/Moscow").date()
        last = old.open_time.iloc[-1].tz_convert("Europe/Moscow").date()
        ranges = []
        if start < first:
            ranges.append((start, first - pd.Timedelta(days=1)))
        tail = max(start, last - pd.Timedelta(days=7))
        if tail <= end:
            ranges.append((tail, end))
    result = old
    for first, last in ranges:
        fresh = download(str(first), str(last), transport)
        if not fresh.empty:
            if not result.empty:
                dates = result.open_time.dt.tz_convert("Europe/Moscow").dt.date
                result = result[~dates.between(first, last)]
            result = pd.concat([result, fresh], ignore_index=True)
    if result.empty:
        raise ValueError("No native daily candles available")
    result = validate_bars(result.sort_values("open_time", ignore_index=True))
    save_cache(path, result, identity)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/native/SNGS.1d.json"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="Europe/Moscow").date()))
    parser.add_argument("--authenticated", action="store_true")
    args = parser.parse_args()
    from .algopack import get_authenticated_json

    bars = update(
        args.output,
        args.start,
        args.end,
        get_authenticated_json if args.authenticated else get_json,
    )
    print(
        json.dumps(
            dict(
                rows=len(bars),
                first=str(bars.open_time.iloc[0]),
                last=str(bars.close_time.iloc[-1]),
            )
        )
    )


if __name__ == "__main__":
    main()
