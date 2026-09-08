"""Incremental, checksummed daily context caches for the signal engine."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .cache import atomic_write
from .events import update_dividend_cache
from .moex import get_json


@dataclass(frozen=True)
class MoexDailySource:
    name: str
    engine: str
    market: str
    board: str
    security: str


SOURCES = (
    MoexDailySource("imoex", "stock", "index", "SNGX", "IMOEX"),
    MoexDailySource("rtsi", "stock", "index", "RTSI", "RTSI"),
    MoexDailySource("rgbI", "stock", "index", "SNGX", "RGBI"),
    MoexDailySource("usd_rub", "currency", "selt", "CETS", "USD000UTSTOM"),
    MoexDailySource("cny_rub", "currency", "selt", "CETS", "CNYRUB_TOM"),
    MoexDailySource("oil", "stock", "index", "RTSI", "BRFOB"),
    MoexDailySource("lkoh", "stock", "shares", "TQBR", "LKOH"),
    MoexDailySource("rosn", "stock", "shares", "TQBR", "ROSN"),
    MoexDailySource("tatn", "stock", "shares", "TQBR", "TATN"),
    MoexDailySource("bane", "stock", "shares", "TQBR", "BANE"),
    MoexDailySource("gazp", "stock", "shares", "TQBR", "GAZP"),
    MoexDailySource("nvtk", "stock", "shares", "TQBR", "NVTK"),
)


def _dates(value) -> date:
    return pd.Timestamp(value).date()


def download_moex_daily(source, start, end, transport=get_json):
    """Read every page. Empty valid tables are allowed; malformed tables are errors."""
    first, last = _dates(start), _dates(end)
    if first > last:
        raise ValueError("start must not follow end")
    frames, offset, previous = [], 0, None
    for _ in range(10000):
        query = urlencode(
            dict(
                start=offset,
                interval=24,
                **{"from": str(first), "till": str(last), "iss.meta": "off", "iss.only": "candles"},
            )
        )
        url = (
            f"https://iss.moex.com/iss/engines/{source.engine}/markets/{source.market}/"
            f"boards/{source.board}/securities/{source.security}/candles.json?{query}"
        )
        response = transport(url)
        block = response.get("candles")
        if not isinstance(block, dict) or not {"columns", "data"} <= block.keys():
            raise ValueError(f"Invalid candles response: {source.name}")
        if not {"begin", "close"} <= set(block["columns"]):
            raise ValueError("Missing daily candle columns")
        if not block["data"]:
            break
        raw = pd.DataFrame(block["data"], columns=block["columns"])
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw.begin).dt.date,
                source.name: pd.to_numeric(raw.close, errors="raise"),
            }
        )
        if (
            frame.date.duplicated().any()
            or not frame.date.is_monotonic_increasing
            or not np.isfinite(frame[source.name]).all()
            or (frame[source.name] <= 0).any()
        ):
            raise ValueError("Invalid daily observations")
        if previous is not None and frame.date.iloc[0] <= previous:
            raise ValueError("Daily pagination did not advance")
        previous = frame.date.iloc[-1]
        frames.append(frame)
        offset += len(frame)
        if previous >= last:
            break
    else:
        raise ValueError("Daily pagination limit exceeded")
    if not frames:
        return pd.DataFrame(columns=["date", source.name])
    result = pd.concat(frames, ignore_index=True)
    return result[result.date.between(first, last)].reset_index(drop=True)


def load_context_cache(path, identity):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in (1, 2) or payload.get("identity") != identity:
        raise ValueError("Context cache identity/schema mismatch")
    if hashlib.sha256(payload["csv"].encode()).hexdigest() != payload["sha256"]:
        raise ValueError("Context cache checksum mismatch")
    frame = pd.read_csv(io.StringIO(payload["csv"]), float_precision="round_trip")
    frame["date"] = pd.to_datetime(frame.date).dt.date
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("Invalid context dates")
    return frame, _dates(payload["range_start"]), _dates(payload["range_end"])


def save_context_cache(path, frame, identity, coverage=None):
    data = frame.sort_values("date").reset_index(drop=True)
    if data.date.duplicated().any():
        raise ValueError("Duplicate context dates")
    if coverage is None:
        if data.empty:
            raise ValueError("Empty cache requires explicit queried coverage")
        coverage = (data.date.iloc[0], data.date.iloc[-1])
    csv = data.to_csv(index=False, lineterminator="\n")
    payload = dict(
        schema=2,
        identity=identity,
        range_start=str(coverage[0]),
        range_end=str(coverage[1]),
        sha256=hashlib.sha256(csv.encode()).hexdigest(),
        csv=csv,
        coverage_kind="queried_dates",
        fetched_at=str(pd.Timestamp.now(tz="UTC")),
    )
    atomic_write(path, json.dumps(payload, sort_keys=True, allow_nan=False).encode())


def update_context_cache(path, source, start, end, transport=get_json):
    identity = dict(
        provider="moex-iss-daily-v1",
        source=source.name,
        engine=source.engine,
        market=source.market,
        board=source.board,
        security=source.security,
    )
    first, last = _dates(start), _dates(end)
    if first > last:
        raise ValueError("start must not follow end")
    if path.exists():
        cached, left, right = load_context_cache(path, identity)
        chunks = [cached]
        ranges = []
        if first < left:
            ranges.append((first, left - timedelta(days=1)))
        if last > right:
            ranges.append((right + timedelta(days=1), last))
        if not ranges:
            return cached[cached.date.between(first, last)].reset_index(drop=True)
        for a, b in ranges:
            chunks.append(download_moex_daily(source, a, b, transport))
        data = pd.concat(chunks, ignore_index=True)
        coverage = (min(first, left), max(last, right))
    else:
        data = download_moex_daily(source, first, last, transport)
        coverage = (first, last)
    save_context_cache(path, data, identity, coverage)
    return data[data.date.between(first, last)].sort_values("date").reset_index(drop=True)


def update_all(cache_dir, start, end, transport=get_json):
    """Refresh independent feeds; persist explicit failures, never synthetic healthy values."""
    frames, status = [], {}
    # A candle for the current Moscow date is not yet a closed daily observation.
    end = min(_dates(end), pd.Timestamp.now(tz="Europe/Moscow").date() - timedelta(days=1))
    for source in SOURCES:
        try:
            frame = update_context_cache(
                cache_dir / f"{source.name}.json", source, start, end, transport
            )
            frames.append(frame)
            status[source.name] = dict(
                status="available" if len(frame) else "empty",
                rows=len(frame),
                last_date=str(frame.date.iloc[-1]) if len(frame) else None,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            status[source.name] = dict(status="error", error=str(error))
    try:
        dividends = update_dividend_cache(cache_dir / "dividends_v2.json")
        status["dividends"] = dict(status="unverified_dates", rows=len(dividends))
    except (OSError, ValueError, KeyError) as error:
        status["dividends"] = dict(status="error", error=str(error))
    for name in ("corporate_announcements", "calendar", "spread", "news", "key_rate", "wti"):
        status[name] = dict(status="unavailable")
    atomic_write(cache_dir / "status.json", json.dumps(status, indent=2).encode())
    if not frames:
        raise ValueError("No market context available; see status.json")
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer", validate="one_to_one")
    result = result.sort_values("date").reset_index(drop=True)
    result["event_risk"] = "UNKNOWN"
    result["event_penalty"] = 0.0
    result["spread_bps"] = np.nan
    save_context_cache(
        cache_dir / "sngs_context_v2.json",
        result,
        {"provider": "robomoex-context-v2", "sources": [s.name for s in SOURCES]},
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh context; missing feeds block entries")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/context_cache_v2"))
    parser.add_argument("--authenticated", action="store_true")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=str(date.today()))
    args = parser.parse_args(argv)
    try:
        from .algopack import get_authenticated_json

        transport = get_authenticated_json if args.authenticated else get_json
        frame = update_all(args.cache_dir, args.start, args.end, transport)
        print(
            json.dumps(
                dict(
                    rows=len(frame),
                    trading_ready=False,
                    action="NO_TRADE",
                    status_path=str(args.cache_dir / "status.json"),
                )
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps(dict(error=str(error), trading_ready=False)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
