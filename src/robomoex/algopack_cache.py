"""AlgoPack history and observed snapshots. SQLite transactions checkpoint each window.

Raw provider fields/units are preserved. Historical revisions are retained but are
not a vendor point-in-time archive. Credentials are exclusively in the transport.
"""

import argparse
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from .algopack import get_authenticated_json

BASE = "https://apim.moex.com/iss/"
MSK = ZoneInfo("Europe/Moscow")


@dataclass(frozen=True)
class Dataset:
    name: str
    route: str
    table: str = "data"
    dated: bool = True
    cursor: bool = True


def registry(equities=("SNGS",), futures=()):
    datasets = []
    for market, symbols, metrics in (
        ("eq", equities, ("tradestats", "obstats", "orderstats", "hi2", "alerts")),
        ("fx", ("CNYRUB_TOM", "USD000UTSTOM"), ("tradestats", "obstats", "orderstats")),
        ("fo", futures, ("tradestats", "obstats")),
    ):
        for ticker in symbols:
            if not ticker.replace("_", "").isalnum():
                raise ValueError("Invalid instrument code")
            for metric in metrics:
                datasets.append(
                    Dataset(
                        f"{market}.{ticker}.{metric}",
                        f"datashop/algopack/{market}/{metric}/{ticker}.json",
                    )
                )
    for ticker in ("BR", "Si", "RI", "CR"):
        datasets.append(
            Dataset(
                f"futoi.{ticker}",
                f"analyticalproducts/futoi/securities/{ticker}.json",
                "futoi",
                cursor=False,
            )
        )
    return datasets


SNAPSHOTS = (
    Dataset("calendar.stock", "calendars/stock.json?show_all_days=1", "off_days", False, False),
    Dataset("calendar.sessions", "calendars/stock/session.json", "session_schedule", False, False),
    Dataset(
        "calendar.suspended",
        "calendars/stock/securities/suspended/details.json",
        "suspended",
        False,
    ),
    Dataset(
        "reference.futures",
        "engines/futures/markets/forts/securities.json",
        "securities",
        False,
        False,
    ),
    Dataset(
        "reference.SNGS",
        "engines/stock/markets/shares/boards/TQBR/securities/SNGS.json",
        "securities",
        False,
        False,
    ),
    Dataset(
        "snapshot.SNGS.orderbook",
        "engines/stock/markets/shares/boards/TQBR/securities/SNGS/orderbook.json",
        "orderbook",
        False,
        False,
    ),
    Dataset(
        "snapshot.SNGS.trades",
        "engines/stock/markets/shares/boards/TQBR/securities/SNGS/trades.json",
        "trades",
        False,
        False,
    ),
)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS windows (
            dataset TEXT, identity TEXT, start TEXT, end TEXT, observed_at TEXT,
            sha256 TEXT, payload TEXT, PRIMARY KEY(dataset, start, end));
        CREATE TABLE IF NOT EXISTS revisions (
            dataset TEXT, start TEXT, end TEXT, observed_at TEXT, sha256 TEXT,
            payload TEXT, UNIQUE(dataset,start,end,sha256));
    """)
    return db


def fetch(dataset, start, end, transport=get_authenticated_json):
    """Cursor integrity checked; no-cursor FUTOI requested in one-day windows."""
    offset, rows, columns, seen = 0, [], None, set()
    expected_total = None
    for _ in range(10000):
        params = {"iss.meta": "off", "start": offset}
        if dataset.dated:
            params.update({"from": start, "till": end})
        url = BASE + dataset.route + ("&" if "?" in dataset.route else "?") + urlencode(params)
        for attempt in range(3):
            try:
                payload = transport(url)
                break
            except ValueError as error:
                if attempt == 2 or not any(
                    s in str(error) for s in ("429", "500", "502", "503", "504", "transport")
                ):
                    raise
                time.sleep(2**attempt)
        block = payload.get(dataset.table)
        if (
            not isinstance(block, dict)
            or not isinstance(block.get("columns"), list)
            or not isinstance(block.get("data"), list)
        ):
            raise ValueError("Expected ISS table missing")
        if columns is None:
            columns = block["columns"]
        if columns != block["columns"] or len(set(columns)) != len(columns):
            raise ValueError("ISS schema changed during pagination")
        page = block["data"]
        if any(not isinstance(r, list) or len(r) != len(columns) for r in page):
            raise ValueError("Malformed ISS row")
        if dataset.dated:
            if "tradedate" not in columns:
                raise ValueError("Historical table has no tradedate")
            idx = columns.index("tradedate")
            if any(not start <= r[idx] <= end for r in page):
                raise ValueError("ISS ignored requested date range")
            identifier = "ticker" if dataset.table == "futoi" else "secid"
            expected = dataset.route.rsplit("/", 1)[-1].removesuffix(".json")
            if identifier not in columns or any(
                r[columns.index(identifier)] != expected for r in page
            ):
                raise ValueError("ISS returned a different instrument")
        digest = hashlib.sha256(encode(page).encode()).hexdigest()
        if page and digest in seen:
            raise ValueError("ISS pagination repeated a page")
        seen.add(digest)
        rows.extend(page)
        if not dataset.cursor:
            break
        cursor = payload.get(dataset.table + ".cursor")
        if not cursor or len(cursor.get("data", [])) != 1:
            raise ValueError("Expected ISS cursor missing")
        info = dict(zip(cursor["columns"], cursor["data"][0], strict=True))
        total = int(info["TOTAL"])
        if int(info["INDEX"]) != offset or total < len(rows):
            raise ValueError("Inconsistent ISS cursor")
        if expected_total is not None and total != expected_total:
            raise ValueError("ISS total changed; retry this window")
        expected_total = total
        if len(rows) == total:
            break
        if not page:
            raise ValueError("ISS pagination ended early")
        offset += len(page)
    else:
        raise ValueError("ISS pagination limit exceeded")
    return {"columns": columns, "data": rows}


def update(db, dataset, start, end, transport=get_authenticated_json, today=None, refresh=False):
    today = today or datetime.now(MSK).date()
    start, end = date.fromisoformat(start), min(date.fromisoformat(end), today)
    if start > end:
        raise ValueError("Invalid date range")
    identity = encode(dataset.__dict__)
    calls, added = 0, 0
    # A previous successful range is coverage, including valid empty responses.
    covered = set()
    for a, b, ident in db.execute(
        "SELECT start,end,identity FROM windows WHERE dataset=?", (dataset.name,)
    ):
        if ident != identity:
            raise ValueError("Dataset cache identity mismatch")
        day = date.fromisoformat(a)
        while day <= date.fromisoformat(b):
            covered.add(day)
            day += timedelta(days=1)
    day = start
    while day <= end:
        if day in covered and day < today - timedelta(days=1) and not refresh:
            day += timedelta(days=1)
            continue
        last = day
        # Bound no-cursor history to a single day; regular data to missing 7-day runs.
        if dataset.cursor:
            while last < min(end, day + timedelta(days=6)):
                nxt = last + timedelta(days=1)
                if nxt in covered and nxt < today - timedelta(days=1) and not refresh:
                    break
                last = nxt
        result = fetch(dataset, str(day), str(last), transport)
        body = encode(result)
        observed = datetime.now(MSK).isoformat()
        digest = hashlib.sha256(body.encode()).hexdigest()
        with db:
            db.execute(
                "INSERT OR IGNORE INTO revisions VALUES (?,?,?,?,?,?)",
                (
                    dataset.name,
                    str(day),
                    str(last),
                    observed,
                    digest,
                    body,
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO windows VALUES (?,?,?,?,?,?,?)",
                (
                    dataset.name,
                    identity,
                    str(day),
                    str(last),
                    observed,
                    digest,
                    body,
                ),
            )
        calls += 1
        added += len(result["data"])
        day = last + timedelta(days=1)
    return {"windows_fetched": calls, "rows_fetched": added}


def read_rows(db, dataset):
    # Latest observed overlapping windows win. Keep raw revisions for audit.
    unique = {}
    for body, digest, start, end in db.execute(
        "SELECT payload,sha256,start,end FROM windows WHERE dataset=? ORDER BY observed_at",
        (dataset,),
    ):
        if hashlib.sha256(body.encode()).hexdigest() != digest:
            raise ValueError("AlgoPack cache checksum mismatch")
        block = json.loads(body)
        unique = {k: r for k, r in unique.items() if not start <= r["tradedate"] <= end}
        for values in block["data"]:
            row = dict(zip(block["columns"], values, strict=True))
            keys = (
                "tradedate",
                "tradetime",
                "secid",
                "ticker",
                "clgroup",
                "metric",
                "alert_type",
                "threshold",
                "sess_id",
                "seqnum",
            )
            key = tuple(row.get(k) for k in keys)
            unique[key] = row
    return list(unique.values())


def snapshot(db, dataset, transport=get_authenticated_json):
    result = fetch(dataset, None, None, transport)
    observed = datetime.now(MSK).isoformat()
    body = encode(result)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with db:
        db.execute(
            "INSERT OR IGNORE INTO revisions VALUES (?,?,?,?,?,?)",
            (
                dataset.name,
                observed,
                observed,
                observed,
                digest,
                body,
            ),
        )
    return {"rows": len(result["data"]), "observed_at": observed}


def inventory(db):
    result = {}
    names = [r[0] for r in db.execute("SELECT DISTINCT dataset FROM windows ORDER BY dataset")]
    for name in names:
        rows = read_rows(db, name)
        spans = list(
            db.execute("SELECT start,end FROM windows WHERE dataset=? ORDER BY start", (name,))
        )
        result[name] = dict(
            rows=len(rows),
            first_row=min((r["tradedate"] for r in rows), default=None),
            last_row=max((r["tradedate"] for r in rows), default=None),
            queried_ranges=spans,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Incremental authenticated AlgoPack ingestion")
    parser.add_argument("--db", type=Path, default=Path("results/algopack/cache.sqlite"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=str(datetime.now(MSK).date()))
    parser.add_argument(
        "--equities", nargs="+", default=["SNGS", "LKOH", "ROSN", "TATN", "BANE", "GAZP", "NVTK"]
    )
    parser.add_argument("--futures", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", help="Exact dataset names; default all")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--status", action="store_true", help="Offline cache inventory")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-query history, retaining revisions"
    )
    args = parser.parse_args(argv)
    report = {}
    with connect(args.db) as db:
        datasets = registry(args.equities, args.futures)
        if args.only and set(args.only) - {d.name for d in datasets}:
            parser.error("Unknown --only dataset; include its equity/futures instrument")
        if args.status:
            print(encode(inventory(db)))
            return 0
        for dataset in datasets:
            if args.only and dataset.name not in args.only:
                continue
            try:
                result = update(db, dataset, args.start, args.end, refresh=args.refresh)
                rows = read_rows(db, dataset.name)
                report[dataset.name] = dict(
                    status="available" if rows else "empty", rows=len(rows), **result
                )
            except (ValueError, OSError) as error:
                report[dataset.name] = dict(status="failed", error=str(error))
            print(encode({dataset.name: report[dataset.name]}), flush=True)
        if args.snapshots:
            for dataset in SNAPSHOTS:
                try:
                    report[dataset.name] = dict(status="snapshot", **snapshot(db, dataset))
                except (ValueError, OSError) as error:
                    report[dataset.name] = dict(status="failed", error=str(error))
                print(encode({dataset.name: report[dataset.name]}), flush=True)
        cached = inventory(db)
    from .cache import atomic_write

    atomic_write(args.db.parent / "last_run.json", encode(report).encode())
    atomic_write(args.db.parent / "inventory.json", encode(cached).encode())
    return 2 if any(r["status"] == "failed" for r in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
