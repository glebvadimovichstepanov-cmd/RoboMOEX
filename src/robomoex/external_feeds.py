"""Incremental FRED oil and GDELT news archives, with observation-time provenance.

Latest-vintage oil and search results are not a verified historical event archive.
No MOEX credentials are sent to these public endpoints.
"""

import argparse
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import pandas as pd

OIL = {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU"}
NEWS_QUERY = '(Surgutneftegas OR "Russian oil" OR OPEC OR "Russia sanctions")'


def get_bytes(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "fred.stlouisfed.org",
        "api.gdeltproject.org",
    }:
        raise ValueError("Unsupported public feed URL")
    with urlopen(url, timeout=30) as response:
        return response.read()


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS observations (
        source TEXT, key TEXT, observed_at TEXT, payload TEXT, sha256 TEXT,
        PRIMARY KEY(source,key,sha256))""")
    db.execute("""CREATE TABLE IF NOT EXISTS coverage (
        source TEXT, day TEXT, observed_at TEXT, status TEXT,
        PRIMARY KEY(source,day))""")
    db.commit()
    return db


def save(db, source, key, observed, payload):
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    db.execute(
        "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)",
        (source, key, observed, raw, digest),
    )


def oil_rows(raw, series, first, last):
    frame = pd.read_csv(io.BytesIO(raw), na_values=["."])
    if set(frame.columns) != {"observation_date", series}:
        raise ValueError("Unexpected FRED schema")
    dates = pd.to_datetime(frame.observation_date, errors="raise").dt.date
    if dates.duplicated().any() or not dates.between(first, last).all():
        raise ValueError("Invalid FRED observation dates")
    values = pd.to_numeric(frame[series], errors="raise")
    # Negative WTI spot prices in 2020 are real; do not reject or log-transform them.
    if not values.dropna().map(lambda value: float("-inf") < value < float("inf")).all():
        raise ValueError("Nonfinite FRED price")
    return [
        dict(
            date=str(day),
            value=None if pd.isna(value) else float(value),
            series=series,
            source_url=f"https://fred.stlouisfed.org/series/{series}",
            historical_vintage_verified=False,
        )
        for day, value in zip(dates, values, strict=True)
    ]


def update_oil(db, start, end, transport=get_bytes, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Timezone required")
    first, last = pd.Timestamp(start).date(), min(pd.Timestamp(end).date(), now.date())
    reports = {}
    for name, series in OIL.items():
        source = f"fred.{name}"
        coverage = {
            row[0] for row in db.execute("SELECT day FROM coverage WHERE source=?", (source,))
        }
        missing = [
            day.date()
            for day in pd.date_range(first, last)
            if str(day.date()) not in coverage or day.date() >= now.date() - pd.Timedelta(days=14)
        ]
        if not missing:
            reports[name] = dict(status="cached", downloaded=0)
            continue
        # Prefix/backfill and revision overlap; responses remain atomic per series.
        begin, finish = min(missing), max(missing)
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(
            dict(id=series, cosd=begin, coed=finish)
        )
        try:
            raw = transport(url)
            rows = oil_rows(raw, series, begin, finish)
            if not rows:
                raise ValueError("Empty FRED result; coverage not advanced")
            with db:
                save(
                    db,
                    source + ".raw",
                    f"{begin}/{finish}",
                    now.isoformat(),
                    dict(url=url, csv=raw.decode("utf-8-sig")),
                )
                for row in rows:
                    save(db, source, row["date"], now.isoformat(), row)
                for day in pd.date_range(begin, finish):
                    db.execute(
                        "INSERT OR REPLACE INTO coverage VALUES (?,?,?,?)",
                        (source, str(day.date()), now.isoformat(), "latest_vintage"),
                    )
            reports[name] = dict(
                status="available",
                downloaded=len(rows),
                last_observation=rows[-1]["date"],
                point_in_time=False,
            )
        except (OSError, ValueError, KeyError) as error:
            reports[name] = dict(
                status="unavailable",
                error=type(error).__name__,
                http_status=error.code if isinstance(error, HTTPError) else None,
            )
    return reports


def update_news(db, start, end, transport=get_bytes, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Timezone required")
    first, last = pd.Timestamp(start).date(), min(pd.Timestamp(end).date(), now.date())
    source = "gdelt." + hashlib.sha256(NEWS_QUERY.encode()).hexdigest()[:12]
    reports = []
    for day in pd.date_range(first, last):
        cached = db.execute(
            "SELECT status FROM coverage WHERE source=? AND day=?", (source, str(day.date()))
        ).fetchone()
        if (
            cached
            and cached[0] == "partial_search"
            and day.date() < now.date() - pd.Timedelta(days=2)
        ):
            continue
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(
            dict(
                query=NEWS_QUERY,
                mode="artlist",
                format="json",
                maxrecords=250,
                startdatetime=day.strftime("%Y%m%d000000"),
                enddatetime=day.strftime("%Y%m%d235959"),
                sort="datedesc",
            )
        )
        try:
            raw = transport(url)
            payload = json.loads(raw)
            articles = payload.get("articles")
            if not isinstance(articles, list):
                raise ValueError("GDELT article list missing")
            validated = []
            for article in articles:
                link = article["url"]
                if urlsplit(link).scheme not in {"http", "https"}:
                    raise ValueError("Invalid article URL")
                seen = pd.to_datetime(article["seendate"], utc=True)
                if seen.date() != day.date():
                    raise ValueError("GDELT article outside window")
                validated.append(
                    dict(
                        url=link,
                        title=article.get("title", ""),
                        seen_at=seen.isoformat(),
                        known_at=now.isoformat(),
                        verified=False,
                        event_risk="UNKNOWN",
                    )
                )
            status = "truncated" if len(articles) >= 250 else "partial_search"
            with db:
                save(db, source + ".raw", str(day.date()), now.isoformat(), payload)
                for row in validated:
                    save(db, source, row["url"], now.isoformat(), row)
                db.execute(
                    "INSERT OR REPLACE INTO coverage VALUES (?,?,?,?)",
                    (source, str(day.date()), now.isoformat(), status),
                )
            reports.append(dict(day=str(day.date()), status=status, articles=len(articles)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            reports.append(
                dict(
                    day=str(day.date()),
                    status="unavailable",
                    error=type(error).__name__,
                    http_status=error.code if isinstance(error, HTTPError) else None,
                )
            )
            break  # Do not hammer an unavailable/rate-limited service.
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("results/external/cache.sqlite"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="UTC").date()))
    parser.add_argument("--news-days", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.news_days <= 90:
        parser.error("--news-days must be 1..90; full historical coverage is not implied")
    with connect(args.db) as db:
        oil = update_oil(db, args.start, args.end)
        end = min(pd.Timestamp(args.end).date(), pd.Timestamp.now(tz="UTC").date())
        news = update_news(db, end - pd.Timedelta(days=args.news_days - 1), end)
    from .operations import Journal

    with Journal(args.db.parent / "operations") as journal:
        for name, status in oil.items():
            journal.emit(
                "source_refresh",
                "ERROR" if status["status"] == "unavailable" else "INFO",
                source=name,
                **status,
            )
        for status in news:
            journal.emit(
                "source_refresh",
                "ERROR" if status["status"] == "unavailable" else "INFO",
                source="gdelt",
                **status,
            )
    print(json.dumps(dict(oil=oil, news=news, event_risk="UNKNOWN")))


if __name__ == "__main__":
    main()
