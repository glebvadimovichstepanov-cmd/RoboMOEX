"""Public e-disclosure ingestion. 403/captcha is unavailable, never empty coverage."""

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from .cache import atomic_write

COMPANY = "https://www.e-disclosure.ru/portal/company.aspx?id=312"


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links, self.text, self.published = [], [], None
        self.hidden = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])
        if tag == "meta" and attributes.get("property") == "article:published_time":
            self.published = attributes.get("content")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, value):
        if not self.hidden:
            self.text.append(value.strip())


def allowed(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc not in {"www.e-disclosure.ru", "e-disclosure.ru"}:
        raise ValueError("Disclosure URL must remain on the official HTTPS host")
    return url


def get_html(url):
    allowed(url)
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "RoboMOEX/0.1 research"}), timeout=20
        ) as response:
            allowed(response.url)
            content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError("Disclosure page exceeds size limit")
            charset = response.headers.get_content_charset() or "utf-8"
            return content.decode(charset)
    except HTTPError as error:
        raise ValueError(f"Disclosure HTTP {error.code}; no coverage asserted") from None


def event_links(html):
    page = Page(html)
    if "сургутнефтегаз" not in " ".join(page.text).lower():
        raise ValueError("Company identity missing or access challenge")
    result = set()
    for link in page.links:
        url = urljoin(COMPANY, link)
        parsed = urlsplit(url)
        if parsed.path.lower() == "/portal/event.aspx" and "EventId" in parse_qs(parsed.query):
            result.add(allowed(url))
    if not result:
        raise ValueError("No event links; layout/archive not verified")
    return sorted(result)


def parse_event(html, url):
    allowed(url)
    event_id = parse_qs(urlsplit(url).query).get("EventId", [])
    page = Page(html)
    content = " ".join(filter(None, page.text))
    if len(event_id) != 1 or "сургутнефтегаз" not in content.lower():
        raise ValueError("Event identity is not verified")
    published = page.published
    if published is None:
        match = re.search(
            r"(?:Дата|Время)\s+публикации\s*:?\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}(?::\d{2})?)",
            content,
            re.I,
        )
        if match:
            published = pd.Timestamp(pd.to_datetime(match[1], dayfirst=True)).tz_localize(
                "Europe/Moscow"
            )
    timestamp = pd.Timestamp(published) if published is not None else None
    if timestamp is None or timestamp.tzinfo is None:
        raise ValueError("Explicit publication timestamp missing; manual review required")
    category = (
        "DIVIDEND_ANNOUNCEMENT" if "дивиденд" in content.lower() else "CORPORATE_ANNOUNCEMENT"
    )
    return dict(
        event_id=event_id[0],
        ticker="SNGS",
        event_type=category,
        published_at=timestamp.tz_convert("UTC").isoformat(),
        source_url=url,
        text=content,
        date_verified=False,
    )


def ingest(cache_dir, url, html):
    root = Path(cache_dir)
    allowed(url)
    digest = hashlib.sha256(html.encode()).hexdigest()
    atomic_write(root / "raw" / f"{digest}.html", html.encode())
    event = parse_event(html, url)
    key = hashlib.sha256(event["event_id"].encode()).hexdigest()
    path = root / "events" / key / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    event["observed_at"] = datetime.now(UTC).isoformat()
    event["raw_sha256"] = digest
    # Later revisions cannot inherit the original publication time in PIT replay.
    prior = list(path.parent.glob("*.json")) if path.parent.exists() else []
    event["known_at"] = event["observed_at"] if prior else event["published_at"]
    event["historical_version_unverified"] = not prior
    atomic_write(root / "raw" / f"{digest}.html", html.encode())
    atomic_write(path, json.dumps(event, ensure_ascii=False, indent=2).encode())
    return event


def refresh(cache_dir, transport=get_html):
    root = Path(cache_dir)
    report = dict(
        source=COMPANY,
        observed_at=datetime.now(UTC).isoformat(),
        archive_complete=False,
        events_ingested=0,
        errors=[],
    )
    try:
        html = transport(COMPANY)
        atomic_write(root / "company.html", html.encode())
        for url in event_links(html):
            try:
                ingest(root, url, transport(url))
                report["events_ingested"] += 1
            except (ValueError, OSError) as error:
                report["errors"].append(str(error))
            time.sleep(0.25)
    except (ValueError, OSError) as error:
        report["errors"].append(str(error))
    report["status"] = "unavailable" if report["errors"] else "partial_archive"
    atomic_write(root / "status.json", json.dumps(report, indent=2).encode())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cache corporate disclosures, preserving provenance"
    )
    parser.add_argument("--cache", type=Path, default=Path("results/disclosure"))
    parser.add_argument("--import-html", type=Path)
    parser.add_argument("--url")
    args = parser.parse_args(argv)
    if args.import_html:
        if not args.url:
            parser.error("--url is required for an HTML import")
        item = ingest(args.cache, args.url, args.import_html.read_text(encoding="utf-8-sig"))
        print(json.dumps(dict(event_id=item["event_id"], date_verified=False)))
        return 0
    report = refresh(args.cache)
    print(json.dumps(report))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
