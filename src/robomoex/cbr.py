"""Official CBR key-rate history, with effective dates kept distinct from announcements."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pandas as pd

from .cache import atomic_write

URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"


def download(start, end, transport=urlopen):
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    body = (
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        '<soap:Body><KeyRateXML xmlns="http://web.cbr.ru/">'
        f"<fromDate>{start}T00:00:00</fromDate><ToDate>{end}T00:00:00</ToDate>"
        "</KeyRateXML></soap:Body></soap:Envelope>"
    )
    request = Request(
        URL,
        data=body.encode(),
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": "http://web.cbr.ru/KeyRateXML",
        },
    )
    with transport(request, timeout=20) as response:
        raw = response.read()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("XML entities are not accepted")
    root = ET.fromstring(raw)
    rows = []
    for node in root.iter():
        fields = {child.tag.rsplit("}", 1)[-1].lower(): child.text for child in node}
        if "dt" in fields and "rate" in fields:
            day = pd.Timestamp(fields["dt"]).date()
            value = float(fields["rate"].replace(",", "."))
            if not start <= day <= end or not 0 <= value <= 100:
                raise ValueError("Invalid CBR rate or date range")
            rows.append(dict(date=str(day), rate=value))
    if not rows and not any(node.tag.rsplit("}", 1)[-1] == "KeyRate" for node in root.iter()):
        raise ValueError("CBR response has no recognized rate records")
    frame = pd.DataFrame(rows, columns=["date", "rate"]).sort_values("date")
    if frame.date.duplicated().any():
        raise ValueError("Duplicate CBR rate dates")
    return frame


def update(path, start, end, transport=urlopen):
    path = Path(path)
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if old and (old.get("source") != URL or old.get("schema") != 1):
        raise ValueError("CBR cache identity mismatch")
    if (
        old
        and old.get("sha256")
        != hashlib.sha256(json.dumps(old["rows"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("CBR cache checksum mismatch")
    ranges = [(start, end)] if old is None else []
    if old:
        first, last = date.fromisoformat(old["from"]), date.fromisoformat(old["till"])
        if start < first:
            ranges.append((start, first - timedelta(days=1)))
        if end > last:
            ranges.append((last + timedelta(days=1), end))
    rows = list(old["rows"]) if old else []
    for first, last in ranges:
        rows += download(first, last, transport).to_dict("records")
    result = dict(
        schema=1,
        source=URL,
        rows=sorted(rows, key=lambda r: r["date"]),
        observed_at=str(pd.Timestamp.now(tz="UTC")),
        **{
            "from": min(str(start), old["from"] if old else str(start)),
            "till": max(str(end), old["till"] if old else str(end)),
        },
    )
    result["sha256"] = hashlib.sha256(
        json.dumps(result["rows"], sort_keys=True).encode()
    ).hexdigest()
    atomic_write(path, json.dumps(result, indent=2).encode())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Incremental official CBR key-rate cache")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="Europe/Moscow").date()))
    parser.add_argument("--output", type=Path, default=Path("results/cbr/key_rate.json"))
    args = parser.parse_args(argv)
    try:
        result = update(args.output, args.start, args.end)
        print(json.dumps(dict(status="available", rows=len(result["rows"]))))
        return 0
    except (ValueError, OSError, ET.ParseError) as error:
        print(json.dumps(dict(status="unavailable", error=str(error))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
