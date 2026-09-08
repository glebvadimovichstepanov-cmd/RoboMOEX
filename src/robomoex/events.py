"""Corporate-event and dividend feed adapters with atomic local caching."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .cache import atomic_write

DIVIDEND_URL = "https://stocksru.ru/dividends/dividends_sngs/"


def download_dividends(url: str = DIVIDEND_URL, transport=urlopen) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "RoboMOEX/0.1 (+research)"})
    with transport(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    rows = []
    for table_row in re.findall(r"<tr.*?</tr>", html, re.IGNORECASE | re.DOTALL):
        dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", table_row)
        amounts = re.findall(r"([0-9]+(?:[.,][0-9]+)?)\s*RUB", table_row, re.IGNORECASE)
        if not dates or not amounts:
            continue
        rows.append(
            {
                "event_date": pd.to_datetime(dates[0], dayfirst=True).date(),
                "event_type": "DIVIDEND_DATE_UNVERIFIED",
                "ticker": "SNGS",
                "amount_rub": float(amounts[0].replace(",", ".")),
                "source_url": url,
                "observed_at": str(pd.Timestamp.now(tz="UTC")),
                "date_verified": False,
            }
        )
    if not rows:
        raise ValueError("Dividend feed returned no parseable events")
    return (
        pd.DataFrame(rows).drop_duplicates(["event_date", "event_type"]).sort_values("event_date")
    )


def save_event_cache(path: Path, frame: pd.DataFrame, identity: dict) -> None:
    data = frame.copy()
    required = {"event_date", "event_type", "ticker"}
    if data.empty or required - set(data):
        raise ValueError("Event cache requires event_date, event_type and ticker")
    data["event_date"] = pd.to_datetime(data["event_date"]).dt.date
    data = data.sort_values(["event_date", "event_type"]).drop_duplicates(
        ["event_date", "event_type", "ticker", "observed_at"], keep="last"
    )
    csv = data.to_csv(index=False, lineterminator="\n")
    payload = {
        "schema": 1,
        "identity": identity,
        "sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "csv": csv,
    }
    atomic_write(path, json.dumps(payload, sort_keys=True).encode())


def load_event_cache(path: Path, identity: dict) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1 or payload.get("identity") != identity:
        raise ValueError("Event cache identity/schema mismatch")
    if hashlib.sha256(payload["csv"].encode()).hexdigest() != payload["sha256"]:
        raise ValueError("Event cache checksum mismatch")
    data = pd.read_csv(io.StringIO(payload["csv"]))
    data["event_date"] = pd.to_datetime(data["event_date"]).dt.date
    return data


def update_dividend_cache(path: Path) -> pd.DataFrame:
    identity = {"provider": "stocksru-dividends-v2", "ticker": "SNGS"}
    fresh = download_dividends()
    if path.exists():
        old = load_event_cache(path, identity)
        # Preserve first observation for unchanged records. Never backdate revisions.
        keys = ["event_date", "event_type", "ticker", "amount_rub"]
        known = set(old[keys].itertuples(index=False, name=None))
        fresh = fresh[
            [tuple(row) not in known for row in fresh[keys].itertuples(index=False, name=None)]
        ]
        fresh = pd.concat([old, fresh], ignore_index=True)
    save_event_cache(path, fresh, identity)
    return fresh


def event_context(events: pd.DataFrame) -> pd.DataFrame:
    """Unverified web dates must never become retrospective ex-dividend signals."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(events.event_date).dt.date,
            "event_penalty": 0.0,
            "event_risk": "UNKNOWN",
            "event_type": events.event_type.to_numpy(),
        }
    ).drop_duplicates("date")
