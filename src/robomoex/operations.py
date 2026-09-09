"""Local structured logs and durable alert outbox; no network notifications."""

import hashlib
import json
import logging
import re
import sqlite3
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(part in key.lower() for part in ("token", "secret", "authorization", "password"))
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
        return re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


class Journal:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.Logger("robomoex.operations")
        self.handler = RotatingFileHandler(
            directory / "runtime.jsonl", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        self.logger.addHandler(self.handler)
        self.db = sqlite3.connect(directory / "alerts.sqlite")
        self.db.execute("""CREATE TABLE IF NOT EXISTS alerts (
            fingerprint TEXT PRIMARY KEY, first_seen TEXT, last_seen TEXT,
            occurrences INTEGER, payload TEXT, acknowledged INTEGER DEFAULT 0)""")
        self.db.commit()

    def emit(self, event, severity="INFO", **fields):
        timestamp = pd.Timestamp.now(tz="UTC").isoformat()
        payload = redact(dict(event=event, severity=severity, **fields))
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        self.logger.log(
            getattr(logging, severity),
            json.dumps(dict(timestamp=timestamp, **payload), ensure_ascii=False, allow_nan=False),
        )
        if severity in {"WARNING", "ERROR", "CRITICAL"}:
            key = hashlib.sha256(raw.encode()).hexdigest()
            with self.db:
                self.db.execute(
                    """INSERT INTO alerts VALUES (?,?,?,?,?,0)
                    ON CONFLICT(fingerprint) DO UPDATE SET last_seen=excluded.last_seen,
                    occurrences=alerts.occurrences+1, acknowledged=0""",
                    (key, timestamp, timestamp, 1, raw),
                )

    def close(self):
        self.handler.close()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
