"""Content-checked, atomic local cache. Corrupt cache is an error, not market data."""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .data import validate_bars


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def save_cache(path: Path, bars: pd.DataFrame, query: dict) -> None:
    csv = validate_bars(bars).to_csv(index=False, lineterminator="\n")
    payload = {
        "schema": 1,
        "query": query,
        "sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "csv": csv,
    }
    atomic_write(path, json.dumps(payload, sort_keys=True).encode())


def load_cache(path: Path, query: dict) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != 1 or payload["query"] != query:
        raise ValueError("Cache query/schema mismatch")
    if hashlib.sha256(payload["csv"].encode()).hexdigest() != payload["sha256"]:
        raise ValueError("Cache checksum mismatch")
    return validate_bars(pd.read_csv(io.StringIO(payload["csv"]), float_precision="round_trip"))
