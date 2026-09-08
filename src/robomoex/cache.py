"""Content-checked, atomic local cache. Corrupt cache is an error, not market data."""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .data import aggregate, utc, validate_bars


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


def load_incremental_cache(
    path: Path, identity: dict
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load a date-ranged cache whose query identity is stable across extensions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == 1:
        old_query = payload.get("query", {})
        if any(old_query.get(key) != value for key, value in identity.items()):
            raise ValueError("Incremental cache identity/schema mismatch")
        if hashlib.sha256(payload["csv"].encode()).hexdigest() != payload["sha256"]:
            raise ValueError("Cache checksum mismatch")
        bars = validate_bars(pd.read_csv(io.StringIO(payload["csv"]), float_precision="round_trip"))
        return bars, bars.open_time.iloc[0], bars.close_time.iloc[-1]
    if payload.get("schema") != 2 or payload.get("identity") != identity:
        raise ValueError("Incremental cache identity/schema mismatch")
    if hashlib.sha256(payload["csv"].encode()).hexdigest() != payload["sha256"]:
        raise ValueError("Cache checksum mismatch")
    bars = validate_bars(pd.read_csv(io.StringIO(payload["csv"]), float_precision="round_trip"))
    return bars, utc(payload["range_start"]), utc(payload["range_end"])


def save_incremental_cache(path: Path, bars: pd.DataFrame, identity: dict) -> None:
    data = validate_bars(bars)
    csv = data.to_csv(index=False, lineterminator="\n")
    payload = {
        "schema": 2,
        "identity": identity,
        "range_start": str(data.open_time.iloc[0]),
        "range_end": str(data.close_time.iloc[-1]),
        "sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "csv": csv,
    }
    atomic_write(path, json.dumps(payload, sort_keys=True).encode())


def merge_bars(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge downloaded chunks, rejecting conflicting duplicate timestamps."""
    if not frames:
        raise ValueError("At least one bar frame is required")
    data = pd.concat([validate_bars(frame) for frame in frames], ignore_index=True)
    if data.open_time.duplicated().any():
        # Identical overlap is safe; conflicting OHLC is not.
        grouped = data.groupby("open_time", sort=False)
        if grouped["close"].nunique().max() > 1:
            raise ValueError("Conflicting bars in cache merge")
        data = data.drop_duplicates("open_time", keep="first")
    return validate_bars(data.sort_values("open_time", ignore_index=True))


def save_timeframe_caches(
    path: Path, bars: pd.DataFrame, sessions: pd.DataFrame
) -> dict[str, Path]:
    """Persist derived 15m/1h/1d datasets next to the 1m cache."""
    result = {}
    stem = path.with_suffix("")
    for label, minutes in (("15m", 15), ("1h", 60), ("1d", None)):
        derived = aggregate(bars, sessions, minutes)
        target = stem.with_name(f"{stem.name}.{label}.json")
        save_cache(target, derived, {"source": str(path.name), "timeframe": label})
        result[label] = target
    return result

