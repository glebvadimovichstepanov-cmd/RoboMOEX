"""Model-independent future-inference boundary, compatible with GluonTS predictors.

Only regular, explicit time grids are accepted. Exchange gaps require a separate
trading-time model; they must not be filled with invented overnight returns.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import utc


def verify_artifact(path: str | Path, expected_sha256: str) -> Path:
    artifact = Path(path)
    if len(expected_sha256) != 64:
        raise ValueError("A pinned SHA256 is required")
    with artifact.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("Model artifact checksum mismatch")
    return artifact


def future_forecast(
    predictor, log_returns: pd.Series, last_price: float, *, frequency: str, steps: int
) -> dict:
    """Use predictor.predict on the FULL context, never evaluation holdout helpers.

    Predictor construction/loading remains outside the core; no pickle loading,
    remote model download or global torch patch is performed here.
    """
    if log_returns.empty or not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError("Finite nonempty context required")
    if not np.isfinite(last_price) or last_price <= 0 or steps < 1:
        raise ValueError("Invalid price or forecast horizon")
    index = pd.DatetimeIndex([utc(t) for t in log_returns.index])
    expected = pd.date_range(index[0], periods=len(index), freq=frequency)
    if not index.equals(expected):
        raise ValueError("Irregular time grid: do not forward-fill exchange closures")
    start = pd.Period(index[0].tz_localize(None), freq=frequency)
    dataset = [{"start": start, "target": log_returns.to_numpy(dtype=np.float32).copy()}]
    predictions = list(predictor.predict(dataset))
    if len(predictions) != 1:
        raise ValueError("Expected exactly one forecast")
    forecast = predictions[0]
    if forecast.start_date != start + len(index):
        raise ValueError("Forecast must start strictly after the complete context")
    samples = np.asarray(forecast.samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != steps or samples.shape[0] < 2:
        raise ValueError("Expected sample paths shaped [samples, steps]")
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite forecast samples")
    # Transform each whole sample path before computing price quantiles.
    with np.errstate(over="raise", invalid="raise"):
        paths = last_price * np.exp(np.cumsum(samples, axis=1))
    return {
        "start": str(forecast.start_date),
        "steps": steps,
        "mean": paths.mean(axis=0).tolist(),
        "q10": np.quantile(paths, 0.1, axis=0).tolist(),
        "q90": np.quantile(paths, 0.9, axis=0).tolist(),
    }
