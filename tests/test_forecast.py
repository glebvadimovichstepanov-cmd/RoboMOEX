import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robomoex.forecast import future_forecast, verify_artifact


class Predictor:
    def __init__(self, shift=0):
        self.shift = shift
        self.seen = None

    def predict(self, dataset):
        self.seen = dataset[0]
        yield SimpleNamespace(
            start_date=self.seen["start"] + len(self.seen["target"]) + self.shift,
            samples=np.array([[0.1, -0.1], [-0.1, 0.1]]),
        )


def context():
    return pd.Series(
        [0.01, -0.02, 0.03, 0.04],
        index=pd.date_range("2025-01-06T10:00:00Z", periods=4, freq="min"),
    )


def test_entire_context_is_used_and_paths_transformed_before_quantiles():
    predictor = Predictor()
    result = future_forecast(predictor, context(), 100, frequency="min", steps=2)
    np.testing.assert_allclose(predictor.seen["target"], context().values)
    assert result["start"] == "2025-01-06 10:04"
    # Both paths return to 100; summing per-step quantiles would produce a false interval.
    assert result["q10"][-1] == pytest.approx(100)
    assert result["q90"][-1] == pytest.approx(100)
    assert result["mean"][-1] == pytest.approx(100)


def test_historical_evaluation_window_is_rejected():
    with pytest.raises(ValueError, match="strictly after"):
        future_forecast(Predictor(shift=-2), context(), 100, frequency="min", steps=2)


def test_closure_gap_never_filled():
    with pytest.raises(ValueError, match="Irregular"):
        future_forecast(
            Predictor(), context().drop(context().index[1]), 100, frequency="min", steps=2
        )


def test_checksum_before_model_loading(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"test fixture, not a model")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert verify_artifact(artifact, digest) == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact(artifact, digest)
