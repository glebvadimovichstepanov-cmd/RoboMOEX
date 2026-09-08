"""Optional ridge return model with chronological purging and train-only preprocessing."""

import numpy as np

FEATURES = (
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "rsi14",
    "realized_vol20",
    "volume_ratio20",
    "relative_imoex20",
    "relative_peers20",
    "cny_rub_ret5",
)


def purged_windows(length, train=260, test=60, horizon=5, embargo=2):
    if min(train, test, horizon) <= 0 or embargo < 0:
        raise ValueError("Invalid validation lengths")
    for boundary in range(train, length, test):
        # A label ending at/after the embargo boundary must not enter training.
        yield (
            np.arange(max(0, boundary - horizon - embargo)),
            np.arange(boundary, min(length, boundary + test)),
        )


def fit_ridge(x, y, alpha=1.0):
    lo, hi = np.quantile(x, [0.01, 0.99], axis=0)
    clipped = np.clip(x, lo, hi)
    center = np.median(clipped, axis=0)
    scale = np.quantile(clipped, 0.75, axis=0) - np.quantile(clipped, 0.25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (clipped - center) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return dict(lo=lo, hi=hi, center=center, scale=scale, coefficients=coefficients)


def predict(model, x):
    z = (np.clip(x, model["lo"], model["hi"]) - model["center"]) / model["scale"]
    return np.column_stack([np.ones(len(z)), z]) @ model["coefficients"]


def evaluate(frame, train=260, test=60, horizon=5, embargo=2):
    names = [name for name in FEATURES if name in frame]
    if not names:
        return dict(status="RULES_FALLBACK", reason="no_features", folds=[])
    x = frame[names].to_numpy(dtype=float)
    price = frame.get("signal_close", frame.close)
    target = (price.shift(-horizon) / price - 1).to_numpy()
    folds = []
    for training, testing in purged_windows(len(frame), train, test, horizon, embargo):
        valid_train = training[np.isfinite(x[training]).all(axis=1) & np.isfinite(target[training])]
        valid_test = testing[np.isfinite(x[testing]).all(axis=1) & np.isfinite(target[testing])]
        if len(valid_train) < 100 or len(valid_test) < 10:
            folds.append(
                dict(
                    status="RULES_FALLBACK",
                    reason="insufficient_complete_observations",
                    train_rows=len(valid_train),
                    test_rows=len(valid_test),
                )
            )
            continue
        model = fit_ridge(x[valid_train], target[valid_train])
        estimates = predict(model, x[valid_test])
        truth = target[valid_test]
        mse = float(np.mean((estimates - truth) ** 2))
        baseline = float(np.mean((target[valid_train].mean() - truth) ** 2))
        drift = np.abs((np.mean(x[valid_test], axis=0) - model["center"]) / model["scale"])
        folds.append(
            dict(
                status="EVALUATED",
                train_last=int(valid_train[-1]),
                test_first=int(valid_test[0]),
                test_last=int(valid_test[-1]),
                mse=mse,
                baseline_mse=baseline,
                improves_baseline=mse < baseline,
                coefficients=dict(zip(names, model["coefficients"][1:].tolist(), strict=True)),
                feature_drift_iqr=dict(zip(names, drift.tolist(), strict=True)),
                predictions=[
                    dict(row=int(i), expected_return=float(p))
                    for i, p in zip(valid_test, estimates, strict=True)
                ],
            )
        )
    valid = [fold for fold in folds if fold["status"] == "EVALUATED"]
    return dict(
        status="RESEARCH_ONLY" if valid else "RULES_FALLBACK",
        features=names,
        horizon=horizon,
        embargo=embargo,
        folds=folds,
        trading_enabled=False,
        caveat="Overlapping labels; dependent fold metrics. Model cannot override risk gates.",
    )


def safe_prediction(model, values):
    """Explicit fallback for missing features, invalid models and numerical failures."""
    try:
        array = np.asarray(values, dtype=float).reshape(1, -1)
        result = predict(model, array)[0]
        if not np.isfinite(array).all() or not np.isfinite(result):
            raise ValueError("Non-finite model input/output")
        return dict(status="MODEL_ESTIMATE", expected_return=float(result))
    except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError):
        return dict(status="RULES_FALLBACK", expected_return=None)
