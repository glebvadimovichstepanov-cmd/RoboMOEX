"""Explicit research ablation and baselines; never a trading-readiness certificate."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write
from .research import replay
from .signal_engine import CompositeConfig, RiskLimits, make_signal


def benchmark(frame, weight=1.0, fee=0.001, slip=0.0005):
    """Buy once at first open, sell final close; price-only, round lots, costs included."""
    capital = 100000.0
    entry = float(frame.open.iloc[0]) * (1 + slip)
    qty = int(capital * weight / (entry * (1 + fee)) / 100) * 100
    cash = capital - qty * entry * (1 + fee)
    values = cash + qty * frame.close.to_numpy()
    values[-1] -= qty * float(frame.close.iloc[-1]) * (slip + fee * (1 - slip))
    values = np.r_[capital, values]
    return dict(
        total_return=float(values[-1] / capital - 1),
        max_drawdown=float((1 - values / np.maximum.accumulate(values)).max()),
        shares=qty,
        dividends_included=False,
    )


def evaluate(features):
    signals = [make_signal(row, equity=100000) for _, row in features.iterrows()]
    warnings = Counter(warning for signal in signals for warning in signal["warnings"])
    missing = Counter(name for signal in signals for name in signal["missing_data"])
    # Predeclared price-only hypothesis, not parameters optimized against test outcomes.
    config = CompositeConfig(
        require_context=False,
        trend_weight=0.55,
        momentum_weight=0.40,
        oil_weight=0,
        fx_weight=0,
        rate_weight=0,
        relative_weight=0,
        liquidity_weight=0.05,
    )
    market = features.copy()
    market["event_risk"], market["event_penalty"] = "LOW", 1.0
    # Explicit assumed cost for this ablation only; actual spread coverage stays in strict report.
    market["spread_bps"] = 10.0
    market["avg_turnover20"] = (market.close * market.volume).shift().rolling(20).mean()
    for flag in (
        "dividend_entry_block",
        "is_dividend_cutoff",
        "open_entry_block",
        "sanctions_event",
        "geopolitical_event",
        "tax_event",
        "opec_event",
        "report_event",
    ):
        market[flag] = False
    windows = []
    boundary = len(market) - 60
    for start in range(260, boundary, 60):
        sample = market.iloc[start : min(start + 60, boundary)]
        windows.append(
            dict(
                start=str(sample.open_time.iloc[0]),
                end=str(sample.close_time.iloc[-1]),
                **replay(sample, config)["metrics"],
            )
        )
    sensitivity = []
    for fee, slip, stop in (
        (0.001, 0.0005, 2.5),
        (0.002, 0.001, 2.5),
        (0.001, 0.0005, 2.0),
        (0.001, 0.0005, 3.0),
    ):
        result = replay(
            market.iloc[260:],
            config,
            fee=fee,
            slippage=slip,
            risk=replace(RiskLimits(), stop_atr=stop),
        )
        sensitivity.append(dict(fee=fee, slippage=slip, stop_atr=stop, **result["metrics"]))
    holdout = market.iloc[boundary:]
    years = pd.to_datetime(market.close_time, utc=True).dt.year
    periods = {
        str(year): replay(market[years == year], config)["metrics"]
        for year in sorted(years.unique())
    }
    return dict(
        status="ABLATION_ONLY",
        trading_ready=False,
        assumptions=[
            "Event and macro filters disabled only in this diagnostic",
            "10 bps spread assumed; daily OHLC fills, no order queue",
            "No verified historical dividends, taxes, borrow or lot changes",
            "Fixed weights 0.55 trend / 0.40 momentum / 0.05 liquidity; threshold 0.50",
            "Independent 100000 RUB per fold; no stitched portfolio claimed",
            "Previously inspected history is not a pristine holdout",
        ],
        strict_exclusions=dict(warnings=dict(warnings), missing=dict(missing)),
        windows=windows,
        holdout=replay(holdout, config),
        sensitivity=sensitivity,
        periods=periods,
        baseline_holdout_full=benchmark(holdout),
        baseline_holdout_5pct=benchmark(holdout, 0.05),
        baseline_oos_full=benchmark(market.iloc[260:]),
        baseline_oos_5pct=benchmark(market.iloc[260:], 0.05),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    for name in ("open_time", "close_time"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if len(frame) < 380:
        raise ValueError("Need at least 380 observations")
    result = evaluate(frame)
    result["features_sha256"] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    atomic_write(args.output, json.dumps(result, indent=2, allow_nan=False).encode())
    print(
        json.dumps(
            dict(
                status=result["status"],
                windows=len(result["windows"]),
                holdout=result["holdout"]["metrics"],
                sensitivity=result["sensitivity"],
            )
        )
    )


if __name__ == "__main__":
    main()
