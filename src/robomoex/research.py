"""Reproducible daily signal replay. Missing context blocks new entries."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write, load_cache
from .context import SOURCES, load_context_cache, update_all
from .signal_engine import CompositeConfig, RiskLimits, build_features, make_signal, position_size


def replay(frame, config=None, fee=0.001, slippage=0.0005, lot=100, risk=None):
    """Fill close decisions at next open; stops use adverse gap prices; mark daily equity."""
    if frame.empty or not 0 <= fee < 1 or not 0 <= slippage < 1 or lot < 1:
        raise ValueError("Invalid replay input")
    config, risk = config or CompositeConfig(), risk or RiskLimits()
    cash, quantity, pending, stop, entry, entry_fee = 100000.0, 0, None, 0.0, 0.0, 0.0
    held, peak, month_peak, month, halted = 0, cash, cash, None, False
    trades, curve = [], []
    turnover, monthly_halted = 0.0, False
    alerts = []
    receivables, entry_dividends = [], 0.0
    last_spread = slippage * 20000
    for row in frame.itertuples(index=False):
        data = pd.Series(row._asdict())
        opening_time = pd.Timestamp(row.open_time)
        cash += sum(amount for payment, amount in receivables if payment <= opening_time)
        receivables = [
            (payment, amount) for payment, amount in receivables if payment > opening_time
        ]
        split = float(data.get("split_ratio", 1.0))
        if split != 1 and quantity:
            if not float(quantity * split).is_integer():
                raise ValueError("Fractional split entitlement needs verified cash-in-lieu")
            quantity = int(quantity * split)
            entry, stop = entry / split, stop / split
        dividend = float(data.get("dividend_cash_per_share", 0.0)) * quantity
        if dividend:
            payment = pd.Timestamp(data["dividend_payment_at"])
            if payment.tzinfo is None:
                raise ValueError("Dividend payment time needs timezone")
            if payment <= opening_time:
                cash += dividend
            else:
                receivables.append((payment, dividend))
            entry_dividends += dividend
        outstanding = sum(amount for _, amount in receivables)
        current_month = pd.Timestamp(row.open_time).tz_convert("Europe/Moscow").strftime("%Y-%m")
        opening_equity = cash + quantity * row.open + outstanding
        if current_month != month:
            month, month_peak = current_month, opening_equity
            monthly_halted = False
        prior_equity = curve[-1]["equity"] if curve else 100000.0
        daily_breach = opening_equity < prior_equity * (1 - risk.max_daily_loss)
        halted = halted or opening_equity < peak * (1 - risk.max_drawdown_stop)
        monthly_halted = monthly_halted or opening_equity < month_peak * (
            1 - risk.max_monthly_drawdown
        )
        monthly_breach = monthly_halted
        risk_exit = daily_breach or monthly_breach or halted
        if (
            pending
            and pending[0] == "BUY"
            and not quantity
            and not risk_exit
            and not data.get("open_entry_block", False)
        ):
            _, atr, volatility, spread, known_turnover = pending
            atr /= split
            effective_slippage = max(slippage, spread / 20000)
            entry = row.open * (1 + effective_slippage)
            shares = position_size(cash, entry, atr, volatility, risk)
            shares = min(shares, known_turnover * risk.max_participation / entry)
            quantity = int(min(shares, cash / (entry * (1 + fee))) // lot) * lot
            if quantity:
                entry_fee = quantity * entry * fee
                cash -= quantity * entry + entry_fee
                turnover += quantity * entry
                stop, held, entry_dividends = entry - risk.stop_atr * atr, 0, 0.0
        exit_reason, raw_exit = None, None
        if quantity:
            held += 1
            risk_floor = max(
                prior_equity * (1 - risk.max_daily_loss),
                month_peak * (1 - risk.max_monthly_drawdown),
                peak * (1 - risk.max_drawdown_stop),
            )
            # Account-level floor evaluated using current holdings; fill remains adverse at gaps.
            floor_price = (risk_floor - cash - outstanding) / quantity
            active_stop = max(stop, floor_price)
            if row.open <= stop:
                exit_reason, raw_exit = "stop_gap", row.open
            elif risk_exit or (pending and pending[0] == "EXIT"):
                exit_reason, raw_exit = "risk" if risk_exit else "signal_or_time", row.open
            elif row.low <= active_stop:
                exit_reason, raw_exit = (
                    ("account_risk", min(row.open, active_stop))
                    if floor_price > stop
                    else ("stop", stop)
                )
            if raw_exit is not None:
                price = raw_exit * (1 - max(slippage, last_spread / 20000))
                pnl = (
                    quantity * (price - entry)
                    - entry_fee
                    - quantity * price * fee
                    + entry_dividends
                )
                cash += quantity * price * (1 - fee)
                turnover += quantity * price
                trades.append(
                    dict(
                        exit_time=str(
                            row.close_time
                            if exit_reason in {"stop", "account_risk"}
                            else row.open_time
                        ),
                        exit_time_precision="bar"
                        if exit_reason in {"stop", "account_risk"}
                        else "open",
                        quantity=quantity,
                        net_pnl=pnl,
                        holding_bars=held,
                        reason=exit_reason,
                        gross_dividends=entry_dividends,
                    )
                )
                quantity = 0
        equity = cash + quantity * row.close + outstanding
        peak, month_peak = max(peak, equity), max(month_peak, equity)
        halted = halted or equity < peak * (1 - risk.max_drawdown_stop)
        daily_breach = (
            daily_breach
            or equity < prior_equity * (1 - risk.max_daily_loss)
            or exit_reason == "account_risk"
        )
        monthly_halted = monthly_halted or equity < month_peak * (1 - risk.max_monthly_drawdown)
        monthly_breach = monthly_halted
        if daily_breach or monthly_breach or halted or exit_reason == "account_risk":
            alerts.append(
                dict(
                    timestamp=str(row.close_time),
                    daily_breach=bool(daily_breach),
                    monthly_breach=bool(monthly_breach),
                    halted=bool(halted),
                    reason=exit_reason or "drawdown_limit",
                )
            )
        signal = make_signal(
            data, equity=equity, position="LONG" if quantity else "FLAT", config=config, risk=risk
        )
        pending = None
        if quantity:
            if np.isfinite(row.atr14):
                stop = max(stop, row.close - risk.trailing_atr * row.atr14)
            if (
                signal["action"] == "EXIT"
                or held >= risk.time_stop_bars
                or equity < prior_equity * (1 - risk.max_daily_loss)
                or equity < month_peak * (1 - risk.max_monthly_drawdown)
                or equity < peak * (1 - risk.max_drawdown_stop)
            ):
                pending = ("EXIT",)
        elif signal["action"] == "BUY" and not halted and not monthly_breach and not daily_breach:
            pending = ("BUY", row.atr14, row.realized_vol20, data.spread_bps, data.avg_turnover20)
        curve.append(
            dict(
                close_time=str(row.close_time),
                equity=equity,
                quantity=quantity,
                signal=signal["action"],
            )
        )
        if np.isfinite(data.get("spread_bps", np.nan)):
            last_spread = max(0.0, float(data.spread_bps))
    # Explicit terminal liquidation convention, charged both slippage and commission.
    if quantity:
        price = float(frame.close.iloc[-1]) * (1 - max(slippage, last_spread / 20000))
        pnl = quantity * (price - entry) - entry_fee - quantity * price * fee + entry_dividends
        cash += quantity * price * (1 - fee)
        turnover += quantity * price
        trades.append(
            dict(
                exit_time=str(frame.close_time.iloc[-1]),
                quantity=quantity,
                net_pnl=pnl,
                holding_bars=held,
                reason="terminal_liquidation",
                gross_dividends=entry_dividends,
            )
        )
        curve[-1]["equity"], curve[-1]["quantity"] = (
            cash + sum(amount for _, amount in receivables),
            0,
        )
    values = np.array([100000.0] + [x["equity"] for x in curve])
    returns = np.diff(values) / values[:-1]
    pnl = np.array([t["net_pnl"] for t in trades])
    std = returns.std(ddof=1) if len(returns) > 1 else 0
    metrics = dict(
        trades=len(trades),
        total_return=float(values[-1] / values[0] - 1),
        max_drawdown=float((1 - values / np.maximum.accumulate(values)).max()),
        sharpe_daily_252=float(returns.mean() / std * np.sqrt(252)) if std > 0 else None,
        win_rate=float((pnl > 0).mean()) if len(pnl) else None,
        profit_factor=float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()) if (pnl < 0).any() else None,
        average_holding_bars=float(np.mean([t["holding_bars"] for t in trades]))
        if trades
        else None,
        traded_notional=turnover,
        turnover_on_initial_equity=turnover / 100000,
        ending_dividend_receivable=sum(amount for _, amount in receivables),
    )
    for name in ("imoex", "oil"):
        benchmark = (
            frame[name].pct_change(fill_method=None).to_numpy()
            if name in frame
            else np.full(len(returns), np.nan)
        )
        valid = np.isfinite(benchmark) & np.isfinite(returns)
        variance = np.var(benchmark[valid], ddof=1) if valid.sum() > 20 else 0
        metrics[f"beta_{name}"] = (
            float(np.cov(returns[valid], benchmark[valid], ddof=1)[0, 1] / variance)
            if variance > 0
            else None
        )
    return dict(metrics=metrics, trades=trades, equity=curve, alerts=alerts)


def walk_forward(features, train=260, test=60, holdout=60):
    if min(train, test, holdout) <= 0 or len(features) < train + test + holdout:
        raise ValueError("Insufficient observations for train/test/holdout")
    boundary = len(features) - holdout

    def calibrate(history):
        candidates = []
        for threshold in (0.25, 0.35, 0.50, 0.60):
            config = replace(CompositeConfig(), buy_threshold=threshold)
            result = replay(history, config)["metrics"]
            if result["trades"] >= 3:
                candidates.append((result["total_return"], threshold))
        if not candidates:
            return CompositeConfig(), "insufficient_training_trades"
        return replace(CompositeConfig(), buy_threshold=max(candidates)[1]), "calibrated"

    windows = []
    for start in range(train, boundary, test):
        config, status = calibrate(features.iloc[:start])
        sample = features.iloc[start : min(start + test, boundary)]
        windows.append(
            dict(
                start=str(sample.open_time.iloc[0]),
                end=str(sample.close_time.iloc[-1]),
                threshold=config.buy_threshold,
                calibration=status,
                **replay(sample, config),
            )
        )
    config, status = calibrate(features.iloc[:boundary])
    final = replay(features.iloc[boundary:], config)
    return dict(
        windows=windows,
        holdout=dict(threshold=config.buy_threshold, calibration=status, **final),
        holdout_caveat=(
            "Chronologically separated; history was inspected in prior research. "
            "Not a pristine holdout."
        ),
    )


def diagnostics(features):
    sensitivity = []
    for fee, slip, stop in (
        (0.001, 0.0005, 2.5),
        (0.002, 0.001, 2.5),
        (0.001, 0.0005, 2.0),
        (0.001, 0.0005, 3.0),
    ):
        result = replay(features, fee=fee, slippage=slip, risk=replace(RiskLimits(), stop_atr=stop))
        sensitivity.append(dict(fee=fee, slippage=slip, stop_atr=stop, **result["metrics"]))
    years = pd.to_datetime(features.close_time, utc=True).dt.year
    periods = {}
    for label, first, last in (
        ("pre2020", 1900, 2019),
        ("2020", 2020, 2020),
        ("2021", 2021, 2021),
        ("2022", 2022, 2022),
        ("2023-2024", 2023, 2024),
        ("2025+", 2025, 9999),
    ):
        sample = features[years.between(first, last)]
        periods[label] = (
            dict(status="NO_DATA", rows=0)
            if sample.empty
            else dict(status="RESEARCH", rows=len(sample), metrics=replay(sample)["metrics"])
        )
    return dict(
        sensitivity=sensitivity,
        periods=periods,
        caveat="No trades means sensitivity and profitability are not established.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily SNGS research with explicit data readiness")
    parser.add_argument("--daily", type=Path, default=Path("results/cache/SNGS.TQBR.1d.json"))
    parser.add_argument("--context", type=Path, default=Path("results/context_cache_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/research_v2"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--authenticated", action="store_true")
    parser.add_argument("--events", type=Path, help="Verified point-in-time corporate event JSON")
    parser.add_argument("--cbr-cache", type=Path, default=Path("results/cbr/key_rate.json"))
    parser.add_argument("--ml", action="store_true", help="Evaluate optional purged ridge model")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--algopack-db", type=Path, help="Use cached historical microstructure factors"
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="Europe/Moscow").date()))
    args = parser.parse_args(argv)
    try:
        if args.refresh:
            from .algopack import get_authenticated_json
            from .moex import get_json

            update_all(
                args.context,
                args.start,
                args.end,
                get_authenticated_json if args.authenticated else get_json,
            )
            from .cbr import update as update_rates
            from .disclosure import refresh as update_disclosures

            source_status = {"disclosure": update_disclosures(Path("results/disclosure"))}
            try:
                source_status["cbr"] = dict(
                    status="available",
                    rows=len(update_rates(args.cbr_cache, args.start, args.end)["rows"]),
                )
            except (ValueError, OSError) as error:
                source_status["cbr"] = dict(status="unavailable", error=str(error))
            atomic_write(args.output / "source_refresh.json", json.dumps(source_status).encode())
        payload = json.loads(args.daily.read_text(encoding="utf8"))
        daily = load_cache(args.daily, payload["query"])
        cutoff = pd.Timestamp(args.end, tz="Europe/Moscow")
        daily = daily[
            (daily.close_time <= cutoff)
            & (daily.open_time >= pd.Timestamp(args.start, tz="Europe/Moscow"))
        ]
        context_path = args.context / "sngs_context_v2.json"
        context, _, _ = load_context_cache(
            context_path, dict(provider="robomoex-context-v2", sources=[s.name for s in SOURCES])
        )
        if args.cbr_cache.is_file():
            rates = json.loads(args.cbr_cache.read_text(encoding="utf-8"))
            if (
                rates.get("sha256")
                != hashlib.sha256(json.dumps(rates["rows"], sort_keys=True).encode()).hexdigest()
            ):
                raise ValueError("CBR cache checksum mismatch")
            rate_frame = pd.DataFrame(rates["rows"])
            rate_frame["date"] = pd.to_datetime(rate_frame.date).dt.date
            context = context.drop(columns=["rate"], errors="ignore").merge(
                rate_frame, on="date", how="left", validate="one_to_one"
            )
        events = None
        if args.events:
            from .corporate_actions import load_events

            events = load_events(args.events)
        features = build_features(daily, context, corporate_actions=events)
        if events is not None:
            from .corporate_actions import execution_actions, flags

            features = execution_actions(features, events)

            sessions = (
                pd.to_datetime(features.open_time).dt.tz_convert("Europe/Moscow").dt.date.tolist()
            )
            event_flags = pd.DataFrame(
                [flags(events, timestamp, sessions=sessions) for timestamp in features.close_time]
            )
            for name in event_flags:
                features[name] = event_flags[name].to_numpy()
            features["open_entry_block"] = [
                any(flags(events, timestamp, sessions=sessions).values())
                for timestamp in features.open_time
            ]
        algopack_info = None
        if args.algopack_db:
            if not args.algopack_db.is_file():
                raise ValueError("AlgoPack cache does not exist")
            from .algopack_cache import connect
            from .algopack_features import attach_features, daily_features

            with connect(args.algopack_db) as db:
                factors = daily_features(db)
            features = attach_features(features, factors)
            atomic_write(args.output / "algopack_factors.csv", factors.to_csv(index=False).encode())
            algopack_info = dict(
                daily_factor_rows=len(factors),
                rows_with_spread=int(
                    features.get("spread_bps", pd.Series(dtype=float)).notna().sum()
                ),
                factors_sha256=hashlib.sha256(factors.to_csv(index=False).encode()).hexdigest(),
                historical_revision_caveat=True,
            )
        report = dict(
            status="RESEARCH_ONLY",
            trading_ready=False,
            blockers=[
                (
                    "native daily labels are not exact fill timestamps"
                    if payload.get("query", {}).get("provider") == "MOEX ISS"
                    else "unverified daily aggregation/session coverage"
                ),
                "daily spread proxy is not executable bid/ask at fill time",
                "missing point-in-time corporate/news coverage",
                "dividend cashflows not validated",
                "current SNGS lot size 100; historical lot-size changes not verified",
            ],
            data_sha256=hashlib.sha256(args.daily.read_bytes()).hexdigest(),
            context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
            algopack=algopack_info,
            results=walk_forward(features),
        )
        if args.ml:
            from .model_validation import evaluate

            report["model_validation"] = evaluate(features)
        if args.diagnostics:
            report["diagnostics"] = diagnostics(features)
        from .operations import Journal

        with Journal(args.output / "operations") as journal_log:
            journal_log.emit(
                "research_completed", rows=len(features), windows=len(report["results"]["windows"])
            )
            journal_log.emit("trading_blocked", "WARNING", blockers=report["blockers"])
        journal = [make_signal(row, equity=100000) for _, row in features.iterrows()]
        atomic_write(
            args.output / "signals.jsonl",
            "\n".join(json.dumps(s, allow_nan=False) for s in journal).encode(),
        )
        atomic_write(args.output / "features.csv", features.to_csv(index=False).encode())
        atomic_write(
            args.output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode()
        )
        print(
            json.dumps(
                dict(
                    status=report["status"],
                    trading_ready=False,
                    report=str(args.output / "report.json"),
                    windows=len(report["results"]["windows"]),
                    holdout=report["results"]["holdout"]["metrics"],
                )
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(error=str(error), trading_ready=False)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
