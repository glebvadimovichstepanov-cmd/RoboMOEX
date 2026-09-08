"""Safe CLI. Live fails before reading files, tokens, models or network data."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .cache import (
    atomic_write,
    load_incremental_cache,
    merge_bars,
    save_incremental_cache,
    save_timeframe_caches,
)
from .calendar import download_stock_calendar
from .config import load_config
from .data import assert_fresh, closed_minutes, utc, validate_sessions
from .demo import dataset
from .engine import simulate
from .moex import download_minutes
from .strategy import signals


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="RoboMOEX: safe research and paper replay")
    result.add_argument("--mode", choices=["backtest", "paper", "live"], default="paper")
    source = result.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Minute OHLCV CSV")
    source.add_argument("--download", action="store_true", help="Read public MOEX ISS candles")
    source.add_argument("--demo", action="store_true", help="Synthetic deterministic replay")
    result.add_argument("--sessions", type=Path, help="Explicit session schedule CSV")
    result.add_argument(
        "--calendar-open", default="09:50:00", help="MOEX calendar session start (Moscow time)"
    )
    result.add_argument(
        "--calendar-close", default="18:50:00", help="MOEX calendar session end (Moscow time)"
    )
    result.add_argument("--symbol", default="SBER")
    result.add_argument("--board", default="TQBR")
    result.add_argument("--start", help="Timezone-aware start for download")
    result.add_argument("--as-of", help="Historical cutoff; backtest only")
    result.add_argument("--cache", type=Path, help="Immutable historical download cache")
    result.add_argument("--config", type=Path)
    result.add_argument("--output", type=Path, default=Path("results/latest"))
    return result


def run(args: argparse.Namespace) -> dict:
    if args.mode == "live":
        raise ValueError("LIVE_DISABLED: broker lifecycle and risk controls are stages 3-4")
    if args.mode == "paper" and args.as_of:
        raise ValueError("paper uses the current clock; historical --as-of requires backtest")
    strategy, execution = load_config(args.config)
    now = pd.Timestamp(datetime.now(UTC))
    demo = args.demo or not (args.input or args.download)
    if demo:
        if args.sessions or args.start or args.cache or args.as_of:
            raise ValueError("Demo cannot be combined with external data options")
        bars, schedule = dataset()
        cutoff = utc(bars.close_time.iloc[-1])
        source = "SYNTHETIC_DEMO"
    else:
        if args.sessions is None and not args.download:
            raise ValueError("--sessions is required for --input")
        schedule = (
            validate_sessions(pd.read_csv(args.sessions)) if args.sessions is not None else None
        )
        cutoff = utc(args.as_of) if args.as_of else now
        if args.download:
            if not args.start:
                raise ValueError("--start is required for download")
            query = {
                "symbol": args.symbol,
                "board": args.board,
                "provider": "moex-iss-v1",
            }
            if args.cache and args.mode == "paper":
                raise ValueError("Historical cache cannot be used for current paper input")
            cache_hit = bool(args.cache and args.cache.exists())
            if cache_hit and args.cache:
                bars, cached_start, cached_end = load_incremental_cache(args.cache, query)
                requested_start = utc(args.start)
                chunks = [bars]
                if requested_start < cached_start:
                    chunks.insert(
                        0,
                        download_minutes(
                            args.symbol, requested_start, cached_start, board=args.board
                        ),
                    )
                if cutoff > cached_end:
                    chunks.append(
                        download_minutes(args.symbol, cached_end, cutoff, board=args.board)
                    )
                bars = merge_bars(*chunks)
            else:
                bars = download_minutes(args.symbol, args.start, cutoff, board=args.board)
            if schedule is None:
                schedule = download_stock_calendar(
                    args.start,
                    cutoff,
                    open_at=args.calendar_open,
                    close_at=args.calendar_close,
                )
            # Validation happens before persisting any downloaded dataset.
            bars = closed_minutes(bars, schedule, cutoff)
            if args.cache:
                save_incremental_cache(args.cache, bars, query)
                save_timeframe_caches(args.cache, bars, schedule)
            source = "MOEX_ISS"
        else:
            if args.start or args.cache:
                raise ValueError("--start/--cache apply only to --download")
            bars = pd.read_csv(args.input, float_precision="round_trip")
            if args.mode == "backtest" and not args.as_of:
                cutoff = utc(bars.close_time.iloc[-1])
            source = "USER_CSV"
    bars = closed_minutes(bars, schedule, cutoff)
    if args.mode == "paper" and not demo:
        assert_fresh(bars, schedule, now)
    decisions = signals(bars, schedule, strategy)
    outcome = simulate(bars, decisions, execution, strategy)
    data_csv = bars.to_csv(index=False, lineterminator="\n")
    session_csv = schedule.to_csv(index=False, lineterminator="\n")
    report = {
        "version": __version__,
        "mode": args.mode,
        "execution": "SIMULATED_REPLAY",
        "source": source,
        "symbol": "SYNTHETIC" if demo else args.symbol,
        "as_of": str(cutoff),
        "last_closed_bar": str(bars.close_time.iloc[-1]),
        "data_sha256": hashlib.sha256(data_csv.encode()).hexdigest(),
        "sessions_sha256": hashlib.sha256(session_csv.encode()).hexdigest(),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "strategy": asdict(strategy),
        "execution_config": asdict(execution),
        "bars": len(bars),
        "entry_signals": int(decisions.enter.sum()),
        "metrics": outcome.metrics(execution.initial_cash),
    }
    atomic_write(
        args.output / "report.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode(),
    )
    atomic_write(
        args.output / "trades.json", json.dumps(outcome.trades, indent=2, allow_nan=False).encode()
    )
    atomic_write(args.output / "equity.csv", outcome.equity.to_csv(index=False).encode())
    return report


def main(argv=None) -> int:
    try:
        report = run(parser().parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"RoboMOEX: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
