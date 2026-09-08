"""Explicit long-only OHLC execution; shared by historical and paper replay."""

from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import numpy as np
import pandas as pd

from .config import ExecutionConfig, StrategyConfig
from .data import validate_bars


def tick_round(value: float, tick: float, *, up: bool) -> float:
    quantum = Decimal(str(tick))
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    return float((Decimal(str(value)) / quantum).to_integral_value(rounding=rounding) * quantum)


@dataclass
class Position:
    entry_time: str
    entry_price: float
    quantity: int
    entry_fee: float
    stop: float
    target: float


@dataclass
class Result:
    equity: pd.DataFrame
    trades: list[dict]
    open_position: dict | None
    rejected_entries: int

    def metrics(self, initial_cash: float) -> dict:
        values = self.equity.equity
        peak = values.cummax().clip(lower=initial_cash)
        drawdown = 1 - values / peak
        daily = (
            pd.Series(
                values.to_numpy(),
                index=pd.DatetimeIndex(self.equity.close_time).tz_convert("Europe/Moscow"),
            )
            .groupby(lambda stamp: stamp.date())
            .last()
        )
        returns = daily.pct_change()
        returns.iloc[0] = daily.iloc[0] / initial_cash - 1
        deviation = returns.std(ddof=1)
        sharpe = (
            float(returns.mean() / deviation * np.sqrt(252))
            if len(returns) >= 2 and pd.notna(deviation) and deviation > 0
            else None
        )
        return {
            "final_equity": float(values.iloc[-1]),
            "total_return": float(values.iloc[-1] / initial_cash - 1),
            "max_drawdown": float(drawdown.max()),
            "sharpe_daily_252": sharpe,
            "closed_trades": len(self.trades),
            "win_rate": (
                sum(t["net_pnl"] > 0 for t in self.trades) / len(self.trades)
                if self.trades
                else None
            ),
            "open_position": self.open_position,
            "rejected_entries": self.rejected_entries,
        }


def simulate(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    execution: ExecutionConfig | None = None,
    strategy: StrategyConfig | None = None,
) -> Result:
    execution = execution or ExecutionConfig()
    strategy = strategy or StrategyConfig()
    data = validate_bars(bars)
    if len(signals) != len(data) or set(("enter", "atr")) - set(signals.columns):
        raise ValueError("Signals must match bars one-for-one")
    if not signals.index.equals(data.index):
        raise ValueError("Signals must use the validated bars' RangeIndex")
    if signals.enter.isna().any() or not pd.api.types.is_bool_dtype(signals.enter):
        raise ValueError("Entry signals must be boolean")
    if ((~np.isfinite(signals.atr) | (signals.atr <= 0)) & signals.enter).any():
        raise ValueError("Entry needs a finite positive ATR")
    cash, position, pending, rejected = execution.initial_cash, None, None, 0
    trades, equity = [], []
    quantity = execution.quantity_lots * execution.lot_size

    def fill(price, buy):
        multiplier = 1 + (1 if buy else -1) * execution.slippage_bps / 10_000
        return tick_round(price * multiplier, execution.tick_size, up=buy)

    for i, bar in enumerate(data.itertuples()):
        if pending is not None and position is None:
            atr, expected_open = pending
            # Signal expires at a discontinuity; never enter using yesterday's signal.
            if bar.open_time == expected_open:
                price = fill(bar.open, True)
                fee = price * quantity * execution.fee_rate
                stop = tick_round(price - strategy.sl_atr_mult * atr, execution.tick_size, up=False)
                target = tick_round(
                    price + strategy.tp_atr_mult * atr, execution.tick_size, up=True
                )
                if cash >= price * quantity + fee and 0 < stop < price < target:
                    cash -= price * quantity + fee
                    position = Position(str(bar.open_time), price, quantity, fee, stop, target)
                else:
                    rejected += 1
            pending = None

        exited = False
        # Protective exits are unconditional: entry filters cannot bypass them.
        if position is not None:
            reason, raw_price = None, None
            if bar.open <= position.stop:
                reason, raw_price = "stop_gap", bar.open
            elif bar.open >= position.target:
                reason, raw_price = "target_gap", bar.open
            elif bar.low <= position.stop:
                # If both levels occur in one OHLC bar, use the adverse (stop) outcome.
                reason, raw_price = "stop", position.stop
            elif bar.high >= position.target:
                reason, raw_price = "target", position.target
            if reason:
                price = fill(raw_price, False)
                fee = price * quantity * execution.fee_rate
                cash += price * quantity - fee
                trades.append(
                    {
                        **asdict(position),
                        "exit_bar_open": str(bar.open_time),
                        "exit_bar_close": str(bar.close_time),
                        "exit_price": price,
                        "exit_fee": fee,
                        "reason": reason,
                        "net_pnl": (price - position.entry_price) * quantity
                        - fee
                        - position.entry_fee,
                    }
                )
                position, exited = None, True

        equity.append(
            {
                "close_time": bar.close_time,
                "cash": cash,
                "equity": cash + (position.quantity * bar.close if position else 0),
            }
        )
        if position is None and not exited and signals.enter.iloc[i]:
            # Decision at close can only fill at the NEXT bar's open.
            pending = (signals.atr.iloc[i], bar.close_time)

    return Result(pd.DataFrame(equity), trades, asdict(position) if position else None, rejected)
