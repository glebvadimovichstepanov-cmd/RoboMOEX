import pandas as pd
import pytest

from robomoex.config import ExecutionConfig, StrategyConfig
from robomoex.engine import simulate, tick_round


def frame(prices):
    index = pd.date_range("2025-01-06T10:00:00+03:00", periods=len(prices), freq="min")
    return pd.DataFrame(
        [
            {
                "open_time": t,
                "close_time": t + pd.Timedelta(minutes=1),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 1000,
            }
            for t, (o, h, lo, c) in zip(index, prices, strict=True)
        ]
    )


def run(prices, entries=None, **kwargs):
    bars = frame(prices)
    sig = pd.DataFrame({"enter": entries or [True] + [False] * (len(bars) - 1), "atr": 2.0})
    execution = ExecutionConfig(
        initial_cash=1000, lot_size=1, quantity_lots=2, fee_rate=0, slippage_bps=0, **kwargs
    )
    return simulate(bars, sig, execution, StrategyConfig(sl_atr_mult=1, tp_atr_mult=2))


def test_entry_next_open_and_exact_pnl():
    result = run([(100, 101, 99, 100), (101, 102, 100, 101), (102, 106, 101, 105)])
    assert result.equity.cash.iloc[0] == 1000
    trade = result.trades[0]
    assert trade["entry_price"] == 101
    assert trade["exit_price"] == 105
    assert trade["net_pnl"] == 8
    assert result.equity.equity.iloc[-1] == 1008


def test_exit_works_when_entry_filter_false():
    result = run([(100, 100, 100, 100), (100, 101, 99, 100), (99, 100, 97, 98)])
    assert result.trades[0]["reason"] == "stop"
    assert result.trades[0]["net_pnl"] == -4


def test_adverse_priority_when_both_levels_hit():
    result = run([(100, 100, 100, 100), (100, 105, 97, 100)])
    assert result.trades[0]["reason"] == "stop"
    assert result.trades[0]["exit_price"] == 98


def test_gap_stop_and_positive_drawdown():
    result = run([(100, 100, 100, 100), (100, 101, 99, 100), (90, 92, 89, 91)])
    assert result.trades[0]["reason"] == "stop_gap"
    assert result.trades[0]["exit_price"] == 90
    assert result.metrics(1000)["max_drawdown"] == pytest.approx(0.02)


def test_fees_and_slippage_accounting():
    bars = frame([(100, 100, 100, 100), (100, 101, 100, 101), (102, 106, 101, 105)])
    result = simulate(
        bars,
        pd.DataFrame({"enter": [True, False, False], "atr": 2.0}),
        ExecutionConfig(
            initial_cash=1000, lot_size=1, quantity_lots=2, fee_rate=0.001, slippage_bps=10
        ),
        StrategyConfig(sl_atr_mult=1, tp_atr_mult=2),
    )
    trade = result.trades[0]
    assert trade["entry_price"] == 100.1
    assert trade["exit_price"] == 103.99
    expected = (103.99 - 100.1) * 2 - (103.99 + 100.1) * 2 * 0.001
    assert trade["net_pnl"] == pytest.approx(expected)
    assert result.equity.equity.iloc[-1] == pytest.approx(1000 + expected)


def test_signal_expires_across_session_gap():
    bars = frame([(100, 100, 100, 100), (100, 101, 99, 100)])
    bars.loc[1, ["open_time", "close_time"]] += pd.Timedelta(days=1)
    result = simulate(bars, pd.DataFrame({"enter": [True, False], "atr": 2.0}))
    assert result.open_position is None
    assert result.equity.cash.iloc[-1] == 100_000


def test_insufficient_cash_no_partial_or_short():
    result = run([(100, 100, 100, 100), (1000, 1001, 999, 1000)])
    assert result.rejected_entries == 1
    assert result.open_position is None
    assert result.equity.cash.iloc[-1] == 1000


def test_open_position_marked_not_fictitiously_closed():
    result = run([(100, 100, 100, 100), (100, 101, 99, 101)])
    assert result.open_position["quantity"] == 2
    assert result.trades == []
    assert result.equity.equity.iloc[-1] == 1002


def test_no_trades_metrics_and_tick_grid():
    result = run([(100, 100, 100, 100)], [False])
    assert result.metrics(1000)["sharpe_daily_252"] is None
    assert result.metrics(1000)["win_rate"] is None
    assert tick_round(1.234, 0.05, up=True) == 1.25
    assert tick_round(1.234, 0.05, up=False) == 1.2
