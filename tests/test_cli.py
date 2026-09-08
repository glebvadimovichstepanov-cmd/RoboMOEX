import json

import pandas as pd
import pytest

from robomoex.cli import main, parser, run
from robomoex.config import ExecutionConfig, StrategyConfig, load_config
from robomoex.demo import dataset


def test_live_refused_before_files_or_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Live mode must not access external resources")

    monkeypatch.setattr("robomoex.cli.load_config", forbidden)
    monkeypatch.setattr("robomoex.cli.download_minutes", forbidden)
    assert main(["--mode", "live", "--config", str(tmp_path / "missing.toml")]) == 2


def test_default_offline_and_backtest_paper_parity(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Default command must remain offline")

    monkeypatch.setattr("robomoex.cli.download_minutes", forbidden)
    first = run(parser().parse_args(["--output", str(tmp_path / "paper")]))
    second = run(
        parser().parse_args(
            ["--mode", "backtest", "--demo", "--output", str(tmp_path / "backtest")]
        )
    )
    repeat = run(parser().parse_args(["--output", str(tmp_path / "repeat")]))
    assert first == repeat
    assert first["mode"] == "paper"
    assert first["source"] == "SYNTHETIC_DEMO"
    assert first["metrics"] == second["metrics"]
    assert first["data_sha256"] == second["data_sha256"]
    assert (tmp_path / "paper" / "trades.json").read_bytes() == (
        tmp_path / "backtest" / "trades.json"
    ).read_bytes()
    assert (
        json.loads((tmp_path / "paper" / "report.json").read_text())["metrics"] == first["metrics"]
    )


def test_external_paper_rejects_stale_data_and_clock_override(tmp_path):
    bars, sessions = dataset(1)
    bars.to_csv(tmp_path / "bars.csv", index=False)
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    args = [
        "--input",
        str(tmp_path / "bars.csv"),
        "--sessions",
        str(tmp_path / "sessions.csv"),
        "--output",
        str(tmp_path / "out"),
    ]
    assert main(args) == 2
    assert main(args + ["--as-of", "2025-01-06T10:30:00Z"]) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fee_rate": -1},
        {"lot_size": 0.5},
        {"initial_cash": 0},
        {"tick_size": float("nan")},
        {"slippage_bps": 10000},
    ],
)
def test_execution_config_validation(kwargs):
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)


def test_config_typo_is_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[execution]\nlot_szie = 10\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    with pytest.raises(ValueError):
        StrategyConfig(rsi_1m_thresh=101)


def test_external_backtest_exports(tmp_path):
    bars, sessions = dataset(1)
    bars.to_csv(tmp_path / "bars.csv", index=False)
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    assert (
        main(
            [
                "--mode",
                "backtest",
                "--input",
                str(tmp_path / "bars.csv"),
                "--sessions",
                str(tmp_path / "sessions.csv"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    curve = pd.read_csv(tmp_path / "out" / "equity.csv")
    assert len(curve) == 60
