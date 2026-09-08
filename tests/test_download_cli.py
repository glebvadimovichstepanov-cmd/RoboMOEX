import pandas as pd

from robomoex.cli import main, parser, run
from robomoex.demo import dataset


def test_download_and_cache_have_identical_results(tmp_path, monkeypatch):
    bars, sessions = dataset(1)
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    calls = []

    def download(*args, **kwargs):
        calls.append(args)
        return bars

    monkeypatch.setattr("robomoex.cli.download_minutes", download)
    argv = [
        "--mode",
        "backtest",
        "--download",
        "--sessions",
        str(tmp_path / "sessions.csv"),
        "--start",
        str(bars.open_time.iloc[0]),
        "--as-of",
        str(bars.close_time.iloc[-1]),
        "--cache",
        str(tmp_path / "cache.json"),
        "--output",
        str(tmp_path / "out"),
    ]
    first = run(parser().parse_args(argv))
    second = run(parser().parse_args(argv))
    assert len(calls) == 1
    assert first == second


def test_incomplete_download_not_written_to_cache(tmp_path, monkeypatch):
    bars, sessions = dataset(1)
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    monkeypatch.setattr("robomoex.cli.download_minutes", lambda *a, **kw: bars.drop(index=4))
    assert (
        main(
            [
                "--mode",
                "backtest",
                "--download",
                "--sessions",
                str(tmp_path / "sessions.csv"),
                "--start",
                str(bars.open_time.iloc[0]),
                "--as-of",
                str(bars.close_time.iloc[-1]),
                "--cache",
                str(tmp_path / "cache.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "cache.json").exists()
    assert not (tmp_path / "out").exists()


def test_future_input_does_not_trade_forming_bar(tmp_path):
    bars, sessions = dataset(1)
    bars.to_csv(tmp_path / "bars.csv", index=False)
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    report = run(
        parser().parse_args(
            [
                "--mode",
                "backtest",
                "--input",
                str(tmp_path / "bars.csv"),
                "--sessions",
                str(tmp_path / "sessions.csv"),
                "--as-of",
                str(bars.close_time.iloc[10] + pd.Timedelta(seconds=30)),
                "--output",
                str(tmp_path / "out"),
            ]
        )
    )
    assert report["bars"] == 11
