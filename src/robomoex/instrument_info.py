"""Exact MOEX instrument resolution; exchange metadata is not a borrow inventory."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from .algopack import get_authenticated_json
from .cache import atomic_write


def records(payload, table):
    block = payload.get(table, {})
    if not {"columns", "data"} <= block.keys():
        raise ValueError(f"Missing instrument table: {table}")
    return [dict(zip(block["columns"], row, strict=True)) for row in block["data"]]


def resolve(ticker, board="TQBR", transport=get_authenticated_json):
    ticker = ticker.strip().upper()
    if not ticker.isascii() or not ticker.isalnum():
        raise ValueError("Ticker must be an exact ASCII instrument code")
    base = "https://apim.moex.com/iss/"
    matches = []
    seen = set()
    for start in range(0, 10000, 100):
        rows = records(
            transport(base + f"securities.json?q={quote(ticker)}&start={start}&iss.meta=off"),
            "securities",
        )
        signature = json.dumps(rows, sort_keys=True)
        if rows and signature in seen:
            raise ValueError("Instrument search pagination stalled")
        seen.add(signature)
        matches.extend(r for r in rows if r["secid"] == ticker)
        if len(rows) < 100:
            break
    if len(matches) != 1:
        raise ValueError("Instrument not found or ambiguous; use exact SECID")
    item = matches[0]
    if item.get("group") != "stock_shares":
        raise ValueError("Instrument is not an equity")
    details = transport(base + f"securities/{ticker}.json?iss.meta=off")
    boards = [
        r for r in records(details, "boards") if r["boardid"] == board and r["is_traded"] == 1
    ]
    if len(boards) != 1:
        raise ValueError("Requested board is not uniquely active")
    trading = records(
        transport(
            base
            + f"engines/stock/markets/shares/boards/{quote(board)}/securities/{ticker}.json"
            + "?iss.meta=off&iss.only=securities"
        ),
        "securities",
    )
    if len(trading) != 1 or trading[0]["SECID"] != ticker:
        raise ValueError("Trading metadata does not match instrument")
    lot, step = int(trading[0]["LOTSIZE"]), float(trading[0]["MINSTEP"])
    if lot <= 0 or step <= 0 or not item.get("isin"):
        raise ValueError("Missing or invalid ISIN/lot/tick")
    return dict(
        ticker=ticker,
        isin=item["isin"],
        share_type=item["type"],
        board=board,
        currency=boards[0]["currencyid"],
        lot_size=lot,
        min_step=step,
        short_allowed=False,
        borrow_status="UNKNOWN_REQUIRES_BROKER",
        observed_at=datetime.now(UTC).isoformat(),
        source=base + f"securities/{ticker}.json",
    )


def refresh(path, ticker="SNGS", board="TQBR", transport=get_authenticated_json):
    info = resolve(ticker, board, transport)
    content = json.dumps(info, sort_keys=True)
    payload = dict(schema=1, sha256=hashlib.sha256(content.encode()).hexdigest(), instrument=info)
    atomic_write(Path(path), json.dumps(payload, indent=2).encode())
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve and cache exact MOEX equity metadata")
    parser.add_argument("--ticker", default="SNGS")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--output", type=Path, default=Path("results/instrument/SNGS.json"))
    args = parser.parse_args(argv)
    print(json.dumps(refresh(args.output, args.ticker, args.board)))


if __name__ == "__main__":
    main()
