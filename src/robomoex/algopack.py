"""Explicit AlgoPack authentication; credentials never enter URLs or result caches."""

import argparse
import json
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .cache import atomic_write


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Authenticated redirects are disabled")


def get_token():
    token = os.environ.get("MOEX_API_TOKEN")
    if not token:
        path = Path(os.environ.get("MOEX_TOKEN_FILE", ".secrets/moex.token"))
        if not path.is_file():
            raise ValueError("Set MOEX_API_TOKEN or MOEX_TOKEN_FILE")
        token = path.read_text(encoding="utf-8").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("Invalid MOEX token format")
    return token


def get_authenticated_json(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in {"iss.moex.com", "apim.moex.com"}
        or not parsed.path.startswith("/iss/")
    ):
        raise ValueError("Authentication is restricted to MOEX ISS HTTPS endpoints")
    target = urlunsplit(("https", "apim.moex.com", parsed.path, parsed.query, ""))
    request = Request(
        target, headers={"Authorization": "Bearer " + get_token(), "Accept": "application/json"}
    )
    context = ssl.create_default_context()
    ca_file = os.environ.get("MOEX_CA_FILE")
    if ca_file is None and Path(".secrets/moex-ca.pem").is_file():
        ca_file = ".secrets/moex-ca.pem"
    if ca_file:
        context.load_verify_locations(cafile=ca_file)
    try:
        with build_opener(NoRedirect(), HTTPSHandler(context=context)).open(
            request, timeout=20
        ) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(f"MOEX HTTP {error.code}; check endpoint and subscription") from None
    except URLError as error:
        raise ValueError(f"MOEX transport failure ({type(error.reason).__name__})") from None


def probe(output):
    base = "https://apim.moex.com/iss/"
    routes = {
        "candles": (
            "engines/stock/markets/shares/boards/TQBR/securities/SNGS/candles.json?"
            "from=2026-09-01&till=2026-09-02&interval=1&iss.meta=off",
            "candles",
        ),
        "trades": (
            "engines/stock/markets/shares/boards/TQBR/securities/SNGS/trades.json?iss.meta=off",
            "trades",
        ),
        "orderbook": (
            "engines/stock/markets/shares/boards/TQBR/securities/SNGS/orderbook.json?iss.meta=off",
            "orderbook",
        ),
        "obstats": ("datashop/algopack/eq/obstats/SNGS.json?date=2026-09-01&iss.meta=off", "data"),
        "calendar": (
            "calendars/stock.json?from=2026-09-01&till=2026-09-08&show_all_days=1&iss.meta=off",
            "off_days",
        ),
    }
    report = {}
    for name, (path, table) in routes.items():
        try:
            data = get_authenticated_json(base + path)
            block = data.get(table)
            if not isinstance(block, dict) or not {"columns", "data"} <= block.keys():
                raise ValueError("Expected data table missing")
            report[name] = dict(
                status="available", rows=len(block["data"]), columns=block["columns"]
            )
        except (OSError, ValueError, KeyError) as error:
            report[name] = dict(status="unavailable", error=str(error))
    atomic_write(output, json.dumps(report, indent=2).encode())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe MOEX token access without disclosing it")
    parser.add_argument("--output", type=Path, default=Path("results/algopack_access.json"))
    args = parser.parse_args(argv)
    report = probe(args.output)
    print(json.dumps(report))
    return 0 if any(x["status"] == "available" for x in report.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
