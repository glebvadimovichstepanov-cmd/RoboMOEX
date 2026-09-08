"""Verified point-in-time event input. An empty feed never asserts event coverage."""

import json
import math
from pathlib import Path

import pandas as pd

TYPES = {"DIVIDEND", "SPLIT", "REPORT", "OPEC", "SANCTIONS", "TAX", "GEOPOLITICS"}


def validate_events(rows):
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    required = {
        "event_id",
        "event_type",
        "ticker",
        "effective_at",
        "known_at",
        "source_url",
        "verified",
    }
    if required - set(events):
        raise ValueError("Event feed missing required provenance fields")
    if events.event_id.duplicated().any():
        raise ValueError("Duplicate event IDs; resolve revisions before import")
    if (
        not events.event_type.isin(TYPES).all()
        or not events.verified.map(lambda x: x is True).all()
    ):
        raise ValueError("Unsupported or unverified event")
    for column in ("effective_at", "known_at"):
        if any(pd.Timestamp(v).tzinfo is None for v in events[column]):
            raise ValueError("Event timestamps must include timezone")
        events[column] = pd.to_datetime(events[column], utc=True)
    if not events.source_url.str.startswith("https://").all():
        raise ValueError("Event requires HTTPS source provenance")
    for row in events.itertuples():
        if row.event_type == "DIVIDEND" and not math.isfinite(
            float(getattr(row, "amount_rub", -1))
        ):
            raise ValueError("Invalid dividend amount")
        if row.event_type == "SPLIT" and not math.isfinite(float(getattr(row, "split_ratio", 0))):
            raise ValueError("Invalid split ratio")
        if row.event_type == "DIVIDEND" and not float(getattr(row, "amount_rub", -1)) >= 0:
            raise ValueError("Invalid dividend amount")
        if row.event_type == "SPLIT" and not float(getattr(row, "split_ratio", 0)) > 0:
            raise ValueError("Invalid split ratio")
    return events.sort_values("known_at")


def load_events(path):
    return validate_events(json.loads(Path(path).read_text(encoding="utf-8")))


def flags(events, decision, ticker="SNGS", sessions=None):
    decision = pd.Timestamp(decision)
    if decision.tzinfo is None:
        raise ValueError("Decision time must include timezone")
    today = decision.tz_convert("Europe/Moscow").date()
    future_dates = sorted(
        {pd.Timestamp(d).date() for d in sessions or [] if pd.Timestamp(d).date() > today}
    )
    boundary = future_dates[1] if len(future_dates) >= 2 else today + pd.Timedelta(days=2)
    result = dict(
        is_dividend_cutoff=False,
        dividend_entry_block=False,
        report_event=False,
        opec_event=False,
        sanctions_event=False,
        tax_event=False,
        geopolitical_event=False,
    )
    if events.empty:
        return result
    known = events[(events.known_at <= decision) & events.ticker.isin([ticker, "*"])]
    mapping = {
        "REPORT": "report_event",
        "OPEC": "opec_event",
        "SANCTIONS": "sanctions_event",
        "TAX": "tax_event",
        "GEOPOLITICS": "geopolitical_event",
    }
    for event in known.itertuples():
        day = event.effective_at.tz_convert("Europe/Moscow").date()
        if event.event_type == "DIVIDEND":
            result["is_dividend_cutoff"] |= day == today
            result["dividend_entry_block"] |= today <= day <= boundary
        elif event.event_type in mapping:
            result[mapping[event.event_type]] |= today <= day <= boundary
    return result


def adjust_prices(daily, events, ticker="SNGS"):
    """Forward total-return index, leaving raw execution prices intact.

    Only actions already known by the effective bar are used. Late-announced
    actions cause an error instead of retroactively rewriting historical signals.
    Cash here adjusts the trend index; execution cash accounting is separate.
    """
    x = daily.copy()
    signal, previous_raw, previous_index = [], None, None
    for row in x.itertuples():
        close_time = pd.Timestamp(row.close_time)
        day = pd.Timestamp(row.open_time).tz_convert("Europe/Moscow").date()
        actions = (
            events
            if events.empty
            else events[
                (events.ticker == ticker)
                & (events.effective_at.dt.tz_convert("Europe/Moscow").dt.date == day)
                & events.event_type.isin(["DIVIDEND", "SPLIT"])
            ]
        )
        cash, split = 0.0, 1.0
        for event in actions.itertuples():
            if event.known_at > close_time:
                raise ValueError("Corporate action was not known at effective bar")
            if event.event_type == "DIVIDEND":
                cash += float(event.amount_rub)
            else:
                split *= float(event.split_ratio)
        if cash and split != 1:
            raise ValueError("Simultaneous split/dividend requires verified unit ordering")
        current = (
            row.close
            if previous_raw is None
            else previous_index * (row.close * split + cash) / previous_raw
        )
        signal.append(current)
        previous_raw, previous_index = row.close, current
    x["signal_close"] = signal
    return x


def execution_actions(daily, events, ticker="SNGS"):
    """Attach verified economic actions; ex-date entitlement is not spendable cash."""
    x = daily.copy()
    x["dividend_cash_per_share"] = 0.0
    x["split_ratio"] = 1.0
    x["dividend_payment_at"] = ""
    if events.empty:
        return x
    dates = pd.to_datetime(x.open_time).dt.tz_convert("Europe/Moscow").dt.date
    used = set()
    for event in events[events.ticker == ticker].itertuples():
        if event.event_type not in {"DIVIDEND", "SPLIT"}:
            continue
        day = event.effective_at.tz_convert("Europe/Moscow").date()
        mask = dates == day
        if not mask.any():
            continue
        if day in used:
            raise ValueError("Multiple same-day actions require explicit unit ordering")
        used.add(day)
        if event.known_at > pd.to_datetime(x.loc[mask, "open_time"], utc=True).min():
            raise ValueError("Action was not known before effective session")
        if event.event_type == "SPLIT":
            x.loc[mask, "split_ratio"] = float(event.split_ratio)
        else:
            payment = pd.Timestamp(getattr(event, "payment_at", None))
            if pd.isna(payment) or payment.tzinfo is None or payment < event.effective_at:
                raise ValueError("Dividend accounting requires verified timezone-aware payment_at")
            x.loc[mask, "dividend_cash_per_share"] = float(event.amount_rub)
            x.loc[mask, "dividend_payment_at"] = payment.isoformat()
    return x
