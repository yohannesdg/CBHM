"""Parse observed discharge files into monthly means per gauge.

Two layouts are accepted:
* CSV with a date column (YYYY-MM or YYYY-MM-DD) and one column per gauge (m3/s);
* a GRDC single-station export: '#' header lines (with 'Station:'), then rows 'YYYY-MM-DD;hh:mm;value',
  with -999 for missing values. Daily or monthly GRDC files both work.
Daily values become monthly means when at least 80 % of the month's days are present.
"""
from __future__ import annotations

import calendar
import csv
import io
import re

import numpy as np

MISSING = {"", "na", "nan", "-9999", "-999", "-999.0", "-999.000", "-9999.0"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", (name or "").split("(")[0].split(" ")[0].lower())


def _monthly(daily: dict, months: list) -> np.ndarray:
    """daily: {'YYYY-MM': [sum, count, is_daily]} -> monthly means on `months`."""
    q = np.full(len(months), np.nan)
    for t, ym in enumerate(months):
        if ym in daily:
            tot, cnt, is_daily = daily[ym]
            nd = calendar.monthrange(int(ym[:4]), int(ym[5:7]))[1]
            if cnt and (not is_daily or cnt >= 0.8 * nd):
                q[t] = tot / cnt
    return q


def _add(acc: dict, date: str, value: str):
    v = value.strip()
    if v.lower() in MISSING:
        return
    try:
        x = float(v)
    except ValueError:
        return
    if x < 0:
        return
    s = acc.setdefault(date[:7], [0.0, 0, len(date.strip()) > 7 and date[8:10] not in ("00", "")])
    s[0] += x
    s[1] += 1


def parse_grdc(text: str):
    """Return (station_name, {'YYYY-MM': [sum, n, daily]}) or None when not a GRDC export."""
    if "GRDC" not in text[:2000] and not re.search(r"^#\s*Station:", text, re.M):
        return None
    station, acc = "", {}
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.match(r"#\s*Station:\s*(.+)", line)
            if m:
                station = m.group(1).strip()
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 3 and re.match(r"\d{4}-\d{2}", parts[0]):
            _add(acc, parts[0], parts[-1])
    return station, acc


def parse_observed(raw: bytes, gauges: list, months: list, gauge_id=None):
    """Return (qobs (G, T) with NaN where nothing was supplied, list of matched gauge names, message)."""
    text = raw.decode("utf-8-sig", errors="replace")
    q = np.full((len(gauges), len(months)), np.nan)
    keys = [_key(g["name"]) for g in gauges]
    target = None
    if gauge_id not in (None, ""):
        target = next((i for i, g in enumerate(gauges) if str(g["id"]) == str(gauge_id)), None)
        if target is None:
            raise ValueError(f"gauge {gauge_id} is not in this dataset")

    grdc = parse_grdc(text)
    if grdc is not None:
        station, acc = grdc
        i = target if target is not None else next((j for j, k in enumerate(keys) if k and k == _key(station)), None)
        if i is None:
            raise ValueError(f"GRDC station '{station}' does not match any gauge "
                             f"({', '.join(g['name'] for g in gauges)}). Pick the gauge from the list and upload again.")
        q[i] = _monthly(acc, months)
        return q, [gauges[i]["name"]]

    rd = csv.reader(io.StringIO(text), delimiter=";" if text.count(";") > text.count(",") else ",")
    rows = [r for r in rd if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("the file is empty")
    head = [c.strip() for c in rows[0]]
    dcol = next((j for j, c in enumerate(head) if c.lower() in ("date", "time", "month", "yyyy-mm-dd")), None)
    if dcol is None:
        raise ValueError("no 'date' column found. Expected a header like: date, Bamboi, Nawuni")
    vcols = [j for j in range(len(head)) if j != dcol]
    match = {}
    if target is not None:
        if len(vcols) != 1:
            named = [j for j in vcols if _key(head[j]) == keys[target]]
            if not named:
                raise ValueError("with a gauge chosen, the file should have one value column (or one named after the gauge)")
            vcols = named
        match[target] = vcols[0]
    else:
        for i, k in enumerate(keys):
            j = next((j for j in vcols if k and _key(head[j]) == k), None)
            if j is not None:
                match[i] = j
    if not match:
        raise ValueError(f"no column matches a gauge name ({', '.join(g['name'] for g in gauges)}). "
                         "Rename the columns or pick the gauge from the list.")
    accs = {i: {} for i in match}
    for r in rows[1:]:
        if len(r) <= dcol or not re.match(r"\s*\d{4}-\d{1,2}", r[dcol]):
            continue
        d = r[dcol].strip()
        y, m = d.split("-")[:2]
        d = f"{y}-{int(m):02d}" + d[len(y) + 1 + len(m):]
        for i, j in match.items():
            if j < len(r):
                _add(accs[i], d, r[j])
    for i in match:
        q[i] = _monthly(accs[i], months)
    return q, [gauges[i]["name"] for i in match]
