"""Local observed-flow database: gauging stations and monthly discharge from several sources, in one SQLite file.

    data/flowdb/flows.sqlite
      sources  (source, name, licence, citation, url, imported, stations, months)
      stations (id, source, source_id, name, river, country, lon, lat, area_km2, kind, flag, comment, dams, first, last, months)
      series   (station, ym, q, missing_days)              monthly mean discharge, m3/s

Station ids are "<SOURCE>:<source id>", e.g. "ADHI:ADHI_520", "GRDC:1531100", "USER:nawuni".

Importers: ADHI (all 1,466 stations from the files the app downloads), GRDC export files (daily or monthly),
and your own CSV files (a station list plus series, in wide or long layout). Daily values become monthly
means when at least 80 % of the month's days are present.

When two sources describe the same gauge, both are kept; candidates() offers the one with the higher
source priority (USER > GRDC > ADHI) and longer record, and fills its gaps from the others when added.
"""
from __future__ import annotations

import calendar
import csv
import io
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PRIORITY = {"USER": 3, "GRDC": 2, "ADHI": 1}
MISSING_TEXT = {"", "na", "nan", "null", "-", "-9999", "-999", "-999.0", "-999.000", "-9999.0", "-9999.000"}
MIN_DAY_FRACTION = 0.8

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source TEXT PRIMARY KEY, name TEXT, licence TEXT, citation TEXT, url TEXT,
  imported REAL, stations INTEGER, months INTEGER);
CREATE TABLE IF NOT EXISTS stations (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, source_id TEXT, name TEXT, river TEXT, country TEXT,
  lon REAL, lat REAL, area_km2 REAL, kind TEXT DEFAULT 'discharge', flag INTEGER DEFAULT 0,
  comment TEXT DEFAULT '', dams INTEGER DEFAULT 0, first TEXT, last TEXT, months INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS st_lonlat ON stations (lon, lat);
CREATE TABLE IF NOT EXISTS series (
  station TEXT NOT NULL, ym TEXT NOT NULL, q REAL NOT NULL, missing_days INTEGER DEFAULT 0,
  PRIMARY KEY (station, ym));
"""

RESERVOIR = re.compile(r"volume|mm3|barrage|retenue|r[eé]servoir", re.I)


def split_name(label: str):
    """'VOLTA BLANCHE a NAWUNI' -> ('Nawuni', 'Volta Blanche'); 'X at Y (123)' -> ('Y', 'X')."""
    s = re.sub(r"\s*\(\d+\)\s*$", "", label or "").strip()
    m = re.match(r"(.+?)\s+(?:a|à|at|au|en)\s+(.+)", s, re.I)
    river, place = (m.group(1), m.group(2)) if m else ("", s)
    nice = lambda x: " ".join(w.capitalize() if w.isupper() else w for w in x.split())
    return nice(place), nice(river)


class FlowDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ writing
    def _upsert_station(self, c, st: dict):
        cols = ["id", "source", "source_id", "name", "river", "country", "lon", "lat", "area_km2", "kind",
                "flag", "comment", "dams"]
        vals = [st.get(k) for k in cols]
        c.execute(f"INSERT INTO stations ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
                  f"ON CONFLICT(id) DO UPDATE SET {','.join(f'{k}=excluded.{k}' for k in cols[1:])}", vals)

    def _put_series(self, c, sid: str, monthly: dict, replace=True):
        """monthly: {'YYYY-MM': (q, missing_days)}"""
        if replace:
            c.execute("DELETE FROM series WHERE station=?", (sid,))
        c.executemany("INSERT OR REPLACE INTO series (station, ym, q, missing_days) VALUES (?,?,?,?)",
                      [(sid, ym, float(q), int(md)) for ym, (q, md) in sorted(monthly.items())])

    def _refresh_station(self, c, sid: str):
        r = c.execute("SELECT MIN(ym), MAX(ym), COUNT(*) FROM series WHERE station=?", (sid,)).fetchone()
        c.execute("UPDATE stations SET first=?, last=?, months=? WHERE id=?", (r[0], r[1], r[2], sid))

    def _source(self, c, source, **info):
        n = c.execute("SELECT COUNT(*), COALESCE(SUM(months),0) FROM stations WHERE source=?", (source,)).fetchone()
        c.execute("INSERT INTO sources (source, name, licence, citation, url, imported, stations, months) "
                  "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET name=excluded.name, licence=excluded.licence, "
                  "citation=excluded.citation, url=excluded.url, imported=excluded.imported, stations=excluded.stations, "
                  "months=excluded.months",
                  (source, info.get("name", source), info.get("licence", ""), info.get("citation", ""), info.get("url", ""),
                   time.time(), n[0], n[1]))

    def add_station(self, st: dict, monthly: dict, replace=True):
        """Add or update one station and its monthly series. Returns the station id."""
        st = dict(st)
        st["id"] = st.get("id") or f"{st['source']}:{st['source_id']}"
        for k, v in (("flag", 0), ("dams", 0), ("comment", ""), ("river", ""), ("country", "")):
            if st.get(k) is None:
                st[k] = v
        st.setdefault("kind", "reservoir" if RESERVOIR.search(st.get("name", "") + " " + st.get("river", "")) else "discharge")
        with self.conn() as c:
            self._upsert_station(c, st)
            self._put_series(c, st["id"], monthly, replace)
            self._refresh_station(c, st["id"])
        return st["id"]

    # ------------------------------------------------------------------ importers
    def import_adhi(self, stations_tab: Path, monthly_zip: Path, log=print, citation="", max_missing_days=6):
        """All ADHI stations with their monthly series. Months with more than `max_missing_days` missing are left out."""
        import zipfile
        text = Path(stations_tab).read_text(encoding="utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
        n = 0
        with zipfile.ZipFile(monthly_zip) as z, self.conn() as c:
            names = {Path(nm).stem.replace("monthly_", ""): nm for nm in z.namelist() if nm.endswith(".txt")}
            for r in rows:
                r = {k.strip().strip('"'): (v or "").strip().strip('"') for k, v in r.items() if k}
                aid = r.get("ID", "")
                if aid not in names:
                    continue
                try:
                    label = r.get("Name", "")
                    place, river = split_name(label)
                    st = {"id": f"ADHI:{aid}", "source": "ADHI", "source_id": aid, "name": place, "river": river,
                          "country": r.get("Country", ""), "lon": float(r["Longitude"]), "lat": float(r["Latitude"]),
                          "area_km2": float(r["Catchment"]), "flag": int(float(r.get("Flag") or 0)),
                          "comment": " ".join(x for x in (r.get("Comment", "").strip(), f"code {r.get('Station_co', '')}",
                                                         f"via {r.get('Data_sourc', '')}") if x.strip()),
                          "dams": int(float(r.get("Number_dams") or 0)),
                          "kind": "reservoir" if RESERVOIR.search(label) else "discharge"}
                except (KeyError, ValueError):
                    continue
                monthly = {}
                for line in z.read(names[aid]).decode("utf-8", errors="replace").splitlines():
                    p = line.split(",")
                    if len(p) < 6:
                        continue
                    try:
                        y, m, mean, miss = int(p[0]), int(p[1]), float(p[2]), float(p[5])
                    except ValueError:
                        continue
                    if np.isfinite(mean) and mean >= 0 and miss <= max_missing_days:
                        monthly[f"{y}-{m:02d}"] = (mean, int(miss))
                self._upsert_station(c, st)
                self._put_series(c, st["id"], monthly)
                self._refresh_station(c, st["id"])
                n += 1
            self._source(c, "ADHI", name="African Database of Hydrometric Indices", licence="CC BY 4.0",
                         citation=citation, url="https://doi.org/10.23708/LXGXQ9")
        log(f"ADHI: {n} stations imported")
        return n

    def import_grdc(self, text: str, filename="", overrides: dict | None = None):
        """One GRDC export file (daily or monthly). Station metadata come from its header. Returns the station id."""
        head = {}
        rows = []
        cols = None
        for line in text.splitlines():
            if line.startswith("#"):
                m = re.match(r"#\s*([^:]+?)\s*:\s*(.*)$", line)
                if m:
                    head[m.group(1).strip().lower()] = m.group(2).strip()
                continue
            if cols is None and re.match(r"\s*YYYY", line, re.I):
                cols = [x.strip().lower() for x in line.split(";")]
                continue
            if re.match(r"\s*\d{4}-\d{2}", line):
                rows.append([x.strip() for x in line.split(";")])
        if not rows:
            raise ValueError(f"{filename or 'file'}: no GRDC data rows found")
        number = head.get("grdc-no.") or head.get("grdc no.") or head.get("grdc-no") or ""
        if not number:
            raise ValueError(f"{filename or 'file'} has no 'GRDC-No.' header line")
        area = head.get("catchment area (km²)") or head.get("catchment area (km2)") or head.get("catchment area (km²)")
        num = lambda s: float(re.sub(r"[^0-9eE.+-]", "", s)) if s and re.search(r"\d", s) else None
        content = (head.get("data set content") or head.get("data set content:") or "").lower()
        daily = "daily" in content or (rows and rows[0][0][8:10] not in ("00", ""))
        vi = None
        if cols:
            for name in ("original", "value", "calculated"):
                if name in cols:
                    vi = cols.index(name)
                    break
        alt = cols.index("calculated") if cols and "calculated" in cols else None
        acc = {}
        for r in rows:
            v = _num(r[vi] if vi is not None and vi < len(r) else r[-1])
            if v is None and alt is not None and alt < len(r):
                v = _num(r[alt])
            if v is None:
                continue
            ym = r[0][:7]
            s = acc.setdefault(ym, [0.0, 0])
            s[0] += v
            s[1] += 1
        monthly = _to_monthly(acc, daily)
        st = {"source": "GRDC", "source_id": number, "name": _title(head.get("station", filename)),
              "river": _title(head.get("river", "")), "country": head.get("country", ""),
              "lon": num(head.get("longitude (dd)") or head.get("longitude", "")),
              "lat": num(head.get("latitude (dd)") or head.get("latitude", "")), "area_km2": num(area or ""),
              "comment": f"GRDC export {filename}".strip()}
        st.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})
        if st["lon"] is None or st["lat"] is None:
            raise ValueError(f"{filename or 'file'}: no station coordinates in the header")
        sid = self.add_station(st, monthly)
        with self.conn() as c:
            self._source(c, "GRDC", name="Global Runoff Data Centre", licence="GRDC data policy (no redistribution)",
                         url="https://portal.grdc.bafg.de")
        return sid

    def import_user(self, stations_csv: str | None, series_csv: str, source_name="your files", log=print):
        """Your own records.

        stations_csv: id, name, lon, lat, area_km2 [, river, country] (one row per station). Optional when the
        series file is long-format with those columns.
        series_csv: wide (date, <id1>, <id2>, ...) or long (id/station, date, q/value/discharge), in m3/s,
        daily or monthly dates (YYYY-MM or YYYY-MM-DD; DD/MM/YYYY also accepted)."""
        meta = {}
        if stations_csv:
            for r in csv.DictReader(io.StringIO(stations_csv.lstrip("﻿")), delimiter=_delim(stations_csv)):
                r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                sid = r.get("id") or r.get("station") or r.get("code") or r.get("name")
                if not sid:
                    continue
                meta[sid] = r
        text = series_csv.lstrip("﻿")
        rd = list(csv.reader(io.StringIO(text), delimiter=_delim(text)))
        rd = [r for r in rd if r and any(x.strip() for x in r)]
        head = [h.strip() for h in rd[0]]
        low = [h.lower() for h in head]
        dcol = next((i for i, h in enumerate(low) if h in ("date", "time", "month", "datetime")), None)
        if dcol is None:
            raise ValueError("the series file needs a 'date' column")
        idcol = next((i for i, h in enumerate(low) if h in ("id", "station", "code", "station_id")), None)
        vcol = next((i for i, h in enumerate(low) if h in ("q", "value", "discharge", "flow", "q_m3s")), None)
        acc = {}
        if idcol is not None and vcol is not None:          # long
            for r in rd[1:]:
                if len(r) <= max(dcol, idcol, vcol):
                    continue
                ym, day = _ym(r[dcol])
                v = _num(r[vcol])
                if ym and v is not None:
                    a = acc.setdefault(r[idcol].strip(), {}).setdefault(ym, [0.0, 0, day])
                    a[0] += v; a[1] += 1
        else:                                               # wide
            for r in rd[1:]:
                if len(r) <= dcol:
                    continue
                ym, day = _ym(r[dcol])
                if not ym:
                    continue
                for j, h in enumerate(head):
                    if j == dcol or j >= len(r):
                        continue
                    v = _num(r[j])
                    if v is not None:
                        a = acc.setdefault(h, {}).setdefault(ym, [0.0, 0, day])
                        a[0] += v; a[1] += 1
        long_meta = {}
        if idcol is not None:                               # station metadata repeated in a long file
            for r in rd[1:]:
                if len(r) == len(head):
                    rr = dict(zip(low, (x.strip() for x in r)))
                    long_meta.setdefault(rr[low[idcol]], rr)
        added, problems = [], []
        for sid, months in acc.items():
            m = {**long_meta.get(sid, {}), **meta.get(sid, {})}
            try:
                lon = float(m.get("lon") or m.get("longitude"))
                lat = float(m.get("lat") or m.get("latitude"))
            except (TypeError, ValueError):
                problems.append(f"{sid}: no lon/lat in the station list")
                continue
            area = _num(m.get("area_km2") or m.get("area") or m.get("catchment") or "")
            daily = any(x[2] for x in months.values())
            monthly = _to_monthly({k: v[:2] for k, v in months.items()}, daily)
            st = {"source": "USER", "source_id": re.sub(r"\s+", "_", sid.strip()), "name": m.get("name") or sid,
                  "river": m.get("river", ""), "country": m.get("country", ""), "lon": lon, "lat": lat,
                  "area_km2": area, "comment": source_name}
            added.append(self.add_station(st, monthly))
        with self.conn() as c:
            self._source(c, "USER", name="Your files", licence="as provided")
        log(f"USER: {len(added)} stations imported" + (f"; skipped {len(problems)}" if problems else ""))
        return added, problems

    # ------------------------------------------------------------------ reading
    def summary(self) -> dict:
        with self.conn() as c:
            src = [dict(r) for r in c.execute("SELECT * FROM sources ORDER BY source")]
            tot = c.execute("SELECT COUNT(*), COALESCE(SUM(months),0) FROM stations").fetchone()
        return {"path": str(self.path), "sources": src, "stations": tot[0], "months": tot[1]}

    def stations(self, bbox=None, min_months=1, source=None) -> list:
        q = "SELECT * FROM stations WHERE months >= ?"
        a = [min_months]
        if bbox:
            w, s, e, n = bbox
            q += " AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?"
            a += [w, e, s, n]
        if source:
            q += " AND source = ?"
            a.append(source)
        with self.conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY area_km2 DESC", a)]

    def series(self, sid: str, months: list | None = None):
        """{'YYYY-MM': q} or, with `months`, an array on those months (NaN where missing)."""
        with self.conn() as c:
            d = {r[0]: r[1] for r in c.execute("SELECT ym, q FROM series WHERE station=?", (sid,))}
        if months is None:
            return d
        return np.array([d.get(m, np.nan) for m in months])

    def delete_station(self, sid: str):
        with self.conn() as c:
            c.execute("DELETE FROM series WHERE station=?", (sid,))
            c.execute("DELETE FROM stations WHERE id=?", (sid,))

    def export_csv(self, ids: list | None = None) -> str:
        """Long format: id, source, name, river, lon, lat, area_km2, date, q_m3s."""
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "source", "name", "river", "lon", "lat", "area_km2", "date", "q_m3s"])
        with self.conn() as c:
            sel = "SELECT s.id, s.source, s.name, s.river, s.lon, s.lat, s.area_km2, r.ym, r.q FROM series r " \
                  "JOIN stations s ON s.id = r.station"
            args = []
            if ids:
                sel += f" WHERE s.id IN ({','.join('?' * len(ids))})"
                args = ids
            for row in c.execute(sel + " ORDER BY s.id, r.ym", args):
                w.writerow(list(row))
        return out.getvalue()


# ---------------------------------------------------------------------- helpers
def _num(s):
    s = (s or "").strip()
    if s.lower() in MISSING_TEXT:
        return None
    try:
        v = float(s.replace(",", ".")) if s.count(",") == 1 and "." not in s else float(s)
    except ValueError:
        return None
    return v if np.isfinite(v) and v >= 0 else None


def _ym(s):
    """(YYYY-MM, is_daily) from 'YYYY-MM', 'YYYY-MM-DD', 'YYYY/MM/DD' or 'DD/MM/YYYY'."""
    s = (s or "").strip()
    m = re.match(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", s)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), m.group(3)
        return (f"{y}-{mo:02d}", bool(d) and int(d) != 0) if 1 <= mo <= 12 else (None, False)
    m = re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}", True
    return None, False


def _to_monthly(acc: dict, daily: bool) -> dict:
    out = {}
    for ym, (tot, cnt) in acc.items():
        if not cnt:
            continue
        if daily:
            nd = calendar.monthrange(int(ym[:4]), int(ym[5:7]))[1]
            if cnt < MIN_DAY_FRACTION * nd:
                continue
            out[ym] = (tot / cnt, nd - cnt)
        else:
            out[ym] = (tot / cnt, 0)
    return out


def _delim(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return max((";", ",", "\t"), key=first.count)


def _title(s: str) -> str:
    return " ".join(w.capitalize() if w.isupper() else w for w in (s or "").split())
