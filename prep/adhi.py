"""Observed discharge from ADHI, the African Database of Hydrometric Indices (Tramblay et al., 2021).

1,466 stations with at least 10 years of daily data (1950-2018), as monthly mean discharge series in m3/s.
Licence CC BY 4.0. https://doi.org/10.23708/LXGXQ9

    stations, monthly zip -> data/raw/adhi/
    candidates(dataset)   -> ADHI stations inside the model basin, snapped to its river network
    add(dataset, choices) -> gauges and qobs added to (or merged into) the dataset
"""
from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from hydro.routing import Router, snap_to_stream
from prep import prep_volta as pv

DOI = "doi:10.23708/LXGXQ9"
DATAVERSE = "https://dataverse.ird.fr"
FILES = {"stations": ("ADHI_stations.tab", 4944), "monthly": ("ADHI_MonthlySeries.zip", 4916)}
CITATION = ("Tramblay, Y. et al. (2021). ADHI: the African Database of Hydrometric Indices (1950-2018). "
            "Earth System Science Data 13, 1547-1560. Data: https://doi.org/10.23708/LXGXQ9 (CC BY 4.0)")
MAX_MISSING_DAYS = 6          # a month counts if at least ~80 % of its days were measured
AREA_TOL = 1.25               # accept a station when model and ADHI catchment areas agree within 25 %
MATCH_TOL = 1.10              # ... and treat it as an existing gauge within 10 % and a few cells


def adhi_dir() -> Path:
    return pv.RAW / "adhi"


def _file_ids():
    """Current file ids of the dataset (they change with new versions); fall back to the known ones."""
    try:
        url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
        req = urllib.request.Request(url, headers={"User-Agent": pv.UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            files = json.load(r)["data"]["latestVersion"]["files"]
        by_name = {f["dataFile"]["filename"].strip(): f["dataFile"]["id"] for f in files}
        return {k: by_name.get(name, fid) for k, (name, fid) in FILES.items()}
    except Exception:  # noqa: BLE001
        return {k: fid for k, (_, fid) in FILES.items()}


def ensure_files(log=print):
    d = adhi_dir()
    d.mkdir(parents=True, exist_ok=True)
    need = {k: d / name for k, (name, _) in FILES.items() if not _ok(d / name)}
    if need:
        ids = _file_ids()
        for k, dest in need.items():
            log(f"downloading {dest.name} from the IRD Dataverse (ADHI, CC BY 4.0)")
            pv.download(f"{DATAVERSE}/api/access/datafile/{ids[k]}", dest, resume=False, retries=4, timeout=60)
    return d / FILES["stations"][0], d / FILES["monthly"][0]


def _ok(p: Path):
    if not p.exists() or p.stat().st_size < 200:
        return False
    if p.suffix == ".zip":
        return zipfile.is_zipfile(p)
    return p.read_bytes()[:200].count(b"\t") > 3


def files_ready() -> bool:
    d = adhi_dir()
    return all(_ok(d / name) for name, _ in FILES.values())


def stations(path: Path | None = None) -> list:
    path = path or adhi_dir() / FILES["stations"][0]
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    out = []
    for r in rows:
        r = {k.strip().strip('"'): (v or "").strip().strip('"') for k, v in r.items() if k}
        try:
            out.append({"adhi": r["ID"], "code": r.get("Station_co", ""), "label": r.get("Name", ""),
                        "lon": float(r["Longitude"]), "lat": float(r["Latitude"]), "area_km2": float(r["Catchment"]),
                        "start": int(float(r["Starting_y"])), "end": int(float(r["Ending_yea"])),
                        "country": r.get("Country", ""), "source": r.get("Data_sourc", ""),
                        "flag": int(float(r.get("Flag") or 0)), "comment": r.get("Comment", "").strip(),
                        "dams": int(float(r.get("Number_dams") or 0))})
        except (KeyError, ValueError):
            continue
    return out


def split_name(label: str):
    """'VOLTA BLANCHE a NAWUNI' -> ('Nawuni', 'Volta Blanche'); 'X at Y (123)' -> ('Y', 'X')."""
    s = re.sub(r"\s*\(\d+\)\s*$", "", label).strip()
    m = re.match(r"(.+?)\s+(?:a|à|at|au|en)\s+(.+)", s, re.I)
    river, place = (m.group(1), m.group(2)) if m else ("", s)
    nice = lambda x: " ".join(w.capitalize() if w.isupper() else w for w in x.split())
    return nice(place), nice(river)


def monthly_series(adhi_id: str, months: list, zip_path: Path | None = None) -> np.ndarray:
    """Monthly mean discharge (m3/s) on `months`, NaN where missing or with too many missing days."""
    zip_path = zip_path or adhi_dir() / FILES["monthly"][0]
    num = adhi_id.split("_")[-1]
    with zipfile.ZipFile(zip_path) as z:
        name = next((n for n in z.namelist() if n.endswith(f"monthly_ADHI_{num}.txt")), None)
        if name is None:
            raise KeyError(f"{adhi_id} has no monthly series")
        text = z.read(name).decode("utf-8", errors="replace")
    vals = {}
    for line in text.splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        try:
            y, m, mean, miss = int(p[0]), int(p[1]), float(p[2]), float(p[5])
        except ValueError:
            continue
        if np.isfinite(mean) and mean >= 0 and miss <= MAX_MISSING_DAYS:
            vals[f"{y}-{m:02d}"] = mean
    return np.array([vals.get(m, np.nan) for m in months])


def _acc_grid(data):
    area = data.grid.cell_area_m2().ravel()[data.cells] / 1e6
    acc, _ = Router(data.ds).accumulate(area)
    return data.to_grid(acc)


def candidates(data, stations_list=None, zip_path=None) -> list:
    """ADHI stations inside the basin, snapped to the model river network, with their overlap and a suggestion."""
    st = stations_list if stations_list is not None else stations()
    g = data.grid
    acc2d = _acc_grid(data)
    west, south, east, north = g.west, g.south, g.east, g.north
    radius = max(2, int(round(0.25 / g.res)))
    basin_area = float(np.nanmax(acc2d))
    out = []
    for s in st:
        if not (west <= s["lon"] <= east and south <= s["lat"] <= north) or s["area_km2"] > basin_area * AREA_TOL:
            continue
        try:
            r, c = g.rowcol(s["lon"], s["lat"])
            r, c = snap_to_stream(acc2d, data.basin, r, c, radius=radius, area_hint=s["area_km2"])
        except ValueError:
            continue
        a = float(acc2d[r, c])
        if not np.isfinite(a) or abs(np.log(a / s["area_km2"])) > np.log(AREA_TOL):
            continue
        cell = data.compact_index(r, c)
        q = monthly_series(s["adhi"], data.dates, zip_path)
        n = int(np.isfinite(q).sum())
        if n == 0:
            continue
        place, river = split_name(s["label"])
        yrs = [int(m[:4]) for m, v in zip(data.dates, q) if np.isfinite(v)]
        out.append({**s, "name": place, "river": river, "row": int(r), "col": int(c), "cell": cell,
                    "lon_model": float(g.lon[c]), "lat_model": float(g.lat[r]), "area_model_km2": a,
                    "months": n, "first": min(yrs), "last": max(yrs), "mean_m3s": float(np.nanmean(q)), "q": q})
    _suggest(data, out, radius, basin_area, ds=data.ds)
    for c_ in out:
        c_.pop("q")
    out.sort(key=lambda x: -x["area_model_km2"])
    return out


def _key(name):
    """First word of a station name, letters only: 'Senchi (below Akosombo)' -> 'senchi'."""
    w = re.sub(r"\(.*?\)", "", name or "").strip().split()
    return re.sub(r"[^a-z]", "", w[0].lower()) if w else ""


# French reservoir records ("Barrage de Bagré volume en Mm3"). Not "dam": "Bui amont dam" is upstream of a dam.
RESERVOIR = re.compile(r"volume|mm3|barrage|retenue|r[eé]servoir", re.I)


def _downstream(ds, cell, limit=100000):
    path, c = [], cell
    while c >= 0 and len(path) < limit:
        path.append(c)
        c = int(ds[c])
    return set(path)


def _suggest(data, cands, radius, basin_area, ds=None):
    """Decide a default for each station and say why. Fills match, already, suggest, notes.

    ds (compact downstream index) lets "nearly the same catchment" be judged along the river: only a station
    on the same flow path as a chosen gauge counts. Without it, stations within 25 cells are compared."""
    paths = {}

    def nested(a, b):
        if ds is None:
            return near(a, b, 25)
        for x, y in ((a, b), (b, a)):
            if x["cell"] not in paths:
                paths[x["cell"]] = _downstream(ds, x["cell"])
            if y["cell"] in paths[x["cell"]]:
                return True
        return False
    near = lambda a, b, k: abs(a["row"] - b["row"]) <= k and abs(a["col"] - b["col"]) <= k
    lr = lambda x, y: abs(np.log(x / y))
    min_area = max(1000.0, 0.02 * basin_area)
    min_months = 84
    for c in cands:
        c["notes"] = []
        c["reservoir"] = bool(RESERVOIR.search(c["label"]))
        if c["reservoir"]:
            c["notes"].append("reservoir record: may be storage volume or regulated outflow, not river flow")
        c["match"] = None
        c["already"] = False
    # existing gauges: each takes at most one station, the closest in catchment area
    for i, gg in enumerate(data.gauges):
        key = _key(gg["name"])
        opts = [c for c in cands if not c["reservoir"] and (c["cell"] == gg["cell"] or
                (near(c, gg, radius + 1) and lr(c["area_model_km2"], gg["area_km2"]) < np.log(MATCH_TOL)) or
                (key and _key(c["name"]) == key and lr(c["area_model_km2"], gg["area_km2"]) < np.log(1.15)))]
        for c in opts:
            c.setdefault("could_match", []).append({"id": gg["id"], "name": gg["name"]})
        if not opts:
            continue
        best = min(opts, key=lambda c: (_key(c["name"]) != key, lr(c["area_model_km2"], gg["area_km2"]), -c["months"]))
        have = data.qobs[i]
        best["match"] = {"id": gg["id"], "name": gg["name"], "months_have": int(np.isfinite(have).sum())}
        best["already"] = not (np.isfinite(best["q"]) & ~np.isfinite(have)).any()
    # the same station listed twice (e.g. from both SIEREM and GRDC): keep the longer record
    by_size = sorted(cands, key=lambda c: (c["match"] is None, -c["months"], c["flag"]))
    kept = []
    for c in by_size:
        dup = next((k for k in kept if near(c, k, 3) and lr(c["area_model_km2"], k["area_model_km2"]) < np.log(1.05)), None)
        if dup is not None:
            c["dup_of"] = dup["name"]
            c["notes"].append(f"same station as {dup['name']} ({dup['adhi']}), which has the longer record"
                              if _key(c["name"]) == _key(dup["name"]) else
                              f"shares a model cell with {dup['name']}, which has the longer record")
        else:
            kept.append(c)
    used = {gg["cell"] for gg in data.gauges}
    chosen = [{"name": gg["name"], "row": gg["row"], "col": gg["col"], "cell": gg["cell"], "area_km2": gg["area_km2"]}
              for gg in data.gauges]
    for c in sorted(cands, key=lambda c: (-c["months"], -c["area_model_km2"])):
        if c["already"]:
            c["suggest"] = "skip"
        elif c["match"]:
            c["suggest"] = str(c["match"]["id"])
        else:
            why = []
            if c["reservoir"] or c.get("dup_of"):
                why.append("")
            if c["flag"] >= 2:
                why.append("flagged as suspicious in ADHI")
            if c["months"] < min_months:
                why.append(f"short record in the model period (under {min_months // 12} years)")
            if c["area_model_km2"] < min_area:
                why.append(f"small catchment for a basin-scale calibration (under {min_area:,.0f} km²)")
            if c["cell"] in used and not c.get("dup_of"):
                why.append("on the same model cell as a gauge already chosen")
            twin = next((t for t in chosen if lr(c["area_model_km2"], t["area_km2"]) < np.log(1.15) and nested(c, t)), None)
            if twin is not None and not why:
                why.append(f"drains almost the same area as {twin['name']} ({twin['area_km2']:,.0f} km²), "
                           "so it would add a very small calibration zone")
            c["notes"] += [w for w in why if w]
            c["suggest"] = "skip" if why else "new"
            if c["suggest"] == "new":
                used.add(c["cell"])
                chosen.append({"name": c["name"], "row": c["row"], "col": c["col"], "cell": c["cell"],
                               "area_km2": c["area_model_km2"]})


def add(data, choices: dict, zip_path=None, cands=None) -> dict:
    """choices: {adhi id: "new" | "<existing gauge id>"}. Changes `data` in place; returns a summary."""
    cands = {c["adhi"]: c for c in (cands if cands is not None else candidates(data, zip_path=zip_path))}
    added, merged = [], []
    gauges = list(data.gauges)
    q = [row for row in data.qobs]
    next_id = max([gg["id"] for gg in gauges] + [0]) + 1
    used_cells = {gg["cell"] for gg in gauges}
    for aid, action in choices.items():
        c = cands.get(aid)
        if c is None or action in (None, "", "skip"):
            continue
        series = monthly_series(aid, data.dates, zip_path)
        if action == "new":
            if c["cell"] in used_cells:
                raise ValueError(f"{c['name']} falls on the same model cell as an existing gauge; use it for that gauge instead")
            gauges.append({"id": next_id, "name": c["name"], "river": c["river"], "lon": c["lon_model"], "lat": c["lat_model"],
                           "lon_given": c["lon"], "lat_given": c["lat"], "row": c["row"], "col": c["col"], "cell": c["cell"],
                           "area_km2": c["area_model_km2"], "source": f"ADHI {aid} ({c['code']})"})
            q.append(series)
            used_cells.add(c["cell"])
            added.append(f"{c['name']} ({int(np.isfinite(series).sum())} months)")
            next_id += 1
        else:
            i = next((k for k, gg in enumerate(gauges) if str(gg["id"]) == str(action)), None)
            if i is None:
                raise ValueError(f"gauge {action} not found")
            fill = ~np.isfinite(q[i]) & np.isfinite(series)
            if not fill.any():
                continue
            q[i] = np.where(fill, series, q[i])
            gauges[i] = {**gauges[i], "source": f"{gauges[i].get('source', '')} ADHI {aid}".strip()}
            merged.append(f"{gauges[i]['name']} from {c['name']} ({int(fill.sum())} months)")
    order = sorted(range(len(gauges)), key=lambda k: -gauges[k]["area_km2"])
    data.gauges = [gauges[k] for k in order]
    data.qobs = np.array([q[k] for k in order]).reshape(len(gauges), len(data.dates))
    data.meta = {**data.meta, "discharge": CITATION}
    return {"added": added, "merged": merged}
