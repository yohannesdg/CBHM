"""Match stations from the observed-flow database to a model basin, and add their records as gauges.

candidates(data, db) -> stations inside the basin, snapped to the model river network (catchment areas must
                        agree within 25 %), each with a suggested action and the reasons for it
add(data, db, choices) -> gauges and observed series added to (or merged into) the dataset, in place
"""
from __future__ import annotations

import re

import numpy as np

from .flowdb import PRIORITY
from .routing import Router, snap_to_stream

AREA_TOL = 1.25        # station accepted when model and reported catchment areas agree within 25 %
MATCH_TOL = 1.10       # ... offered for an existing gauge within a few cells and 10 % of its area
NAME_TOL = 1.15        # ... or with the same name and within 15 %
TWIN_TOL = 1.15        # a station on the same river with almost the same catchment as a chosen gauge is skipped
MIN_MONTHS = 84
MIN_AREA_FRACTION = 0.02


def _acc_grid(data):
    area = data.grid.cell_area_m2().ravel()[data.cells] / 1e6
    acc, _ = Router(data.ds).accumulate(area)
    return data.to_grid(acc)


def _key(name):
    """First word of a station name, letters only: 'Senchi (below Akosombo)' -> 'senchi'."""
    w = re.sub(r"\(.*?\)", "", name or "").strip().split()
    return re.sub(r"[^a-z]", "", w[0].lower()) if w else ""


def _downstream(ds, cell, limit=100000):
    path, c = [], cell
    while c >= 0 and len(path) < limit:
        path.append(c)
        c = int(ds[c])
    return set(path)


def candidates(data, db, include_reservoirs=True) -> list:
    g = data.grid
    acc2d = _acc_grid(data)
    radius = max(2, int(round(0.25 / g.res)))
    basin_area = float(np.nanmax(acc2d))
    out = []
    for s in db.stations(bbox=(g.west, g.south, g.east, g.north)):
        if not s.get("area_km2") or s["area_km2"] > basin_area * AREA_TOL:
            continue
        if s["kind"] == "reservoir" and not include_reservoirs:
            continue
        if s["last"] < data.dates[0] or s["first"] > data.dates[-1]:
            continue
        try:
            r, c = g.rowcol(s["lon"], s["lat"])
            r, c = snap_to_stream(acc2d, data.basin, r, c, radius=radius, area_hint=s["area_km2"])
        except ValueError:
            continue
        a = float(acc2d[r, c])
        if not np.isfinite(a) or abs(np.log(a / s["area_km2"])) > np.log(AREA_TOL):
            continue
        q = db.series(s["id"], data.dates)
        n = int(np.isfinite(q).sum())
        if n == 0:
            continue
        yrs = [int(m[:4]) for m, v in zip(data.dates, q) if np.isfinite(v)]
        out.append({**s, "row": int(r), "col": int(c), "cell": data.compact_index(r, c),
                    "lon_model": float(g.lon[c]), "lat_model": float(g.lat[r]), "area_model_km2": a,
                    "months": n, "first": min(yrs), "last": max(yrs), "mean_m3s": float(np.nanmean(q)), "q": q})
    _suggest(data, out, radius, basin_area, ds=data.ds)
    for c_ in out:
        c_.pop("q"); c_.pop("q_all", None)
    out.sort(key=lambda x: -x["area_model_km2"])
    return out


def _suggest(data, cands, radius, basin_area, ds=None):
    """Decide a default for each station and say why. Fills match, already, suggest, notes, fill_from."""
    paths = {}
    near = lambda a, b, k: abs(a["row"] - b["row"]) <= k and abs(a["col"] - b["col"]) <= k
    lr = lambda x, y: abs(np.log(x / y))
    prio = lambda c: PRIORITY.get(c.get("source"), 0)

    def nested(a, b):
        if ds is None:
            return near(a, b, 25)
        for x, y in ((a, b), (b, a)):
            if x["cell"] not in paths:
                paths[x["cell"]] = _downstream(ds, x["cell"])
            if y["cell"] in paths[x["cell"]]:
                return True
        return False

    min_area = max(1000.0, MIN_AREA_FRACTION * basin_area)
    for c in cands:
        c["notes"] = []
        c["reservoir"] = c.get("kind") == "reservoir"
        if c["reservoir"]:
            c["notes"].append("reservoir record: may be storage volume or regulated outflow, not river flow")
        c["match"] = None
        c["already"] = False
        c["fill_from"] = []
        c["q_all"] = c["q"].copy()
    # existing gauges: each takes at most one station: same name first, then closest area, source, length
    for i, gg in enumerate(data.gauges):
        key = _key(gg["name"])
        opts = [c for c in cands if not c["reservoir"] and (c["cell"] == gg["cell"] or
                (near(c, gg, radius + 1) and lr(c["area_model_km2"], gg["area_km2"]) < np.log(MATCH_TOL)) or
                (key and _key(c["name"]) == key and lr(c["area_model_km2"], gg["area_km2"]) < np.log(NAME_TOL)))]
        for c in opts:
            c.setdefault("could_match", []).append({"id": gg["id"], "name": gg["name"]})
        if not opts:
            continue
        best = min(opts, key=lambda c: (_key(c["name"]) != key, round(lr(c["area_model_km2"], gg["area_km2"]), 2),
                                        -prio(c), -c["months"]))
        have = data.qobs[i]
        best["match"] = {"id": gg["id"], "name": gg["name"], "months_have": int(np.isfinite(have).sum())}
        best["already"] = not (np.isfinite(best["q"]) & ~np.isfinite(have)).any()
    # the same station from two sources: keep one (matched, then higher priority, then longer record) and
    # remember the others to fill its gaps
    order = sorted(cands, key=lambda c: (c["match"] is None, -prio(c), -c["months"], c["flag"]))
    kept = []
    for c in order:
        dup = next((k for k in kept if near(c, k, 3) and lr(c["area_model_km2"], k["area_model_km2"]) < np.log(1.05)), None)
        if dup is None:
            kept.append(c)
            continue
        c["dup_of"] = dup["name"]
        if _key(c["name"]) == _key(dup["name"]):
            c["notes"].append(f"same station as {dup['name']} ({dup['source']}); its records fill that one's gaps")
            dup["fill_from"].append(c["id"])
            dup["q_all"] = np.where(np.isfinite(dup["q_all"]), dup["q_all"], c["q"])
            if dup["already"] and (np.isfinite(c["q"]) & ~np.isfinite(dup["q"])).any():
                dup["already"] = False
        else:
            c["notes"].append(f"shares a model cell with {dup['name']}, which has the longer record")
    for c in cands:
        c["months_total"] = c["months"] + int(np.isfinite(c["q_all"]).sum() - np.isfinite(c["q"]).sum())
    used = {gg["cell"] for gg in data.gauges}
    chosen = [{"name": gg["name"], "cell": gg["cell"], "row": gg["row"], "col": gg["col"], "area_km2": gg["area_km2"]}
              for gg in data.gauges]
    for c in sorted(cands, key=lambda c: (-prio(c), -c["months"], -c["area_model_km2"])):
        if c["already"]:
            c["suggest"] = "skip"
        elif c["match"]:
            c["suggest"] = str(c["match"]["id"])
        else:
            why = []
            if c["reservoir"] or c.get("dup_of"):
                why.append("")
            if c["flag"] >= 2:
                why.append("flagged as suspicious by its source")
            if c["months_total"] < MIN_MONTHS:
                why.append(f"short record in the model period (under {MIN_MONTHS // 12} years)")
            if c["area_model_km2"] < min_area:
                why.append(f"small catchment for a basin-scale calibration (under {min_area:,.0f} km²)")
            if c["cell"] in used and not c.get("dup_of"):
                why.append("on the same model cell as a gauge already chosen")
            twin = next((t for t in chosen if lr(c["area_model_km2"], t["area_km2"]) < np.log(TWIN_TOL) and nested(c, t)), None)
            if twin is not None and not why:
                why.append(f"drains almost the same area as {twin['name']} ({twin['area_km2']:,.0f} km²), "
                           "so it would add a very small calibration zone")
            c["notes"] += [w for w in why if w]
            c["suggest"] = "skip" if why else "new"
            if c["suggest"] == "new":
                used.add(c["cell"])
                chosen.append({"name": c["name"], "cell": c["cell"], "row": c["row"], "col": c["col"],
                               "area_km2": c["area_model_km2"]})


def add(data, db, choices: dict, cands=None) -> dict:
    """choices: {station id: "new" | "<existing gauge id>"}. Changes `data` in place; returns a summary."""
    cands = {c["id"]: c for c in (cands if cands is not None else candidates(data, db))}
    added, merged, sources = [], [], set()
    gauges = list(data.gauges)
    q = [row for row in data.qobs]
    next_id = max([gg["id"] for gg in gauges] + [0]) + 1
    used_cells = {gg["cell"] for gg in gauges}
    for sid, action in choices.items():
        c = cands.get(sid)
        if c is None or action in (None, "", "skip"):
            continue
        series = db.series(sid, data.dates)
        for other in c.get("fill_from", []):
            o = db.series(other, data.dates)
            series = np.where(np.isfinite(series), series, o)
        sources.add(c["source"])
        if action == "new":
            if c["cell"] in used_cells:
                raise ValueError(f"{c['name']} falls on the same model cell as an existing gauge; use it for that gauge instead")
            gauges.append({"id": next_id, "name": c["name"], "river": c.get("river", ""), "lon": c["lon_model"],
                           "lat": c["lat_model"], "lon_given": c["lon"], "lat_given": c["lat"], "row": c["row"],
                           "col": c["col"], "cell": c["cell"], "area_km2": c["area_model_km2"],
                           "source": " + ".join([sid] + c.get("fill_from", []))})
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
            gauges[i] = {**gauges[i], "source": f"{gauges[i].get('source', '')} {sid}".strip()}
            merged.append(f"{gauges[i]['name']} from {c['name']} ({int(fill.sum())} months)")
    order = sorted(range(len(gauges)), key=lambda k: -gauges[k]["area_km2"])
    data.gauges = [gauges[k] for k in order]
    data.qobs = np.array([q[k] for k in order]).reshape(len(gauges), len(data.dates))
    if sources:
        cites = []
        for src in sorted(sources):
            info = next((x for x in db.summary()["sources"] if x["source"] == src), {})
            cites.append(info.get("citation") or info.get("name") or src)
        old = data.meta.get("discharge", "")
        data.meta = {**data.meta, "discharge": "; ".join(dict.fromkeys(([old] if old and old != "none" else []) + cites))}
    return {"added": added, "merged": merged}
