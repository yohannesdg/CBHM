"""FastAPI backend for the Volta distributed monthly water-balance model.

Run:  uvicorn backend.app:app --port 8000      (from the project root)
Then open http://localhost:8000
"""
from __future__ import annotations

import io
import json
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hydro import __version__
from hydro.abcd import HAVE_NUMBA
from hydro.calibrate import OBJECTIVES, Cancelled, calibrate
from hydro.dataset import Dataset
from hydro import models as wb_models
from hydro.model import FIELD_UNITS, FIELDS, GLOBAL_SPEC, VoltaModel, default_params, merge_params, spec_for

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FRONTEND = ROOT / "frontend"
CONFIG = json.loads((ROOT / "config" / "volta.json").read_text())



class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.name: str | None = None
        self.data: Dataset | None = None
        self.model: VoltaModel | None = None
        self.params = default_params()
        self.params_by_model: dict = {}     # last parameter set used with each model, for switching back
        self.results = None
        self.static: dict = {}
        self.calib = {"running": False}
        self.scenario = None
        self.calib_stop = False


S = State()


# ------------------------------------------------------------------------------- helpers
def _need_model():
    if S.model is None:
        raise HTTPException(409, "No dataset loaded. Generate the demo or load a prepared dataset first.")
    return S.model


def _need_results():
    _need_model()
    if S.results is None:
        _run(S.params)
    return S.results


def _grid_payload(values2d: np.ndarray, units: str = "", decimals: int = 3, **extra):
    v = np.asarray(values2d, float)
    finite = v[np.isfinite(v)]
    flat = np.round(v.ravel(), decimals)
    out = [None if not np.isfinite(x) else float(x) for x in flat]
    lo, hi = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (0.0, 1.0)
    return {"nrows": v.shape[0], "ncols": v.shape[1], "values": out, "units": units,
            "min": float(finite.min()) if finite.size else None, "max": float(finite.max()) if finite.size else None,
            "p2": lo, "p98": hi, **extra}


def _periods():
    """Calibration and validation periods. With observed discharge they follow the records: the years where
    at least a quarter of the measured gauges report are split about 60/40. Otherwise the config periods."""
    d = S.data.dates
    q = S.data.qobs
    if q.size and np.isfinite(q).any():
        measured = np.isfinite(q).any(axis=1)
        frac = np.isfinite(q[measured]).mean(axis=0)
        years = sorted({int(m[:4]) for m, f in zip(d, frac) if f >= 0.25})
        if len(years) >= 6:
            y0, y1 = years[0], years[-1]
            split = y0 + int(round(0.6 * (y1 - y0 + 1))) - 1
            return {"cal": [max(f"{y0}-01", d[0]), f"{split}-12"], "val": [f"{split + 1}-01", min(f"{y1}-12", d[-1])],
                    "source": "observed records"}
    cal = (max(CONFIG["calibration"]["start"], d[0]), min(CONFIG["calibration"]["end"], d[-1]))
    val = (max(CONFIG["validation"]["start"], d[0]), min(CONFIG["validation"]["end"], d[-1]))
    return {"cal": list(cal), "val": list(val), "source": "config"}


def _load(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found in data/")
    with S.lock:
        data = Dataset.load(path)
        model = VoltaModel(data)
        S.name, S.data, S.model = name, data, model
        S.params = default_params(S.params.get("model"))
        S.params_by_model = {}
        S.results = None
        S.scenario = None
        acc, _ = model.router.accumulate(model.area / 1e6)
        S.static = {
            "dem": (data.dem.copy(), "m"),
            "acc": (data.to_grid(acc), "km²"),
            "zone": (data.to_grid(model.zone.astype(float)), "zone id"),
            "lake": (data.to_grid(model.lake.astype(float)), "open water"),
        }
        S.static["dem"][0][~data.basin] = np.nan
        _run(S.params)


def _run(params):
    with S.lock:
        p = merge_params(params)
        t0 = time.time()
        S.results = S.model.run(p)
        S.params = p
        S.params_by_model[p["model"]] = p
        S.results.seconds = round(time.time() - t0, 2)
        return S.results


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _summary(res):
    per = _periods()
    return _clean({"dates": S.data.dates, "params": res.params, "balance": res.balance,
                   "basin": res.basin_series(), "gauges": res.gauge_report(per["cal"], per["val"]),
                   "periods": per, "seconds": getattr(res, "seconds", None)})


def _startup():
    DATA_DIR.mkdir(exist_ok=True)
    files = sorted(DATA_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.name for p in files]
    real = [n for n in names if not n.startswith("synthetic")]
    pick = "volta.npz" if "volta.npz" in names else (real[0] if real else (names[0] if names else None))
    if pick:
        try:
            _load(pick)
        except Exception:
            traceback.print_exc()


@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield


app = FastAPI(title="Volta Hydrologic Model", version=__version__, lifespan=lifespan)


# ------------------------------------------------------------------------------- datasets
@app.get("/api/status")
def status():
    return {"version": __version__, "numba": HAVE_NUMBA, "dataset": S.name,
            "datasets": sorted(p.name for p in DATA_DIR.glob("*.npz")),
            "loaded": S.model is not None, "calibrating": S.calib.get("running", False)}


class SynthReq(BaseModel):
    res: float = 0.05


@app.post("/api/datasets/synthetic")
def make_demo(req: SynthReq):
    from hydro.synthetic import make_synthetic
    if req.res not in (0.05, 0.1):
        raise HTTPException(400, "res must be 0.05 or 0.1")
    data = make_synthetic(res=req.res, log=lambda *a: None)
    name = f"synthetic_volta_{str(req.res).replace('.', 'p')}.npz"
    data.save(DATA_DIR / name)
    _load(name)
    return status()


class LoadReq(BaseModel):
    name: str


@app.post("/api/datasets/load")
def load_dataset(req: LoadReq):
    _load(Path(req.name).name)
    return status()


@app.post("/api/datasets/upload")
async def upload_dataset(request: Request, name: str = "uploaded.npz"):
    """Raw file body (no multipart), file name in ?name= ."""
    name = Path(name).name
    if not name.endswith(".npz"):
        raise HTTPException(400, "Upload a dataset .npz produced by prep/prep_volta.py")
    (DATA_DIR / name).write_bytes(await request.body())
    _load(name)
    return status()


def _spec_json(spec):
    return {k: {"default": v[0], "min": v[1], "max": v[2], "label": v[3]} for k, v in spec.items()}


def _models_json():
    return [{"key": m.key, "label": m.label, "description": m.description, "n_params": len(m.spec),
             "calibrate_default": list(m.calibrate_default)} for m in wb_models.MODELS.values()]


@app.get("/api/meta")
def meta():
    m = _need_model()
    d = S.data
    rr, cc = np.nonzero(d.basin)
    g = d.grid
    basin_bounds = {"north": g.north - rr.min() * g.res, "south": g.north - (rr.max() + 1) * g.res,
                    "west": g.west + cc.min() * g.res, "east": g.west + (cc.max() + 1) * g.res}
    return _clean({"basin_bounds": basin_bounds,
        "name": d.meta.get("name", S.name), "dataset": S.name, "meta": {k: v for k, v in d.meta.items() if k != "true_params"},
        "synthetic": bool(d.meta.get("synthetic")), "grid": d.grid.to_dict(), "bounds": d.grid.bounds(),
        "dates": d.dates, "n_cells": d.n, "gauges": d.gauges, "zones": m.zones_info(),
        "outlet": {"row": int(d.cells[m.outlet] // d.grid.ncols), "col": int(d.cells[m.outlet] % d.grid.ncols)},
        "param_spec": _spec_json(spec_for(S.params)), "model": S.params["model"], "models": _models_json(),
        "global_spec": {k: {"default": v[0], "min": v[1], "max": v[2], "label": v[3]} for k, v in GLOBAL_SPEC.items()},
        "params": S.params, "fields": {k: FIELD_UNITS[k] for k in FIELDS}, "periods": _periods(),
        "objectives": list(OBJECTIVES), "basin_area_km2": float(m.area.sum() / 1e6),
        "lake_cells": int(m.lake.sum()), "numba": HAVE_NUMBA, "hybas": d.meta.get("hybas"),
        "has_gauges": bool(d.gauges),
    })


@app.get("/api/static/{var}")
def static_map(var: str):
    _need_model()
    if var not in S.static:
        raise HTTPException(404, f"unknown static layer {var}")
    arr, units = S.static[var]
    return _grid_payload(arr, units, decimals=1)


# ------------------------------------------------------------------------------- model runs
class RunReq(BaseModel):
    params: dict | None = None


@app.post("/api/run")
def run(req: RunReq):
    _need_model()
    return _summary(_run(req.params or S.params))


@app.get("/api/results")
def results():
    return _summary(_need_results())


class ModelReq(BaseModel):
    model: str


@app.post("/api/model")
def switch_model(req: ModelReq):
    """Switch the water-balance model. Each model keeps its own last parameter set."""
    _need_model()
    try:
        md = wb_models.get(req.model)
    except KeyError as e:
        raise HTTPException(400, str(e))
    if S.calib.get("running"):
        raise HTTPException(409, "a calibration is running; wait for it or cancel it before switching models")
    p = S.params_by_model.get(md.key) or default_params(md.key)
    res = _run(p)
    return {"model": md.key, "param_spec": _spec_json(md.spec), "summary": _summary(res)}


@app.get("/api/compare")
def compare(model: str | None = None):
    """Run another model with its last parameters (defaults if never used) for comparison.
    The current model and its results are not changed."""
    _need_results()
    other = model or next(k for k in wb_models.MODELS if k != S.params["model"])
    try:
        md = wb_models.get(other)
    except KeyError as e:
        raise HTTPException(400, str(e))
    p = S.params_by_model.get(md.key) or default_params(md.key)
    with S.lock:
        t0 = time.time()
        res = S.model.run(p)
        secs = round(time.time() - t0, 2)
    per = _periods()
    gauges = [{"id": g["id"], "name": g["name"], "sim": g["sim"], "metrics": g["metrics"]}
              for g in res.gauge_report(per["cal"], per["val"])]
    return _clean({"model": md.key, "label": md.label, "calibrated": md.key in S.params_by_model and
                   bool(S.params_by_model[md.key]["zones"]), "params": p, "balance": res.balance,
                   "outlet_q": res.basin_series()["outlet_q"], "gauges": gauges, "seconds": secs})


def _map_values(var: str, mode: str, t: int | None, month: int | None):
    res = _need_results()
    if var not in res.fields:
        raise HTTPException(404, f"unknown field {var}")
    f = res.fields[var]
    years = len(S.data.dates) / 12.0
    flux = FIELD_UNITS[var].endswith("/month")
    if mode == "month":
        if t is None or not (0 <= t < f.shape[0]):
            raise HTTPException(400, "t out of range")
        v, units = f[t], FIELD_UNITS[var]
    elif mode == "annual":            # long-term mean annual total (fluxes) or mean (states)
        v = f.sum(0) / years if flux else f.mean(0)
        units = "mm/yr" if flux else FIELD_UNITS[var]
    elif mode == "clim":              # mean of a calendar month
        if month is None or not (1 <= month <= 12):
            raise HTTPException(400, "month must be 1..12")
        v, units = f[S.model.months == month].mean(0), FIELD_UNITS[var]
    elif mode == "anomaly":           # month t minus its calendar-month mean
        if t is None:
            raise HTTPException(400, "t required")
        v = f[t] - f[S.model.months == S.model.months[t]].mean(0)
        units = FIELD_UNITS[var] + " anomaly"
    else:
        raise HTTPException(400, "mode must be month|annual|clim|anomaly")
    return S.data.to_grid(v), units


@app.get("/api/map")
def map_layer(var: str = "runoff", mode: str = "annual", t: int | None = None, month: int | None = None):
    v, units = _map_values(var, mode, t, month)
    if var == "discharge":
        return _grid_payload(v, units, decimals=2, log=True)
    return _grid_payload(v, units, decimals=2, log=False)


@app.get("/api/cell")
def cell(row: int, col: int):
    res = _need_results()
    m = S.model
    k = S.data.compact_index(row, col)
    if k < 0:
        raise HTTPException(404, "cell outside basin")
    zone = int(m.zone[k])
    pa = m.param_arrays(res.params)
    acc = S.static["acc"][0][row, col]
    return _clean({"row": row, "col": col, "lon": float(S.data.grid.lon[col]), "lat": float(S.data.grid.lat[row]),
                   "zone": zone, "lake": bool(m.lake[k]), "dem": float(S.data.dem[row, col]),
                   "upstream_km2": float(acc), "model": res.params["model"],
                   "params": {p: float(pa[p][k]) for p in spec_for(res.params)},
                   "series": {f: np.round(res.fields[f][:, k].astype(float), 3).tolist() for f in FIELDS}})


@app.get("/api/zones/summary")
def zones_summary():
    """Long-term means per parameter zone (each gauge's incremental sub-catchment)."""
    res = _need_results()
    m, d = S.model, S.data
    years = len(d.dates) / 12.0
    per = _periods()
    report = {g["id"]: g for g in res.gauge_report(per["cal"], per["val"])}
    g = d.grid
    rows_all = d.cells // g.ncols
    cols_all = d.cells % g.ncols

    def fluxes(mask):
        w = m.area[mask] / m.area[mask].sum()
        mean = lambda a: float((a[:, mask].sum(0) * w).sum() / years)
        out = {k: mean(res.fields[k]) for k in ("precip", "pet", "et", "runoff", "recharge", "baseflow")}
        out["runoff_coef"] = out["runoff"] / out["precip"] if out["precip"] else None
        # Partition of rainfall, mm/yr over the area (P = ET + quickflow + recharge + dS on land;
        # recharge = baseflow + dG; on lake cells P = evaporation + net lake runoff).
        f = res.fields
        lake = m.lake[None, :]
        zero = np.zeros_like(f["et"])
        et_lake = mean(np.where(lake, f["et"], zero))
        quick = mean(np.where(lake, f["precip"] - f["et"], f["runoff"] - f["baseflow"]))
        dS = out["precip"] - out["et"] - quick - out["recharge"]
        out["partition"] = {"et_land": out["et"] - et_lake, "et_lake": et_lake, "quickflow": quick,
                            "recharge": out["recharge"], "baseflow": out["baseflow"],
                            "d_gw": out["recharge"] - out["baseflow"], "d_soil": dS}
        out["area_km2"] = float(m.area[mask].sum() / 1e6)
        return out

    rows = []
    for z in m.zones_info():
        mask = m.zone == z["id"]
        if not mask.any():
            continue
        r, c = rows_all[mask], cols_all[mask]
        row = {**z, **fluxes(mask),
               "bounds": {"north": g.north - r.min() * g.res, "south": g.north - (r.max() + 1) * g.res,
                          "west": g.west + c.min() * g.res, "east": g.west + (c.max() + 1) * g.res}}
        gi = next((i for i, gg in enumerate(d.gauges) if gg["id"] == z["id"]), None)
        if gi is not None:
            rep = report[z["id"]]
            obs, sim = d.qobs[gi], res.gauge_sim[gi]
            ok = np.isfinite(obs)
            row["gauge"] = {"index": gi, "name": d.gauges[gi]["name"], "area_km2": d.gauges[gi]["area_km2"],
                            "metrics": rep["metrics"],
                            "q_obs": float(obs[ok].mean()) if ok.any() else None,
                            "q_sim": float(sim[ok].mean()) if ok.any() else float(sim.mean())}
        rows.append(row)
    total = {"id": None, "name": "Whole basin", "cells": int(m.zone.size),
             "area_km2": float(m.area.sum() / 1e6), **fluxes(np.ones(m.zone.size, bool))}
    return _clean({"zones": rows, "total": total})


# ------------------------------------------------------------------------------- calibration
class CalReq(BaseModel):
    names: list[str] | None = None
    objective: str = "kge"
    cal_start: str | None = None
    cal_end: str | None = None
    maxiter: int = 30
    popsize: int = 10
    gauges: list[int] | None = None
    seed: int = 42


@app.post("/api/calibrate")
def start_calibration(req: CalReq):
    m = _need_model()
    if S.calib.get("running"):
        raise HTTPException(409, "calibration already running")
    md = wb_models.get(S.params["model"])
    names = req.names or list(md.calibrate_default)
    bad = [n for n in names if n not in md.spec]
    if bad or req.objective not in OBJECTIVES:
        raise HTTPException(400, f"{', '.join(bad) or req.objective} is not a {md.label} parameter or objective")
    per = _periods()
    cal = (req.cal_start or per["cal"][0], req.cal_end or per["cal"][1])
    from hydro.model import period_mask
    mask = period_mask(S.data.dates, *cal)
    usable = [g["name"] for i, g in enumerate(S.data.gauges)
              if (req.gauges is None or g["id"] in req.gauges) and np.isfinite(S.data.qobs[i][mask]).sum() >= 24]
    if not usable:
        have = [f"{g['name']} {int(np.isfinite(S.data.qobs[i]).sum())}" for i, g in enumerate(S.data.gauges)]
        raise HTTPException(400, f"Nothing to calibrate: no selected gauge has 24 months of observed discharge between "
                                 f"{cal[0]} and {cal[1]} (months on record: {', '.join(have) or 'no gauges'}). "
                                 "Add records in the Data tab first.")
    S.calib = {"running": True, "started": time.time(), "progress": None, "log": [], "error": None,
               "result": None, "request": (req.model_dump() if hasattr(req, "model_dump") else req.dict()), "cal": list(cal)}
    S.calib_stop = False
    base = merge_params(S.params)

    def progress(p):
        if "done_zone" in p:
            S.calib["log"].append(p["done_zone"])
        S.calib["progress"] = {k: v for k, v in p.items() if k != "done_zone"}

    def work():
        try:
            out = calibrate(m, base, names=tuple(names), objective=req.objective, cal_period=cal,
                            gauges=req.gauges, maxiter=req.maxiter, popsize=req.popsize, seed=req.seed,
                            progress=progress, should_stop=lambda: S.calib_stop)
            S.calib["result"] = out["params"]
        except Cancelled:
            S.calib["error"] = "cancelled"
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            S.calib["error"] = str(e)
        finally:
            S.calib["running"] = False
            S.calib["finished"] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.get("/api/calibrate/status")
def calibration_status():
    c = dict(S.calib)
    if c.get("started"):
        c["elapsed"] = round((c.get("finished") or time.time()) - c["started"], 1)
    return _clean(c)


@app.post("/api/calibrate/cancel")
def cancel_calibration():
    S.calib_stop = True
    return {"cancelling": True}


@app.post("/api/calibrate/apply")
def apply_calibration():
    if not S.calib.get("result"):
        raise HTTPException(409, "no calibration result")
    return _summary(_run(S.calib["result"]))


# ------------------------------------------------------------------------------- African basins
class Job:
    """One background preparation job at a time (download HydroBASINS, or prepare a basin)."""

    def __init__(self):
        self.state = {"running": False}

    def start(self, kind, fn, **info):
        if self.state.get("running"):
            raise HTTPException(409, "another download or preparation is still running")
        self.cancel = threading.Event()
        self.state = {"running": True, "kind": kind, "step": "starting", "log": [], "error": None,
                      "result": None, "download": None, "started": time.time(), **info}

        def log(*a):
            line = " ".join(str(x) for x in a)
            self.state["log"] = (self.state["log"] + [line])[-40:]

        def hook(name, got, total, rate):
            prev = self.state.get("download") or {}
            since = prev.get("since") if prev.get("file") == name else time.time()
            self.state["download"] = {"file": name, "waiting": got < 0, "mb": round(max(got, 0) / 1e6, 1),
                                      "total_mb": round(total / 1e6, 1) if total else None, "kBps": round(rate),
                                      "since": since, "for_s": round(time.time() - since)}

        def work():
            from prep import prep_volta as pv
            pv.PROGRESS_HOOK = hook
            pv.CANCEL = self.cancel
            pv.LOG_HOOK = log

            def step(st):
                if self.cancel.is_set():
                    raise pv.Cancelled()
                self.state.update(step=st, download=None)
            try:
                self.state["result"] = fn(log, step)
            except pv.Cancelled:
                self.state["error"] = "cancelled"
            except SystemExit as e:
                self.state["error"] = str(e)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.state["error"] = f"{type(e).__name__}: {e}"
            finally:
                pv.PROGRESS_HOOK = None
                pv.CANCEL = None
                pv.LOG_HOOK = None
                self.state["running"] = False
                self.state["finished"] = time.time()

        threading.Thread(target=work, daemon=True).start()


JOB = Job()


def _catalog():
    from hydro.basins import Catalog
    from prep.basin_data import BASIN_DIR
    return Catalog(BASIN_DIR)


@app.get("/api/catalog")
def catalog_status():
    from prep import prep_volta as pv
    cat = _catalog()
    raw = pv.RAW
    return {"levels": cat.available(), "ready": cat.available() == [2, 3, 4, 5],
            "files": {k: (raw / v).exists() for k, v in pv.FILES.items()},
            "hydrobasins_zip": (raw / "hybas_af_lev01-12_v1c.zip").exists(), "job": _job_view()}


@app.post("/api/catalog/build")
def catalog_build():
    from prep import basin_data as bd
    JOB.start("catalog", lambda log, step: (step("downloading HydroBASINS"), bd.ensure_catalog(log=log))[-1])
    return _job_view()


@app.get("/api/catalog/level/{level}")
def catalog_level(level: int):
    cat = _catalog()
    if level not in cat.available():
        raise HTTPException(404, "basin outlines not prepared yet")
    return FileResponse(cat.path(level), media_type="application/json")


@app.get("/api/basins/plan")
def basin_plan(level: int, id: int, upstream: bool = True, res: str = "auto"):
    from prep import basin_data as bd
    try:
        pl = bd.plan(level, id, upstream, res)
    except KeyError:
        raise HTTPException(404, "unknown basin id")
    pl["prepared"] = (DATA_DIR / pl["dataset"]).exists()
    pl.pop("ids", None)
    return _clean(pl)


class PrepareReq(BaseModel):
    level: int
    id: int
    upstream: bool = True
    res: str = "auto"


@app.post("/api/basins/prepare")
def basin_prepare(req: PrepareReq):
    from prep import basin_data as bd

    def run(log, step):
        out = bd.prepare(req.level, req.id, req.upstream, req.res, log=log, progress=step)
        _load(out.name)
        return out.name

    JOB.start("basin", run, level=req.level, id=req.id)
    return _job_view()


def _job_view():
    st = dict(JOB.state)
    if st.get("started"):
        st["elapsed"] = round((st.get("finished") or time.time()) - st["started"], 1)
    return _clean(st)


@app.get("/api/jobs/current")
def job_current():
    st = _job_view()
    d = st.get("download")
    if d and d.get("since"):
        d["for_s"] = round(time.time() - d["since"])
    return st


@app.post("/api/jobs/cancel")
def job_cancel():
    if JOB.state.get("running"):
        JOB.cancel.set()
    return _job_view()


# ------------------------------------------------------------------------------- exports
def _csv(header, rows):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join("" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else
                           (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + "\n")
    return buf.getvalue()


@app.get("/api/export/gauges.csv")
def export_gauges():
    res = _need_results()
    d = S.data
    header = ["date"]
    for g in d.gauges:
        header += [f"{g['name']}_obs_m3s", f"{g['name']}_sim_m3s"]
    rows = []
    for t, date in enumerate(d.dates):
        r = [date]
        for i in range(len(d.gauges)):
            r += [float(d.qobs[i, t]), float(res.gauge_sim[i, t])]
        rows.append(r)
    return PlainTextResponse(_csv(header, rows), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=volta_gauges.csv"})


@app.get("/api/export/basin.csv")
def export_basin():
    res = _need_results()
    b = res.basin_series()
    keys = [k for k in b if k != "outlet_q"]
    header = ["date"] + [f"{k}_{FIELD_UNITS[k].replace('/', 'per').replace(' ', '')}" for k in keys] + ["outlet_q_m3s"]
    rows = [[date] + [b[k][t] for k in keys] + [b["outlet_q"][t]] for t, date in enumerate(S.data.dates)]
    return PlainTextResponse(_csv(header, rows), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=volta_basin_balance.csv"})


@app.get("/api/export/annual.csv")
def export_annual():
    """Calendar-year volumes in million m3: basin fluxes (mm x area) and gauge flows (m3/s x seconds)."""
    import calendar as _cal
    res = _need_results()
    d = S.data
    b = res.basin_series()
    area_km2 = float(S.model.area.sum() / 1e6)
    flux = [k for k in ("precip", "pet", "et", "recharge", "baseflow", "runoff") if k in b]
    secs = np.array([_cal.monthrange(int(x[:4]), int(x[5:7]))[1] * 86400.0 for x in d.dates])
    years = sorted({x[:4] for x in d.dates})
    header = ["year", "months"] + [f"{k}_MCM" for k in flux] + ["outlet_outflow_MCM"]
    for g in d.gauges:
        header += [f"{g['name']}_obs_MCM", f"{g['name']}_sim_MCM"]
    rows = []
    for y in years:
        idx = np.array([i for i, x in enumerate(d.dates) if x[:4] == y])
        full = len(idx) == 12
        r = [y, len(idx)]
        r += [float(np.sum(np.asarray(b[k])[idx]) * area_km2 / 1000.0) for k in flux]
        r.append(float(np.sum(np.asarray(b["outlet_q"])[idx] * secs[idx]) / 1e6))
        for i in range(len(d.gauges)):
            o = d.qobs[i, idx]
            r.append(float(np.sum(o * secs[idx]) / 1e6) if full and np.isfinite(o).all() else None)
            r.append(float(np.sum(res.gauge_sim[i, idx] * secs[idx]) / 1e6))
        rows.append(r)
    return PlainTextResponse(_csv(header, rows), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=annual_volumes_MCM.csv"})


# ------------------------------------------------------------------ open gauge records (ADHI)
@app.get("/api/adhi/candidates")
def adhi_candidates():
    """ADHI stations inside the open basin. 409 until the ADHI files have been downloaded."""
    from prep import adhi
    _need_model()
    if not adhi.files_ready():
        raise HTTPException(409, "ADHI files not downloaded yet")
    c = adhi.candidates(S.data)
    return _clean({"candidates": c, "citation": adhi.CITATION,
                   "gauges": [{"id": g["id"], "name": g["name"],
                               "months": int(np.isfinite(S.data.qobs[i]).sum())} for i, g in enumerate(S.data.gauges)]})


@app.post("/api/adhi/fetch")
def adhi_fetch():
    from prep import adhi

    def run(log, step):
        step("downloading ADHI")
        adhi.ensure_files(log=log)
        return "ok"

    JOB.start("adhi", run)
    return _job_view()


class AdhiAddReq(BaseModel):
    choices: dict


@app.post("/api/adhi/add")
def adhi_add(req: AdhiAddReq):
    """Add or merge ADHI records, save the dataset and reopen it (gauges define the parameter zones)."""
    from prep import adhi
    _need_model()
    if S.calib.get("running"):
        raise HTTPException(409, "a calibration is running")
    if not S.name or S.data.meta.get("synthetic"):
        raise HTTPException(400, "Open a real basin dataset first; the demo basin has its own synthetic records.")
    try:
        summary = adhi.add(S.data, req.choices)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    if not summary["added"] and not summary["merged"]:
        raise HTTPException(400, "Nothing new to add: the selected records are already in this dataset.")
    old = S.params
    S.data.save(DATA_DIR / S.name)
    _load(S.name)
    keep = merge_params({"model": old["model"], "global": old["global"], "kw": old["kw"], "k_route": old["k_route"]})
    res = _run(keep)
    parts = []
    if summary["added"]:
        parts.append("added " + ", ".join(summary["added"]))
    if summary["merged"]:
        parts.append("filled " + ", ".join(summary["merged"]))
    return {"message": "ADHI records: " + "; ".join(parts) + ". Sub-basin values were reset because the gauges "
            "now define the parameter zones.", "summary": _summary(res)}


@app.post("/api/observed")
async def add_observed(request: Request, name: str = "observed.csv", gauge: str | None = None):
    """Raw CSV or GRDC export in the body. Merges monthly means into the gauges' records and saves the dataset."""
    from hydro.observed import parse_observed
    _need_model()
    d = S.data
    if not d.gauges:
        raise HTTPException(400, "This basin has no gauges, so observed discharge can't be attached to it.")
    raw = await request.body()
    try:
        q, names = parse_observed(raw, d.gauges, d.dates, gauge)
    except ValueError as e:
        raise HTTPException(400, f"{Path(name).name}: {e}")
    with S.lock:
        got = np.isfinite(q)
        d.qobs = np.where(got, q, d.qobs)
        if S.name and not d.meta.get("synthetic"):
            d.save(DATA_DIR / S.name)
        res = _run(S.params)
    counts = [f"{n} {int(np.isfinite(q[i]).sum())} months" for i, n in
              ((i, g["name"]) for i, g in enumerate(d.gauges)) if n in names]
    msg = f"Added observed discharge from {Path(name).name}: " + ", ".join(counts) + "."
    if not got.any():
        msg += " None of the dates fall inside the model period (" + d.dates[0] + " to " + d.dates[-1] + ")."
    return {"message": msg, "summary": _summary(res)}


# ------------------------------------------------------------------------------- climate scenarios
class ScenarioReq(BaseModel):
    rain_pct: float = 0.0
    temp_c: float = 0.0
    pet_pct: float = 0.0


def _flow_stats(res):
    """Outlet and gauge flow statistics from a model run."""
    import calendar as _cal
    d = S.data
    q = np.asarray(res.basin_series()["outlet_q"], float)
    secs = np.array([_cal.monthrange(int(x[:4]), int(x[5:7]))[1] * 86400.0 for x in d.dates])
    months = np.array([int(x[5:7]) for x in d.dates])
    clim = [float(q[months == m].mean()) for m in range(1, 13)]
    years = sorted({x[:4] for x in d.dates})
    annual = [float(sum(q[i] * secs[i] for i, x in enumerate(d.dates) if x[:4] == y) / 1e6) for y in years]
    srt = sorted(clim)
    gauges = [{"name": g["name"], "mean_m3s": float(np.nanmean(res.gauge_sim[i]))} for i, g in enumerate(d.gauges)]
    b = res.balance["per_year"]
    return {"per_year": b, "runoff_ratio": res.balance["runoff_ratio"], "outlet_mean_m3s": float(q.mean()),
            "outlet_mcm_per_year": float(np.mean(annual)), "seasonal_m3s": clim, "years": years, "annual_mcm": annual,
            "wettest_month_m3s": srt[-1], "dry_season_m3s": float(np.mean(srt[:3])), "gauges": gauges}


@app.post("/api/scenario")
def scenario(req: ScenarioReq):
    """Run the current model and parameters under changed rainfall, temperature and PET; compare with the baseline."""
    base = _need_results()
    for v, lo, hi, what in ((req.rain_pct, -60, 100, "rainfall change"), (req.temp_c, -5, 8, "temperature change"),
                            (req.pet_pct, -50, 100, "PET change")):
        if not (lo <= v <= hi):
            raise HTTPException(400, f"{what} must be between {lo} and {hi}")
    with S.lock:
        t0 = time.time()
        sc_model = S.model.with_climate(req.rain_pct, req.temp_c, req.pet_pct)
        sc = sc_model.run(S.params)
        secs = round(time.time() - t0, 2)
    b, x = _flow_stats(base), _flow_stats(sc)
    pct = lambda new, old: None if not old else 100.0 * (new / old - 1.0)
    change = {"P": pct(x["per_year"]["P"], b["per_year"]["P"]), "PET": pct(x["per_year"]["PET"], b["per_year"]["PET"]),
              "ET": pct(x["per_year"]["ET"], b["per_year"]["ET"]), "Q": pct(x["per_year"]["Q"], b["per_year"]["Q"]),
              "dry_season": pct(x["dry_season_m3s"], b["dry_season_m3s"]),
              "wettest_month": pct(x["wettest_month_m3s"], b["wettest_month_m3s"])}
    elasticity = None
    if abs(req.rain_pct) >= 1 and abs(req.temp_c) < 1e-9 and abs(req.pet_pct) < 1e-9 and change["Q"] is not None:
        elasticity = change["Q"] / req.rain_pct
    S.scenario = {"request": req.model_dump() if hasattr(req, "model_dump") else req.dict(),
                  "baseline": b, "scenario": x, "dates": S.data.dates,
                  "outlet_base": base.basin_series()["outlet_q"], "outlet_scen": sc.basin_series()["outlet_q"],
                  "gauge_base": base.gauge_sim, "gauge_scen": sc.gauge_sim}
    return _clean({"request": S.scenario["request"], "model": S.params["model"], "baseline": b, "scenario": x,
                   "change_pct": change, "elasticity": elasticity, "seconds": secs})


@app.get("/api/export/scenario.csv")
def export_scenario():
    sc = getattr(S, "scenario", None)
    if not sc:
        raise HTTPException(404, "run a scenario first")
    r = sc["request"]
    header = ["date", "outlet_baseline_m3s", "outlet_scenario_m3s"]
    for g in S.data.gauges:
        header += [f"{g['name']}_baseline_m3s", f"{g['name']}_scenario_m3s"]
    rows = []
    for t, date in enumerate(sc["dates"]):
        row = [date, float(sc["outlet_base"][t]), float(sc["outlet_scen"][t])]
        for i in range(len(S.data.gauges)):
            row += [float(sc["gauge_base"][i, t]), float(sc["gauge_scen"][i, t])]
        rows.append(row)
    name = f"scenario_rain{r['rain_pct']:+g}pct_temp{r['temp_c']:+g}C.csv".replace("+", "p").replace("-", "m")
    return PlainTextResponse(_csv(header, rows), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={name}"})


# ------------------------------------------------------------------------------- saved runs
# A saved run is a named parameter set (and, optionally, a climate change). Its flows are recomputed when
# asked for, so a run always matches the current dataset and its scores use the current observed records.
def _runs_path():
    _need_model()
    return DATA_DIR / "runs" / f"{Path(S.name).stem}.json"


def _runs_read() -> list:
    p = _runs_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return []


def _runs_write(runs: list):
    p = _runs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(runs), indent=1))
    tmp.replace(p)


def _run_or_404(run_id: str) -> dict:
    r = next((x for x in _runs_read() if x["id"] == run_id), None)
    if r is None:
        raise HTTPException(404, "saved run not found")
    return r


def _climate_label(c) -> str:
    if not c:
        return ""
    parts = []
    if c.get("rain_pct"):
        parts.append(f"rain {c['rain_pct']:+g}%")
    if c.get("temp_c"):
        parts.append(f"temp {c['temp_c']:+g}°C")
    if c.get("pet_pct"):
        parts.append(f"PET {c['pet_pct']:+g}%")
    return ", ".join(parts)


class SaveRunReq(BaseModel):
    label: str = ""
    climate: dict | None = None        # {"rain_pct", "temp_c", "pet_pct"}; None keeps today's climate
    note: str = ""


@app.get("/api/runs")
def list_runs():
    return {"runs": _runs_read()}


@app.post("/api/runs")
def save_run(req: SaveRunReq):
    res = _need_results()
    clim = None
    if req.climate:
        clim = {k: float(req.climate.get(k, 0) or 0) for k in ("rain_pct", "temp_c", "pet_pct")}
        if not any(clim.values()):
            clim = None
    runs = _runs_read()
    md = wb_models.get(S.params["model"])
    label = (req.label or "").strip()[:80] or f"Run {len(runs) + 1}"
    rid = f"r{int(time.time() * 1000):x}"
    bal = res.balance
    if clim:
        with S.lock:
            bal = S.model.with_climate(clim["rain_pct"], clim["temp_c"], clim["pet_pct"]).run(S.params).balance
    per = _periods()
    kges = [g["metrics"]["all"]["kge"] for g in res.gauge_report(per["cal"], per["val"])
            if g["metrics"]["all"]["n"]] if not clim else []
    entry = {"id": rid, "label": label, "created": time.strftime("%Y-%m-%d %H:%M"), "model": md.key,
             "model_label": md.label, "params": merge_params(S.params), "climate": clim,
             "climate_label": _climate_label(clim), "note": req.note[:300],
             "summary": {"P": bal["per_year"]["P"], "ET": bal["per_year"]["ET"], "Q": bal["per_year"]["Q"],
                         "runoff_ratio": bal["runoff_ratio"],
                         "median_kge": float(np.median(kges)) if kges else None, "gauges_scored": len(kges)}}
    runs.append(entry)
    _runs_write(runs)
    return entry


class RenameRunReq(BaseModel):
    label: str


@app.patch("/api/runs/{run_id}")
def rename_run(run_id: str, req: RenameRunReq):
    runs = _runs_read()
    r = next((x for x in runs if x["id"] == run_id), None)
    if r is None:
        raise HTTPException(404, "saved run not found")
    r["label"] = req.label.strip()[:80] or r["label"]
    _runs_write(runs)
    return r


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    runs = _runs_read()
    keep = [x for x in runs if x["id"] != run_id]
    if len(keep) == len(runs):
        raise HTTPException(404, "saved run not found")
    _runs_write(keep)
    return {"deleted": run_id}


@app.get("/api/runs/{run_id}")
def run_series(run_id: str):
    """Recompute a saved run for comparison: same shape as /api/compare. Current results are not changed."""
    _need_results()
    r = _run_or_404(run_id)
    p = merge_params(r["params"], r["model"])
    c = r.get("climate") or {}
    with S.lock:
        t0 = time.time()
        m = S.model.with_climate(c.get("rain_pct", 0), c.get("temp_c", 0), c.get("pet_pct", 0)) if c else S.model
        res = m.run(p)
        secs = round(time.time() - t0, 2)
    per = _periods()
    gauges = [{"id": g["id"], "name": g["name"], "sim": g["sim"], "metrics": g["metrics"]}
              for g in res.gauge_report(per["cal"], per["val"])]
    label = r["label"] + (f" ({r['climate_label']})" if r.get("climate_label") else "")
    return _clean({"run_id": r["id"], "model": r["model"], "label": label, "climate": r.get("climate"),
                   "calibrated": bool(p["zones"]), "params": p, "balance": res.balance,
                   "outlet_q": res.basin_series()["outlet_q"], "gauges": gauges, "seconds": secs})


@app.post("/api/runs/{run_id}/apply")
def apply_run(run_id: str):
    """Make a saved run's parameters (and model) the current ones. Climate changes are not applied."""
    _need_model()
    if S.calib.get("running"):
        raise HTTPException(409, "a calibration is running; wait for it or cancel it first")
    r = _run_or_404(run_id)
    md = wb_models.get(r["model"])
    res = _run(merge_params(r["params"], md.key))
    return {"model": md.key, "param_spec": _spec_json(md.spec), "summary": _summary(res)}


@app.get("/api/export/params.json")
def export_params():
    return JSONResponse(S.params, headers={"Content-Disposition": "attachment; filename=volta_params.json"})


@app.get("/api/export/map.asc")
def export_map(var: str = "runoff", mode: str = "annual", t: int | None = None, month: int | None = None):
    v, units = _map_values(var, mode, t, month)
    g = S.data.grid
    head = (f"ncols {g.ncols}\nnrows {g.nrows}\nxllcorner {g.west}\nyllcorner {g.south}\n"
            f"cellsize {g.res}\nNODATA_value -9999\n")
    body = "\n".join(" ".join("-9999" if not np.isfinite(x) else f"{x:.4f}" for x in row) for row in v)
    name = f"volta_{var}_{mode}{'' if t is None else '_' + S.data.dates[t]}.asc"
    return PlainTextResponse(head + body + "\n", headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/api/export/dataset")
def export_dataset():
    _need_model()
    return FileResponse(DATA_DIR / S.name, filename=S.name)


# ------------------------------------------------------------------------------- frontend
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
