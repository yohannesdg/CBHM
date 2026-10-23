"""Distributed model: forcing -> Modified Hargreaves ETo -> abcd per cell -> D8 routing."""
from __future__ import annotations

import calendar
import copy

import numpy as np

from . import metrics
from . import models as wb_models
from .dataset import Dataset
from .pet import modified_hargreaves, monthly_ra
from .routing import Router, linear_reservoir

# name: (default, lower, upper, description). PARAM_SPEC is the abcd set, kept for older callers;
# use spec_for(params) for the model a parameter set belongs to.
PARAM_SPEC = wb_models.MODELS["abcd"].spec
GLOBAL_SPEC = {
    "kw": (1.05, 0.70, 1.30, "Open-water evaporation coefficient (x ETo) for lake cells"),
    "k_route": (0.0, 0.0, 6.0, "Linear-reservoir lag at gauges, months (0 = off)"),
}
FIELDS = ("precip", "pet", "et", "soil", "gw", "recharge", "baseflow", "runoff", "discharge")
FIELD_UNITS = {"precip": "mm/month", "pet": "mm/month", "et": "mm/month", "soil": "mm", "gw": "mm",
               "recharge": "mm/month", "baseflow": "mm/month", "runoff": "mm/month", "discharge": "m³/s"}


def spec_for(params: dict | None) -> dict:
    return wb_models.get((params or {}).get("model")).spec


def default_params(model: str | None = None) -> dict:
    md = wb_models.get(model)
    return {"model": md.key, "global": {k: v[0] for k, v in md.spec.items()},
            "zones": {},
            **{k: v[0] for k, v in GLOBAL_SPEC.items()}}


def merge_params(params: dict | None, model: str | None = None) -> dict:
    """Fill a (partial) parameter set with defaults. Keys that don't belong to the model are dropped."""
    params = params or {}
    p = default_params(model or params.get("model"))
    spec = wb_models.get(p["model"]).spec
    p["global"].update({k: float(v) for k, v in (params.get("global") or {}).items() if k in spec})
    for z, vals in (params.get("zones") or {}).items():
        zv = {k: float(v) for k, v in vals.items() if k in spec and v is not None}
        if zv:
            p["zones"][str(z)] = zv
    for k in GLOBAL_SPEC:
        if k in params and params[k] is not None:
            p[k] = float(params[k])
    return p


def period_mask(dates: list, start: str | None, end: str | None) -> np.ndarray:
    d = np.array(dates)
    m = np.ones(d.size, bool)
    if start:
        m &= d >= start
    if end:
        m &= d <= end
    return m


class Results:
    def __init__(self, model: "VoltaModel", params: dict, fields: dict, gauge_sim: np.ndarray,
                 gauge_raw: np.ndarray, balance: dict):
        self.model = model
        self.params = params
        self.fields = fields          # name -> (T, n) float32
        self.gauge_sim = gauge_sim    # (G, T) after linear reservoir
        self.gauge_raw = gauge_raw    # (G, T) routed, before reservoir
        self.balance = balance

    def basin_series(self) -> dict:
        w = self.model.area / self.model.area.sum()
        out = {k: (self.fields[k] @ w).astype(float).tolist() for k in FIELDS if k != "discharge"}
        out["outlet_q"] = self.fields["discharge"][:, self.model.outlet].astype(float).tolist()
        return out

    def gauge_report(self, cal=(None, None), val=(None, None)) -> list:
        ds = self.model.data
        mc = period_mask(ds.dates, *cal)
        mv = period_mask(ds.dates, *val)
        rep = []
        for i, g in enumerate(ds.gauges):
            sim, obs = self.gauge_sim[i], ds.qobs[i]
            rep.append({**{k: g[k] for k in ("id", "name", "river", "lon", "lat", "area_km2")},
                        "sim": _clean(sim), "obs": _clean(obs),
                        "metrics": {"all": metrics.all_metrics(sim, obs),
                                    "cal": metrics.all_metrics(np.where(mc, sim, np.nan), obs),
                                    "val": metrics.all_metrics(np.where(mv, sim, np.nan), obs)}})
        return rep


def _clean(a):
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), 4) for v in a]


class VoltaModel:
    def __init__(self, data: Dataset):
        self.data = data
        g = data.grid
        cells = data.cells
        self.lake = data.lake.ravel()[cells]
        self.land = ~self.lake
        self.router = Router(data.ds, self.lake)
        self.area = g.cell_area_m2().ravel()[cells]                       # m2
        self.lat = g.lat2d().ravel()[cells]
        y = np.array([int(s[:4]) for s in data.dates])
        m = np.array([int(s[5:7]) for s in data.dates])
        self.months = m
        self.ndays = np.array([calendar.monthrange(a, b)[1] for a, b in zip(y, m)], float)
        self.secs = self.ndays * 86400.0
        self.precip = np.asarray(data.precip, float)
        ra = monthly_ra(self.lat, m)
        self.ra = ra.astype(np.float32)
        self.eto = modified_hargreaves(data.tmax, data.tmin, self.precip, ra, self.ndays)
        self.outlet = int(np.flatnonzero(data.ds < 0)[0])
        gcells = [gg["cell"] for gg in data.gauges]
        gids = [gg["id"] for gg in data.gauges]
        self.zone = self.router.zones_from_gauges(gcells, gids, default=0)
        if data.zone_ids is not None and not gcells:     # HydroBASINS units as parameter zones
            self.zone = np.asarray(data.zone_ids, np.int64)
        pos = np.empty(self.router.n, np.int64)
        pos[self.router.order] = np.arange(self.router.n)
        # upstream gauges first
        self.gauge_order = sorted(range(len(gcells)), key=lambda i: pos[gcells[i]])
        self.upstream_gauges = {}
        for i, gg in enumerate(data.gauges):
            self.upstream_gauges[gg["id"]] = [
                j for j, h in enumerate(data.gauges)
                if j != i and data.ds[h["cell"]] >= 0 and self.zone[data.ds[h["cell"]]] == gg["id"]]

    # ---------------------------------------------------------------------------------
    def param_arrays(self, params: dict, idx: np.ndarray | None = None) -> dict:
        p = merge_params(params)
        zone = self.zone if idx is None else self.zone[idx]
        out = {}
        for k in spec_for(p):
            arr = np.full(zone.size, p["global"][k])
            for z, vals in p["zones"].items():
                if k in vals:
                    arr[zone == int(z)] = vals[k]
            out[k] = arr
        return out

    def zones_info(self) -> list:
        if self.data.zone_ids is not None and not self.data.gauges:
            names = self.data.zone_names or {}
            out = []
            for z in np.unique(self.zone):
                m = self.zone == z
                out.append({"id": int(z), "name": names.get(int(z), f"Zone {z}"), "cells": int(m.sum()),
                            "area_km2": float(self.area[m].sum() / 1e6), "upstream_gauges": []})
            return out
        info = [{"id": 0, "name": "Ungauged / downstream", "cells": int((self.zone == 0).sum()),
                 "area_km2": float(self.area[self.zone == 0].sum() / 1e6)}]
        for g in self.data.gauges:
            m = self.zone == g["id"]
            info.append({"id": g["id"], "name": f"{g['name']} (incremental)", "cells": int(m.sum()),
                         "area_km2": float(self.area[m].sum() / 1e6),
                         "upstream_gauges": [self.data.gauges[j]["name"] for j in self.upstream_gauges[g["id"]]]})
        return info

    def local_runoff(self, params: dict, idx: np.ndarray):
        """Run the parameter set's water-balance model on cells `idx`.

        Returns (runoff_mm (T,k), lake_demand_mm (T,k), model outputs, pa, pet)."""
        p = merge_params(params)
        md = wb_models.get(p["model"])
        pa = self.param_arrays(p, idx)
        P = self.precip[:, idx]
        eto = self.eto[:, idx]
        lake = self.lake[idx]
        land = ~lake
        pet = eto * pa["kc"][None, :]
        runoff = np.zeros_like(P)
        res = md.run(P[:, land], pet[:, land], {k: v[land] for k, v in pa.items()})
        runoff[:, land] = res["runoff"]
        demand = np.zeros_like(P)
        demand[:, lake] = p["kw"] * eto[:, lake]
        runoff[:, lake] = P[:, lake] - demand[:, lake]
        return runoff, demand, res, pa, pet

    def run(self, params: dict | None = None) -> Results:
        p = merge_params(params)
        n = self.router.n
        idx = np.arange(n)
        runoff, demand, res, pa, pet = self.local_runoff(p, idx)
        conv = self.area[None, :] / 1000.0 / self.secs[:, None]                  # mm/month -> m3/s
        Q, deficit = self.router.accumulate((runoff * conv).T)
        Q = Q.T
        land, lake = self.land, self.lake
        et = np.zeros_like(runoff)
        et[:, land] = res["et"]
        if deficit is not None:
            et[:, lake] = demand[:, lake] - (deficit.T[:, lake] / conv[:, lake])
        soil = np.zeros_like(runoff); soil[:, land] = res["soil"]
        gw = np.zeros_like(runoff); gw[:, land] = res["gw"]
        rch = np.zeros_like(runoff); rch[:, land] = res["recharge"]
        bf = np.zeros_like(runoff); bf[:, land] = res["baseflow"]
        f32 = lambda a: np.asarray(a, np.float32)
        fields = {"precip": f32(self.precip), "pet": f32(pet), "et": f32(et), "soil": f32(soil), "gw": f32(gw),
                  "recharge": f32(rch), "baseflow": f32(bf),
                  "runoff": f32(np.where(lake[None, :], self.precip - et, runoff)), "discharge": f32(Q)}
        gcells = [g["cell"] for g in self.data.gauges]
        raw = Q[:, gcells].T if gcells else np.zeros((0, Q.shape[0]))
        sim = linear_reservoir(raw, p["k_route"]) if len(gcells) else raw

        # ---- basin water balance (mm over basin area, whole period) ----
        A = self.area.sum()
        w = self.area / A
        P_tot = float((self.precip.sum(0) * w).sum())
        ET_tot = float((et.sum(0) * w).sum())
        Qout_mm = float((Q[:, self.outlet] * self.secs).sum() / A * 1000.0)
        s_end = soil[-1]; g_end = gw[-1]
        s0 = np.zeros(n); g0 = np.zeros(n)
        s0[land] = res["s0"]; g0[land] = res["g0"]
        dS = float(((s_end - s0) * w).sum())
        dG = float(((g_end - g0) * w).sum())
        years = len(self.data.dates) / 12.0
        bal = {"P": P_tot, "ET": ET_tot, "Q": Qout_mm, "dS": dS, "dG": dG,
               "residual": P_tot - ET_tot - Qout_mm - dS - dG,
               "years": years, "basin_area_km2": float(A / 1e6),
               "per_year": {"P": P_tot / years, "ET": ET_tot / years, "Q": Qout_mm / years,
                            "PET": float((pet.sum(0) * w).sum()) / years},
               "runoff_ratio": Qout_mm / P_tot if P_tot else None,
               "lake_evap_unmet_mm": float(((deficit.T / conv).sum(0) * w).sum()) if deficit is not None else 0.0}
        return Results(self, p, fields, sim, raw, bal)

    def with_climate(self, rain_pct: float = 0.0, temp_c: float = 0.0, pet_pct: float = 0.0) -> "VoltaModel":
        """The same basin under a simple climate change: rainfall scaled by rain_pct %, Tmax and Tmin raised
        by temp_c degC (Modified Hargreaves PET recomputed, so the rain term of the formula follows the new
        rainfall), and PET then scaled by pet_pct %. Routing, lakes and zones are shared with this model."""
        sc = copy.copy(self)
        sc.precip = self.precip * (1.0 + rain_pct / 100.0)
        eto = modified_hargreaves(self.data.tmax + temp_c, self.data.tmin + temp_c, sc.precip, self.ra, self.ndays)
        sc.eto = eto * (1.0 + pet_pct / 100.0)
        sc.climate = {"rain_pct": rain_pct, "temp_c": temp_c, "pet_pct": pet_pct}
        return sc

    def copy_params(self, p):
        return copy.deepcopy(merge_params(p))
