"""Zone-by-zone calibration with differential evolution, upstream gauges first.

Each gauge's incremental sub-catchment (zone) gets its own parameter set. When zone g is
calibrated, its upstream gauges keep their already-calibrated parameters. Their routed flows
are injected as inflow, so each objective evaluation only runs the cells of zone g. The
result is exact (lake clipping included) and fast.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution

from . import metrics
from . import models as wb_models
from .model import VoltaModel, merge_params, period_mask
from .abcd import climatology
from .routing import Router, linear_reservoir

OBJECTIVES = {
    "kge": lambda s, o: 1.0 - metrics.kge(s, o),
    "nse": lambda s, o: 1.0 - metrics.nse(s, o),
    "log_nse": lambda s, o: 1.0 - metrics.log_nse(s, o),
    "kge_lognse": lambda s, o: 1.0 - 0.5 * (metrics.kge(s, o) + metrics.log_nse(s, o)),
}


class Cancelled(Exception):
    pass


class ZoneProblem:
    def __init__(self, model: VoltaModel, gauge_index: int, base_params: dict, inflow_q: dict):
        self.model = model
        g = model.data.gauges[gauge_index]
        self.gid = g["id"]
        self.zkey = str(self.gid)
        self.idx = np.flatnonzero(model.zone == self.gid)
        local = np.full(model.router.n, -1, np.int64)
        local[self.idx] = np.arange(self.idx.size)
        d = model.data.ds[self.idx]
        sub_ds = np.where(d >= 0, local[np.maximum(d, 0)], -1)
        self.lake = model.lake[self.idx]
        self.router = Router(sub_ds, self.lake)
        self.gcell = int(local[g["cell"]])
        self.base = merge_params(base_params)
        T = model.precip.shape[0]
        self.inflow = np.zeros((self.idx.size, T))
        for j, q in inflow_q.items():         # inject upstream gauge flow where it enters the zone
            entry = local[model.data.ds[model.data.gauges[j]["cell"]]]
            self.inflow[entry] += q
        land_mask = ~self.lake
        land = self.idx[land_mask]
        self.P = np.ascontiguousarray(model.precip[:, land])
        self.E = np.ascontiguousarray(model.eto[:, land])
        self.pclim = climatology(self.P)
        self.eclim = climatology(self.E)
        self.secs = model.secs
        # --- compress the zone network to lake nodes (+ gauge node) -------------------
        nz = self.idx.size
        order = self.router.order
        next_lake = np.full(nz, -1, np.int64)
        for cc in order[::-1]:
            dn = sub_ds[cc]
            if dn >= 0:
                next_lake[cc] = dn if self.lake[dn] else next_lake[dn]
        self.lake_order = order[self.lake[order]]              # topological
        nl = self.lake_order.size
        kpos = np.full(nz, -1, np.int64)
        kpos[self.lake_order] = np.arange(nl)
        node = np.where(next_lake >= 0, kpos[np.maximum(next_lake, 0)], nl)   # where outflow goes
        self.nnodes = nl + 1
        self.node_land = node[land_mask]
        self.lake_dest = node[self.lake_order]
        self.w_land = model.area[land] / 1000.0
        lk = self.idx[self.lake_order]
        self.P_lake = model.precip[:, lk].T
        self.E_lake = model.eto[:, lk].T
        self.conv_lake = (model.area[lk] / 1000.0)[:, None] / model.secs[None, :]
        # upstream inflow enters at the entry cell's node (a lake cell itself, or its next node)
        self.inflow_nodes = np.zeros((self.nnodes, T))
        for cc in np.flatnonzero(self.inflow.any(axis=1)):
            k = kpos[cc] if self.lake[cc] else node[cc]
            self.inflow_nodes[k] += self.inflow[cc]

    def simulate(self, params: dict) -> np.ndarray:
        p = merge_params(params)
        zp = {**p["global"], **p["zones"].get(self.zkey, {})}
        kc = zp["kc"]
        md = wb_models.get(p["model"])
        X = md.zone_nodes(self.P, self.E * kc, self.pclim, self.eclim * kc, zp, self.w_land, self.node_land,
                          self.nnodes)
        X /= self.secs[None, :]
        X += self.inflow_nodes
        nl = self.nnodes - 1
        if nl:
            X[:nl] += (self.P_lake - p["kw"] * self.E_lake) * self.conv_lake
            for k in range(nl):                              # lakes in topological order
                X[self.lake_dest[k]] += np.maximum(X[k], 0.0)
        return linear_reservoir(X[nl], p["k_route"])


def calibrate(model: VoltaModel, params: dict | None = None, names=None,
              objective="kge", cal_period=(None, None), gauges=None, maxiter=40, popsize=12,
              seed=42, progress=None, should_stop=None, bounds_override=None) -> dict:
    """Returns {"params": calibrated params, "log": [...per zone results...]}."""
    params = merge_params(params)
    md = wb_models.get(params["model"])
    names = tuple(names) if names else md.calibrate_default
    unknown = [k for k in names if k not in md.spec]
    if unknown:
        raise ValueError(f"{', '.join(unknown)} not parameters of {md.label}")
    obj = OBJECTIVES[objective]
    mask = period_mask(model.data.dates, *cal_period)
    bounds = [tuple((bounds_override or {}).get(k, md.spec[k][1:3])) for k in names]
    todo = [i for i in model.gauge_order
            if gauges is None or model.data.gauges[i]["id"] in gauges or model.data.gauges[i]["name"] in gauges]
    todo = [i for i in todo if np.isfinite(model.data.qobs[i][mask]).sum() >= 24]
    log = []
    evals_per_zone = (maxiter + 1) * popsize * len(names)
    for zi, gi in enumerate(todo):
        g = model.data.gauges[gi]
        t0 = time.time()
        full = model.run(params)                      # upstream flows with current parameters
        inflow = {j: full.gauge_raw[j] for j in model.upstream_gauges[g["id"]]}
        prob = ZoneProblem(model, gi, params, inflow)
        obs = np.where(mask, model.data.qobs[gi], np.nan)
        zkey = str(g["id"])
        start = {**params["global"], **params["zones"].get(zkey, {})}
        counter = {"n": 0, "best": np.inf}

        def f(x):
            if should_stop and should_stop():
                raise Cancelled()
            p = merge_params(params)
            p["zones"][zkey] = {**p["zones"].get(zkey, {}), **dict(zip(names, map(float, x)))}
            val = obj(prob.simulate(p), obs)
            val = 1e3 if not np.isfinite(val) else float(val)
            counter["n"] += 1
            counter["best"] = min(counter["best"], val)
            if progress and counter["n"] % 25 == 0:
                progress({"zone": g["name"], "zone_index": zi, "zones_total": len(todo),
                          "evals": counter["n"], "evals_est": evals_per_zone, "best": 1.0 - counter["best"]})
            return val

        x0 = np.clip([start[k] for k in names], [b[0] for b in bounds], [b[1] for b in bounds])
        r = differential_evolution(f, bounds, maxiter=maxiter, popsize=popsize, seed=seed, tol=1e-4,
                                   polish=False, init="latinhypercube", x0=x0)
        params["zones"][zkey] = {**params["zones"].get(zkey, {}), **dict(zip(names, map(float, r.x)))}
        entry = {"gauge": g["name"], "zone": g["id"], "cells": int(prob.idx.size),
                 "objective": objective, "score": float(1.0 - r.fun), "evals": int(r.nfev),
                 "seconds": round(time.time() - t0, 1), "params": params["zones"][zkey]}
        log.append(entry)
        if progress:
            progress({"zone": g["name"], "zone_index": zi + 1, "zones_total": len(todo), "done_zone": entry})
    return {"params": params, "log": log}
