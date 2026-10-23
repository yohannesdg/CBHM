"""Water-balance models the distributed framework can run in every cell.

Each model supplies its parameters (default, lower, upper, description) and two functions:

* run(P, PET, params) -> {"et", "soil", "gw", "direct", "baseflow", "recharge", "runoff", "s0", "g0"}
  for (T, n) monthly rainfall and PET in mm, with params as {name: (n,) array};
* zone_nodes(P, PET, pclim, eclim, params, weight, node, nnodes) -> (nnodes, T), the calibration helper
  that sums runoff * weight per routing node for one parameter set.

Everything else (PET, lakes, routing, zones, calibration, maps, balance) is shared, so a new model only
has to provide these. "soil" and "gw" are the model's soil-type and groundwater storages; with s0/g0 (the
starting values) they close the water balance P = ET + runoff + dS + dG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import abcd, nam


@dataclass(frozen=True)
class ModelDef:
    key: str
    label: str
    description: str
    spec: dict                     # name -> (default, lower, upper, description)
    run: Callable
    zone_nodes: Callable
    calibrate_default: tuple = field(default=())


_ABCD_SPEC = {
    "a": (0.995, 0.85, 1.00, "Runoff before soil saturation (abcd a)"),
    "b": (700.0, 100.0, 3000.0, "Soil water + ET capacity, mm (abcd b)"),
    "c": (0.40, 0.00, 1.00, "Fraction of surplus recharging groundwater (abcd c)"),
    "d": (0.10, 0.005, 1.00, "Groundwater recession constant, 1/month (abcd d)"),
    "kc": (1.00, 0.60, 1.30, "PET multiplier on Modified-Hargreaves ETo"),
}

_NAM_SPEC = {
    "umax": (15.0, 1.0, 60.0, "Surface storage capacity, mm (NAM Umax)"),
    "lmax": (300.0, 20.0, 1500.0, "Root-zone storage capacity, mm (NAM Lmax)"),
    "cqof": (0.40, 0.0, 1.0, "Overland flow coefficient (NAM CQOF)"),
    "ckif": (1.0, 0.1, 20.0, "Interflow time constant, months (NAM CKIF)"),
    "tof": (0.30, 0.0, 0.95, "Root-zone threshold for overland flow (NAM TOF)"),
    "tif": (0.30, 0.0, 0.95, "Root-zone threshold for interflow (NAM TIF)"),
    "tg": (0.20, 0.0, 0.95, "Root-zone threshold for recharge (NAM TG)"),
    "ckbf": (6.0, 0.5, 60.0, "Baseflow time constant, months (NAM CKBF)"),
    "kc": (1.00, 0.60, 1.30, "PET multiplier on Modified-Hargreaves ETo"),
}


def _abcd_run(P, PET, pa):
    return abcd.run_abcd(P, PET, pa["a"], pa["b"], pa["c"], pa["d"])


def _abcd_nodes(P, E, pclim, eclim, zp, weight, node, nnodes):
    return abcd.zone_runoff_nodes(P, E, pclim, eclim, zp["a"], zp["b"], zp["c"], zp["d"], weight, node, nnodes)


def _nam_run(P, PET, pa):
    return nam.run_nam(P, PET, {k: pa[k] for k in nam.NAMES})


def _nam_nodes(P, E, pclim, eclim, zp, weight, node, nnodes):
    return nam.zone_runoff_nodes(P, E, pclim, eclim, {k: zp[k] for k in nam.NAMES}, weight, node, nnodes)


MODELS = {
    "abcd": ModelDef("abcd", "abcd", "Thomas (1981) monthly water balance: one soil store and one groundwater store, "
                     "4 parameters.", _ABCD_SPEC, _abcd_run, _abcd_nodes, ("a", "b", "c", "d")),
    "nam": ModelDef("nam", "NAM (monthly)", "DHI NAM structure on a monthly step: surface, root-zone and groundwater "
                    "stores with overland flow, interflow and baseflow, 8 parameters.", _NAM_SPEC, _nam_run,
                    _nam_nodes, ("umax", "lmax", "cqof", "ckif", "tof", "tg", "ckbf")),
}
DEFAULT_MODEL = "abcd"


def get(key: str | None) -> ModelDef:
    if key in (None, ""):
        key = DEFAULT_MODEL
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'; choose from {', '.join(MODELS)}")
    return MODELS[key]
