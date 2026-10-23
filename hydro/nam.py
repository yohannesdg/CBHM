"""NAM (Nedbør-Afstrømnings-Model) adapted to a monthly step, run in every cell.

Based on the structure of the DHI NAM and the Python version in models/NAM_Model-master. Per cell and
month, with rainfall P and potential evaporation E in mm:

  Surface storage U (0..Umax):   U1 = U + P;  Eu = min(U1, E)
  Root zone L (0..Lmax):         El = (E - Eu) * L/Lmax              (limited to L)
  Interflow:                     QIF = (1 - exp(-1/CKIF)) * f(L/Lmax, TIF) * min(U1 - Eu, Umax)
  Net rainfall:                  PN = what is left above Umax after Eu and QIF
  Overland flow:                 QOF = CQOF * f(L/Lmax, TOF) * PN
  Recharge:                      G   = f(L/Lmax, TG) * (PN - QOF)    (+ any overflow above Lmax)
  Root zone:                     L  += PN - QOF - G - El
  Groundwater (linear store):    GW  = (GW + G) / (1 + 1/CKBF);   BF = GW / CKBF

where f(x, T) = max(0, (x - T) / (1 - T)). PN is applied in chunks of at most PMM mm so that L/Lmax
follows the root zone as it fills, as in the original code.

Changes from the daily original, made for a monthly step:
* time constants CKIF and CKBF are in months; CK12 (routing of overland flow and interflow) is
  dropped, because those flows reach the river within the month;
* snow is left out (it double-counted rain between 0 degC and the snow threshold, and plays no role in
  African basins);
* the root zone cannot go below zero, and the groundwater store is solved implicitly, so
  P = ET + QOF + QIF + BF + dU + dL + dGW holds exactly in every cell and month.
"""
from __future__ import annotations

import os

import numpy as np

NAMES = ("umax", "lmax", "cqof", "ckif", "tof", "tif", "tg", "ckbf")
PMM = 25.0          # mm of net rainfall per internal chunk
OUTPUTS = ("et", "soil", "gw", "direct", "baseflow", "recharge", "runoff")


def _clip_params(n, prm):
    lo = {"umax": 0.1, "lmax": 1.0, "cqof": 0.0, "ckif": 0.05, "tof": 0.0, "tif": 0.0, "tg": 0.0, "ckbf": 0.1}
    hi = {"cqof": 1.0, "tof": 0.99, "tif": 0.99, "tg": 0.99}
    out = []
    for k in NAMES:
        a = np.asarray(prm[k], float)
        a = np.maximum(a, lo[k])
        if k in hi:
            a = np.minimum(a, hi[k])
        out.append(np.ascontiguousarray(np.broadcast_to(a, (n,))))
    return out


# ---------------------------------------------------------------------------------- NumPy version
def _step_np(p, e, U, L, G, umax, lmax, cqof, ckif, tof, tif, tg, ckbf):
    lf = L / lmax
    U1 = U + p
    eu = np.minimum(U1, e)
    U1 = U1 - eu
    el = np.minimum((e - eu) * lf, L)
    fi = np.where(lf > tif, (lf - tif) / (1.0 - tif), 0.0)
    qif = fi * np.minimum(U1, umax) * (1.0 - np.exp(-1.0 / ckif))
    U1 = U1 - qif
    pn = np.maximum(U1 - umax, 0.0)
    U = np.minimum(U1, umax)
    n = np.floor(pn / PMM).astype(np.int64) + 1
    elk = el / n
    qof = np.zeros_like(p); g = np.zeros_like(p); el_act = np.zeros_like(p)
    for k in range(int(n.max())):
        act = k < n
        chunk = np.where(k < n - 1, PMM, pn - (n - 1) * PMM) * act
        lf = L / lmax
        qk = np.where(lf > tof, cqof * (lf - tof) / (1.0 - tof), 0.0) * chunk
        gk = np.where(lf > tg, (lf - tg) / (1.0 - tg), 0.0) * (chunk - qk)
        ek = elk * act
        L = L + chunk - qk - gk - ek
        over = np.maximum(L - lmax, 0.0)
        gk = gk + over
        L = L - over
        short = np.minimum(L, 0.0)              # evaporation cannot take more than the root zone holds
        ek = ek + short
        L = L - short
        qof += qk; g += gk; el_act += ek
    G = (G + g) / (1.0 + 1.0 / ckbf)
    bf = G / ckbf
    return U, L, G, eu + el_act, qif + qof, bf, g


def _spinup_np(pclim, eclim, prm):
    n = pclim.shape[1]
    umax, lmax = prm[0], prm[1]
    U = 0.5 * umax; L = 0.5 * lmax; G = np.zeros(n)
    for _ in range(3):
        for m in range(12):
            U, L, G, *_ = _step_np(pclim[m], eclim[m], U, L, G, *prm)
    return U, L, G


def _climatology(x, years=5):
    yrs = max(1, min(years, x.shape[0] // 12))
    return np.ascontiguousarray(x[: yrs * 12].reshape(yrs, 12, x.shape[1]).mean(axis=0))


def run_nam_numpy(precip, pet, prm_arrays, dtype=np.float32):
    T, n = precip.shape
    pclim, eclim = _climatology(precip), _climatology(pet)
    U, L, G = _spinup_np(pclim, eclim, prm_arrays)
    s0, g0 = U + L, G.copy()
    out = {k: np.empty((T, n), dtype) for k in OUTPUTS}
    for t in range(T):
        U, L, G, et, quick, bf, rch = _step_np(precip[t], pet[t], U, L, G, *prm_arrays)
        out["et"][t] = et; out["soil"][t] = U + L; out["gw"][t] = G
        out["direct"][t] = quick; out["baseflow"][t] = bf; out["recharge"][t] = rch; out["runoff"][t] = quick + bf
    out["s0"] = s0; out["g0"] = g0
    return out


# ---------------------------------------------------------------------------------- numba version
try:
    if os.environ.get("VOLTA_NO_NUMBA"):
        raise ImportError
    from numba import get_num_threads, njit, prange

    @njit(cache=True, inline="always")
    def _cell_step(p, e, U, L, G, umax, lmax, cqof, ckif, tof, tif, tg, ckbf):
        lf = L / lmax
        U1 = U + p
        eu = U1 if U1 < e else e
        U1 -= eu
        el = (e - eu) * lf
        if el > L:
            el = L
        qif = 0.0
        if lf > tif:
            qif = (lf - tif) / (1.0 - tif) * (U1 if U1 < umax else umax) * (1.0 - np.exp(-1.0 / ckif))
        U1 -= qif
        if U1 > umax:
            pn = U1 - umax
            U = umax
        else:
            pn = 0.0
            U = U1
        n = int(np.floor(pn / PMM)) + 1
        elk = el / n
        qof = 0.0; g = 0.0; el_act = 0.0
        for k in range(n):
            chunk = PMM if k < n - 1 else pn - (n - 1) * PMM
            lf = L / lmax
            qk = cqof * (lf - tof) / (1.0 - tof) * chunk if lf > tof else 0.0
            gk = (lf - tg) / (1.0 - tg) * (chunk - qk) if lf > tg else 0.0
            ek = elk
            L = L + chunk - qk - gk - ek
            if L > lmax:
                gk += L - lmax
                L = lmax
            if L < 0.0:
                ek += L
                L = 0.0
            qof += qk; g += gk; el_act += ek
        G = (G + g) / (1.0 + 1.0 / ckbf)
        bf = G / ckbf
        return U, L, G, eu + el_act, qif + qof, bf, g

    @njit(cache=True)
    def _spinup_cell(pclim, eclim, i, umax, lmax, cqof, ckif, tof, tif, tg, ckbf):
        U = 0.5 * umax; L = 0.5 * lmax; G = 0.0
        for _ in range(3):
            for m in range(12):
                U, L, G, _a, _b, _c, _d = _cell_step(pclim[m, i], eclim[m, i], U, L, G,
                                                     umax, lmax, cqof, ckif, tof, tif, tg, ckbf)
        return U, L, G

    @njit(cache=True)
    def _kernel(precip, pet, pclim, eclim, umax, lmax, cqof, ckif, tof, tif, tg, ckbf,
                s0, g0, et, soil, gw, quick, bfo, rch):
        T, n = precip.shape
        for i in range(n):
            U, L, G = _spinup_cell(pclim, eclim, i, umax[i], lmax[i], cqof[i], ckif[i], tof[i], tif[i], tg[i], ckbf[i])
            s0[i] = U + L; g0[i] = G
            for t in range(T):
                U, L, G, e_, q_, b_, r_ = _cell_step(precip[t, i], pet[t, i], U, L, G, umax[i], lmax[i], cqof[i],
                                                     ckif[i], tof[i], tif[i], tg[i], ckbf[i])
                et[t, i] = e_; soil[t, i] = U + L; gw[t, i] = G
                quick[t, i] = q_; bfo[t, i] = b_; rch[t, i] = r_

    @njit(parallel=True, cache=True)
    def _lean_nodes(precip, pet, pclim, eclim, umax, lmax, cqof, ckif, tof, tif, tg, ckbf,
                    weight, node, nnodes, nchunks):
        T, n = precip.shape
        buf = np.zeros((nchunks, nnodes, T))
        size = (n + nchunks - 1) // nchunks
        for ch in prange(nchunks):
            for i in range(ch * size, min(n, (ch + 1) * size)):
                U, L, G = _spinup_cell(pclim, eclim, i, umax, lmax, cqof, ckif, tof, tif, tg, ckbf)
                k = node[i]; wi = weight[i]
                for t in range(T):
                    U, L, G, e_, q_, b_, r_ = _cell_step(precip[t, i], pet[t, i], U, L, G,
                                                         umax, lmax, cqof, ckif, tof, tif, tg, ckbf)
                    buf[ch, k, t] += (q_ + b_) * wi
        out = np.zeros((nnodes, T))
        for ch in range(nchunks):
            out += buf[ch]
        return out

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def run_nam(precip, pet, prm: dict, dtype=np.float32, use_numba=None):
    """precip, pet (T, n) mm/month; prm: {name: scalar or (n,)}. Same outputs as hydro.abcd.run_abcd."""
    precip = np.ascontiguousarray(precip, dtype=float)
    pet = np.ascontiguousarray(pet, dtype=float)
    T, n = precip.shape
    arrs = _clip_params(n, prm)
    if not (HAVE_NUMBA if use_numba is None else use_numba):
        return run_nam_numpy(precip, pet, arrs, dtype)
    pclim, eclim = _climatology(precip), _climatology(pet)
    s0 = np.empty(n); g0 = np.empty(n)
    bufs = [np.empty((T, n)) for _ in range(6)]
    _kernel(precip, pet, pclim, eclim, *arrs, s0, g0, *bufs)
    et, soil, gw, quick, bf, rch = bufs
    vals = {"et": et, "soil": soil, "gw": gw, "direct": quick, "baseflow": bf, "recharge": rch, "runoff": quick + bf}
    out = {k: vals[k].astype(dtype, copy=False) for k in OUTPUTS}
    out["s0"] = s0; out["g0"] = g0
    return out


def zone_runoff_nodes(precip, pet, pclim, eclim, prm: dict, weight, node, nnodes):
    """Calibration helper: sum of runoff * weight per routing node (nnodes, T), scalar parameters."""
    T, n = precip.shape
    if HAVE_NUMBA:
        vals = [float(a[0]) for a in _clip_params(1, prm)]
        nchunks = int(max(1, min(get_num_threads() * 4, 8_000_000 // max(1, nnodes * T), n)))
        return _lean_nodes(precip, pet, pclim, eclim, *vals, np.ascontiguousarray(weight, float),
                           np.ascontiguousarray(node, np.int64), int(nnodes), nchunks)
    r = run_nam_numpy(precip, pet, _clip_params(n, prm), float)["runoff"] * weight[None, :]
    out = np.zeros((nnodes, T))
    np.add.at(out, node, r.T)
    return out
