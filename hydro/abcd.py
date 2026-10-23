"""Vectorised abcd monthly water-balance model (Thomas, 1981; Alley, 1984).

Per cell and month t, with precipitation P and potential ET PET (mm):

    W_t  = P_t + S_{t-1}                                    available water
    Y_t  = (W+b)/(2a) - sqrt( ((W+b)/(2a))^2 - W*b/a )      ET opportunity
    S_t  = Y_t * exp(-PET_t / b)                            soil moisture
    ET_t = Y_t - S_t                                        actual ET
    Rch  = c * (W_t - Y_t)                                  groundwater recharge
    Qd   = (1-c) * (W_t - Y_t)                              direct runoff
    G_t  = (G_{t-1} + Rch) / (1 + d)                        groundwater storage
    Qb   = d * G_t                                          baseflow
    Q    = Qd + Qb                                          cell runoff

Parameters: a (0-1] runoff before saturation, b [mm] upper limit on Y (soil+ET capacity),
c [0-1] recharge fraction, d [0-1] groundwater recession constant.
Parameters may be scalars or arrays of shape (n,) (one value per cell).

Mass balance per step (exact):  P = ET + Qd + Qb + dS + dG.
"""
from __future__ import annotations

import os

import numpy as np

OUTPUTS = ("et", "soil", "gw", "direct", "baseflow", "recharge", "runoff")


def _step(p, pet, s_prev, g_prev, a, b, c, d):
    w = p + s_prev
    half = (w + b) / (2.0 * a)
    y = half - np.sqrt(np.maximum(half * half - w * b / a, 0.0))
    y = np.minimum(y, w)                       # guard against round-off
    s = y * np.exp(-pet / b)
    et = y - s
    surplus = w - y
    rch = c * surplus
    qd = surplus - rch
    g = (g_prev + rch) / (1.0 + d)
    qb = d * g
    return s, g, et, qd, qb, rch


def initial_states(precip, pet, a, b, c, d, spinup_years=5, cycles=3):
    """Spin up S and G by cycling the mean annual cycle of the first `spinup_years`."""
    n = precip.shape[1]
    s = 0.5 * np.broadcast_to(np.asarray(b, float), (n,)).copy()
    g = np.zeros(n)
    yrs = max(1, min(spinup_years, precip.shape[0] // 12))
    p_clim = precip[: yrs * 12].reshape(yrs, 12, n).mean(axis=0)
    e_clim = pet[: yrs * 12].reshape(yrs, 12, n).mean(axis=0)
    for _ in range(cycles):
        for m in range(12):
            s, g, *_ = _step(p_clim[m], e_clim[m], s, g, a, b, c, d)
    return s, g


try:  # optional JIT acceleration (~10x); falls back to NumPy if numba is not installed
    if os.environ.get("VOLTA_NO_NUMBA"):
        raise ImportError
    from numba import njit, prange, get_num_threads

    @njit(cache=True, fastmath=False)
    def _kernel(precip, pet, a, b, c, d, s0, g0, et, soil, gw, qd, qb, rch):
        T, n = precip.shape
        for i in range(n):
            s = s0[i]; g = g0[i]
            ai = a[i]; bi = b[i]; ci = c[i]; di = d[i]
            for t in range(T):
                w = precip[t, i] + s
                half = (w + bi) / (2.0 * ai)
                disc = half * half - w * bi / ai
                y = half - np.sqrt(disc if disc > 0.0 else 0.0)
                if y > w:
                    y = w
                s = y * np.exp(-pet[t, i] / bi)
                sur = w - y
                r = ci * sur
                g = (g + r) / (1.0 + di)
                et[t, i] = y - s; soil[t, i] = s; gw[t, i] = g
                qd[t, i] = sur - r; qb[t, i] = di * g; rch[t, i] = r

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def run_abcd(precip, pet, a, b, c, d, s0=None, g0=None, outputs=OUTPUTS, dtype=np.float32):
    """Run the model.

    precip, pet: (T, n) mm/month.  Returns dict of (T, n) arrays for the requested outputs
    plus 's0'/'g0' initial states (n,).
    """
    precip = np.asarray(precip, dtype=float)
    pet = np.asarray(pet, dtype=float)
    T, n = precip.shape
    a = np.ascontiguousarray(np.broadcast_to(np.clip(np.asarray(a, float), 1e-6, 1.0), (n,)))
    b = np.ascontiguousarray(np.broadcast_to(np.maximum(np.asarray(b, float), 1e-3), (n,)))
    c = np.ascontiguousarray(np.broadcast_to(np.clip(np.asarray(c, float), 0.0, 1.0), (n,)))
    d = np.ascontiguousarray(np.broadcast_to(np.clip(np.asarray(d, float), 0.0, 1.0), (n,)))
    if s0 is None or g0 is None:
        s_init, g_init = initial_states(precip, pet, a, b, c, d)
        s0 = s_init if s0 is None else s0
        g0 = g_init if g0 is None else g0
    if HAVE_NUMBA:
        arrs = [np.empty((T, n)) for _ in range(6)]
        _kernel(np.ascontiguousarray(precip), np.ascontiguousarray(pet), a, b, c, d,
                np.asarray(s0, float), np.asarray(g0, float), *arrs)
        et, soil, gw, qd, qb, rch = arrs
        vals = {"et": et, "soil": soil, "gw": gw, "direct": qd, "baseflow": qb, "recharge": rch,
                "runoff": qd + qb}
        out = {k: vals[k].astype(dtype, copy=False) for k in outputs}
        out["s0"] = np.asarray(s0, float)
        out["g0"] = np.asarray(g0, float)
        return out
    s, g = np.asarray(s0, float).copy(), np.asarray(g0, float).copy()
    out = {k: np.empty((T, n), dtype=dtype) for k in outputs}
    for t in range(T):
        s, g, et, qd, qb, rch = _step(precip[t], pet[t], s, g, a, b, c, d)
        vals = {"et": et, "soil": s, "gw": g, "direct": qd, "baseflow": qb,
                "recharge": rch, "runoff": qd + qb}
        for k in outputs:
            out[k][t] = vals[k]
    out["s0"] = np.asarray(s0, float)
    out["g0"] = np.asarray(g0, float)
    return out


# ---------------------------------------------------------------------------------------
# Lean entry points used by calibration (scalar parameters for a whole zone)
# ---------------------------------------------------------------------------------------
def climatology(x: np.ndarray, years: int = 5) -> np.ndarray:
    yrs = max(1, min(years, x.shape[0] // 12))
    return np.ascontiguousarray(x[: yrs * 12].reshape(yrs, 12, x.shape[1]).mean(axis=0))


if HAVE_NUMBA:
    @njit(cache=True)
    def _lean(precip, pet, pclim, eclim, a, b, c, d, cycles, weight, out_sum, out_full, full):
        T, n = precip.shape
        for i in range(n):
            s = 0.5 * b
            g = 0.0
            for _ in range(cycles):
                for m in range(12):
                    w = pclim[m, i] + s
                    half = (w + b) / (2.0 * a)
                    disc = half * half - w * b / a
                    y = half - np.sqrt(disc if disc > 0.0 else 0.0)
                    if y > w:
                        y = w
                    s = y * np.exp(-eclim[m, i] / b)
                    g = (g + c * (w - y)) / (1.0 + d)
            wi = weight[i]
            for t in range(T):
                w = precip[t, i] + s
                half = (w + b) / (2.0 * a)
                disc = half * half - w * b / a
                y = half - np.sqrt(disc if disc > 0.0 else 0.0)
                if y > w:
                    y = w
                s = y * np.exp(-pet[t, i] / b)
                sur = w - y
                r = c * sur
                g = (g + r) / (1.0 + d)
                q = sur - r + d * g
                if full:
                    out_full[t, i] = q
                else:
                    out_sum[t] += q * wi


def zone_runoff(precip, pet, pclim, eclim, a, b, c, d, weight=None, summed=False, cycles=3):
    """Runoff for cells sharing one parameter set (scalars).

    summed=True returns sum_i runoff[t, i] * weight[i] (T,), otherwise runoff (T, n).
    """
    T, n = precip.shape
    a = float(np.clip(a, 1e-6, 1.0)); b = float(max(b, 1e-3))
    c = float(np.clip(c, 0, 1)); d = float(np.clip(d, 0, 1))
    if HAVE_NUMBA:
        out_sum = np.zeros(T)
        out_full = np.empty((T, n)) if not summed else np.empty((1, 1))
        wt = np.ones(n) if weight is None else np.ascontiguousarray(weight, float)
        _lean(precip, pet, pclim, eclim, a, b, c, d, cycles, wt, out_sum, out_full, not summed)
        return out_sum if summed else out_full
    s = np.full(n, 0.5 * b); g = np.zeros(n)
    for _ in range(cycles):
        for m in range(12):
            s, g, *_ = _step(pclim[m], eclim[m], s, g, a, b, c, d)
    out = np.empty((T, n))
    for t in range(T):
        s, g, et, qd, qb, rch = _step(precip[t], pet[t], s, g, a, b, c, d)
        out[t] = qd + qb
    if summed:
        return out @ (np.ones(n) if weight is None else weight)
    return out


if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _lean_nodes(precip, pet, pclim, eclim, a, b, c, d, cycles, weight, node, nnodes, nchunks):
        T, n = precip.shape
        buf = np.zeros((nchunks, nnodes, T))
        size = (n + nchunks - 1) // nchunks
        for ch in prange(nchunks):
            for i in range(ch * size, min(n, (ch + 1) * size)):
                s = 0.5 * b
                g = 0.0
                for _ in range(cycles):
                    for m in range(12):
                        w = pclim[m, i] + s
                        half = (w + b) / (2.0 * a)
                        disc = half * half - w * b / a
                        y = half - np.sqrt(disc if disc > 0.0 else 0.0)
                        if y > w:
                            y = w
                        s = y * np.exp(-eclim[m, i] / b)
                        g = (g + c * (w - y)) / (1.0 + d)
                k = node[i]
                wi = weight[i]
                for t in range(T):
                    w = precip[t, i] + s
                    half = (w + b) / (2.0 * a)
                    disc = half * half - w * b / a
                    y = half - np.sqrt(disc if disc > 0.0 else 0.0)
                    if y > w:
                        y = w
                    s = y * np.exp(-pet[t, i] / b)
                    sur = w - y
                    r = c * sur
                    g = (g + r) / (1.0 + d)
                    buf[ch, k, t] += (sur - r + d * g) * wi
        out = np.zeros((nnodes, T))
        for ch in range(nchunks):
            out += buf[ch]
        return out


def zone_runoff_nodes(precip, pet, pclim, eclim, a, b, c, d, weight, node, nnodes, cycles=3):
    """sum over cells of runoff*weight, grouped by node -> (nnodes, T). Scalar parameters."""
    T, n = precip.shape
    a = float(np.clip(a, 1e-6, 1.0)); b = float(max(b, 1e-3))
    c = float(np.clip(c, 0, 1)); d = float(np.clip(d, 0, 1))
    if HAVE_NUMBA:
        nchunks = int(max(1, min(get_num_threads() * 4, 8_000_000 // max(1, nnodes * T), n)))
        return _lean_nodes(precip, pet, pclim, eclim, a, b, c, d, cycles,
                           np.ascontiguousarray(weight, float), np.ascontiguousarray(node, np.int64),
                           int(nnodes), nchunks)
    r = zone_runoff(precip, pet, pclim, eclim, a, b, c, d, cycles=cycles) * weight[None, :]
    out = np.zeros((nnodes, T))
    np.add.at(out, node, r.T)
    return out
