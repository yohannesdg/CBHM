"""Synthetic Volta-like basin for demos and tests.

It reproduces the geography and climate of the real basin in a rough way: Sahel-to-Guinea
rainfall gradient, a unimodal north and bimodal south, Black/White Volta and Oti valleys,
Lake Volta, and Atakora and Kwahu highlands. It is NOT real data. Use prep/prep_volta.py with
CHIRPS + TerraClimate + a DEM for the real basin.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .dataset import build_dataset
from .grid import Grid

OUTLINE = [(-5.3, 11.0), (-4.6, 12.6), (-3.4, 13.6), (-2.0, 14.3), (-0.6, 13.9), (0.4, 13.0), (1.4, 12.0),
           (2.1, 11.0), (1.9, 9.8), (1.3, 8.6), (0.9, 7.4), (0.8, 6.3), (0.68, 5.72), (0.3, 6.0), (-0.3, 6.5),
           (-1.2, 7.2), (-2.2, 7.6), (-3.0, 8.2), (-3.6, 9.0), (-4.4, 9.6), (-5.2, 10.2)]
_FIXED = {(0.68, 5.72), (0.8, 6.3), (0.3, 6.0)}
OUTLINE = [p if p in _FIXED else (-1.0 + 0.935 * (p[0] + 1.0), 10.0 + 0.935 * (p[1] - 10.0)) for p in OUTLINE]
OUTLET = (0.63, 5.80)
RIVERS = {  # listed mouth -> source; tributaries start at a point on their parent
    "main": [(0.63, 5.80), (0.2, 6.0), (0.07, 6.3), (0.0, 6.9), (-0.3, 7.6), (-0.9, 8.4)],
    "oti": [(-0.3, 7.6), (-0.1, 8.4), (0.15, 9.0), (0.33, 9.71), (0.5, 10.4), (0.9, 10.9), (1.6, 11.4)],
    "white": [(-0.9, 8.4), (-1.0, 9.0), (-1.08, 9.70), (-0.85, 10.6), (-0.9, 11.0), (-1.2, 12.0), (-1.4, 13.0)],
    "black": [(-0.9, 8.4), (-1.4, 8.25), (-2.03, 8.15), (-2.6, 8.6), (-2.8, 9.8), (-2.9, 11.0), (-3.3, 12.3),
              (-3.9, 12.0), (-4.5, 11.3)],
    "sissili": [(-1.0, 9.0), (-1.6, 10.0), (-2.0, 11.2)],
    "nakambe_trib": [(-0.9, 11.0), (-0.2, 12.2), (0.3, 12.8)],
    "pendjari": [(0.9, 10.9), (1.5, 10.6), (1.9, 10.2)],
    "afram": [(0.0, 6.9), (-0.8, 6.9), (-1.5, 7.2)],
}
TRUE_PARAMS = {"global": {"a": 0.997, "b": 900, "c": 0.5, "d": 0.12, "kc": 0.95},
               "zones": {"1": {"a": 0.998, "b": 820, "c": 0.45, "d": 0.05, "kc": 0.95},
                         "2": {"a": 0.998, "b": 560, "c": 0.30, "d": 0.18, "kc": 0.95},
                         "3": {"a": 0.993, "b": 520, "c": 0.40, "d": 0.08, "kc": 0.95},
                         "4": {"a": 0.997, "b": 900, "c": 0.55, "d": 0.12, "kc": 0.95}},
               "kw": 1.05, "k_route": 0.6}
GAUGES = [{"name": "Bamboi", "river": "Black Volta", "lon": -2.03, "lat": 8.15},
          {"name": "Nawuni", "river": "White Volta", "lon": -1.08, "lat": 9.70},
          {"name": "Saboba", "river": "Oti", "lon": 0.33, "lat": 9.71},
          {"name": "Senchi (below Akosombo)", "river": "Volta", "lon": 0.10, "lat": 6.20}]


def _in_poly(x, y, poly):
    px = np.array([p[0] for p in poly]); py = np.array([p[1] for p in poly])
    inside = np.zeros(x.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= cond
        j = i
    return inside


def _km(lon, lat):
    return np.stack([lon * 111.32 * np.cos(np.deg2rad(8.5)), lat * 110.57], axis=-1)


def _river_points():
    pts, chain, name = [], [], []
    chain_at = {}
    for rname, line in RIVERS.items():
        c0 = chain_at.get(line[0], 0.0)
        acc = c0
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            seg = float(np.linalg.norm(_km(np.array(x1), np.array(y1)) - _km(np.array(x0), np.array(y0))))
            k = max(2, int(seg / 1.5))
            for t in np.linspace(0, 1, k, endpoint=False):
                pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                chain.append(acc + t * seg)
                name.append(rname)
            acc += seg
            chain_at.setdefault((x1, y1), acc)
    return np.array(pts), np.array(chain), np.array(name)


def make_synthetic(res: float = 0.05, start: str = "1981-01", end: str = "2022-12", seed: int = 7, log=print):
    from .model import VoltaModel  # local import to avoid cycle
    rng = np.random.default_rng(seed)
    grid = Grid(west=-5.6, north=14.8, res=res, nrows=int(round((14.8 - 5.5) / res)),
                ncols=int(round((2.6 + 5.6) / res)))
    lon, lat = grid.lon2d(), grid.lat2d()
    inside = _in_poly(lon, lat, OUTLINE)

    pts, chain, rname = _river_points()
    tree = cKDTree(_km(pts[:, 0], pts[:, 1]))
    dist, nearest = tree.query(_km(lon, lat).reshape(-1, 2))
    dist = dist.reshape(lon.shape); nearest = nearest.reshape(lon.shape)
    sigma_cells = max(1.0, 0.25 / res)
    noise = gaussian_filter(rng.normal(size=lon.shape), sigma_cells) * 25 / max(1e-6, np.std(gaussian_filter(rng.normal(size=lon.shape), sigma_cells)))
    dem = (2.0 + 0.27 * chain[nearest] + 1.1 * dist + noise * 0.4
           + 380 * np.exp(-((lon - 1.0) / 0.3) ** 2 - ((lat - 9.8) / 0.9) ** 2)        # Atakora
           + 260 * np.exp(-((lon + 1.0) / 0.5) ** 2 - ((lat - 6.9) / 0.25) ** 2))      # Kwahu
    # Lake Volta: wide reach of the main stem + lower arms
    ch = chain[nearest]
    rn = rname[nearest]
    akosombo_chain = float(chain[np.argmin(np.hypot(pts[:, 0] - 0.07, pts[:, 1] - 6.3))])
    lake = inside & (dist < 11) & (((rn == "main") & (ch > akosombo_chain)) |
                                   ((rn == "black") & (ch < 470)) | ((rn == "oti") & (ch < 330)) |
                                   ((rn == "white") & (ch < 420)) | ((rn == "afram") & (ch < 160)))
    dem = np.where(lake, 2.0 + 0.27 * akosombo_chain + 0.5, dem)

    # --- time axis
    y0, m0 = int(start[:4]), int(start[5:7]); y1, m1 = int(end[:4]), int(end[5:7])
    nmon = (y1 - y0) * 12 + (m1 - m0) + 1
    months = (np.arange(nmon) + m0 - 1) % 12 + 1
    years = y0 + (np.arange(nmon) + m0 - 1) // 12
    dates = [f"{y}-{m:02d}" for y, m in zip(years, months)]

    # --- rainfall climatology (mm/yr) and seasonality
    lat_k = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    p_k = [1350, 1300, 1250, 1150, 1100, 1060, 950, 800, 650, 500, 400]
    pann = np.interp(lat, lat_k, p_k) * (1 + 0.12 * np.exp(-((lon - 1.0) / 0.5) ** 2 - ((lat - 9.8) / 1.2) ** 2)
                                         + 0.10 * np.exp(-((lon + 1.0) / 0.6) ** 2 - ((lat - 6.9) / 0.4) ** 2))
    m = np.arange(1, 13)[:, None, None]
    north = np.exp(-0.5 * ((m - 8.0) / 1.25) ** 2)
    south = 0.55 * np.exp(-0.5 * ((m - 6.0) / 1.1) ** 2) + 0.45 * np.exp(-0.5 * ((m - 9.8) / 1.0) ** 2) + 0.04
    f = np.clip((lat - 7.5) / 3.0, 0, 1)[None]
    shape = (1 - f) * south / south.sum(0) + f * north / north.sum(0)
    clim = pann[None] * shape                                   # (12, nr, nc)
    big = lambda s: gaussian_filter(rng.normal(size=lon.shape), s)
    year_fac = {}
    for y in range(y0, y1 + 1):
        trend = 0.88 if y < 1991 else 1.0 + 0.002 * (y - 1991)
        fld = big(30 * 0.05 / res); fld /= fld.std() + 1e-9
        year_fac[y] = trend * np.exp(rng.normal(0, 0.10) + 0.07 * fld)
    P = np.empty((nmon, grid.nrows, grid.ncols), np.float32)
    for t in range(nmon):
        g = big(6 * 0.05 / res); g /= g.std() + 1e-9
        mult = np.exp(0.35 * g - 0.5 * 0.35 ** 2)
        P[t] = np.maximum(clim[months[t] - 1] * year_fac[years[t]] * mult, 0)

    # --- temperatures
    tx_n = np.array([33, 36, 39, 41, 40, 37, 33, 31, 33, 36, 36, 34.0])
    tx_s = np.array([33, 34, 34, 33, 32, 30, 28, 28, 29, 31, 32, 32.0])
    tn_n = np.array([17, 20, 24, 27, 27, 25, 23, 22, 22, 22, 19, 17.0])
    tn_s = np.array([22, 23, 23, 23, 23, 22, 22, 21, 22, 22, 22, 21.0])
    f2 = np.clip((lat - 6.5) / 6.5, 0, 1)
    lapse = -6.5 * (np.clip(dem, 0, None) - 200) / 1000.0
    TX = np.empty_like(P); TN = np.empty_like(P)
    for t in range(nmon):
        k = months[t] - 1
        anom = rng.normal(0, 0.5) + 0.025 * (years[t] - 1981)
        wet = (P[t] - clim[k]) / (clim[k] + 30)
        TX[t] = (1 - f2) * tx_s[k] + f2 * tx_n[k] + lapse + anom - 1.2 * np.clip(wet, -1, 1)
        TN[t] = (1 - f2) * tn_s[k] + f2 * tn_n[k] + lapse + anom * 0.8

    data = build_dataset(grid, np.where(inside, dem, np.nan), P, TX, TN, dates, OUTLET, GAUGES,
                         basin_hint=inside, lake=lake, log=log,
                         meta={"synthetic": True, "name": "Synthetic Volta (demo)",
                               "note": "Synthetic data shaped like the Volta Basin. Not observations.",
                               "precip": "synthetic", "temperature": "synthetic", "pet": "Modified Hargreaves"})
    # --- synthetic "observed" discharge from a hidden parameter set + noise and gaps
    model = VoltaModel(data)
    truth = model.run(TRUE_PARAMS)
    q = truth.gauge_sim * np.exp(rng.normal(0, 0.12, truth.gauge_sim.shape))
    for i in range(q.shape[0]):
        q[i, rng.random(q.shape[1]) < 0.06] = np.nan
        if q.shape[1] > 180:                      # one multi-year gap, as in real records
            s = rng.integers(60, q.shape[1] - 80)
            q[i, s:s + rng.integers(12, 48)] = np.nan
    data.qobs = q
    data.meta["true_params"] = TRUE_PARAMS
    return data
