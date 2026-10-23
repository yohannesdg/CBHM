"""Model dataset container (compact, basin-only arrays) plus the builder used by prep & demo."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import Grid
from .routing import Router, compact_ds, priority_flood_d8, snap_to_stream, upstream_of


@dataclass
class Dataset:
    grid: Grid
    basin: np.ndarray          # (nr, nc) bool
    dem: np.ndarray            # (nr, nc) float, NaN outside data
    lake: np.ndarray           # (nr, nc) bool open water
    ds: np.ndarray             # (n,) compact downstream index, -1 = outlet
    dates: list                # ["YYYY-MM", ...]
    precip: np.ndarray         # (T, n) mm/month
    tmax: np.ndarray           # (T, n) degC
    tmin: np.ndarray           # (T, n) degC
    gauges: list               # [{id, name, river, lon, lat, row, col, cell, area_km2}]
    qobs: np.ndarray           # (G, T) m3/s, NaN = missing
    meta: dict = field(default_factory=dict)
    zone_ids: np.ndarray | None = None   # (n,) parameter zone per cell (HydroBASINS units); None = from gauges
    zone_names: dict | None = None       # {zone id: label}

    # ---- derived -------------------------------------------------------------------
    @property
    def cells(self) -> np.ndarray:
        return np.flatnonzero(self.basin.ravel())

    @property
    def n(self) -> int:
        return int(self.basin.sum())

    @property
    def T(self) -> int:
        return len(self.dates)

    def to_grid(self, values: np.ndarray, fill=np.nan) -> np.ndarray:
        out = np.full(self.grid.nrows * self.grid.ncols, fill, dtype=float)
        out[self.cells] = values
        return out.reshape(self.grid.nrows, self.grid.ncols)

    def compact_index(self, row: int, col: int) -> int:
        flat = row * self.grid.ncols + col
        pos = np.searchsorted(self.cells, flat)
        if pos >= self.cells.size or self.cells[pos] != flat:
            return -1
        return int(pos)

    # ---- io ------------------------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, grid=json.dumps(self.grid.to_dict()), basin=self.basin, dem=self.dem.astype(np.float32),
            lake=self.lake, ds=self.ds, dates=np.array(self.dates), precip=self.precip.astype(np.float32),
            tmax=self.tmax.astype(np.float32), tmin=self.tmin.astype(np.float32),
            gauges=json.dumps(self.gauges), qobs=self.qobs.astype(np.float32), meta=json.dumps(self.meta),
            **({} if self.zone_ids is None else {"zone_ids": self.zone_ids.astype(np.int64),
                                                "zone_names": json.dumps({str(k): v for k, v in (self.zone_names or {}).items()})}))

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        return cls(grid=Grid(**json.loads(str(z["grid"]))), basin=z["basin"].astype(bool),
                   dem=z["dem"].astype(float), lake=z["lake"].astype(bool), ds=z["ds"].astype(np.int64),
                   dates=[str(d) for d in z["dates"]], precip=z["precip"], tmax=z["tmax"], tmin=z["tmin"],
                   gauges=json.loads(str(z["gauges"])), qobs=z["qobs"].astype(float).reshape(-1, len(z["dates"])),
                   meta=json.loads(str(z["meta"])),
                   zone_ids=z["zone_ids"].astype(np.int64) if "zone_ids" in z.files else None,
                   zone_names={int(k): v for k, v in json.loads(str(z["zone_names"])).items()} if "zone_names" in z.files else None)


def build_dataset(grid: Grid, dem: np.ndarray, precip3d: np.ndarray, tmax3d: np.ndarray, tmin3d: np.ndarray,
                  dates: list, outlet_lonlat: tuple, gauges: list, qobs: np.ndarray | None = None,
                  basin_hint: np.ndarray | None = None, lake: np.ndarray | None = None,
                  meta: dict | None = None, snap_radius: int = 3, log=print,
                  outlet_area_km2: float | None = None, outlet_radius: int = 8,
                  zone_grid: np.ndarray | None = None, zone_names: dict | None = None) -> Dataset:
    """Delineate the basin, snap gauges and pack forcing into a compact Dataset.

    precip3d/tmax3d/tmin3d: (T, nr, nc) on `grid`. gauges: [{name, lon, lat, river?, area_km2?}].
    basin_hint: optional boolean mask (e.g. rasterised HydroBASINS polygon). If it is given, all hint
    cells are forced to drain to the outlet. Otherwise the basin is everything upstream of the outlet.
    """
    nr, nc = grid.nrows, grid.ncols
    valid = np.isfinite(dem) & np.all(np.isfinite(precip3d), axis=0)
    area_km2 = grid.cell_area_m2() / 1e6
    if outlet_lonlat is None:
        if basin_hint is None:
            raise ValueError("an outlet point or a basin polygon is needed")
        # natural spill point: flood from the polygon edge, take the edge cell draining the most area
        dom = valid & basin_hint
        ds0, _ = priority_flood_d8(dem, dom)
        c0 = compact_ds(ds0, dom.ravel())
        acc0, _ = Router(c0).accumulate(area_km2.ravel()[dom.ravel()])
        k = int(np.argmax(acc0))
        flat = np.flatnonzero(dom.ravel())[k]
        orow, ocol = divmod(int(flat), nc)
        log(f"outlet found at ({grid.lon[ocol]:.3f}, {grid.lat[orow]:.3f}), draining {acc0[k]:,.0f} km2")
    else:
        orow, ocol = grid.rowcol(*outlet_lonlat)

    if basin_hint is not None:
        dom = valid & basin_hint
        # nearest domain cell to the outlet point
        rr, cc = np.nonzero(dom)
        k = np.argmin((rr - orow) ** 2 + (cc - ocol) ** 2)
        orow, ocol = int(rr[k]), int(cc[k])
        ds_flat, filled = priority_flood_d8(dem, dom, seeds=[orow * nc + ocol])
        basin = dom.ravel() & (upstream_of(ds_flat, orow * nc + ocol))
    else:
        ds_flat, filled = priority_flood_d8(dem, valid)
        allc = compact_ds(ds_flat, valid.ravel())
        acc_c, _ = Router(allc).accumulate(area_km2.ravel()[valid.ravel()])
        acc = np.full(nr * nc, np.nan)
        acc[np.flatnonzero(valid.ravel())] = acc_c
        orow, ocol = snap_to_stream(acc.reshape(nr, nc), valid, orow, ocol, radius=outlet_radius,
                                    area_hint=outlet_area_km2)
        basin = upstream_of(ds_flat, orow * nc + ocol)
    basin2d = basin.reshape(nr, nc)
    ds = compact_ds(ds_flat, basin)
    cells = np.flatnonzero(basin)
    router = Router(ds)
    acc_b, _ = router.accumulate(area_km2.ravel()[cells])
    acc2d = np.full(nr * nc, np.nan)
    acc2d[cells] = acc_b
    acc2d = acc2d.reshape(nr, nc)
    log(f"basin: {cells.size} cells, {acc_b.max():,.0f} km2 at outlet ({grid.lon[ocol]:.3f}, {grid.lat[orow]:.3f})")
    if outlet_area_km2 and abs(acc_b.max() / outlet_area_km2 - 1) > 0.15:
        log(f"WARNING: delineated area differs from the expected {outlet_area_km2:,.0f} km2 by more than 15%. "
            "Check the outlet point or pass a basin polygon (--basin-mask, e.g. HydroBASINS).")

    out_g = []
    for i, g in enumerate(gauges):
        try:
            r, c = grid.rowcol(g["lon"], g["lat"])
            r, c = snap_to_stream(acc2d, basin2d, r, c, radius=snap_radius, area_hint=g.get("area_km2"))
        except ValueError as e:
            log(f"gauge {g['name']}: skipped ({e})")
            continue
        cell = int(np.searchsorted(cells, r * nc + c))
        out_g.append({"id": len(out_g) + 1, "name": g["name"], "river": g.get("river", ""),
                      "lon": float(grid.lon[c]), "lat": float(grid.lat[r]),
                      "lon_given": g["lon"], "lat_given": g["lat"], "row": int(r), "col": int(c),
                      "cell": cell, "area_km2": float(acc2d[r, c]), "src_index": i})
        log(f"gauge {g['name']}: snapped to ({grid.lon[c]:.3f}, {grid.lat[r]:.3f}), area {acc2d[r, c]:,.0f} km2")
    T = len(dates)
    if qobs is None:
        q = np.full((len(out_g), T), np.nan)
    else:
        q = np.asarray(qobs, float)[[g["src_index"] for g in out_g]]
    lake2d = np.zeros((nr, nc), bool) if lake is None else (lake & basin2d)

    flat = lambda a: a.reshape(a.shape[0], -1)[:, cells].astype(np.float32)
    zid = None
    if zone_grid is not None:
        zid = np.asarray(zone_grid).ravel()[cells].astype(np.int64)
        if (zid == 0).any():   # cells the polygons missed: take the zone of the nearest downstream cell
            order = Router(ds).order[::-1]
            for c in order:
                if zid[c] == 0 and ds[c] >= 0:
                    zid[c] = zid[ds[c]]
            if (zid == 0).any():
                zid[zid == 0] = np.bincount(zid[zid > 0]).argmax()
    return Dataset(grid=grid, basin=basin2d, dem=np.where(valid, dem, np.nan), lake=lake2d, ds=ds,
                   dates=list(dates), precip=flat(precip3d), tmax=flat(tmax3d), tmin=flat(tmin3d),
                   gauges=out_g, qobs=q.reshape(len(out_g), T), meta=meta or {},
                   zone_ids=zid, zone_names=zone_names)
