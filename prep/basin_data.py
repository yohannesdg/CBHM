"""Prepare a model dataset for any HydroBASINS (levels 2-5) catchment in Africa.

    python prep/basin_data.py catalog                    # build the level 2-5 catalogue from the HydroBASINS zip
    python prep/basin_data.py prepare --level 4 --id 1040...  [--res auto] [--no-upstream]

The selected unit plus every unit upstream of it (same level) forms the catchment. Rainfall (CHIRPS) and
Tmax/Tmin (TerraClimate) are downloaded for that box only, at a resolution that keeps the model around
20,000 cells. Large basins use coarser grids, so downloads stay small. DEM and lakes come from the
Africa-wide files in data/raw.
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import urllib.parse
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydro.basins import Catalog, build_catalog  # noqa: E402
from hydro.dataset import build_dataset  # noqa: E402
from hydro.geo import rasterize  # noqa: E402
from hydro.grid import Grid  # noqa: E402
from prep import chc_cog  # noqa: E402
from prep import prep_volta as pv  # noqa: E402

HYDROBASINS_AF = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_af_lev01-12_v1c.zip"
BASIN_DIR = ROOT / "data" / "basins"
RES_CHOICES = (0.05, 0.1, 0.25, 0.5)
MAX_CELLS = 20000
TC_RES = 1 / 24          # TerraClimate native resolution
START, END = "1981-01", "2023-12"


def catalog_ready() -> bool:
    return Catalog(BASIN_DIR).available() == [2, 3, 4, 5]


def ensure_catalog(log=print):
    if catalog_ready():
        return
    z = pv.RAW / "hybas_af_lev01-12_v1c.zip"
    if not pv._usable(z):
        pv.download(HYDROBASINS_AF, z, resume=True)
    build_catalog(z, BASIN_DIR, log=log)


BUDGET_MB = 40          # automatic resolution keeps the download around this size
CHUNK_MB = 12           # rainfall is fetched in chunks of whole years of about this size
CHUNK_YEARS = 3         # ... and at most this many years, so each IRI request is answered in seconds
IRI_TIMEOUT = 45        # seconds without an answer before an IRI request counts as failed


def _grid_for(bbox, r):
    w, s, e, n = bbox
    pad = r
    return Grid(west=round(np.floor((w - pad) / r) * r, 6), north=round(np.ceil((n + pad) / r) * r, 6), res=r,
                nrows=int(np.ceil((n + pad) / r) - np.floor((s - pad) / r)),
                ncols=int(np.ceil((e + pad) / r) - np.floor((w - pad) / r)))


def _sizes(grid: Grid, T: int, key: str):
    r = grid.res
    stride = max(1, int(round(r / TC_RES)))
    tc_cells = (grid.nrows * r / (TC_RES * stride) + 2) * (grid.ncols * r / (TC_RES * stride) + 2)
    have = pv.RAW / "basins" / key
    rain_done = _reuse_volta(grid) is not None or pv._usable(have / f"chirps_chc_{r}.nc") or \
        all(pv._usable(have / f"chirps_{r}_{a}_{b}.nc") for a, b in _year_chunks(grid, START, END))
    rain = 0.0 if rain_done else grid.nrows * grid.ncols * T * 4 / 1e6
    temp = sum(0.0 if pv._usable(have / f"{v}_{r}.nc") else tc_cells * T * 2 / 1e6 for v in ("tmax", "tmin"))
    return rain, temp, stride


def _year_chunks(grid: Grid, start=START, end=END):
    """Split the period into runs of whole years, each roughly CHUNK_MB of rainfall data."""
    y0, y1 = int(start[:4]), int(end[:4])
    per_year = grid.nrows * grid.ncols * 12 * 4 / 1e6
    step = int(max(1, min(y1 - y0 + 1, CHUNK_YEARS, CHUNK_MB // max(per_year, 1e-9))))
    out = []
    for a in range(y0, y1 + 1, step):
        b = min(y1, a + step - 1)
        out.append((f"{a}-01" if a > y0 else start, f"{b}-12" if b < y1 else end))
    return out


def options(bbox, area, lat_mid, T, key):
    out = []
    for r in RES_CHOICES:
        g = _grid_for(bbox, r)
        rain, temp, _ = _sizes(g, T, key)
        cells = int(area / ((111.32 * r) * (110.57 * r) * np.cos(np.deg2rad(lat_mid))))
        out.append({"res": r, "cells": cells, "download_mb": round(rain + temp, 1)})
    return out


def choose_res(opts, res=None):
    if res not in (None, "auto"):
        return float(res)
    for o in opts:
        if o["cells"] <= MAX_CELLS and o["download_mb"] <= BUDGET_MB:
            return o["res"]
    ok = [o for o in opts if o["cells"] <= MAX_CELLS]
    return (ok[-1] if ok else opts[-1])["res"]


def plan(level: int, hid: int, include_upstream=True, res=None, start=START, end=END):
    """What a preparation would do: ids, box, resolution options, estimated cells and download size."""
    cat = Catalog(BASIN_DIR)
    ids = cat.upstream(level, hid) if include_upstream else [int(hid)]
    p = cat.props(level, hid)
    area = sum(cat.props(level, i)["sub_area"] for i in ids)
    bbox = cat.bbox(level, ids)
    key = f"L{level}_{hid}" + ("" if include_upstream else "_unit")
    T = len(pv.month_list(start, end))
    opts = options(bbox, area, 0.5 * (bbox[1] + bbox[3]), T, key)
    r = choose_res(opts, res)
    grid = _grid_for(bbox, r)
    rain, temp, stride = _sizes(grid, T, key)
    mb = rain + temp
    return {"key": key, "level": level, "id": int(hid), "pfaf": p["pfaf"], "ids": ids, "units": len(ids),
            "area_km2": area, "unit_area_km2": p["sub_area"], "bbox": list(bbox), "res": r, "auto_res": choose_res(opts),
            "options": opts, "grid": grid.to_dict(), "cells_est": next(o["cells"] for o in opts if o["res"] == r),
            "download_mb": round(mb, 1), "tc_stride": stride, "start": start, "end": end,
            "chunks": len(_year_chunks(grid, start, end)) if rain else 0,
            "endorheic": bool(p["endo"]), "dataset": f"basin_{key}_{r}.npz", "cached": mb == 0,
            "fallback_mb": round(chc_cog.estimate_mb(pv.RAW, pv.month_list(start, end), r, _bounds(grid))[0], 0) if rain else 0}


def _bounds(grid: Grid):
    return (grid.west, grid.south, grid.east, grid.north)


def _chirps_url(grid: Grid, start, end):
    y0, m0 = map(int, start.split("-")); y1, m1 = map(int, end.split("-"))
    t0 = urllib.parse.quote(f"{calendar.month_abbr[m0]} {y0}"); t1 = urllib.parse.quote(f"{calendar.month_abbr[m1]} {y1}")
    base = pv.IRI_CHIRPS.split("X/")[0]
    r = grid.res
    if abs(r - 0.05) < 1e-9:     # native resolution: just cut the box
        return base + (f"X/{grid.west}/{grid.east}/RANGEEDGES/Y/{grid.south}/{grid.north}/RANGEEDGES/"
                       f"T/({t0})/({t1})/RANGEEDGES/data.nc")
    # coarser: let the Data Library regrid on the server, so only the small grid is downloaded
    x0, x1 = grid.west + r / 2, grid.east - r / 2
    y0_, y1_ = grid.south + r / 2, grid.north - r / 2
    return base + (f"X/{x0:.4f}/{r}/{x1:.4f}/GRID/Y/{y0_:.4f}/{r}/{y1_:.4f}/GRID/"
                   f"T/({t0})/({t1})/RANGEEDGES/data.nc")


def _terraclimate_url(var, grid: Grid, stride, start, end):
    y1, m1 = map(int, end.split("-"))
    pad = 2 * TC_RES * stride
    q = {"var": var, "north": grid.north + pad, "west": grid.west - pad, "east": grid.east + pad,
         "south": grid.south - pad, "disableProjSubset": "on", "horizStride": stride,
         "time_start": f"{start}-01T00:00:00Z", "time_end": f"{end}-{calendar.monthrange(y1, m1)[1]:02d}T23:59:59Z",
         "timeStride": 1, "accept": "netcdf"}
    return pv.TERRACLIMATE_NCSS.format(var=var) + "?" + urllib.parse.urlencode(q)


def _reuse_volta(grid: Grid):
    """The Volta box downloaded by prep_volta covers a basin inside it at 0.05 deg: reuse it."""
    f = pv.RAW / pv.FILES["chirps"]
    if abs(grid.res - 0.05) > 1e-9 or not f.exists():
        return None
    try:
        with pv._open_nc_scipy(f, mmap=True) as nc:
            xs = [v for v in nc.variables if v.lower() in ("x", "lon", "longitude")][0]
            ys = [v for v in nc.variables if v.lower() in ("y", "lat", "latitude")][0]
            x = np.array(nc.variables[xs].data); y = np.array(nc.variables[ys].data)
        if x.min() <= grid.west and x.max() >= grid.east and y.min() <= grid.south and y.max() >= grid.north:
            return f
    except Exception:  # noqa: BLE001
        return None
    return None


def _rain_iri(grid: Grid, raw: Path, start, end, log, step):
    """Rainfall from the IRI Data Library in short chunks. None if the server doesn't answer."""
    chunks = _year_chunks(grid, start, end)
    files = []
    for i, (a, b) in enumerate(chunks, start=1):
        f = raw / f"chirps_{grid.res}_{a}_{b}.nc"
        files.append(f)
        if pv._usable(f):
            continue
        step(f"rainfall {i}/{len(chunks)} ({a[:4]}-{b[:4]})")
        try:
            pv.download(_chirps_url(grid, a, b), f, resume=False, retries=2, timeout=IRI_TIMEOUT)
        except RuntimeError:
            log(f"the IRI Data Library did not deliver rainfall for {a[:4]}-{b[:4]}")
            return None
    return files


def prepare(level: int, hid: int, include_upstream=True, res=None, start=START, end=END, log=print, progress=None):
    step = (lambda s: progress(s)) if progress else (lambda s: None)
    pl = plan(level, hid, include_upstream, res, start, end)
    grid = Grid(**pl["grid"])
    cat = Catalog(BASIN_DIR)
    raw = pv.RAW / "basins" / pl["key"]
    raw.mkdir(parents=True, exist_ok=True)
    for need in ("dem", "lakes"):
        if not pv._usable(pv.RAW / pv.FILES[need]):
            raise SystemExit(f"{pv.FILES[need]} is missing: run download_data.bat first")
    log(f"basin {pl['key']} (PFAF {pl['pfaf']}): {pl['units']} units, {pl['area_km2']:,.0f} km2, "
        f"grid {grid.nrows}x{grid.ncols} at {grid.res} deg, about {pl['download_mb']} MB to download"
        + (f" in {pl['chunks']} rainfall parts" if pl["chunks"] > 1 else ""))

    reuse = _reuse_volta(grid)
    chc_file = raw / f"chirps_chc_{grid.res}.nc"
    if reuse is not None:
        chirps_files = [reuse]
    elif pv._usable(chc_file):
        chirps_files = [chc_file]
    else:
        chirps_files = _rain_iri(grid, raw, start, end, log, step)
        if chirps_files is None:
            step("rainfall from CHC")
            months = pv.month_list(start, end)
            est, lvl, nt = chc_cog.estimate_mb(pv.RAW, months, grid.res, _bounds(grid))
            log(f"switching to the Climate Hazards Center tiles: about {est:.0f} MB "
                f"({nt} tile(s) a month, kept in data/raw/chirps_cog for nearby basins)")

            def prog(name, got, total, rate):
                if pv.PROGRESS_HOOK:
                    pv.PROGRESS_HOOK(f"CHIRPS {name}", got, total, rate)
            try:
                chc_cog.fetch(pv.RAW, months, grid.res, _bounds(grid), chc_file, log=log, progress=prog, cancel=pv.CANCEL)
            except chc_cog.Cancelled:
                raise pv.Cancelled()
            chirps_files = [chc_file]
    step("temperature")
    temp = {}
    for var in ("tmax", "tmin"):
        f = raw / f"{var}_{grid.res}.nc"
        temp[var] = f
        if not pv._usable(f):
            pv.download(_terraclimate_url(var, grid, pl["tc_stride"], start, end), f, resume=False)

    step("building")
    ps = pv.open_series_many(chirps_files, ["precipitation", "precip", "prcp", "pr"])
    tx = pv.open_series(temp["tmax"], ["tmax"])
    tn = pv.open_series(temp["tmin"], ["tmin"])
    have = set(ps.months) & set(tx.months) & set(tn.months)
    months = []
    for m in pv.month_list(start, end):          # longest run of months that every source has
        if m in have:
            months.append(m)
        elif months:
            break
    if len(months) < 24:
        raise SystemExit(f"the downloaded files share only {len(months)} months between {start} and {end}")
    if months[0] != start or months[-1] != end:
        log(f"using {months[0]} to {months[-1]}, the months available in all sources")
    P = pv.to_mm_month(pv.regrid(ps, grid, months, "CHIRPS"), ps.units, months)
    TX = pv.regrid(tx, grid, months, "Tmax")
    TN = pv.regrid(tn, grid, months, "Tmin")
    if np.nanmean(TX) > 200:
        TX -= 273.15; TN -= 273.15
    dem = pv.read_dem(pv.RAW / pv.FILES["dem"], grid)

    # parameter zones: the HydroBASINS units of this level inside the catchment (or the next level down)
    zone_level, zone_ids = level, pl["ids"]
    if len(zone_ids) == 1 and level < 5 and (BASIN_DIR / f"hybas_af_lev{level + 1:02d}.geojson").exists():
        sub = [i for i, p in cat._load(level + 1)["props"].items()
               if str(p["pfaf"]).startswith(str(pl["pfaf"]))]
        if len(sub) > 1:
            zone_level, zone_ids = level + 1, sub
    zgrid = np.zeros((grid.nrows, grid.ncols), np.int64)
    names = {}
    order = sorted(zone_ids, key=lambda i: -cat.props(zone_level, i)["up_area"])
    for k, h in enumerate(order, start=1):
        m = rasterize(cat.polygons(zone_level, [h]), grid)
        zgrid[m & (zgrid == 0)] = k
        p = cat.props(zone_level, h)
        names[k] = f"PFAF {p['pfaf']}"
    hint = zgrid > 0
    if hint.sum() < 10:
        raise SystemExit("the basin covers fewer than 10 model cells; choose a larger basin or a finer resolution")
    lake = pv.read_mask(pv.RAW / pv.FILES["lakes"], grid)          # every Natural Earth lake in the box
    valid = np.isfinite(dem) & hint
    P = pv.fill_nan_nearest(P, valid); TX = pv.fill_nan_nearest(TX, valid); TN = pv.fill_nan_nearest(TN, valid)
    data = build_dataset(grid, dem, P, TX, TN, months, None, [], basin_hint=hint, lake=lake, log=log,
                         zone_grid=zgrid, zone_names=names,
                         meta={"name": f"HydroBASINS level {level} basin, PFAF {pl['pfaf']}",
                               "synthetic": False, "created": date.today().isoformat(),
                               "precip": "CHIRPS v2.0 monthly (IRI Data Library)",
                               "temperature": "TerraClimate monthly Tmax/Tmin", "pet": "Modified Hargreaves (Droogers & Allen, 2002)",
                               "dem": "HydroSHEDS v1 30 arc-second void-filled DEM",
                               "lake": "Natural Earth 1:10m lakes", "basin_mask": f"HydroBASINS v1c level {level} (+ upstream)" if include_upstream else f"HydroBASINS v1c level {level} unit",
                               "hybas": {"level": level, "id": int(hid), "pfaf": pl["pfaf"], "units": pl["units"],
                                         "zone_level": zone_level, "area_km2": pl["area_km2"], "endorheic": pl["endorheic"]},
                               "discharge": "none"})
    out = ROOT / "data" / pl["dataset"]
    data.save(out)
    area = data.grid.cell_area_m2().ravel()[data.cells].sum() / 1e6
    log(f"saved {out.name}: {data.n} cells, {area:,.0f} km2 (HydroBASINS {pl['area_km2']:,.0f} km2), {len(names)} zones")
    step("done")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["catalog", "plan", "prepare"])
    ap.add_argument("--level", type=int, default=4); ap.add_argument("--id", type=int)
    ap.add_argument("--res", default="auto"); ap.add_argument("--no-upstream", action="store_true")
    ap.add_argument("--start", default=START); ap.add_argument("--end", default=END)
    a = ap.parse_args(argv)
    if a.command == "catalog":
        ensure_catalog()
    elif a.command == "plan":
        print(json.dumps(plan(a.level, a.id, not a.no_upstream, a.res, a.start, a.end), indent=1))
    else:
        prepare(a.level, a.id, not a.no_upstream, a.res, a.start, a.end)


if __name__ == "__main__":
    main()
