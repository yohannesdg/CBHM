#!/usr/bin/env python
"""Prepare the Volta model dataset from public data.

  python prep/prep_volta.py download     # fetch everything listed below into data/raw (resumable)
  python prep/prep_volta.py build        # regrid, delineate the basin, snap gauges -> data/volta.npz
  python prep/prep_volta.py all          # both
  python prep/prep_volta.py check        # summary of data/volta.npz

Downloads (public, no account needed):
  * CHIRPS v2.0 monthly rainfall, 0.05 deg   IRI Data Library subset of the Volta box      (~60 MB)
  * TerraClimate monthly Tmax and Tmin       NCSS subsets from the Univ. of Idaho THREDDS (~45 MB each)
  * HydroSHEDS v1 void-filled DEM, 30 arc-s  Africa tile                                  (48 MB)
  * Natural Earth 1:10m lakes (public domain), for the Lake Volta outline                (2 MB)

Only numpy and scipy are required, plus two small pure-Python readers: tifffile (DEM) and pyshp (lakes).
xarray/netCDF4/rasterio are used automatically if installed, for formats the light readers can't open.
Downloads resume where they stopped, which matters on a slow connection.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydro.dataset import Dataset, build_dataset  # noqa: E402
from hydro.grid import Grid  # noqa: E402

RAW = ROOT / "data" / "raw"
PROGRESS_HOOK = None   # optional callable(name, got_bytes, total_bytes, rate_kBps), used by the web app
CANCEL = None          # optional threading.Event: set it to stop a download (raises Cancelled)


class Cancelled(Exception):
    pass

IRI_CHIRPS = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
              "X/{w}/{e}/RANGEEDGES/Y/{s}/{n}/RANGEEDGES/T/({t0})/({t1})/RANGEEDGES/data.nc")
CHC_CHIRPS_TIF = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz"
# TerraClimate v1.1 aggregation (the older ..._1958_CurrentYear_GLOBE.nc address now returns empty files)
TERRACLIMATE_NCSS = ("http://thredds.northwestknowledge.net:8080/thredds/ncss/grid/"
                     "agg_terraclimate_{var}_1950_CurrentYear_GLOBE.nc")
HYDROSHEDS_DEM = "https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_af_dem_30s.zip"
NE_LAKES = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip"

FILES = {  # standard names in data/raw
    "chirps": "chirps_monthly_volta.nc",
    "tmax": "terraclimate_tmax_volta.nc",
    "tmin": "terraclimate_tmin_volta.nc",
    "dem": "hyd_af_dem_30s.zip",
    "lakes": "ne_10m_lakes.zip",
}


# ------------------------------------------------------------------------------------ utils
LOG_HOOK = None     # the app sets this to copy messages into its progress panel


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
    if LOG_HOOK:
        try:
            LOG_HOOK(*a)
        except Exception:  # noqa: BLE001
            pass


def load_config(path):
    return json.loads(Path(path).read_text())


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) volta-hydro/1.2 (research water-balance model)"


def download(url: str, dest: Path, retries: int = 8, timeout: int = 120, resume: bool = True, quiet_retry: bool = False):
    """Download with progress, retries and resume (HTTP Range) for slow or unstable connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if (resume and tmp.exists()) else 0
        try:
            headers = {"User-Agent": UA}
            if have:
                headers["Range"] = f"bytes={have}-"
            log(f"GET {url}" + (f"  (resuming at {have / 1e6:.1f} MB)" if have else ""))
            req = urllib.request.Request(url, headers=headers)
            if PROGRESS_HOOK:
                PROGRESS_HOOK(dest.name, -1, 0, 0)        # waiting for the server to answer
            if CANCEL is not None and CANCEL.is_set():
                raise Cancelled()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                partial = r.status == 206
                if have and not partial:
                    have = 0                       # server ignored Range: start again
                total = int(r.headers.get("Content-Length") or 0) + (have if partial else 0)
                got, t0, last = have, time.time(), 0.0
                with open(tmp, "ab" if partial else "wb") as f:
                    while True:
                        if CANCEL is not None and CANCEL.is_set():
                            raise Cancelled()
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        if PROGRESS_HOOK and now - last > 0.5:
                            PROGRESS_HOOK(dest.name, got, total, (got - have) / max(now - t0, 1e-6) / 1e3)
                        if now - last > 0.5:
                            rate = (got - have) / max(now - t0, 1e-6) / 1e3
                            tot = f" / {total / 1e6:.1f}" if total else ""
                            eta = f", about {(total - got) / max(rate * 1e3, 1) / 60:.0f} min left" if total and rate > 0 else ""
                            print(f"\r   {got / 1e6:7.1f}{tot} MB  {rate:6.0f} kB/s{eta}      ", end="", flush=True)
                            last = now
            print()
            if total and got < total:
                raise IOError(f"connection closed at {got / 1e6:.1f} of {total / 1e6:.1f} MB")
            problem = _check_payload(tmp, dest.suffix.lower())
            if problem:
                tmp.unlink(missing_ok=True)
                raise ValueError(problem)
            tmp.replace(dest)
            log(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            print()
            log(f"  attempt {attempt}/{retries} interrupted: {e}")
            if attempt < retries:
                for _ in range(min(60, 5 * attempt)):
                    if CANCEL is not None and CANCEL.is_set():
                        raise Cancelled()
                    time.sleep(1)
    raise RuntimeError(f"could not download {url}. Run the same command again to resume.")


def _check_payload(path: Path, suffix: str):
    """Return a message if a downloaded file is empty or not what it should be (e.g. an error page)."""
    size = path.stat().st_size if path.exists() else 0
    with open(path, "rb") as f:
        head = f.read(512)
    if size < 200:
        return f"the server sent an empty or tiny response ({size} bytes): {head[:200]!r}"
    if suffix == ".nc" and not (head.startswith(b"CDF") or head.startswith(b"\x89HDF")):
        return "the server did not send NetCDF data: " + head[:300].decode("latin-1", "replace")
    if suffix == ".zip" and not head.startswith(b"PK"):
        return "the server did not send a zip file: " + head[:300].decode("latin-1", "replace")
    return None


def _usable(path: Path) -> bool:
    """A file from an earlier run counts as downloaded only if it looks valid."""
    return path.exists() and _check_payload(path, path.suffix.lower()) is None


def month_list(start: str, end: str):
    y0, m0 = map(int, start.split("-")[:2]); y1, m1 = map(int, end.split("-")[:2])
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def model_grid(cfg, res):
    b = cfg["bbox"]
    eps = 1e-9
    west = np.floor(b["west"] / res + eps) * res
    north = np.ceil(b["north"] / res - eps) * res
    east = np.ceil(b["east"] / res - eps) * res
    south = np.floor(b["south"] / res + eps) * res
    return Grid(west=round(west, 6), north=round(north, 6), res=res,
                nrows=int(round((north - south) / res)), ncols=int(round((east - west) / res)))


# ------------------------------------------------------------------------------------ download
def cmd_download(args, cfg):
    b = cfg["bbox"]
    pad = 0.15
    w, e, s, n = b["west"] - pad, b["east"] + pad, b["south"] - pad, b["north"] + pad
    y0, m0 = map(int, args.start.split("-")); y1, m1 = map(int, args.end.split("-"))
    mon = lambda y, m: f"{calendar.month_abbr[m]} {y}"
    jobs = []
    if args.chirps_source == "iri":
        jobs.append(("chirps", IRI_CHIRPS.format(w=w, e=e, s=s, n=n, t0=urllib.parse.quote(mon(y0, m0)),
                                                t1=urllib.parse.quote(mon(y1, m1)))))
    for var in ("tmax", "tmin"):
        q = {"var": var, "north": n, "west": w, "east": e, "south": s, "disableProjSubset": "on",
             "horizStride": 1, "time_start": f"{args.start}-01T00:00:00Z",
             "time_end": f"{args.end}-{calendar.monthrange(y1, m1)[1]:02d}T23:59:59Z",
             "timeStride": 1, "accept": "netcdf"}
        jobs.append((var, TERRACLIMATE_NCSS.format(var=var) + "?" + urllib.parse.urlencode(q)))
    jobs.append(("dem", HYDROSHEDS_DEM))
    jobs.append(("lakes", NE_LAKES))
    for key, url in jobs:
        dest = RAW / FILES[key]
        if _usable(dest) and not args.force:
            log(f"{dest.name}: already downloaded")
            continue
        if dest.exists():
            log(f"{dest.name}: the earlier download is empty or invalid, fetching it again")
            dest.unlink()
        # dynamic subset services can't resume a half-made file; static files can
        download(url, dest, resume=key in ("dem", "lakes"))
    if args.chirps_source == "chc":
        _download_chc_tifs(args, (w, s, e, n))
    log("all downloads complete")


def _download_chc_tifs(args, box):
    tif_dir = RAW / "chirps_tifs"
    tif_dir.mkdir(parents=True, exist_ok=True)
    for ym in month_list(args.start, args.end):
        y, m = map(int, ym.split("-"))
        tif = tif_dir / f"chirps-v2.0.{y}.{m:02d}.tif"
        if tif.exists():
            continue
        gz = tif.with_suffix(".tif.gz")
        download(CHC_CHIRPS_TIF.format(y=y, m=m), gz)
        with gzip.open(gz) as fi, open(tif, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        gz.unlink()


# ------------------------------------------------------------------------------------ readers
class Series:
    """A monthly gridded variable: data (T, nlat, nlon), lat descending, lon ascending."""

    def __init__(self, data, lon, lat, months, units=""):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        data = np.asarray(data, dtype="float64")
        if lat[0] < lat[-1]:
            lat, data = lat[::-1], data[:, ::-1, :]
        if lon[0] > lon[-1]:
            lon, data = lon[::-1], data[:, :, ::-1]
        self.data, self.lon, self.lat, self.months, self.units = data, lon, lat, list(months), units


def _decode_months(values, units: str):
    units = units.strip()
    v = np.asarray(values, float)
    if units.startswith("months since"):
        by, bm = map(int, units.split("since")[1].strip()[:7].split("-")[:2])
        k = np.floor(v + 1e-6).astype(int)
        return [f"{by + (bm - 1 + kk) // 12}-{(bm - 1 + kk) % 12 + 1:02d}" for kk in k]
    unit, base = units.split(" since ")
    base = np.datetime64(base.strip().replace(" ", "T")[:19] if "T" in base or ":" in base else base.strip()[:10])
    scale = {"days": 86400.0, "day": 86400.0, "hours": 3600.0, "hour": 3600.0, "minutes": 60.0, "seconds": 1.0}[unit.strip().lower()]
    secs = (v * scale).astype("int64")
    t = base.astype("datetime64[s]") + secs.astype("timedelta64[s]")
    return [str(x)[:7] for x in t.astype("datetime64[M]")]


def _true_dims(var):
    """Dimension names of a scipy netcdf variable.

    scipy copies attributes onto the variable object, so an attribute literally named "dimensions"
    (TerraClimate v1.1 has one: "lon lat time") hides the real dimension tuple. _open_nc_scipy records
    the real tuple as _real_dims while the file is read.
    """
    dims = var.__dict__.get("_real_dims", var.dimensions)
    if isinstance(dims, (bytes, str)):        # fallback: attribute text, in Fortran order
        names = (dims.decode() if isinstance(dims, bytes) else dims).split()
        dims = tuple(reversed(names)) if len(names) == len(var.shape) else tuple(names)
    return tuple(d.decode() if isinstance(d, bytes) else d for d in dims)


def _open_nc_scipy(path: Path, mmap: bool = False):
    from scipy.io import _netcdf as snc
    orig = snc.netcdf_variable.__init__

    def init(self, *args, **kwargs):
        dims = kwargs.get("dimensions", args[4] if len(args) > 4 else None)
        orig(self, *args, **kwargs)
        self.__dict__["_real_dims"] = tuple(dims) if dims is not None else self.dimensions

    snc.netcdf_variable.__init__ = init
    try:
        return snc.netcdf_file(path, "r", mmap=mmap)
    finally:
        snc.netcdf_variable.__init__ = orig


def _read_nc_scipy(path: Path, candidates):
    with _open_nc_scipy(path) as nc:
        vars_ = nc.variables
        name = next((v for v in candidates if v in vars_), None)
        if name is None:
            name = next(v for v, x in vars_.items() if len(_true_dims(x)) == 3)
        var = vars_[name]
        vdims = _true_dims(var)
        dims = [d.lower() for d in vdims]
        raw = np.array(var.data, dtype="float64")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in var._attributes.items()}
        for key in ("_FillValue", "missing_value"):
            if key in attrs:
                mv = np.atleast_1d(attrs[key]).astype(float)
                for x in mv:
                    raw[raw == x] = np.nan
        raw = raw * float(attrs.get("scale_factor", 1.0)) + float(attrs.get("add_offset", 0.0))
        raw[np.abs(raw) > 1e20] = np.nan
        tname = vdims[[i for i, d in enumerate(dims) if d in ("t", "time")][0]]
        yname = vdims[[i for i, d in enumerate(dims) if d in ("y", "lat", "latitude")][0]]
        xname = vdims[[i for i, d in enumerate(dims) if d in ("x", "lon", "longitude")][0]]
        order = [vdims.index(tname), vdims.index(yname), vdims.index(xname)]
        raw = np.transpose(raw, order)
        tu = vars_[tname]._attributes.get("units", b"")
        tu = tu.decode() if isinstance(tu, bytes) else tu
        months = _decode_months(vars_[tname].data, tu)
        units = attrs.get("units", "")
        return Series(raw, vars_[xname].data.copy(), vars_[yname].data.copy(), months, units)


def _read_nc_xarray(path: Path, candidates):
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    name = next((v for v in candidates if v in ds.data_vars), None) or [v for v in ds.data_vars if ds[v].ndim == 3][0]
    da = ds[name]
    ren = {}
    for c in da.dims:
        cl = c.lower()
        ren[c] = "lon" if cl in ("x", "lon", "longitude") else "lat" if cl in ("y", "lat", "latitude") else "time"
    da = da.rename(ren).transpose("time", "lat", "lon")
    months = _decode_months(da["time"].values, str(da["time"].attrs.get("units", "")))
    return Series(da.values, da["lon"].values, da["lat"].values, months, da.attrs.get("units", ""))


def _read_tif_stack(folder: Path):
    """Folder of monthly GeoTIFFs named ...YYYY.MM.tif (e.g. CHIRPS)."""
    import re
    arrs, months, lon, lat = [], [], None, None
    for f in sorted(folder.glob("*.tif")):
        mt = re.search(r"(\d{4})[._-](\d{2})", f.name)
        if not mt:
            continue
        a, lo, la = _read_geotiff(f)
        a[a < -1] = np.nan
        arrs.append(a); months.append(f"{mt.group(1)}-{mt.group(2)}")
        lon, lat = lo, la
    return Series(np.stack(arrs), lon, lat, months, "mm/month")


def open_series_many(paths, candidates) -> Series:
    """Several files that split one variable by time (e.g. CHIRPS downloaded in chunks of years)."""
    parts = [open_series(p, candidates) for p in paths]
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    for q in parts[1:]:
        if q.data.shape[1:] != first.data.shape[1:] or not np.allclose(q.lon, first.lon) or not np.allclose(q.lat, first.lat):
            raise SystemExit("rainfall chunks are on different grids; delete data/raw/basins/<basin> and prepare again")
    months, data, seen = [], [], set()
    for q in parts:
        keep = [i for i, m in enumerate(q.months) if m not in seen]
        seen.update(q.months[i] for i in keep)
        months += [q.months[i] for i in keep]
        data.append(q.data[keep])
    sr = Series(np.concatenate(data), first.lon, first.lat, months, first.units)
    order = np.argsort(sr.months)
    sr.data, sr.months = sr.data[order], [sr.months[i] for i in order]
    return sr


def _fix_half_cell(sr: Series) -> Series:
    """IRI labels CHIRPS columns by their western edge (-5.75, -5.70, ...) while rows are cell centres
    (14.925, ...). CHC's own GeoTIFFs put column centres at -5.725, ... Shift the columns by half a cell
    so the grid lines up with CHIRPS and with the model grid."""
    if len(sr.lon) < 2 or len(sr.lat) < 2:
        return sr
    r = float(abs(np.median(np.diff(sr.lon))))
    on_edges = lambda v: np.allclose(np.round(v / r) * r, v, atol=r * 0.02)
    if on_edges(sr.lon) and on_edges(sr.lat - r / 2) and not on_edges(sr.lat):
        log(f"shifting longitudes by half a cell ({r / 2:.3f} deg): the file labels columns by their west edge")
        sr.lon = sr.lon + r / 2
    return sr


def open_series(path, candidates) -> Series:
    p = Path(path)
    if p.is_dir():
        return _read_tif_stack(p)
    try:
        sr = _read_nc_scipy(p, candidates)            # NetCDF-3 (IRI, THREDDS NCSS "netcdf")
        return _fix_half_cell(sr) if any(c in candidates for c in ("precipitation", "prcp")) else sr
    except Exception as e_light:  # noqa: BLE001
        try:
            return _read_nc_xarray(p, candidates)     # NetCDF-4/HDF5 needs xarray + netCDF4
        except ImportError:
            raise SystemExit(f"{p.name} is not NetCDF-3 ({e_light}). Install xarray and netCDF4 to read it.")


def _read_geotiff(path: Path, bbox=None):
    """(array float64 with NaN nodata, lon centres, lat centres) for a north-up EPSG:4326 GeoTIFF.

    bbox=(west, south, east, north) crops before converting to float, to keep memory low.
    """
    try:
        import tifffile
    except ImportError:
        tifffile = None
    if tifffile is None:
        import rasterio
        with rasterio.open(path) as src:
            a = src.read(1)
            nod = src.nodata
            tr = src.transform
            west, north, dx, dy = tr.c, tr.f, tr.a, -tr.e
    else:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            H, W = page.imagelength, page.imagewidth
            tags = page.tags
            dx, dy = tags["ModelPixelScaleTag"].value[:2]
            tie = tags["ModelTiepointTag"].value
            west, north = tie[3] - tie[0] * dx, tie[4] + tie[1] * dy
            nod = tags.get("GDAL_NODATA")
            nod = None if nod is None else str(nod.value).strip().strip("\x00")
        a = None
    r0, r1, c0, c1 = 0, (a.shape[0] if a is not None else H), 0, (a.shape[1] if a is not None else W)
    if bbox is not None:
        w, s_, e, n = bbox
        c0 = max(0, int(np.floor((w - west) / dx))); c1 = min(c1, int(np.ceil((e - west) / dx)))
        r0 = max(0, int(np.floor((north - n) / dy))); r1 = min(r1, int(np.ceil((north - s_) / dy)))
        if c1 <= c0 or r1 <= r0:
            raise SystemExit(f"{Path(path).name} does not cover the model area")
    if a is None:
        from prep.tiffread import read_window          # decodes only the strips/tiles in the window
        t0 = time.time()
        sub = read_window(path, r0, r1, c0, c1).astype("float64")
        log(f"read {r1 - r0} x {c1 - c0} DEM pixels in {time.time() - t0:.0f} s")
    else:
        if a.ndim == 3:
            a = a[..., 0] if a.shape[-1] < a.shape[0] else a[0]
        sub = a[r0:r1, c0:c1].astype("float64")
    if nod not in (None, ""):
        try:
            sub[sub == float(nod)] = np.nan
        except ValueError:
            pass
    lon = west + (np.arange(c0, c1) + 0.5) * dx
    lat = north - (np.arange(r0, r1) + 0.5) * dy
    return sub, lon, lat


def regrid(sr: Series, grid: Grid, months_wanted, kind: str):
    """(T, nrows, ncols) float32 on `grid`: block mean if aligned & integer ratio, else bilinear."""
    idx = {m: i for i, m in enumerate(sr.months)}
    missing = [m for m in months_wanted if m not in idx]
    if missing:
        raise SystemExit(f"{kind}: {len(missing)} months missing from the source, e.g. {missing[:3]}. "
                         f"The source covers {sr.months[0]} to {sr.months[-1]}; adjust --start/--end.")
    data = sr.data[[idx[m] for m in months_wanted]]
    src_res = float(abs(np.median(np.diff(sr.lon))))
    k = grid.res / src_res
    if abs(k - round(k)) < 1e-3 and round(k) >= 1:
        k = int(round(k))
        west_edges = sr.lon - src_res / 2
        north_edges = sr.lat + src_res / 2
        c0 = int(np.argmin(np.abs(west_edges - grid.west)))
        r0 = int(np.argmin(np.abs(north_edges - grid.north)))
        aligned = abs(west_edges[c0] - grid.west) < src_res * 0.01 and abs(north_edges[r0] - grid.north) < src_res * 0.01
        if aligned and r0 + grid.nrows * k <= data.shape[1] and c0 + grid.ncols * k <= data.shape[2]:
            a = data[:, r0:r0 + grid.nrows * k, c0:c0 + grid.ncols * k]
            a = a.reshape(a.shape[0], grid.nrows, k, grid.ncols, k)
            with np.errstate(invalid="ignore"):
                out = np.nanmean(a, axis=(2, 4))
            log(f"{kind}: block-mean regrid x{k}")
            return out.astype("float32")
    log(f"{kind}: bilinear regrid {src_res:.4f} deg -> {grid.res} deg")
    from scipy.interpolate import RegularGridInterpolator
    yy, xx = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    out = np.empty((data.shape[0], grid.nrows, grid.ncols), dtype="float32")
    lat_asc = sr.lat[::-1]
    for t in range(data.shape[0]):
        f = RegularGridInterpolator((lat_asc, sr.lon), data[t, ::-1, :], bounds_error=False, fill_value=None)
        out[t] = f(pts).reshape(grid.nrows, grid.ncols)
    return out


def to_mm_month(arr, units: str, months):
    u = (units or "").lower().replace(" ", "")
    if "day" in u or "/d" in u or "d-1" in u:
        nd = np.array([calendar.monthrange(int(m[:4]), int(m[5:]))[1] for m in months], float)
        log(f"precip units '{units}' -> multiplying by days in month")
        return arr * nd[:, None, None]
    return arr


def fill_nan_nearest(arr3d, valid2d):
    from scipy.ndimage import distance_transform_edt
    out = arr3d.copy()
    bad_any = np.isnan(out).any(axis=0) & valid2d
    if not bad_any.any():
        return out
    for t in range(out.shape[0]):
        a = out[t]
        nan = np.isnan(a)
        if not (nan & valid2d).any():
            continue
        if nan.all():
            raise SystemExit("an entire forcing time step is missing")
        _, (ri, ci) = distance_transform_edt(nan, return_indices=True)
        a[nan] = a[ri[nan], ci[nan]]
    log(f"filled {int(bad_any.sum())} cells with nearest values (coast/lake gaps)")
    return out


def _unzip_member(path: Path, suffixes):
    """Extract the first member of a zip with one of `suffixes` next to it; return its path."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(tuple(suffixes))]
        if not names:
            raise SystemExit(f"{path.name} contains no {suffixes} file")
        out_dir = path.with_suffix("")
        z.extractall(out_dir)
    return out_dir


def read_dem(path, grid: Grid):
    """Average a DEM onto the model grid. GeoTIFF or a .zip containing one (HydroSHEDS)."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        d = _unzip_member(p, (".tif", ".tiff"))
        p = sorted(d.rglob("*.tif"))[0]
    a, lon, lat = _read_geotiff(p, bbox=(grid.west, grid.south, grid.east, grid.north))
    a[(a < -500) | (a > 9000)] = np.nan
    # block average of source pixels whose centres fall in each model cell
    c = np.floor((lon - grid.west) / grid.res).astype(int)
    r = np.floor((grid.north - lat) / grid.res).astype(int)
    cm = (c >= 0) & (c < grid.ncols); rm = (r >= 0) & (r < grid.nrows)
    sub = a[np.ix_(rm, cm)]
    R = np.repeat(r[rm][:, None], cm.sum(), axis=1); C = np.repeat(c[cm][None, :], rm.sum(), axis=0)
    ok = np.isfinite(sub)
    flat = R[ok] * grid.ncols + C[ok]
    s = np.bincount(flat, weights=sub[ok], minlength=grid.nrows * grid.ncols)
    n = np.bincount(flat, minlength=grid.nrows * grid.ncols)
    with np.errstate(invalid="ignore"):
        dem = (s / n).reshape(grid.nrows, grid.ncols)
    dem[n.reshape(grid.nrows, grid.ncols) < 1] = np.nan
    log(f"DEM: {np.isfinite(dem).sum()} cells with data, {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m")
    return dem


def _polygons_from_geojson(gj):
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    out = []
    for f in feats:
        g = f.get("geometry", f)
        if g["type"] == "Polygon":
            out.append(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            out.extend(g["coordinates"])
    return out  # list of polygons, each a list of rings


def _rasterize(polygons, grid: Grid):
    """Cell-centre point-in-polygon (even-odd over all rings, so holes work)."""
    x = grid.lon2d(); y = grid.lat2d()
    inside = np.zeros(x.shape, bool)
    for poly in polygons:
        inp = np.zeros(x.shape, bool)
        for ring in poly:
            ring = np.asarray(ring, float)
            px, py = ring[:, 0], ring[:, 1]
            if px.min() > grid.east or px.max() < grid.west or py.min() > grid.north or py.max() < grid.south:
                continue
            j = len(ring) - 1
            for i in range(len(ring)):
                cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-15) + px[i])
                inp ^= cond
                j = i
        inside |= inp
    return inside


def _shapefile_polygons(path: Path, name_filter=None):
    import shapefile  # pyshp
    if path.suffix.lower() == ".zip":
        d = _unzip_member(path, (".shp",))
        path = sorted(d.rglob("*.shp"))[0]
    rd = shapefile.Reader(str(path))
    fields = [f[0] for f in rd.fields[1:]]
    polys = []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if name_filter and not name_filter(rec):
            continue
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        polys.append([pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)])
    return polys


def read_mask(path, grid: Grid, name_contains: str | None = None):
    """GeoJSON, shapefile (.shp or .zip, via pyshp) or GeoTIFF -> boolean grid."""
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        a, lon, lat = _read_geotiff(p)
        c = np.clip(np.floor((grid.lon - lon[0]) / (lon[1] - lon[0]) + 0.5).astype(int), 0, len(lon) - 1)
        r = np.clip(np.floor((grid.lat - lat[0]) / (lat[1] - lat[0]) + 0.5).astype(int), 0, len(lat) - 1)
        return np.nan_to_num(a[np.ix_(r, c)]) >= 0.5
    if p.suffix.lower() in (".geojson", ".json"):
        polys = _polygons_from_geojson(json.loads(p.read_text()))
    else:
        flt = None
        if name_contains:
            key = name_contains.lower()
            flt = lambda rec: any(key in str(v).lower() for k, v in rec.items() if k.lower().startswith("name"))
        polys = _shapefile_polygons(p, flt)
        if not polys:
            raise SystemExit(f"no polygon named like '{name_contains}' in {p.name}")
    return _rasterize(polys, grid)


def read_qobs(path, gauges, months):
    """CSV: a 'date' column (YYYY-MM or YYYY-MM-DD) and one column per gauge name, in m3/s.

    Daily values are averaged to months; months with < 80 % of days present are dropped.
    """
    names = [g["name"] for g in gauges]
    sums = {n: {} for n in names}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.strip(): c for c in rd.fieldnames}
        date_col = next(c for c in cols if c.lower() in ("date", "time", "month"))
        match = {}
        for n in names:
            key = next((c for c in cols if c.lower() == n.lower() or c.lower().startswith(n.split(" ")[0].lower())), None)
            if key:
                match[n] = cols[key]
        log(f"qobs: matched columns {match}")
        for row in rd:
            d = row[cols[date_col]].strip()
            ym = d[:7]
            for n, c in match.items():
                v = row[c].strip()
                if v in ("", "NA", "NaN", "-9999", "-999"):
                    continue
                s = sums[n].setdefault(ym, [0.0, 0, len(d) > 7])
                s[0] += float(v); s[1] += 1
    q = np.full((len(gauges), len(months)), np.nan)
    for i, n in enumerate(names):
        for t, ym in enumerate(months):
            if ym in sums[n]:
                tot, cnt, daily = sums[n][ym]
                nd = calendar.monthrange(int(ym[:4]), int(ym[5:]))[1]
                if not daily or cnt >= 0.8 * nd:
                    q[i, t] = tot / cnt
        log(f"qobs {n}: {np.isfinite(q[i]).sum()} months")
    return q


# ------------------------------------------------------------------------------------ build
def _default(path_arg, key):
    if path_arg:
        return Path(path_arg)
    p = RAW / FILES[key]
    return p if p.exists() else None


def cmd_build(args, cfg):
    grid = model_grid(cfg, args.res)
    months = month_list(args.start, args.end)
    log(f"grid {grid.nrows}x{grid.ncols} at {grid.res} deg, {len(months)} months ({args.start} to {args.end})")

    chirps = _default(args.chirps, "chirps") or (RAW / "chirps_tifs" if (RAW / "chirps_tifs").exists() else None)
    tmax_p, tmin_p = _default(args.tmax, "tmax"), _default(args.tmin, "tmin")
    dem_p = _default(args.dem, "dem")
    for label, p in (("rainfall (CHIRPS)", chirps), ("Tmax", tmax_p), ("Tmin", tmin_p), ("DEM", dem_p)):
        if p is None or not Path(p).exists():
            raise SystemExit(f"{label} file not found. Run: python prep/prep_volta.py download")

    ps = open_series(chirps, ["precipitation", "precip", "prcp", "pr"])
    P = to_mm_month(regrid(ps, grid, months, "CHIRPS"), ps.units, months)
    TX = regrid(open_series(tmax_p, ["tmax", "tasmax", "tmx", "Tmax"]), grid, months, "Tmax")
    TN = regrid(open_series(tmin_p, ["tmin", "tasmin", "tmn", "Tmin"]), grid, months, "Tmin")
    if np.nanmean(TX) > 200:  # Kelvin
        TX -= 273.15; TN -= 273.15

    dem = read_dem(dem_p, grid)
    hint = read_mask(args.basin_mask, grid) if args.basin_mask else None
    lake = None
    lake_src = "none"
    if args.lake == "auto":
        lp = RAW / FILES["lakes"]
        if lp.exists():
            lake = read_mask(lp, grid, name_contains="volta")
            lake_src = "Natural Earth 1:10m lakes (Lake Volta)"
            log(f"Lake Volta: {lake.sum()} cells")
    elif args.lake and args.lake != "none":
        lake = read_mask(args.lake, grid)
        lake_src = Path(args.lake).name
    valid = np.isfinite(dem) if hint is None else (np.isfinite(dem) & hint)
    P = fill_nan_nearest(P, valid); TX = fill_nan_nearest(TX, valid); TN = fill_nan_nearest(TN, valid)

    qobs = read_qobs(args.qobs, cfg["gauges"], months) if args.qobs else None
    o = cfg["outlet"]
    src = cfg["sources"]
    data = build_dataset(grid, dem, P, TX, TN, months, (o["lon"], o["lat"]), cfg["gauges"], qobs=qobs,
                         basin_hint=hint, lake=lake, log=log, snap_radius=args.snap_radius,
                         outlet_area_km2=o.get("area_km2"),
                         meta={"name": cfg["name"], "synthetic": False, "created": date.today().isoformat(),
                               "precip": src["precip"] if not args.chirps else f"CHIRPS from {Path(args.chirps).name}",
                               "temperature": src["temperature"] if not args.tmax else Path(args.tmax).name,
                               "dem": Path(dem_p).name, "pet": src["pet"],
                               "basin_mask": str(args.basin_mask or "delineated from the DEM"),
                               "lake": lake_src,
                               "discharge": Path(args.qobs).name if args.qobs else "none (no observed flows loaded)"})
    area = data.grid.cell_area_m2().ravel()[data.cells].sum() / 1e6
    log(f"basin area {area:,.0f} km2 (published Volta area is about 400,000 km2)")
    out = Path(args.out)
    data.save(out)
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    _summary(data)


def _summary(data: Dataset):
    from hydro.model import VoltaModel
    m = VoltaModel(data)
    w = m.area / m.area.sum()
    yrs = data.T / 12
    log(f"basin mean rainfall {(m.precip.sum(0) * w).sum() / yrs:,.0f} mm/yr, Modified-Hargreaves PET "
        f"{(m.eto.sum(0) * w).sum() / yrs:,.0f} mm/yr, Tmax {(data.tmax.mean(0) * w).sum():.1f} C, "
        f"Tmin {(data.tmin.mean(0) * w).sum():.1f} C, lake cells {int(data.lake.sum())}")
    for i, g in enumerate(data.gauges):
        hint = next((x.get("area_km2") for x in load_config(ROOT / "config" / "volta.json")["gauges"]
                     if x["name"] == g["name"]), None)
        extra = f" (published {hint:,.0f})" if hint else ""
        log(f"  {g['name']:<28} area {g['area_km2']:>9,.0f} km2{extra}  obs months {np.isfinite(data.qobs[i]).sum()}")


def cmd_check(args, cfg):
    data = Dataset.load(args.out)
    log(f"{args.out}: {data.n} cells, {data.T} months {data.dates[0]}..{data.dates[-1]}")
    _summary(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "build", "all", "check"])
    ap.add_argument("--config", default=str(ROOT / "config" / "volta.json"))
    ap.add_argument("--res", type=float, default=None, help="model resolution in degrees (default from config, 0.05)")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--dem", help="DEM GeoTIFF or HydroSHEDS .zip (default: data/raw/hyd_af_dem_30s.zip)")
    ap.add_argument("--basin-mask", help="optional basin polygon (GeoJSON/shapefile/zip) or raster")
    ap.add_argument("--lake", default="auto", help="'auto' (Lake Volta from Natural Earth), 'none', or a polygon/raster file")
    ap.add_argument("--qobs", help="observed discharge CSV: date + one column per gauge name, m3/s")
    ap.add_argument("--chirps", help="local CHIRPS NetCDF or folder of monthly GeoTIFFs")
    ap.add_argument("--chirps-source", choices=["iri", "chc"], default="iri")
    ap.add_argument("--tmax"); ap.add_argument("--tmin")
    ap.add_argument("--snap-radius", type=int, default=3, help="cells searched when snapping gauges")
    ap.add_argument("--out", default=str(ROOT / "data" / "volta.npz"))
    ap.add_argument("--force", action="store_true", help="download again even if files exist")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    args.res = args.res or cfg.get("resolution_deg", 0.05)
    args.start = args.start or cfg["period"]["start"]
    args.end = args.end or cfg["period"]["end"]
    if args.command in ("download", "all"):
        cmd_download(args, cfg)
    if args.command in ("build", "all"):
        cmd_build(args, cfg)
    if args.command == "check":
        cmd_check(args, cfg)


if __name__ == "__main__":
    main()
