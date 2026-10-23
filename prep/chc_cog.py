"""CHIRPS v2.0 monthly rainfall read straight from the Climate Hazards Center's cloud-optimized GeoTIFFs.

https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/chirps-v2.0.YYYY.MM.cog

Each file is the global 0.05 deg grid (7200 x 2000, 50 S to 50 N) cut into 512 x 512 LZW tiles, with
overviews at 0.1, 0.2, 0.4 and 0.8 deg. HTTP range requests fetch only the file header and the tiles that
overlap the basin, so no sign-in and no server-side processing are needed. Tiles are kept in
data/raw/chirps_cog and reused by every basin that falls in them.
"""
from __future__ import annotations

import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from prep.tiffread import _decode

COG_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/cogs/chirps-v2.0.{y}.{m:02d}.cog"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) volta-hydro/1.2 (research water-balance model)"
HEADER_BYTES = 16384
TILE_MB_GUESS = 0.8          # typical compressed tile size, for estimates before anything is downloaded
WORKERS = 4


class Cancelled(Exception):
    pass


def _get(url, a, b, timeout=40, retries=5, cancel=None):
    """Bytes a..b (inclusive) of url."""
    last = None
    for k in range(retries):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.status == 200 and len(data) > b - a + 1:      # server ignored Range
                    data = data[a:b + 1]
                return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(20, 2 * (k + 1)))
    raise IOError(f"{url}: {last}")


def parse_header(buf: bytes):
    """Levels of a little-endian classic TIFF COG: [{w, h, tw, th, offsets_at, counts_at, n, res, west, north, comp}]."""
    if buf[:4] != b"II*\x00":
        raise ValueError("not a little-endian classic TIFF")
    u16 = lambda o: struct.unpack_from("<H", buf, o)[0]
    u32 = lambda o: struct.unpack_from("<I", buf, o)[0]
    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
    levels, off = [], u32(4)
    while off and off + 2 <= len(buf):
        n = u16(off)
        tags = {}
        for i in range(n):
            p = off + 2 + 12 * i
            tag, typ, cnt = u16(p), u16(p + 2), u32(p + 4)
            nbytes = size.get(typ, 1) * cnt
            vo = p + 8 if nbytes <= 4 else u32(p + 8)
            tags[tag] = (typ, cnt, vo)
        off = u32(off + 2 + 12 * n)
        val = lambda t: (u16(tags[t][2]) if tags[t][0] == 3 else u32(tags[t][2])) if t in tags else None
        lv = {"w": val(256), "h": val(257), "comp": val(259) or 1, "pred": val(317) or 1, "tw": val(322), "th": val(323),
              "n": tags[324][1], "offsets_at": tags[324][2], "counts_at": tags[325][2], "otype": tags[324][0]}
        if 33550 in tags:
            lv["scale"] = struct.unpack_from("<3d", buf, tags[33550][2])
            lv["tie"] = struct.unpack_from("<6d", buf, tags[33922][2])
        levels.append(lv)
    base = levels[0]
    r0 = round(base["scale"][0], 6)
    west, north = base["tie"][3] - base["tie"][0] * r0, base["tie"][4] + base["tie"][1] * r0
    for lv in levels:
        lv["res"] = round(r0 * base["w"] / lv["w"], 6)
        lv["west"], lv["north"] = west, north
    return levels


def tile_table(buf: bytes, lv):
    n = lv["n"]
    if lv["otype"] != 4 or lv["offsets_at"] + 4 * n > len(buf) or lv["counts_at"] + 4 * n > len(buf):
        raise ValueError("tile tables are outside the header block")
    offs = struct.unpack_from(f"<{n}I", buf, lv["offsets_at"])
    cnts = struct.unpack_from(f"<{n}I", buf, lv["counts_at"])
    return offs, cnts


def window(lv, bbox):
    """Pixel window (r0, r1, c0, c1) covering bbox=(west, south, east, north), clipped to the raster."""
    w, s, e, n = bbox
    r = lv["res"]
    c0 = max(0, int(np.floor((w - lv["west"]) / r + 1e-9)))
    c1 = min(lv["w"], int(np.ceil((e - lv["west"]) / r - 1e-9)))
    r0 = max(0, int(np.floor((lv["north"] - n) / r + 1e-9)))
    r1 = min(lv["h"], int(np.ceil((lv["north"] - s) / r - 1e-9)))
    if c1 <= c0 or r1 <= r0:
        raise SystemExit("the basin lies outside CHIRPS coverage (50 S to 50 N)")
    return r0, r1, c0, c1


def tiles_for(lv, win):
    r0, r1, c0, c1 = win
    across = -(-lv["w"] // lv["tw"])
    return [ty * across + tx for ty in range(r0 // lv["th"], (r1 - 1) // lv["th"] + 1)
            for tx in range(c0 // lv["tw"], (c1 - 1) // lv["tw"] + 1)]


# The tile layout is the same in every monthly file, so it can be planned from fixed numbers.
LAYOUT = [{"w": 7200 // f, "h": 2000 // f, "tw": 512, "th": 512, "res": round(0.05 * f, 6), "west": -180.0, "north": 50.0}
          for f in (1, 2, 4, 8, 16)]


def choose_level(target_res, bbox):
    """Overview level for a model grid: fewest tiles per month, at least two source pixels per model cell."""
    best = None
    for i, lv in enumerate(LAYOUT):
        if lv["res"] > target_res / 2 + 1e-9 and i > 0:
            break
        k = len(tiles_for(lv, window(lv, bbox)))
        if best is None or k < best[1]:
            best = (i, k)
    return best


def cache_dir(raw: Path):
    return raw / "chirps_cog"


def estimate_mb(raw: Path, months, target_res, bbox):
    lvl, k = choose_level(target_res, bbox)
    lv = LAYOUT[lvl]
    tiles = tiles_for(lv, window(lv, bbox))
    d = cache_dir(raw) / f"L{lvl}"
    missing = sum(1 for m in months for t in tiles if not (d / f"{m.replace('-', '')}_{t}.lzw").exists())
    return missing * TILE_MB_GUESS, lvl, len(tiles)


def fetch(raw: Path, months, target_res, bbox, out_nc: Path, log=print, progress=None, cancel=None):
    """Read the bbox for every month into a NetCDF-3 file (precipitation in mm/month, cell-centre coordinates)."""
    lvl, _ = choose_level(target_res, bbox)
    cdir = cache_dir(raw)
    (cdir / "hdr").mkdir(parents=True, exist_ok=True)
    (cdir / f"L{lvl}").mkdir(parents=True, exist_ok=True)
    lv0 = LAYOUT[lvl]
    win = window(lv0, bbox)
    r0, r1, c0, c1 = win
    tiles = tiles_for(lv0, win)
    log(f"CHIRPS from CHC: {len(months)} months, {r1 - r0} x {c1 - c0} pixels at {lv0['res']} deg, "
        f"{len(tiles)} tile(s) a month")
    out = np.full((len(months), r1 - r0, c1 - c0), np.nan, np.float32)
    lock = threading.Lock()
    state = {"bytes": 0, "done": 0, "t0": time.time(), "fresh": 0}

    def one(i, ym):
        y, m = int(ym[:4]), int(ym[5:7])
        url = COG_URL.format(y=y, m=m)
        tag = f"{y}{m:02d}"
        hf = cdir / "hdr" / f"{tag}.bin"
        need = [t for t in tiles if not (cdir / f"L{lvl}" / f"{tag}_{t}.lzw").exists()]
        got = 0
        buf = hf.read_bytes() if hf.exists() else None
        if need and buf is None:
            buf = _get(url, 0, HEADER_BYTES - 1, cancel=cancel)
            hf.write_bytes(buf); got += len(buf)
        if buf is None:
            buf = hf.read_bytes() if hf.exists() else _get(url, 0, HEADER_BYTES - 1, cancel=cancel)
        lv = {**parse_header(buf)[lvl]}
        if (lv["w"], lv["tw"]) != (lv0["w"], lv0["tw"]):
            raise SystemExit(f"unexpected tile layout in {url}")
        offs, cnts = tile_table(buf, lv)
        across = -(-lv["w"] // lv["tw"])
        for t in tiles:
            tf = cdir / f"L{lvl}" / f"{tag}_{t}.lzw"
            if tf.exists():
                raw_t = tf.read_bytes()
            else:
                raw_t = _get(url, offs[t], offs[t] + cnts[t] - 1, cancel=cancel)
                tmp = tf.with_suffix(".part"); tmp.write_bytes(raw_t); tmp.replace(tf)
                got += len(raw_t)
            a = np.frombuffer(_decode(raw_t, lv["comp"]), "<f4")[: lv["tw"] * lv["th"]].reshape(lv["th"], lv["tw"])
            ty, tx = divmod(t, across)
            ya, yb = max(r0, ty * lv["th"]), min(r1, (ty + 1) * lv["th"])
            xa, xb = max(c0, tx * lv["tw"]), min(c1, (tx + 1) * lv["tw"])
            out[i, ya - r0:yb - r0, xa - c0:xb - c0] = a[ya - ty * lv["th"]:yb - ty * lv["th"], xa - tx * lv["tw"]:xb - tx * lv["tw"]]
        with lock:
            state["bytes"] += got; state["done"] += 1; state["fresh"] += bool(got)
            done = state["done"]
            if progress:
                per = state["bytes"] / max(state["fresh"], 1)
                total = state["bytes"] + per * (len(months) - done) if state["fresh"] else 0
                progress(f"months {done}/{len(months)}", state["bytes"], total,
                         state["bytes"] / max(time.time() - state["t0"], 1e-6) / 1e3)
            if done % 60 == 0 or done == len(months):
                log(f"  rainfall {done}/{len(months)} months, {state['bytes'] / 1e6:.0f} MB downloaded")

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, i, ym) for i, ym in enumerate(months)]
        for f in futs:
            f.result()

    out[(out < 0) | ~np.isfinite(out)] = np.nan
    r = lv0["res"]
    lon = lv0["west"] + (np.arange(c0, c1) + 0.5) * r
    lat = lv0["north"] - (np.arange(r0, r1) + 0.5) * r
    _write_nc(out_nc, out, lon, lat, months)
    return out_nc


def _write_nc(path: Path, data, lon, lat, months):
    from scipy.io import netcdf_file
    tmp = path.with_suffix(".part")
    with netcdf_file(tmp, "w", version=2) as nc:
        nc.createDimension("time", len(months)); nc.createDimension("lat", len(lat)); nc.createDimension("lon", len(lon))
        v = nc.createVariable("lon", "f8", ("lon",)); v[:] = lon; v.units = "degrees_east"
        v = nc.createVariable("lat", "f8", ("lat",)); v[:] = lat; v.units = "degrees_north"
        t = nc.createVariable("time", "f8", ("time",))
        t[:] = [(int(m[:4]) - 1960) * 12 + int(m[5:7]) - 1 + 0.5 for m in months]
        t.units = "months since 1960-01-01"
        p = nc.createVariable("precipitation", "f4", ("time", "lat", "lon"))
        p[:] = np.nan_to_num(data, nan=-9999.0); p.units = "mm/month"; p.missing_value = np.float32(-9999.0)
        p._FillValue = np.float32(-9999.0)
        nc.source = "CHIRPS v2.0 monthly, Climate Hazards Center cloud-optimized GeoTIFFs"
    tmp.replace(path)
