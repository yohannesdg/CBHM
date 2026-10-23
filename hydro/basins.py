"""HydroBASINS (HydroSHEDS) Africa catalogue for levels 2-5.

build_catalog() reads the official all-levels zip (hybas_af_lev01-12_v1c.zip) once and writes, per level:
  * hybas_af_levNN.geojson   light outlines for the map (Douglas-Peucker simplified)
  * hybas_af_levNN_raster.npz  more detailed rings used to rasterise a basin onto the model grid
Catalog then answers: features, upstream units (via NEXT_DOWN) and polygons for a selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geo import extract_zip_members, read_shapefile, snap_simplify

LEVELS = (2, 3, 4, 5)
DISPLAY_TOL = {2: 0.04, 3: 0.03, 4: 0.02, 5: 0.015}   # degrees, for the browser
RASTER_TOL = 0.008                                    # degrees, for gridding (model cells >= 0.05)
FIELDS = ("HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "SUB_AREA", "UP_AREA", "PFAF_ID", "ENDO", "COAST")


def douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Douglas-Peucker for a closed ring (first == last)."""
    n = len(pts)
    if n < 5:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    # split the ring at the point farthest from the start to avoid a degenerate first segment
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    keep[far] = True
    stack = [(0, far), (far, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b]
        d = q - p
        L = np.hypot(*d)
        if L == 0:
            dist = np.hypot(*(seg - p).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k)); stack.append((k, b))
    return pts[keep]


def _ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_catalog(zip_path: Path, out_dir: Path, levels=LEVELS, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"hybas_af_lev{lv:02d}_v1c{e}" for lv in levels for e in (".shp", ".shx", ".dbf")}
    log(f"extracting levels {list(levels)} from {zip_path.name}")
    files = extract_zip_members(zip_path, wanted, out_dir / "_shp")
    for lv in levels:
        shp = files.get(f"hybas_af_lev{lv:02d}_v1c.shp")
        if shp is None:
            raise SystemExit(f"level {lv} not found in {zip_path.name}")
        feats = read_shapefile(shp, fields=FIELDS)
        gj, xy, ring_off, feat_ring = [], [], [0], [0]
        for rec, rings in feats:
            disp, rast = [], []
            for ring in rings:
                r = np.asarray(ring, float)
                sr = snap_simplify(r, RASTER_TOL)
                if sr is not None and len(sr) >= 4:
                    rast.append(sr)
                dr = douglas_peucker(r, DISPLAY_TOL[lv])
                if len(dr) >= 4 and _ring_area(dr) > (DISPLAY_TOL[lv] ** 2) * 2:
                    disp.append(np.round(dr, 3).tolist())
            if not rast:
                continue
            for r in rast:
                xy.append(r.astype("float32")); ring_off.append(ring_off[-1] + len(r))
            feat_ring.append(feat_ring[-1] + len(rast))
            if not disp:   # keep tiny units clickable: use the largest raster ring
                big = max(rast, key=_ring_area)
                disp = [np.round(big, 3).tolist()]
            props = {"id": int(rec["HYBAS_ID"]), "next_down": int(rec["NEXT_DOWN"]), "main_bas": int(rec["MAIN_BAS"]),
                     "sub_area": float(rec["SUB_AREA"]), "up_area": float(rec["UP_AREA"]),
                     "pfaf": str(rec["PFAF_ID"]), "endo": int(rec["ENDO"]), "coast": int(rec["COAST"])}
            gj.append({"type": "Feature", "properties": props,
                       "geometry": {"type": "Polygon" if len(disp) == 1 else "MultiPolygon",
                                    "coordinates": disp if len(disp) == 1 else [[r] for r in disp]}})
        ids = np.array([f["properties"]["id"] for f in gj], np.int64)
        (out_dir / f"hybas_af_lev{lv:02d}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "level": lv, "features": gj}, separators=(",", ":")))
        np.savez_compressed(out_dir / f"hybas_af_lev{lv:02d}_raster.npz", ids=ids, xy=np.vstack(xy),
                            ring_off=np.array(ring_off, np.int64), feat_ring=np.array(feat_ring, np.int64))
        size = (out_dir / f"hybas_af_lev{lv:02d}.geojson").stat().st_size / 1e6
        log(f"level {lv}: {len(gj)} basins, map file {size:.1f} MB")


class Catalog:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._cache = {}

    def available(self):
        return [lv for lv in LEVELS if (self.folder / f"hybas_af_lev{lv:02d}.geojson").exists()]

    def path(self, level):
        return self.folder / f"hybas_af_lev{level:02d}.geojson"

    def _load(self, level):
        if level not in self._cache:
            gj = json.loads(self.path(level).read_text())
            props = {f["properties"]["id"]: f["properties"] for f in gj["features"]}
            kids = {}
            for i, p in props.items():
                kids.setdefault(p["next_down"], []).append(i)
            r = np.load(self.folder / f"hybas_af_lev{level:02d}_raster.npz")
            idx = {int(v): k for k, v in enumerate(r["ids"])}
            self._cache[level] = {"props": props, "kids": kids, "r": {k: r[k] for k in r.files}, "idx": idx}
        return self._cache[level]

    def props(self, level, hid):
        return self._load(level)["props"][int(hid)]

    def upstream(self, level, hid):
        c = self._load(level)
        out, stack = [], [int(hid)]
        while stack:
            h = stack.pop()
            out.append(h)
            stack.extend(c["kids"].get(h, []))
        return out

    def polygons(self, level, ids):
        """Rings for each id: list of polygons (each a list of rings)."""
        c = self._load(level)
        r = c["r"]
        polys = []
        for h in ids:
            k = c["idx"][int(h)]
            rings = [r["xy"][r["ring_off"][j]:r["ring_off"][j + 1]] for j in range(r["feat_ring"][k], r["feat_ring"][k + 1])]
            # all rings of one unit form one polygon under the even-odd rule: islands add, holes subtract
            polys.append(rings)
        return polys

    def unit_polygons(self, level, hid):
        return self.polygons(level, [hid])

    def bbox(self, level, ids):
        polys = self.polygons(level, ids)
        xy = np.vstack([ring for poly in polys for ring in poly])
        return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())
