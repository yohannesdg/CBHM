"""Small vector helpers: shapefile reading (pyshp), ring simplification, fast polygon rasterisation."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from .grid import Grid


def snap_simplify(ring, tol: float):
    """Snap a ring to a tol-degree lattice and drop repeated points. Fast and adequate for display."""
    a = np.asarray(ring, float)
    if tol <= 0 or len(a) < 8:
        return a
    s = np.round(a / tol) * tol
    keep = np.ones(len(s), bool)
    keep[1:] = np.any(np.diff(s, axis=0) != 0, axis=1)
    s = s[keep]
    if len(s) < 4:
        return None
    if not np.allclose(s[0], s[-1]):
        s = np.vstack([s, s[:1]])
    return s


def rasterize(polygons, grid: Grid) -> np.ndarray:
    """Boolean mask of cell centres inside any polygon (even-odd rule over each polygon's rings).

    polygons: list of polygons, each a list of rings [[x, y], ...]. Scan-line algorithm:
    cost ~ rows x edges, so continental polygons are fine.
    """
    out = np.zeros((grid.nrows, grid.ncols), bool)
    lat = grid.lat
    for poly in polygons:
        edges = []
        for ring in poly:
            r = np.asarray(ring, float)
            if len(r) < 3:
                continue
            if not np.allclose(r[0], r[-1]):
                r = np.vstack([r, r[:1]])
            edges.append(np.column_stack([r[:-1], r[1:]]))
        if not edges:
            continue
        e = np.vstack(edges)                      # x1 y1 x2 y2
        ymin = np.minimum(e[:, 1], e[:, 3]); ymax = np.maximum(e[:, 1], e[:, 3])
        if e[:, [0, 2]].max() < grid.west or e[:, [0, 2]].min() > grid.east or ymax.max() < grid.south or ymin.min() > grid.north:
            continue
        rows = np.flatnonzero((lat >= ymin.min()) & (lat <= ymax.max()))
        tmp = np.zeros_like(out)
        for i in rows:
            y = lat[i]
            m = (e[:, 1] > y) != (e[:, 3] > y)
            if not m.any():
                continue
            x1, y1, x2, y2 = e[m].T
            xs = np.sort(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            # fill cells whose centres lie between pairs of crossings
            c0 = np.ceil((xs[0::2] - grid.west) / grid.res - 0.5).astype(int)
            c1 = np.floor((xs[1::2] - grid.west) / grid.res - 0.5).astype(int)
            for a, b in zip(c0, c1):
                a = max(a, 0); b = min(b, grid.ncols - 1)
                if b >= a:
                    tmp[i, a:b + 1] = True
        out |= tmp
    return out


def extract_zip_members(zip_path: Path, wanted, out_dir: Path):
    """Extract only members whose basename is in `wanted` (a set of names); return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            base = Path(info.filename).name
            if base in wanted:
                dest = out_dir / base
                if not dest.exists() or dest.stat().st_size != info.file_size:
                    with z.open(info) as src, open(dest, "wb") as dst:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                got[base] = dest
    return got


def read_shapefile(path: Path, name_filter=None, fields=None):
    """[(record dict, [rings])] from a polygon shapefile (or a .zip containing one)."""
    import shapefile  # pyshp
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            shp = next(n for n in z.namelist() if n.lower().endswith(".shp"))
        stem = Path(shp).stem
        files = extract_zip_members(path, {stem + e for e in (".shp", ".shx", ".dbf")}, path.with_suffix(""))
        path = files[stem + ".shp"]
    rd = shapefile.Reader(str(path))
    names = [f[0] for f in rd.fields[1:]]
    out = []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record))
        if fields:
            rec = {k: rec[k] for k in fields if k in rec}
        if name_filter and not name_filter(rec):
            continue
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        out.append((rec, rings))
    return out
