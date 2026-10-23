"""Flow directions, basin delineation and monthly flow accumulation.

* Flow directions come from a priority-flood (Barnes et al., 2014) over the DEM. Each cell
  drains to the neighbour that first reached it from the outlet(s). The result is free of pits
  and every cell has a path to an outlet. If a single outlet is given and the
  domain is restricted to a basin mask, every masked cell drains to that outlet.
* Monthly routing assumes all runoff reaches the gauge within the month (realistic
  for monthly steps on the Volta). An optional linear reservoir (k months) at each gauge
  represents channel/floodplain attenuation.
* Open-water (lake) cells add P - Kw*ETo to the flow passing through them. The flow is
  floored at zero, and evaporation that cannot be met is recorded as a deficit.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.signal import lfilter

D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood_d8(dem: np.ndarray, valid: np.ndarray, seeds=None):
    """Return (downstream flat index array with -1 at outlets/invalid, filled dem).

    dem, valid: 2-D arrays. seeds: iterable of flat indices to start from (outlets).
    If None, all valid cells touching the grid edge or an invalid cell are seeds.
    """
    nr, nc = dem.shape
    n = nr * nc
    z = np.where(valid, dem, np.inf).astype(float).ravel()
    filled = z.copy()
    ds = np.full(n, -1, dtype=np.int64)
    visited = ~valid.ravel().copy()
    if seeds is None:
        pad = np.pad(valid, 1, constant_values=False)
        edge = np.zeros_like(valid)
        for dr, dc in D8:
            edge |= ~pad[1 + dr:1 + dr + nr, 1 + dc:1 + dc + nc]
        seeds = np.flatnonzero(edge & valid)
    heap = []
    counter = 0
    for s in seeds:
        s = int(s)
        if visited[s] and valid.ravel()[s]:
            continue
        visited[s] = True
        heapq.heappush(heap, (filled[s], counter, s))
        counter += 1
    while heap:
        e, _, c = heapq.heappop(heap)
        r, cc = divmod(c, nc)
        for dr, dc in D8:
            rr, c2 = r + dr, cc + dc
            if 0 <= rr < nr and 0 <= c2 < nc:
                nb = rr * nc + c2
                if not visited[nb]:
                    visited[nb] = True
                    ds[nb] = c
                    if filled[nb] < e:
                        filled[nb] = e
                    heapq.heappush(heap, (filled[nb], counter, nb))
                    counter += 1
    return ds, filled.reshape(nr, nc)


def children_csr(ds: np.ndarray):
    n = ds.size
    has = ds >= 0
    order = np.argsort(np.where(has, ds, n), kind="stable")
    counts = np.bincount(ds[has], minlength=n)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return ptr, order[: ptr[-1]]


def upstream_of(ds: np.ndarray, outlet: int) -> np.ndarray:
    """Boolean array of all cells draining to `outlet` (inclusive)."""
    ptr, kids = children_csr(ds)
    up = np.zeros(ds.size, dtype=bool)
    stack = [int(outlet)]
    up[outlet] = True
    while stack:
        c = stack.pop()
        for k in kids[ptr[c]:ptr[c + 1]]:
            if not up[k]:
                up[k] = True
                stack.append(int(k))
    return up


def compact_ds(ds_flat: np.ndarray, basin_flat: np.ndarray) -> np.ndarray:
    """Translate flat downstream indices to indices within the basin cell list."""
    cells = np.flatnonzero(basin_flat)
    idx = np.full(ds_flat.size, -1, dtype=np.int64)
    idx[cells] = np.arange(cells.size)
    d = ds_flat[cells]
    out = np.where(d >= 0, idx[np.maximum(d, 0)], -1)
    return out


class Router:
    """Topologically ordered accumulation on a compact (basin-only) cell list."""

    def __init__(self, ds: np.ndarray, lake: np.ndarray | None = None):
        self.ds = np.asarray(ds, dtype=np.int64)
        n = self.ds.size
        self.n = n
        self.lake = np.zeros(n, bool) if lake is None else np.asarray(lake, bool)
        indeg = np.bincount(self.ds[self.ds >= 0], minlength=n)
        levels = []
        cur = np.flatnonzero(indeg == 0)
        seen = 0
        while cur.size:
            levels.append(cur)
            seen += cur.size
            nxt = self.ds[cur]
            nxt = nxt[nxt >= 0]
            np.subtract.at(indeg, nxt, 1)
            cand = np.unique(nxt)
            cur = cand[indeg[cand] == 0]
        if seen != n:
            raise ValueError("flow network contains a cycle")
        self.levels = levels
        self.order = np.concatenate(levels) if levels else np.zeros(0, np.int64)
        self._lv = []
        for cells in levels:
            d = self.ds[cells]
            m = d >= 0
            lk = cells[self.lake[cells]]
            self._lv.append((cells[m], d[m], lk))

    def accumulate(self, q_local: np.ndarray):
        """q_local: (n,) or (n, T). Returns (Q, deficit) with the same shape."""
        Q = np.array(q_local, dtype=float, copy=True)
        deficit = np.zeros_like(Q) if self.lake.any() else None
        for src, dst, lk in self._lv:
            if lk.size:
                neg = np.minimum(Q[lk], 0.0)
                deficit[lk] = -neg
                Q[lk] -= neg
            if src.size:
                np.add.at(Q, dst, Q[src])
        return Q, deficit

    def upstream(self, cell: int) -> np.ndarray:
        return upstream_of(self.ds, cell)

    def zones_from_gauges(self, gauge_cells, gauge_ids, default=0) -> np.ndarray:
        """Assign each cell the id of the first gauge downstream of it (incremental areas)."""
        zone = np.full(self.n, default, dtype=np.int64)
        is_g = np.full(self.n, -1, dtype=np.int64)
        for c, gid in zip(gauge_cells, gauge_ids):
            is_g[c] = gid
        for c in self.order[::-1]:          # outlets first
            if is_g[c] >= 0:
                zone[c] = is_g[c]
            elif self.ds[c] >= 0:
                zone[c] = zone[self.ds[c]]
        return zone


def linear_reservoir(q: np.ndarray, k: float) -> np.ndarray:
    """Exponential (linear reservoir) smoothing along the last axis; k in months."""
    q = np.asarray(q, dtype=float)
    if k is None or k <= 1e-6:
        return q.copy()
    alpha = np.exp(-1.0 / k)
    zi = np.array([alpha * q[..., 0]]) if q.ndim == 1 else (alpha * q[..., :1])
    y, _ = lfilter([1.0 - alpha], [1.0, -alpha], q, axis=-1, zi=zi)
    return y


def snap_to_stream(acc2d: np.ndarray, basin2d: np.ndarray, row: int, col: int,
                   radius: int = 3, area_hint: float | None = None):
    """Move a gauge to the best cell within `radius` cells.

    Picks the cell with the largest accumulation, or, if area_hint (same units as acc)
    is given, the one whose accumulation is closest to the hint.
    """
    nr, nc = acc2d.shape
    r0, r1 = max(0, row - radius), min(nr, row + radius + 1)
    c0, c1 = max(0, col - radius), min(nc, col + radius + 1)
    win = np.where(basin2d[r0:r1, c0:c1], acc2d[r0:r1, c0:c1], np.nan)
    if np.all(np.isnan(win)):
        raise ValueError("no basin cell near gauge location")
    if area_hint:
        score = -np.abs(np.log(np.maximum(win, 1e-9) / area_hint))
    else:
        score = win
    k = np.nanargmax(score)
    rr, cc = divmod(int(k), win.shape[1])
    return r0 + rr, c0 + cc
