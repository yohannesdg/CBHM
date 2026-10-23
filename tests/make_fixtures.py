"""Create source files in the same formats as the real downloads, from the synthetic basin.

  * IRI-style CHIRPS NetCDF   (X, Y ascending, T = 'months since 1960-01-01', 360_day calendar)
  * TerraClimate-style NetCDF (lat descending, lon, time = 'days since 1900-01-01', int16 + scale_factor)
  * GeoTIFF DEM at 0.01 deg, GeoJSON basin polygon, daily discharge CSV

Used by tests/test_prep.py to exercise prep/prep_volta.py end-to-end without network access.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def make(out_dir: Path, data, nc_format: str = "NETCDF4"):
    import netCDF4
    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import distance_transform_edt

    from hydro.synthetic import OUTLINE

    out_dir.mkdir(parents=True, exist_ok=True)
    g = data.grid
    T = data.T

    def full(comp):
        a = np.stack([data.to_grid(comp[t]) for t in range(T)])
        nan = np.isnan(a[0])
        _, (ri, ci) = distance_transform_edt(nan, return_indices=True)
        a[:, nan] = a[:, ri[nan], ci[nan]]
        return a

    P, TX, TN = full(data.precip), full(data.tmax), full(data.tmin)

    # --- IRI CHIRPS-like (Y ascending, padded by 2 cells of 0.05)
    pad = 2
    Pp = np.pad(P, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    lon = g.west + (np.arange(g.ncols + 2 * pad) - pad + 0.5) * g.res
    lat = g.north - (np.arange(g.nrows + 2 * pad) - pad + 0.5) * g.res
    with netCDF4.Dataset(out_dir / "chirps_iri.nc", "w", format=nc_format) as nc:
        nc.createDimension("T", T); nc.createDimension("Y", lat.size); nc.createDimension("X", lon.size)
        t = nc.createVariable("T", "f4", ("T",)); t.units = "months since 1960-01-01"; t.calendar = "360_day"
        y0 = int(data.dates[0][:4]); m0 = int(data.dates[0][5:])
        t[:] = (y0 - 1960) * 12 + (m0 - 1) + np.arange(T) + 0.5
        nc.createVariable("Y", "f4", ("Y",))[:] = lat[::-1]
        nc.createVariable("X", "f4", ("X",))[:] = lon
        v = nc.createVariable("precipitation", "f4", ("T", "Y", "X"), fill_value=-999.0)
        v.units = "mm/month"
        v[:] = Pp[:, ::-1, :]

    # --- TerraClimate-like at 1/24 deg (bilinear-ish via nearest upsample) covering the bbox
    res = 1 / 24
    lon_t = np.arange(g.west - 0.1 + res / 2, g.east + 0.1, res)
    lat_t = np.arange(g.north + 0.1 - res / 2, g.south - 0.1, -res)
    ci = np.clip(((lon_t - g.west) / g.res).astype(int), 0, g.ncols - 1)
    ri = np.clip(((g.north - lat_t) / g.res).astype(int), 0, g.nrows - 1)
    for name, arr in (("tmax", TX), ("tmin", TN)):
        with netCDF4.Dataset(out_dir / f"terraclimate_{name}.nc", "w", format=nc_format) as nc:
            nc.createDimension("time", T); nc.createDimension("lat", lat_t.size); nc.createDimension("lon", lon_t.size)
            tv = nc.createVariable("time", "f8", ("time",)); tv.units = "days since 1900-01-01"
            days = [(np.datetime64(f"{d}-01") - np.datetime64("1900-01-01")).astype(int) for d in data.dates]
            tv[:] = days
            nc.createVariable("lat", "f8", ("lat",))[:] = lat_t
            nc.createVariable("lon", "f8", ("lon",))[:] = lon_t
            v = nc.createVariable(name, "i2", ("time", "lat", "lon"), fill_value=-32768)
            v.scale_factor = 0.01; v.add_offset = 0.0; v.units = "degC"
            v[:] = arr[:, ri][:, :, ci]

    # --- DEM GeoTIFF at 0.01 deg
    k = int(round(g.res / 0.01))
    coarse = np.where(np.isfinite(data.dem), data.dem, 600.0)
    coarse[(g.lat2d() < 5.85) & ~np.isfinite(data.dem)] = 0.0          # "sea" south of the outlet
    dem = np.kron(coarse, np.ones((k, k))).astype("float32")
    with rasterio.open(out_dir / "dem.tif", "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(g.west, g.north, 0.01, 0.01),
                       nodata=-9999) as dst:
        dst.write(dem, 1)

    # --- the same DEM, zipped like the HydroSHEDS download (GeoTIFF inside a folder in a zip)
    import zipfile
    with zipfile.ZipFile(out_dir / "hyd_af_dem_30s.zip", "w") as z:
        z.write(out_dir / "dem.tif", "hyd_af_dem_30s/hyd_af_dem_30s.tif")

    # --- lakes shapefile like Natural Earth (one feature named "Lake Volta", one decoy)
    import shapefile
    shp = out_dir / "ne_10m_lakes"
    with shapefile.Writer(str(shp), shapeType=shapefile.POLYGON) as w:
        w.field("name", "C", 40)
        rings = []
        rr, cc = np.nonzero(data.lake)
        for r, c in zip(rr, cc):
            x0, y0 = g.west + c * g.res, g.north - r * g.res
            rings.append([[x0, y0], [x0 + g.res, y0], [x0 + g.res, y0 - g.res], [x0, y0 - g.res], [x0, y0]])
        w.poly(rings); w.record("Lake Volta")
        w.poly([[[-3, 13], [-2.5, 13], [-2.5, 12.5], [-3, 12.5], [-3, 13]]]); w.record("Decoy Lake")
    with zipfile.ZipFile(out_dir / "ne_10m_lakes.zip", "w") as z:
        for ext in (".shp", ".shx", ".dbf"):
            z.write(str(shp) + ext, "ne_10m_lakes" + ext)

    # --- basin polygon + lake raster
    ring = [list(p) for p in OUTLINE] + [list(OUTLINE[0])]
    (out_dir / "basin.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Volta"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    with rasterio.open(out_dir / "lake.tif", "w", driver="GTiff", height=g.nrows, width=g.ncols, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(g.west, g.north, g.res, g.res)) as dst:
        dst.write(data.lake.astype("uint8"), 1)

    # --- daily discharge CSV for the gauges (monthly value repeated each day)
    with open(out_dir / "qobs_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [gg["name"] for gg in data.gauges])
        for t, ym in enumerate(data.dates):
            y, m = int(ym[:4]), int(ym[5:])
            nd = int((np.datetime64(f"{y + (m == 12)}-{(m % 12) + 1:02d}-01") - np.datetime64(f"{y}-{m:02d}-01")).astype(int))
            for dd in range(1, nd + 1):
                w.writerow([f"{ym}-{dd:02d}"] + ["" if not np.isfinite(data.qobs[i, t]) else f"{data.qobs[i, t]:.3f}"
                                                  for i in range(len(data.gauges))])
    return out_dir


def make_hydrobasins(out_zip: Path, data):
    """Fake hybas_af_lev01-12_v1c.zip (levels 2-5) from the synthetic basin's gauge zones.

    Level 2: one unit (the whole basin). Levels 3-5: the five gauge zones as units, each a multipart
    polygon of cell squares, with NEXT_DOWN links Bamboi/Nawuni/Saboba -> Senchi -> ungauged outlet.
    """
    import zipfile
    import shapefile
    from hydro.model import VoltaModel
    from hydro.synthetic import OUTLINE
    m = VoltaModel(data)
    g = data.grid
    zone2d = data.to_grid(m.zone.astype(float), fill=-1).astype(int)
    tmp = out_zip.parent / "_hybas_src"
    tmp.mkdir(parents=True, exist_ok=True)
    fields = [("HYBAS_ID", "N", 11, 0), ("NEXT_DOWN", "N", 11, 0), ("NEXT_SINK", "N", 11, 0), ("MAIN_BAS", "N", 11, 0),
              ("SUB_AREA", "F", 12, 1), ("UP_AREA", "F", 12, 1), ("PFAF_ID", "N", 13, 0), ("ENDO", "N", 2, 0), ("COAST", "N", 2, 0)]
    area = {z: float(m.area[m.zone == z].sum() / 1e6) for z in range(5)}
    up = {0: sum(area.values()), 4: area[4] + area[1] + area[2] + area[3], 1: area[1], 2: area[2], 3: area[3]}
    nxt = {1: 4, 2: 4, 3: 4, 4: 0, 0: None}
    for lv in (2, 3, 4, 5):
        base = 1_000_000_000 * lv
        name = tmp / f"hybas_af_lev{lv:02d}_v1c"
        with shapefile.Writer(str(name), shapeType=shapefile.POLYGON) as w:
            for f in fields:
                w.field(*f)
            if lv == 2:
                ring = [list(p) for p in OUTLINE] + [list(OUTLINE[0])]
                w.poly([ring]); w.record(base + 1, 0, 0, base + 1, up[0], up[0], 11, 0, 0)
                continue
            for z in range(5):
                rr, cc = np.nonzero(zone2d == z)
                rings = []
                for r, c in zip(rr, cc):
                    x0, y0 = g.west + c * g.res, g.north - r * g.res
                    rings.append([[x0, y0], [x0 + g.res, y0], [x0 + g.res, y0 - g.res], [x0, y0 - g.res], [x0, y0]])
                hid = base + 10 + z
                nd = 0 if nxt[z] is None else base + 10 + nxt[z]
                pf = int("11" + str(z + 1) + "1" * (lv - 3))
                w.poly(rings); w.record(hid, nd, base + 10, base + 10, area[z], up[z], pf, 0, 1 if z == 0 else 0)
    with zipfile.ZipFile(out_zip, "w") as zz:
        for lv in (2, 3, 4, 5):
            for e in (".shp", ".shx", ".dbf"):
                zz.write(tmp / f"hybas_af_lev{lv:02d}_v1c{e}", f"hybas_af_lev01-12_v1c/hybas_af_lev{lv:02d}_v1c{e}")
    return out_zip
