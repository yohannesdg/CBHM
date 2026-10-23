import json
from pathlib import Path

import numpy as np
import pytest

from hydro.dataset import Dataset
from hydro.synthetic import make_synthetic


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    ds = make_synthetic(res=0.1, start="2000-01", end="2003-12", log=lambda *a: None)
    ds.save(d / "synthetic_tiny.npz")
    return d, ds


def test_api_endpoints(tiny, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tiny[0])
    with TestClient(appmod.app) as c:
        st = c.get("/api/status").json()
        assert not st["loaded"] and st["dataset"] is None and "synthetic_tiny.npz" in st["datasets"]   # opens on the basin browser
        assert c.get("/api/meta").status_code == 409
        c.post("/api/datasets/load", json={"name": "synthetic_tiny.npz"})
        st = c.get("/api/status").json()
        assert st["loaded"] and st["dataset"] == "synthetic_tiny.npz"
        meta = c.get("/api/meta").json()
        assert len(meta["dates"]) == 48 and len(meta["gauges"]) == 4
        r = c.post("/api/run", json={"params": {"global": {"b": 900}, "zones": {"2": {"a": 0.99}}}}).json()
        assert abs(r["balance"]["residual"]) < 1e-6 and r["params"]["zones"]["2"]["a"] == 0.99
        mp = c.get("/api/map", params={"var": "et", "mode": "clim", "month": 8}).json()
        assert mp["nrows"] * mp["ncols"] == len(mp["values"])
        assert c.get("/api/map", params={"var": "discharge", "mode": "anomaly", "t": 5}).status_code == 200
        g = meta["gauges"][0]
        cell = c.get("/api/cell", params={"row": g["row"], "col": g["col"]}).json()
        assert cell["zone"] == g["id"] and len(cell["series"]["runoff"]) == 48
        assert c.get("/api/export/gauges.csv").text.startswith("date,")
        assert c.get("/api/export/map.asc", params={"var": "runoff"}).text.startswith("ncols")
        assert c.get("/api/static/zone").status_code == 200
        assert c.post("/api/calibrate", json={"names": ["zz"]}).status_code == 400


def test_prep_end_to_end(tiny, tmp_path):
    pytest.importorskip("rasterio"); pytest.importorskip("xarray"); pytest.importorskip("netCDF4")
    from tests.make_fixtures import make
    import prep.prep_volta as prep
    src = tiny[1]
    fx = make(tmp_path / "fx", src)
    cfg_path = tmp_path / "cfg.json"
    cfg = json.loads((prep.ROOT / "config" / "volta.json").read_text())
    cfg["bbox"] = src.grid.bounds()
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "volta.npz"
    prep.main(["build", "--config", str(cfg_path), "--res", "0.1", "--start", "2000-01", "--end", "2003-12",
               "--chirps", str(fx / "chirps_iri.nc"), "--tmax", str(fx / "terraclimate_tmax.nc"),
               "--tmin", str(fx / "terraclimate_tmin.nc"), "--dem", str(fx / "dem.tif"),
               "--basin-mask", str(fx / "basin.geojson"), "--lake", str(fx / "lake.tif"),
               "--qobs", str(fx / "qobs_daily.csv"), "--out", str(out)])
    d = Dataset.load(out)
    assert d.n == src.n and np.array_equal(d.basin, src.basin) and np.array_equal(d.lake, src.lake)
    assert np.allclose(d.precip, src.precip, rtol=1e-5)               # CHIRPS path reproduces exactly
    assert np.nanmax(np.abs(d.tmax - src.tmax)) < 0.6                 # 1/24 deg -> grid interpolation
    assert np.allclose(np.nan_to_num(d.qobs), np.nan_to_num(src.qobs), atol=2e-3)


def test_upload_raw_body(tiny, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tiny[0])
    raw = (tiny[0] / "synthetic_tiny.npz").read_bytes()
    with TestClient(appmod.app) as c:
        r = c.post("/api/datasets/upload", params={"name": "copy.npz"}, content=raw)
        assert r.status_code == 200 and r.json()["dataset"] == "copy.npz"
        assert c.post("/api/datasets/upload", params={"name": "x.txt"}, content=b"x").status_code == 400


def test_prep_light_readers(tiny, tmp_path, monkeypatch):
    """Same build with only numpy/scipy + tifffile + pyshp (no xarray/netCDF4/rasterio at run time)."""
    pytest.importorskip("netCDF4"); pytest.importorskip("rasterio")   # only to *write* the fixtures
    pytest.importorskip("tifffile"); pytest.importorskip("shapefile")
    import shutil
    import sys
    from tests.make_fixtures import make
    import prep.prep_volta as prep
    src = tiny[1]
    fx = make(tmp_path / "fx", src, nc_format="NETCDF3_CLASSIC")
    raw = tmp_path / "raw"; raw.mkdir()
    for a, b in (("chirps_iri.nc", "chirps_monthly_volta.nc"), ("terraclimate_tmax.nc", "terraclimate_tmax_volta.nc"),
                 ("terraclimate_tmin.nc", "terraclimate_tmin_volta.nc"), ("hyd_af_dem_30s.zip", "hyd_af_dem_30s.zip"),
                 ("ne_10m_lakes.zip", "ne_10m_lakes.zip")):
        shutil.copy(fx / a, raw / b)
    monkeypatch.setattr(prep, "RAW", raw)
    for mod in ("xarray", "netCDF4", "rasterio"):
        monkeypatch.setitem(sys.modules, mod, None)                    # import now raises ImportError
    cfg = json.loads((prep.ROOT / "config" / "volta.json").read_text())
    cfg["bbox"] = src.grid.bounds()
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "volta.npz"
    prep.main(["build", "--config", str(tmp_path / "cfg.json"), "--res", "0.1", "--start", "2000-01", "--end", "2003-12",
               "--basin-mask", str(fx / "basin.geojson"), "--qobs", str(fx / "qobs_daily.csv"), "--out", str(out)])
    d = Dataset.load(out)
    assert np.array_equal(d.basin, src.basin)
    assert np.array_equal(d.lake, src.lake)                            # Lake Volta picked by name, decoy ignored
    assert np.allclose(d.precip, src.precip, rtol=1e-5)
    assert np.nanmax(np.abs(d.tmax - src.tmax)) < 0.6
    assert np.allclose(d.dem[d.basin], src.dem[src.basin], atol=0.01)  # zipped GeoTIFF read by tifffile


def test_month_decoding():
    import prep.prep_volta as prep
    assert prep._decode_months([252.5, 253.5], "months since 1960-01-01") == ["1981-01", "1981-02"]
    assert prep._decode_months([29600.0], "days since 1900-01-01 00:00:00") == ["1981-01"]


def test_african_basin_workflow(tiny, tmp_path, monkeypatch):
    """HydroBASINS catalogue -> upstream tracing -> on-demand basin preparation -> model run."""
    pytest.importorskip("netCDF4"); pytest.importorskip("rasterio"); pytest.importorskip("shapefile")
    import shutil
    from tests.make_fixtures import make, make_hydrobasins
    import prep.prep_volta as pv
    import prep.basin_data as bd
    from hydro.basins import Catalog
    from hydro.model import VoltaModel, default_params
    src = make_synthetic(res=0.05, start="2000-01", end="2001-12", log=lambda *a: None)
    fx = make(tmp_path / "fx", src, nc_format="NETCDF3_CLASSIC")
    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(fx / "hyd_af_dem_30s.zip", raw / "hyd_af_dem_30s.zip")
    shutil.copy(fx / "ne_10m_lakes.zip", raw / "ne_10m_lakes.zip")
    shutil.copy(fx / "chirps_iri.nc", raw / "chirps_monthly_volta.nc")       # reused for basins inside the box
    make_hydrobasins(raw / "hybas_af_lev01-12_v1c.zip", src)
    monkeypatch.setattr(pv, "RAW", raw)
    monkeypatch.setattr(bd, "BASIN_DIR", tmp_path / "basins")
    monkeypatch.setattr(bd, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()

    def fake_download(url, dest, **kw):                                        # TerraClimate subsets
        var = "tmax" if "tmax" in url else "tmin"
        shutil.copy(fx / f"terraclimate_{var}.nc", dest)
        return dest
    monkeypatch.setattr(pv, "download", fake_download)

    bd.ensure_catalog(log=lambda *a: None)
    cat = Catalog(tmp_path / "basins")
    assert cat.available() == [2, 3, 4, 5]
    senchi = 4_000_000_000 + 14
    assert sorted(cat.upstream(4, senchi)) == sorted([senchi, 4_000_000_011, 4_000_000_012, 4_000_000_013])
    pl = bd.plan(4, senchi, start="2000-01", end="2001-12")
    assert pl["units"] == 4 and [o["res"] for o in pl["options"]] == [0.05, 0.1, 0.25, 0.5]
    assert pl["options"][0]["cells"] > pl["options"][1]["cells"] > pl["options"][2]["cells"]
    assert pl["auto_res"] in (0.05, 0.1, 0.25, 0.5)
    pl5 = bd.plan(4, senchi, res="0.05", start="2000-01", end="2001-12")
    assert pl5["res"] == 0.05 and pl5["dataset"].endswith("_0.05.npz")

    out = bd.prepare(4, senchi, res="0.05", start="2000-01", end="2001-12", log=lambda *a: None)
    d = Dataset.load(out)
    assert len(d.zone_names) == 4 and not d.gauges
    # catchment upstream of Senchi = synthetic basin minus the small ungauged zone below it
    srcm = VoltaModel(src)
    want = src.to_grid((srcm.zone != 0).astype(float), fill=0) > 0
    cells = lambda grid, mask: {(round(x, 3), round(y, 3)) for x, y in zip(grid.lon2d()[mask], grid.lat2d()[mask])}
    got, ref = cells(d.grid, d.basin), cells(src.grid, want)
    assert len(got & ref) / len(ref) > 0.97 and len(got) < len(ref) * 1.03
    m = VoltaModel(d)
    assert len(m.zones_info()) == 4
    r = m.run(default_params())
    assert abs(r.balance["residual"]) < 1e-6 * r.balance["P"]

    # IRI does not answer: rainfall comes from the CHC tiles instead
    from prep import chc_cog
    iri_calls, chc_calls = [], []

    def dl2(url, dest, **kw):
        if "iridl" in url:
            iri_calls.append(url)
            raise RuntimeError("no answer")
        return fake_download(url, dest, **kw)

    def fake_fetch(raw_, months, res, bbox, out_nc, **kw):
        chc_calls.append((res, bbox))
        sr = pv.open_series(fx / "chirps_iri.nc", ["precipitation"])
        idx = [sr.months.index(m) for m in months]
        chc_cog._write_nc(out_nc, sr.data[idx].astype("float32"), sr.lon, sr.lat, months)
        return out_nc
    monkeypatch.setattr(pv, "download", dl2)
    monkeypatch.setattr(chc_cog, "fetch", fake_fetch)
    out2 = bd.prepare(4, senchi, res="0.1", start="2000-01", end="2001-12", log=lambda *a: None)
    assert len(iri_calls) == 1 and len(chc_calls) == 1
    d2 = Dataset.load(out2)
    assert d2.grid.res == 0.1 and abs(np.nanmean(d2.precip) - np.nanmean(src.precip)) / np.nanmean(src.precip) < 0.05


def test_download_rejects_empty_and_error_pages(tmp_path):
    import prep.prep_volta as pv
    (tmp_path / "a.nc").write_bytes(b"")
    (tmp_path / "b.nc").write_bytes(b"<html>Error 500 internal</html>" * 20)
    (tmp_path / "c.nc").write_bytes(b"CDF\x01" + b"\0" * 400)
    (tmp_path / "d.zip").write_bytes(b"PK\x03\x04" + b"\0" * 400)
    assert not pv._usable(tmp_path / "a.nc") and not pv._usable(tmp_path / "b.nc")
    assert pv._usable(tmp_path / "c.nc") and pv._usable(tmp_path / "d.zip")
    assert not pv._usable(tmp_path / "missing.nc")


def test_real_terraclimate_v11_sample():
    """A real 3-month, 7x7 subset from the TerraClimate v1.1 NCSS service. Its tmax variable carries an
    attribute named "dimensions" ("lon lat time") that hides the real (time, lat, lon) order in scipy."""
    from pathlib import Path
    import prep.prep_volta as pv
    sr = pv.open_series(Path(__file__).parent / "data" / "terraclimate_v11_sample.nc", ["tmax"])
    assert sr.data.shape == (3, 7, 7) and sr.months == ["2000-01", "2000-02", "2000-03"]
    assert sr.lat[0] > sr.lat[-1] and sr.lon[0] < sr.lon[-1]
    assert abs(sr.data[0, 0, 0] - 34.9) < 0.01 and abs(sr.data[0, 1, 1] - 34.6) < 0.01
    assert 20 < np.nanmean(sr.data) < 45


def test_real_iri_chirps_sample_registration():
    """Real IRI CHIRPS subset: columns are labelled by their west edge; after the fix they line up with a
    0.05-degree model grid whose edges are multiples of 0.05, so the exact block mean is used."""
    from pathlib import Path
    import prep.prep_volta as pv
    from hydro.grid import Grid
    sr = pv.open_series(Path(__file__).parent / "data" / "chirps_iri_sample.nc", ["precipitation", "precip"])
    assert np.allclose(sr.lon[:2], [-5.725, -5.675], atol=1e-4)
    assert sr.months == ["1981-01", "1981-02"] and sr.units == "mm/month"
    g = Grid(west=-5.75, north=14.95, res=0.05, nrows=4, ncols=6)
    out = pv.regrid(sr, g, ["1981-01", "1981-02"], "CHIRPS")
    assert np.allclose(out, sr.data, equal_nan=True)                   # exact copy, no interpolation


@pytest.mark.parametrize("compress,predictor,tiled", [("lzw", 2, False), ("lzw", 1, True), ("deflate", 2, True),
                                                       ("packbits", 1, False), (None, 1, False), ("lzw", 2, True)])
def test_windowed_tiff_decoder_matches_rasterio(tmp_path, compress, predictor, tiled):
    rasterio = pytest.importorskip("rasterio"); pytest.importorskip("tifffile")
    from rasterio.transform import from_origin
    from prep.tiffread import read_window
    rng = np.random.default_rng(3)
    a = (np.cumsum(rng.integers(-3, 4, (300, 700)), axis=1) + 200).astype("int16")   # DEM-like texture
    a[:20, :50] = -32768
    kw = dict(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="int16", crs="EPSG:4326",
              transform=from_origin(-20, 38, 1 / 120, 1 / 120), nodata=-32768)
    if compress:
        kw["compress"] = compress
        if predictor == 2:
            kw["predictor"] = 2
    if tiled:
        kw.update(tiled=True, blockxsize=128, blockysize=128)
    else:
        kw["blockysize"] = 7
    p = tmp_path / "dem.tif"
    with rasterio.open(p, "w", **kw) as dst:
        dst.write(a, 1)
    for (r0, r1, c0, c1) in [(0, 300, 0, 700), (13, 211, 99, 650), (290, 300, 690, 700)]:
        assert np.array_equal(read_window(p, r0, r1, c0, c1), a[r0:r1, c0:c1])


def test_lzw_speed_volta_window(tmp_path):
    """A HydroSHEDS-like LZW strip GeoTIFF: the Volta window decodes in well under a minute."""
    rasterio = pytest.importorskip("rasterio")
    import time
    from rasterio.transform import from_origin
    from prep.tiffread import read_window
    rng = np.random.default_rng(4)
    a = (np.cumsum(rng.integers(-2, 3, (1200, 8880)), axis=1) % 900).astype("int16")
    p = tmp_path / "big.tif"
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="int16",
                       crs="EPSG:4326", transform=from_origin(-20, 38, 1 / 120, 1 / 120), compress="lzw",
                       predictor=2, blockysize=1) as dst:
        dst.write(a, 1)
    t0 = time.time()
    w = read_window(p, 10, 1126, 1700, 2700)
    assert np.array_equal(w, a[10:1126, 1700:2700])
    assert time.time() - t0 < 60


def test_rainfall_chunks_and_multi_file_series(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    import prep.prep_volta as pv
    import prep.basin_data as bd
    from hydro.grid import Grid
    g = Grid(west=10, north=5, res=0.05, nrows=200, ncols=200)
    ch = bd._year_chunks(g, "1981-01", "2023-12")
    assert ch[0][0] == "1981-01" and ch[-1][1] == "2023-12" and len(ch) > 1
    years = [int(b[:4]) - int(a[:4]) + 1 for a, b in ch]
    assert sum(years) == 43 and max(years) * 200 * 200 * 12 * 4 / 1e6 <= bd.CHUNK_MB + 2
    files = []
    for k, (y0, n) in enumerate([(1981, 24), (1983, 12)]):          # two files split by time
        f = tmp_path / f"c{k}.nc"
        with netCDF4.Dataset(f, "w", format="NETCDF3_CLASSIC") as nc:
            nc.createDimension("T", n); nc.createDimension("Y", 3); nc.createDimension("X", 4)
            t = nc.createVariable("T", "f4", ("T",)); t.units = "months since 1960-01-01"
            t[:] = (y0 - 1960) * 12 + np.arange(n) + 0.5
            nc.createVariable("Y", "f4", ("Y",))[:] = [1.025, 1.075, 1.125]
            nc.createVariable("X", "f4", ("X",))[:] = [2.025, 2.075, 2.125, 2.175]
            v = nc.createVariable("precipitation", "f4", ("T", "Y", "X")); v.units = "mm/month"
            v[:] = np.arange(n)[:, None, None] + y0 + np.zeros((n, 3, 4))
        files.append(f)
    sr = pv.open_series_many(files[::-1], ["precipitation"])
    assert sr.months[0] == "1981-01" and sr.months[-1] == "1983-12" and len(sr.months) == 36
    assert sr.data[0, 0, 0] == 1981 and sr.data[24, 0, 0] == 1983


def test_observed_upload_and_annual_export(tiny, monkeypatch, tmp_path):
    """A dataset with no observations gets records from a CSV and a GRDC export; annual MCM export adds up."""
    from fastapi.testclient import TestClient
    import backend.app as appmod
    src = tiny[1]
    ds = Dataset.load(tiny[0] / "synthetic_tiny.npz")
    truth = ds.qobs.copy()
    ds.qobs[:] = np.nan
    ds.meta = {**ds.meta, "synthetic": False}
    ds.save(tmp_path / "noobs.npz")
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    g0, g1 = src.gauges[0], src.gauges[1]
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "noobs.npz"})
        assert all(m["metrics"]["all"]["n"] == 0 for m in c.get("/api/results").json()["gauges"])
        # monthly CSV, column named after gauge 0 (first word only), with a missing value
        lines = ["date," + g0["name"].split(" ")[0]] + [
            f"{d},{'' if t == 3 else f'{truth[0, t]:.4f}'}" for t, d in enumerate(ds.dates)]
        r = c.post("/api/observed", params={"name": "obs.csv"}, content="\n".join(lines).encode())
        assert r.status_code == 200, r.text
        rep = r.json()["summary"]["gauges"]
        assert rep[0]["metrics"]["all"]["n"] == np.isfinite(truth[0]).sum() - 1 and rep[1]["metrics"]["all"]["n"] == 0
        # GRDC daily export for gauge 1, station name not matching, so the gauge is chosen explicitly
        import calendar
        body = ["# GRDC STATION DATA FILE", "# Station:             SOMEWHERE", "# Data Set Content:   MEAN DAILY DISCHARGE",
                "YYYY-MM-DD;hh:mm; Value"]
        for t, d in enumerate(ds.dates):
            y, m = int(d[:4]), int(d[5:])
            for day in range(1, calendar.monthrange(y, m)[1] + 1):
                v = truth[1, t] if np.isfinite(truth[1, t]) else -999
                body.append(f"{y}-{m:02d}-{day:02d};--:--;{v:10.3f}")
        raw = "\n".join(body).encode()
        assert c.post("/api/observed", params={"name": "g.txt"}, content=raw).status_code == 400
        r = c.post("/api/observed", params={"name": "g.txt", "gauge": str(g1["id"])}, content=raw)
        assert r.status_code == 200, r.text
        obs1 = np.array([np.nan if v is None else v for v in r.json()["summary"]["gauges"][1]["obs"]])
        assert np.allclose(np.nan_to_num(obs1), np.nan_to_num(truth[1]), atol=2e-3)
        # saved with the dataset
        assert np.isfinite(Dataset.load(tmp_path / "noobs.npz").qobs[1]).sum() == np.isfinite(truth[1]).sum()
        # annual export: outlet volume equals sum of monthly flow x seconds
        txt = c.get("/api/export/annual.csv").text.strip().splitlines()
        head = txt[0].split(",")
        assert head[:2] == ["year", "months"] and "outlet_outflow_MCM" in head and len(txt) == 1 + 4
        res = c.get("/api/results").json()
        q = np.array(res["basin"]["outlet_q"][:12])
        secs = np.array([calendar.monthrange(2000, m)[1] * 86400 for m in range(1, 13)])
        row = txt[1].split(",")
        assert abs(float(row[head.index("outlet_outflow_MCM")]) - (q * secs).sum() / 1e6) < 1e-2
        assert c.post("/api/observed", params={"name": "bad.csv"}, content=b"x,y\n1,2").status_code == 400


def test_zone_partition_closes(tiny, monkeypatch):
    """Rainfall partition per zone adds back to rainfall, and streamflow generated matches the basin outflow."""
    from fastapi.testclient import TestClient
    import backend.app as appmod
    monkeypatch.setattr(appmod, "DATA_DIR", tiny[0])
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "synthetic_tiny.npz"})
        z = c.get("/api/zones/summary").json()
        bal = c.get("/api/results").json()["balance"]["per_year"]
        for row in z["zones"] + [z["total"]]:
            p = row["partition"]
            assert abs(p["et_land"] + p["et_lake"] + p["quickflow"] + p["recharge"] + p["d_soil"] - row["precip"]) < 1e-6
            assert abs(p["baseflow"] + p["d_gw"] - p["recharge"]) < 1e-9
        t = z["total"]["partition"]
        assert abs(t["quickflow"] + t["baseflow"] - bal["Q"]) / bal["Q"] < 0.01


def test_chc_cog_real_header_and_tile():
    """Bytes copied from the real CHC file: the header parser and LZW decoder read them correctly."""
    import base64
    from prep import chc_cog
    from prep.tiffread import _decode
    d = json.loads((Path(__file__).parent / "data" / "chc_cog_sample.json").read_text())
    hb = base64.b64decode(d["header"][:2944])
    lv = chc_cog.parse_header(hb)
    assert [(x["w"], x["h"], x["res"]) for x in lv] == [(m["w"], m["h"], m["res"]) for m in chc_cog.LAYOUT]
    assert lv[0]["west"] == -180 and lv[0]["north"] == 50 and lv[0]["comp"] == 5
    offs, cnts = chc_cog.tile_table(hb, lv[0])
    tb = base64.b64decode(d["tile"])
    assert cnts[14] == len(tb)
    a = np.frombuffer(_decode(tb, 5), "<f4").reshape(512, 512)
    assert np.all(a[:, :32] == -9999) and np.all(a[:, 32:] == 0)      # 179.4-180 E is ocean, rest is padding


def test_chc_cog_fetch_matches_source(tmp_path, monkeypatch):
    """fetch() reads a basin window from COG files over 'HTTP range requests' and caches the tiles."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from prep import chc_cog
    rng = np.random.default_rng(5)
    months = ["2000-01", "2000-02", "2000-03"]
    W, H = 1024, 768
    src = {}
    for i, m in enumerate(months):
        a = (rng.gamma(2.0, 30.0, (H, W)) * (i + 1)).astype("float32")
        a[:40, :60] = -9999
        p = tmp_path / f"chirps-v2.0.{m.replace('-', '.')}.cog"
        with rasterio.open(p, "w", driver="COG", width=W, height=H, count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(-180, 50, 0.05, 0.05), nodata=-9999, compress="LZW",
                           blocksize=512, overview_resampling="average") as dst:
            dst.write(a, 1)
        src[m] = (p, a)
    layout = [{"w": W, "h": H, "tw": 512, "th": 512, "res": 0.05, "west": -180.0, "north": 50.0},
              {"w": W // 2, "h": H // 2, "tw": 512, "th": 512, "res": 0.1, "west": -180.0, "north": 50.0}]
    monkeypatch.setattr(chc_cog, "LAYOUT", layout)
    calls = []

    def fake_get(url, a, b, **kw):
        calls.append((url, a, b))
        name = url.rsplit("/", 1)[1]
        return (tmp_path / name).read_bytes()[a:b + 1]
    monkeypatch.setattr(chc_cog, "_get", fake_get)
    monkeypatch.setattr(chc_cog, "COG_URL", "https://example.test/chirps-v2.0.{y}.{m:02d}.cog")
    raw = tmp_path / "raw"
    bbox = (-157.0, 26.5, -150.0, 33.5)            # crosses the tile boundary at -154.4 E
    out = chc_cog.fetch(raw, months, 0.1, bbox, tmp_path / "rain.nc", log=lambda *a: None)
    import prep.prep_volta as pv
    sr = pv.open_series(out, ["precipitation"])
    r0, r1 = int(round((50 - 33.5) / 0.05)), int(round((50 - 26.5) / 0.05))
    c0, c1 = int(round((-157 + 180) / 0.05)), int(round((-150 + 180) / 0.05))
    for t, m in enumerate(months):
        assert np.allclose(sr.data[t], src[m][1][r0:r1, c0:c1])
    assert sr.months == months and abs(sr.lon[0] - (-156.975)) < 1e-9 and abs(sr.lat[0] - 33.475) < 1e-9
    n_first = len(calls)
    assert n_first == len(months) * 3                 # one header and two tiles per month
    chc_cog.fetch(raw, months, 0.1, bbox, tmp_path / "rain2.nc", log=lambda *a: None)
    assert len(calls) == n_first                      # everything came from the tile cache
    est, lvl, nt = chc_cog.estimate_mb(raw, months, 0.1, bbox)
    assert est == 0 and nt == 2
