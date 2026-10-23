"""Open gauge records from ADHI: station matching, snapping, merging and the API."""
import zipfile

import numpy as np
import pytest

from hydro.dataset import Dataset
from hydro.model import VoltaModel, default_params
from hydro.synthetic import make_synthetic


def _write_adhi(folder, stations, series):
    """stations: list of dicts; series: {adhi id: {"YYYY-MM": value}}."""
    folder.mkdir(parents=True, exist_ok=True)
    cols = ["ID", "Station_co", "Name", "Data_sourc", "Catchment", "Starting_y", "Ending_yea", "Number_dams",
            "Country", "Longitude", "Latitude", "Flag", "Comment"]
    lines = ["\t".join(cols)]
    for s in stations:
        lines.append("\t".join(f'"{s.get(c, "")}"' if c in ("Name", "Comment", "Country") else str(s.get(c, "0")) for c in cols))
    (folder / "ADHI_stations.tab").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(folder / "ADHI_MonthlySeries.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MonthlySeries/", "")
        for sid, vals in series.items():
            rows = []
            for y in range(1950, 2019):
                for m in range(1, 13):
                    v = vals.get(f"{y}-{m:02d}")
                    rows.append(f"{y},{m},{'NaN' if v is None else f'{v:.4g}'},NaN,NaN,{0 if v is not None else 30}")
            z.writestr(f"MonthlySeries/monthly_{sid}.txt", "\n".join(rows) + "\n")


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    import prep.prep_volta as pv
    src = make_synthetic(res=0.1, start="1995-01", end="2004-12", log=lambda *a: None)
    truth = src.qobs.copy()
    src.qobs[:] = np.nan
    src.meta = {**src.meta, "synthetic": False, "name": "test basin"}
    raw = tmp_path / "raw"
    monkeypatch.setattr(pv, "RAW", raw)
    m = VoltaModel(src)
    sim = m.run(default_params())
    g0, g2 = src.gauges[0], src.gauges[2]
    # a new station on the river halfway down the largest tributary: take a cell with ~40 % of gauge 2's area
    acc = src.to_grid(m.router.accumulate(m.area / 1e6)[0])
    target = 0.4 * g2["area_km2"]
    rr, cc = np.unravel_index(np.nanargmin(np.abs(acc - target) + np.where(src.basin, 0, np.inf)), acc.shape)
    new_cell = src.compact_index(rr, cc)
    stations = [
        {"ID": "ADHI_1", "Station_co": "XXQ1", "Name": f"{g0['name'].upper()} RIVER a {g0['name'].upper()}", "Catchment": g0["area_km2"] * 1.02,
         "Longitude": g0["lon"] + 0.03, "Latitude": g0["lat"] - 0.02, "Starting_y": 1990, "Ending_yea": 2004, "Country": "Ghana", "Flag": 0},
        {"ID": "ADHI_2", "Station_co": "XXQ2", "Name": "NEW RIVER a NEWTOWN", "Catchment": float(acc[rr, cc]),
         "Longitude": float(src.grid.lon[cc]), "Latitude": float(src.grid.lat[rr]), "Starting_y": 1995, "Ending_yea": 2004, "Flag": 0},
        {"ID": "ADHI_3", "Station_co": "XXQ3", "Name": "FAR AWAY a ELSEWHERE", "Catchment": 500, "Longitude": 30, "Latitude": -10,
         "Starting_y": 1960, "Ending_yea": 1990, "Flag": 0},
        {"ID": "ADHI_4", "Station_co": "XXQ4", "Name": "WRONG AREA a NOWHERE", "Catchment": g0["area_km2"] * 3,
         "Longitude": g0["lon"], "Latitude": g0["lat"], "Starting_y": 1990, "Ending_yea": 2004, "Flag": 0},
    ]
    months = src.dates
    s1 = {mo: float(v) for mo, v in zip(months, truth[0]) if np.isfinite(v) and mo >= "1998-01"}
    s2 = {mo: float(v) for mo, v in zip(months, sim.fields["discharge"][:, new_cell]) if mo <= "2003-12"}
    _write_adhi(raw / "adhi", stations, {"ADHI_1": s1, "ADHI_2": s2, "ADHI_3": {"1970-01": 5.0}, "ADHI_4": s1})
    return src, truth, s1, s2, new_cell


def test_candidates_snap_match_and_add(setup):
    from prep import adhi
    src, truth, s1, s2, new_cell = setup
    assert adhi.files_ready()
    c = {x["adhi"]: x for x in adhi.candidates(src)}
    assert set(c) == {"ADHI_1", "ADHI_2"}                         # outside the box / wrong area are left out
    assert c["ADHI_1"]["match"]["id"] == src.gauges[0]["id"] and c["ADHI_1"]["suggest"] == str(src.gauges[0]["id"])
    assert c["ADHI_2"]["match"] is None and c["ADHI_2"]["suggest"] == "new" and c["ADHI_2"]["cell"] == new_cell
    assert c["ADHI_2"]["name"] == "Newtown" and c["ADHI_2"]["river"] == "New River"
    assert c["ADHI_1"]["months"] == len(s1) and c["ADHI_2"]["months"] == len(s2)
    n0 = len(src.gauges)
    out = adhi.add(src, {"ADHI_1": c["ADHI_1"]["suggest"], "ADHI_2": "new"})
    assert len(out["added"]) == 1 and len(out["merged"]) == 1 and len(src.gauges) == n0 + 1
    g0 = next(i for i, g in enumerate(src.gauges) if g["id"] == 1)
    assert np.isfinite(src.qobs[g0]).sum() == len(s1)
    ng = next(i for i, g in enumerate(src.gauges) if g["name"] == "Newtown")
    assert src.gauges[ng]["cell"] == new_cell and np.isfinite(src.qobs[ng]).sum() == len(s2)
    assert "ADHI" in src.meta["discharge"]
    m = VoltaModel(src)                                           # the new gauge defines its own zone
    assert any(z["id"] == src.gauges[ng]["id"] and z["cells"] > 0 for z in m.zones_info())
    with pytest.raises(ValueError):
        adhi.add(src, {"ADHI_2": "new"})                          # same cell twice
    again = {x["adhi"]: x for x in adhi.candidates(src)}
    assert all(x["already"] and x["suggest"] == "skip" for x in again.values())
    assert adhi.add(src, {"ADHI_1": str(src.gauges[g0]["id"])}) == {"added": [], "merged": []}


def test_adhi_api(setup, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    src, truth, s1, s2, _ = setup
    data_dir = tmp_path / "data"
    src.save(data_dir / "basin.npz")
    monkeypatch.setattr(appmod, "DATA_DIR", data_dir)
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "basin.npz"})
        r = c.post("/api/calibrate", json={})
        assert r.status_code == 400 and "Nothing to calibrate" in r.json()["detail"]
        c.post("/api/model", json={"model": "nam"})
        c.post("/api/run", json={"params": {"model": "nam", "global": {"lmax": 420}}})
        r = c.get("/api/adhi/candidates").json()
        assert len(r["candidates"]) == 2 and "CC BY 4.0" in r["citation"]
        choices = {x["adhi"]: x["suggest"] for x in r["candidates"]}
        r = c.post("/api/adhi/add", json={"choices": choices})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "Newtown" in body["message"] and body["summary"]["params"]["model"] == "nam"
        assert body["summary"]["params"]["global"]["lmax"] == 420                # global values kept
        gs = body["summary"]["gauges"]
        assert sum(g["metrics"]["all"]["n"] > 0 for g in gs) == 2
        assert len(Dataset.load(data_dir / "basin.npz").gauges) == len(src.gauges) + 1
        assert c.post("/api/adhi/add", json={"choices": {}}).status_code == 400


def test_suggestion_rules_on_volta_like_cases():
    """Real cases from the Volta: name match across differing areas, duplicate records, reservoir volumes,
    short records, nested near-twins (same river) vs similar areas on different rivers."""
    from prep import adhi

    class D:
        pass
    d = D()
    # flow paths: 0 <- 1 <- 2 (river A, Black-Volta-like) ; 0 <- 3 <- 4 (river B) ; 5 -> 3 ; 6 -> 3
    d.ds = np.array([-1, 0, 1, 0, 3, 3, 3])
    d.gauges = [{"id": 1, "name": "Bamboi", "row": 50, "col": 50, "cell": 1, "area_km2": 133000},
                {"id": 2, "name": "Nawuni", "row": 20, "col": 90, "cell": 4, "area_km2": 92900}]
    d.qobs = np.full((2, 3), np.nan)
    def c(aid, name, cell, area, months, row, col, label=None, flag=0):
        return {"adhi": aid, "label": label or f"RIVER a {name.upper()}", "name": name, "cell": cell, "row": row, "col": col,
                "area_model_km2": area, "area_km2": area, "months": months, "flag": flag, "q": np.ones(3)}
    cands = [
        c("A1", "Bui Amont Dam", 1, 133300, 204, 50, 50, label="VOLTA NOIRE a BUI AMONT DAM"),
        c("A2", "Nawuni", 5, 98800, 271, 23, 84),                      # 7 cells away, 6 % more area: same name
        c("A3", "Nawuni", 5, 98900, 271, 23, 85),                      # the same station from a second source
        c("A4", "Bagre volume en Mm3", 3, 36000, 245, 30, 95, label="NAKAMBE au Barrage de BAGRE volume en MM3"),
        c("A5", "Vonkoro", 2, 122000, 235, 60, 40),                    # upstream of Bamboi, 9 % less area
        c("A6", "Pwalagu", 6, 140000, 118, 10, 80),                    # other river, similar area to Bamboi
        c("A7", "Daboya", 3, 100500, 82, 25, 80),                      # under 7 years
    ]
    adhi._suggest(d, cands, radius=5, basin_area=404000, ds=d.ds)
    s = {x["adhi"]: x for x in cands}
    assert s["A1"]["suggest"] == "1" and not s["A1"]["reservoir"]
    assert s["A2"]["suggest"] == "2" or s["A3"]["suggest"] == "2"
    assert {s["A2"]["suggest"], s["A3"]["suggest"]} == {"2", "skip"}
    assert s["A4"]["reservoir"] and s["A4"]["suggest"] == "skip"
    assert s["A5"]["suggest"] == "skip" and "Bamboi" in " ".join(s["A5"]["notes"])
    assert s["A6"]["suggest"] == "new"                                 # not nested with Bamboi
    assert s["A7"]["suggest"] == "skip" and "short record" in " ".join(s["A7"]["notes"])
