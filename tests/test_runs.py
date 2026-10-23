"""Saved runs: save, list, recompute for comparison, apply, rename, delete."""
import pytest

from hydro.synthetic import make_synthetic


@pytest.fixture(scope="module")
def small():
    return make_synthetic(res=0.1, start="1995-01", end="2004-12", log=lambda *a: None)


def test_saved_runs_roundtrip(tmp_path, monkeypatch, small):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    small.save(tmp_path / "small.npz")
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "small.npz"})
        c.post("/api/model", json={"model": "abcd"})
        assert c.get("/api/runs").json()["runs"] == []
        c.post("/api/run", json={"params": {"model": "abcd", "global": {"b": 900}}})
        a = c.post("/api/runs", json={"label": "wet soils"}).json()
        assert a["label"] == "wet soils" and a["model"] == "abcd" and a["params"]["global"]["b"] == 900
        assert a["summary"]["gauges_scored"] == len(small.gauges)
        dry = c.post("/api/runs", json={"label": "", "climate": {"rain_pct": -10}}).json()
        assert dry["label"] == "Run 2" and dry["climate"]["rain_pct"] == -10 and dry["climate_label"] == "rain -10%"
        assert dry["summary"]["Q"] < a["summary"]["Q"]
        assert (tmp_path / "runs" / "small.json").exists()

        # change the current parameters; the saved run still gives the old flows
        cur = c.post("/api/run", json={"params": {"model": "abcd", "global": {"b": 300}}}).json()
        s = c.get(f"/api/runs/{a['id']}").json()
        assert s["params"]["global"]["b"] == 900 and len(s["outlet_q"]) == len(small.dates)
        assert len(s["gauges"]) == len(small.gauges)
        assert c.get("/api/results").json()["params"]["global"]["b"] == 300          # current results untouched
        assert s["outlet_q"] != cur["basin"]["outlet_q"]
        sd = c.get(f"/api/runs/{dry['id']}").json()
        assert "rain -10%" in sd["label"] and sum(sd["outlet_q"]) < sum(s["outlet_q"])

        # a NAM run saved and applied switches the model
        c.post("/api/model", json={"model": "nam"})
        n = c.post("/api/runs", json={"label": "nam default"}).json()
        c.post("/api/model", json={"model": "abcd"})
        ap = c.post(f"/api/runs/{n['id']}/apply").json()
        assert ap["model"] == "nam" and "lmax" in ap["param_spec"]
        assert c.get("/api/meta").json()["model"] == "nam"

        assert c.patch(f"/api/runs/{a['id']}", json={"label": "renamed"}).json()["label"] == "renamed"
        assert c.delete(f"/api/runs/{dry['id']}").status_code == 200
        assert [r["id"] for r in c.get("/api/runs").json()["runs"]] == [a["id"], n["id"]]
        assert c.get("/api/runs/nope").status_code == 404
        assert c.delete("/api/runs/nope").status_code == 404
