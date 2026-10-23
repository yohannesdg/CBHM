"""Climate scenarios: rainfall and temperature changes applied to the forcing."""
import numpy as np
import pytest

from hydro.model import VoltaModel, default_params
from hydro.pet import modified_hargreaves
from hydro.synthetic import make_synthetic


@pytest.fixture(scope="module")
def small():
    return make_synthetic(res=0.1, start="1995-01", end="2004-12", log=lambda *a: None)


def test_with_climate_changes_forcing_only(small):
    m = VoltaModel(small)
    sc = m.with_climate(rain_pct=-10, temp_c=2, pet_pct=5)
    assert np.allclose(sc.precip, 0.9 * m.precip)
    want = modified_hargreaves(small.tmax + 2, small.tmin + 2, 0.9 * m.precip, m.ra, m.ndays) * 1.05
    assert np.allclose(sc.eto, want)
    assert np.allclose(m.precip, small.precip)                       # the baseline model is untouched
    assert sc.router is m.router and sc.zone is m.zone
    base, drier = m.run(default_params()), sc.run(default_params())
    assert abs(drier.balance["residual"]) < 1e-6 * drier.balance["P"]
    assert drier.balance["Q"] < base.balance["Q"] and drier.balance["per_year"]["PET"] > base.balance["per_year"]["PET"]
    same = m.with_climate().run(default_params())
    assert abs(same.balance["Q"] / base.balance["Q"] - 1) < 1e-4


@pytest.mark.parametrize("model", ["abcd", "nam"])
def test_scenario_api(small, tmp_path, monkeypatch, model):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    small.save(tmp_path / "small.npz")
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "small.npz"})
        c.post("/api/model", json={"model": model})
        r = c.post("/api/scenario", json={"rain_pct": -10}).json()
        assert r["model"] == model and r["change_pct"]["P"] == pytest.approx(-10, abs=1e-6)
        assert r["change_pct"]["Q"] < -10 and r["elasticity"] > 1          # runoff amplifies rainfall change
        assert len(r["scenario"]["seasonal_m3s"]) == 12 and len(r["scenario"]["gauges"]) == len(small.gauges)
        assert r["scenario"]["outlet_mean_m3s"] < r["baseline"]["outlet_mean_m3s"]
        w = c.post("/api/scenario", json={"temp_c": 2}).json()
        assert w["change_pct"]["PET"] > 2 and w["change_pct"]["Q"] < 0 and w["elasticity"] is None
        txt = c.get("/api/export/scenario.csv")
        assert txt.status_code == 200 and txt.text.startswith("date,outlet_baseline_m3s,outlet_scenario_m3s")
        assert c.post("/api/scenario", json={"rain_pct": 500}).status_code == 400
        res = c.get("/api/results").json()                                  # the baseline results are unchanged
        assert res["balance"]["P"] == pytest.approx(c.post("/api/run", json={}).json()["balance"]["P"])
