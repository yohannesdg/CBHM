"""The model choice: monthly NAM, the model registry, switching and comparing through the API."""
import numpy as np
import pytest

from hydro import nam
from hydro.calibrate import ZoneProblem, calibrate
from hydro.metrics import kge
from hydro.model import VoltaModel, default_params, merge_params
from hydro.models import MODELS
from hydro.synthetic import make_synthetic


@pytest.fixture(scope="module")
def small():
    return make_synthetic(res=0.1, start="1995-01", end="2004-12", log=lambda *a: None)


def _forcing(T=240, n=60, seed=1):
    rng = np.random.default_rng(seed)
    season = np.clip(np.sin(np.arange(T) / 12 * 2 * np.pi - 1.2), 0, None)
    P = season[:, None] * rng.gamma(2, 90, (T, n))
    E = 150 + 30 * np.cos(np.arange(T) / 12 * 2 * np.pi)[:, None] + np.zeros((T, n))
    return P, E


def test_nam_closes_and_numba_numpy_agree():
    P, E = _forcing()
    prm = {"umax": 12, "lmax": np.linspace(50, 900, P.shape[1]), "cqof": 0.6, "ckif": 0.7, "tof": 0.2,
           "tif": 0.4, "tg": 0.1, "ckbf": 4}
    a = nam.run_nam(P, E, prm, dtype=float, use_numba=False)
    res = P.sum(0) - a["et"].sum(0) - a["runoff"].sum(0) - (a["soil"][-1] - a["s0"]) - (a["gw"][-1] - a["g0"])
    assert np.abs(res).max() < 1e-8
    assert (a["soil"] >= -1e-12).all() and (a["et"] <= E + 1e-9).all()
    assert np.allclose(a["runoff"], a["direct"] + a["baseflow"])
    if nam.HAVE_NUMBA:
        b = nam.run_nam(P, E, prm, dtype=float)
        for k in nam.OUTPUTS:
            assert np.allclose(a[k], b[k], atol=1e-9)


def test_nam_responds_like_nam():
    """Bigger root zone -> less runoff; higher CQOF -> more quickflow; longer CKBF -> smoother baseflow."""
    P, E = _forcing(n=1)
    base = {"umax": 15, "lmax": 300, "cqof": 0.4, "ckif": 1.0, "tof": 0.3, "tif": 0.3, "tg": 0.2, "ckbf": 6}
    run = lambda **kw: nam.run_nam(P, E, {**base, **kw}, dtype=float, use_numba=False)
    r0 = run()
    assert run(lmax=900)["runoff"].sum() < r0["runoff"].sum()
    assert run(cqof=0.9)["direct"].sum() > r0["direct"].sum()
    assert run(ckbf=30)["baseflow"].std() < r0["baseflow"].std()


def test_merge_params_keeps_each_model_to_its_own_keys():
    p = merge_params({"model": "nam", "global": {"lmax": 500, "a": 0.9}, "zones": {"2": {"b": 900}}})
    assert p["model"] == "nam" and p["global"]["lmax"] == 500 and "a" not in p["global"] and p["zones"] == {}
    assert set(p["global"]) == set(MODELS["nam"].spec)
    assert merge_params(None)["model"] == "abcd"
    with pytest.raises(KeyError):
        default_params("hbv")


def test_nam_distributed_balance_and_zone_problem(small):
    m = VoltaModel(small)
    p = default_params("nam")
    p["zones"]["2"] = {"lmax": 450, "cqof": 0.6}
    full = m.run(p)
    b = full.balance
    assert abs(b["residual"]) < 1e-6 * b["P"] and 0 < b["runoff_ratio"] < 0.6
    for gi, g in enumerate(small.gauges):
        inflow = {j: full.gauge_raw[j] for j in m.upstream_gauges[g["id"]]}
        sim = ZoneProblem(m, gi, p, inflow).simulate(p)
        assert np.allclose(sim, full.gauge_sim[gi], rtol=1e-6, atol=1e-5)


def test_nam_calibration_improves_fit(small):
    m = VoltaModel(small)
    p = default_params("nam")
    i = 2
    before = kge(m.run(p).gauge_sim[i], small.qobs[i])
    out = calibrate(m, p, maxiter=5, popsize=5, gauges=[small.gauges[i]["id"]])
    assert out["params"]["model"] == "nam"
    assert set(out["params"]["zones"][str(small.gauges[i]["id"])]) == set(MODELS["nam"].calibrate_default)
    assert kge(m.run(out["params"]).gauge_sim[i], small.qobs[i]) > before
    with pytest.raises(ValueError):
        calibrate(m, p, names=("a",), maxiter=1, popsize=2)


def test_api_switch_and_compare(tmp_path, monkeypatch, small):
    from fastapi.testclient import TestClient
    import backend.app as appmod
    small.save(tmp_path / "small.npz")
    monkeypatch.setattr(appmod, "DATA_DIR", tmp_path)
    with TestClient(appmod.app) as c:
        c.post("/api/datasets/load", json={"name": "small.npz"})
        c.post("/api/model", json={"model": "abcd"})          # the chosen model carries over between datasets
        meta = c.get("/api/meta").json()
        assert meta["model"] == "abcd" and {m["key"] for m in meta["models"]} == {"abcd", "nam"}
        r = c.post("/api/run", json={"params": {"model": "abcd", "global": {"b": 900}}}).json()
        assert r["params"]["global"]["b"] == 900
        sw = c.post("/api/model", json={"model": "nam"}).json()
        assert sw["model"] == "nam" and "lmax" in sw["param_spec"] and sw["summary"]["params"]["model"] == "nam"
        assert c.get("/api/meta").json()["model"] == "nam"
        cmp_ = c.get("/api/compare").json()
        assert cmp_["model"] == "abcd" and cmp_["params"]["global"]["b"] == 900        # abcd kept its values
        assert len(cmp_["gauges"]) == len(small.gauges) and len(cmp_["outlet_q"]) == len(small.dates)
        assert c.post("/api/calibrate", json={"names": ["a"]}).status_code == 400
        cell = c.get("/api/cell", params={"row": small.gauges[0]["row"], "col": small.gauges[0]["col"]}).json()
        assert cell["model"] == "nam" and "ckbf" in cell["params"]
        back = c.post("/api/model", json={"model": "abcd"}).json()
        assert back["summary"]["params"]["global"]["b"] == 900
        assert c.post("/api/model", json={"model": "hbv"}).status_code == 400
