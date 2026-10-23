import numpy as np
import pytest

from hydro import abcd as abcd_mod
from hydro.abcd import run_abcd, zone_runoff_nodes, climatology
from hydro.calibrate import ZoneProblem, calibrate
from hydro.metrics import kge, nse, pbias
from hydro.model import VoltaModel, default_params
from hydro.pet import extraterrestrial_radiation, modified_hargreaves
from hydro.routing import Router, linear_reservoir, priority_flood_d8, upstream_of, compact_ds
from hydro.synthetic import TRUE_PARAMS, make_synthetic


@pytest.fixture(scope="session")
def small():
    return make_synthetic(res=0.1, start="1995-01", end="2004-12", log=lambda *a: None)


# ---------------------------------------------------------------- PET
def test_ra_fao56_example():
    # FAO-56 Example 8: 20 deg S on 3 September (J = 246) -> Ra = 32.2 MJ m-2 day-1
    assert extraterrestrial_radiation(-20.0, 246) == pytest.approx(32.2, abs=0.1)


def test_modified_hargreaves_hand_calc():
    tmax, tmin, p, ra = np.array([[35.0]]), np.array([[22.0]]), np.array([[100.0]]), np.array([[38.0]])
    expected_day = 0.0013 * 0.408 * 38.0 * (28.5 + 17.0) * (13.0 - 1.23) ** 0.76
    out = modified_hargreaves(tmax, tmin, p, ra, np.array([30]))
    assert out[0, 0] == pytest.approx(expected_day * 30, rel=1e-9)


def test_modified_hargreaves_rain_term_clipped():
    out = modified_hargreaves(np.array([[30.0]]), np.array([[25.0]]), np.array([[800.0]]), np.array([[38.0]]), np.array([31]))
    assert out[0, 0] == 0.0


# ---------------------------------------------------------------- abcd
def test_abcd_mass_balance_and_numpy_numba_agree(monkeypatch):
    rng = np.random.default_rng(0)
    P = rng.gamma(0.8, 80, size=(240, 50)); E = rng.uniform(80, 200, size=(240, 50))
    a, b, c, d = 0.98, rng.uniform(200, 1200, 50), 0.4, 0.1
    r = run_abcd(P, E, a, b, c, d, dtype=np.float64)
    dS = r["soil"][-1] - r["s0"]; dG = r["gw"][-1] - r["g0"]
    resid = P.sum(0) - r["et"].sum(0) - r["runoff"].sum(0) - dS - dG
    assert np.abs(resid).max() < 1e-6
    assert (r["et"] <= E + 1e-9).all() and (r["runoff"] >= -1e-12).all()
    monkeypatch.setattr(abcd_mod, "HAVE_NUMBA", False)
    r2 = run_abcd(P, E, a, b, c, d, dtype=np.float64)
    assert np.allclose(r["runoff"], r2["runoff"], rtol=1e-9, atol=1e-9)


def test_lean_kernel_matches_full():
    rng = np.random.default_rng(1)
    P = rng.gamma(0.8, 80, size=(120, 30)); E = rng.uniform(80, 200, size=(120, 30))
    r = run_abcd(P, E, 0.99, 600, 0.3, 0.2, dtype=np.float64)
    w = rng.uniform(1, 2, 30); node = rng.integers(0, 3, 30)
    out = zone_runoff_nodes(np.ascontiguousarray(P), np.ascontiguousarray(E), climatology(P), climatology(E),
                            0.99, 600, 0.3, 0.2, w, node, 3)
    ref = np.zeros((3, 120))
    np.add.at(ref, node, (r["runoff"] * w).T)
    assert np.allclose(out, ref, rtol=1e-9)


# ---------------------------------------------------------------- routing
def test_priority_flood_drains_everything_to_outlet():
    rng = np.random.default_rng(2)
    dem = rng.uniform(0, 100, (30, 40))
    valid = np.ones_like(dem, bool)
    ds, filled = priority_flood_d8(dem, valid, seeds=[0])
    assert upstream_of(ds, 0).all()                 # single outlet, every cell reaches it
    assert (ds >= 0).sum() == dem.size - 1
    r = Router(compact_ds(ds, valid.ravel()))
    acc, _ = r.accumulate(np.ones(dem.size))
    assert acc.max() == pytest.approx(dem.size)


def test_lake_clipping_and_deficit():
    # chain 0 -> 1 -> 2(outlet); cell 1 is a lake with net evaporation larger than inflow
    r = Router(np.array([1, 2, -1]), lake=np.array([False, True, False]))
    Q, deficit = r.accumulate(np.array([[5.0], [-8.0], [2.0]]))
    assert Q[1, 0] == 0 and deficit[1, 0] == pytest.approx(3.0) and Q[2, 0] == pytest.approx(2.0)


def test_linear_reservoir_conserves_volume():
    q = np.zeros(400); q[10] = 100.0
    y = linear_reservoir(q, 2.0)
    assert y.sum() == pytest.approx(100.0, rel=1e-6)
    assert linear_reservoir(q, 0).sum() == 100.0


# ---------------------------------------------------------------- metrics
def test_metrics():
    o = np.array([1.0, 2, 3, 4, np.nan]); assert nse(o, o) == 1 and kge(o, o) == pytest.approx(1)
    assert pbias(o * 1.1, o) == pytest.approx(10)


# ---------------------------------------------------------------- model
def test_model_water_balance_closes(small):
    m = VoltaModel(small)
    res = m.run(default_params())
    b = res.balance
    assert abs(b["residual"]) < 1e-6 * b["P"]
    assert 0 < b["runoff_ratio"] < 0.5
    assert 1200 < b["per_year"]["PET"] < 2600          # Modified Hargreaves, Volta climate
    assert (res.fields["discharge"] >= 0).all()
    zones = {z["id"] for z in m.zones_info()}
    assert zones == {0, 1, 2, 3, 4}


def test_zone_problem_equals_full_run(small):
    m = VoltaModel(small)
    full = m.run(TRUE_PARAMS)
    for gi, g in enumerate(small.gauges):
        inflow = {j: full.gauge_raw[j] for j in m.upstream_gauges[g["id"]]}
        sim = ZoneProblem(m, gi, TRUE_PARAMS, inflow).simulate(TRUE_PARAMS)
        assert np.allclose(sim, full.gauge_sim[gi], rtol=1e-7, atol=1e-6)


def test_calibration_improves_fit(small):
    m = VoltaModel(small)
    before = m.run(default_params())
    out = calibrate(m, default_params(), maxiter=6, popsize=6, gauges=[small.gauges[2]["id"]])
    after = m.run(out["params"])
    i = 2
    assert kge(after.gauge_sim[i], small.qobs[i]) > kge(before.gauge_sim[i], small.qobs[i])
    assert kge(after.gauge_sim[i], small.qobs[i]) > 0.8
