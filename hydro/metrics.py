"""Goodness-of-fit metrics. NaNs in obs (or sim) are ignored pairwise."""
from __future__ import annotations

import numpy as np


def _pair(sim, obs):
    sim = np.asarray(sim, float)
    obs = np.asarray(obs, float)
    m = np.isfinite(sim) & np.isfinite(obs)
    return sim[m], obs[m]


def nse(sim, obs):
    s, o = _pair(sim, obs)
    if o.size < 3 or np.var(o) == 0:
        return np.nan
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def log_nse(sim, obs, eps=None):
    s, o = _pair(sim, obs)
    if o.size < 3:
        return np.nan
    eps = eps if eps is not None else 0.01 * max(o.mean(), 1e-9)
    return nse(np.log(s + eps), np.log(o + eps))


def kge(sim, obs):
    """Kling-Gupta efficiency (Gupta et al., 2009)."""
    s, o = _pair(sim, obs)
    if o.size < 3 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else 0.0
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(sim, obs):
    s, o = _pair(sim, obs)
    if o.size == 0 or o.sum() == 0:
        return np.nan
    return 100.0 * (s.sum() - o.sum()) / o.sum()


def rmse(sim, obs):
    s, o = _pair(sim, obs)
    return float(np.sqrt(np.mean((s - o) ** 2))) if o.size else np.nan


def r2(sim, obs):
    s, o = _pair(sim, obs)
    if o.size < 3 or s.std() == 0:
        return np.nan
    return float(np.corrcoef(s, o)[0, 1] ** 2)


def all_metrics(sim, obs) -> dict:
    s, o = _pair(sim, obs)
    out = {"n": int(o.size), "nse": nse(sim, obs), "kge": kge(sim, obs),
           "log_nse": log_nse(sim, obs), "pbias": pbias(sim, obs),
           "rmse": rmse(sim, obs), "r2": r2(sim, obs)}
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else float(v))
            for k, v in out.items()}
