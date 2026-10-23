"""Potential evapotranspiration.

Modified Hargreaves (Droogers & Allen, 2002, Irrigation and Drainage Systems 16:33-45):

    ETo = 0.0013 * 0.408 * Ra * (Tavg + 17.0) * (TD - 0.0123 * P) ** 0.76     [mm/day]

    Ra   extraterrestrial radiation [MJ m-2 day-1] (FAO-56 eq. 21, mid-month day)
    Tavg (Tmax + Tmin) / 2 [degC]
    TD   Tmax - Tmin, monthly mean diurnal range [degC]
    P    monthly precipitation [mm/month]
    0.408 converts MJ m-2 day-1 to mm day-1 of evaporation.

The rainfall term corrects the over-estimation of the original Hargreaves equation
in humid months, when cloud cover reduces the diurnal temperature range less than
it reduces radiation. If TD - 0.0123 P is negative it is clipped to zero.
"""
from __future__ import annotations

import numpy as np

GSC = 0.0820  # solar constant, MJ m-2 min-1
# Day of year of the 15th of each month (non-leap; error from leap years is negligible)
MID_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def extraterrestrial_radiation(lat_deg: np.ndarray, doy: np.ndarray | int) -> np.ndarray:
    """FAO-56 eq. 21. lat_deg broadcastable with doy. Returns MJ m-2 day-1."""
    phi = np.deg2rad(lat_deg)
    j = np.asarray(doy, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    return (24.0 * 60.0 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def monthly_ra(lat_deg: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Ra for each (month, cell). months: (T,) 1..12; lat_deg: (n,). Returns (T, n)."""
    table = extraterrestrial_radiation(lat_deg[None, :], MID_MONTH_DOY[:, None])  # (12, n)
    return table[np.asarray(months) - 1]


def modified_hargreaves(tmax, tmin, precip_mm_month, ra, ndays):
    """Monthly reference ET [mm/month].

    tmax, tmin, precip_mm_month, ra: arrays (T, n); ndays: (T,) days in month.
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    td = np.maximum(tmax - tmin, 0.0)
    tavg = 0.5 * (tmax + tmin)
    rain_term = np.maximum(td - 0.0123 * np.asarray(precip_mm_month, dtype=float), 0.0)
    eto_day = 0.0013 * 0.408 * ra * np.maximum(tavg + 17.0, 0.0) * rain_term ** 0.76
    return eto_day * np.asarray(ndays, dtype=float)[:, None]


def hargreaves_samani(tmax, tmin, ra, ndays):
    """Original Hargreaves-Samani (1985) for comparison [mm/month]."""
    td = np.maximum(np.asarray(tmax) - np.asarray(tmin), 0.0)
    tavg = 0.5 * (np.asarray(tmax) + np.asarray(tmin))
    eto_day = 0.0023 * 0.408 * ra * (tavg + 17.8) * np.sqrt(td)
    return eto_day * np.asarray(ndays, dtype=float)[:, None]
