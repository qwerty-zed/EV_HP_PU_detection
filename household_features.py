#!/usr/bin/env python3
"""
household_features.py
=====================
The 20 household-level heat-pump features, computed from half-hourly consumption and
weather. Used by create_hp_dataset_v6.py.

Conventions:
- hh_index is 1-based (1..48; slot i covers [(i-1)/2, i/2) hours)
- HDD base 15.5C on daily mean temperature
- winter = {12,1,2}, summer = {6,7,8}
- 'temperature' is sub-daily (per half-hour slot), from the weather join

Entry point
-----------
features_from_long(daily, hh) -> 20-feature DataFrame indexed by household_id
"""
import numpy as np
import pandas as pd

WINTER = {12, 1, 2}
SUMMER = {6, 7, 8}

FEATURE_COLS = [
    "r_hdd", "beta_temp", "delta_cold", "r_ws", "base_summer", "f_zero",
    "i_ampk", "i_pmpk", "r_night", "sigma2_delta", "s_winter", "rho_48",
    "d_midday", "cov_t_x", "base_night", "avg_daily_kwh", "cv_daily_kwh",
    "winter_avg_kwh", "summer_avg_kwh", "consumption_per_hdd",
]


# ── vectorised per-group statistics (product-column versions of the originals) ──
def _pearson(df, x, y, grp):
    sub = df[[grp, x, y]].dropna().copy()
    sub["_xy"] = sub[x] * sub[y]
    sub["_x2"] = sub[x] ** 2
    sub["_y2"] = sub[y] ** 2
    g = sub.groupby(grp, observed=True)
    n = g[x].count(); sx = g[x].sum(); sy = g[y].sum()
    sxy = g["_xy"].sum(); sx2 = g["_x2"].sum(); sy2 = g["_y2"].sum()
    num = n * sxy - sx * sy
    den = np.sqrt((n * sx2 - sx ** 2) * (n * sy2 - sy ** 2))
    return num / den.replace(0, np.nan)


def _slope(df, y, x, grp):
    sub = df[[grp, x, y]].dropna().copy()
    sub["_xy"] = sub[x] * sub[y]
    sub["_x2"] = sub[x] ** 2
    g = sub.groupby(grp, observed=True)
    n = g[x].count(); sx = g[x].sum(); sy = g[y].sum()
    num = n * g["_xy"].sum() - sx * sy
    den = n * g["_x2"].sum() - sx ** 2
    return num / den.replace(0, np.nan)


def _cov(df, x, y, grp):
    sub = df[[grp, x, y]].dropna().copy()
    sub["_xy"] = sub[x] * sub[y]
    g = sub.groupby(grp, observed=True)
    n = g[x].count()
    return g["_xy"].sum() / n - (g[x].sum() / n) * (g[y].sum() / n)


def features_from_long(daily: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    """
    daily: household_id, date, daily_kwh, mean_temp, month
    hh   : household_id, date, hh_index (1-based), kwh, temperature, month
    """
    hid = "household_id"
    daily = daily.copy()
    daily["HDD"] = np.maximum(0, 15.5 - daily["mean_temp"].fillna(15.5))
    daily["is_winter"] = daily["month"].isin(WINTER)
    daily["is_summer"] = daily["month"].isin(SUMMER)
    hh = hh.copy()
    hh["is_winter"] = hh["month"].isin(WINTER)
    hh["is_summer"] = hh["month"].isin(SUMMER)

    f1 = _pearson(daily, "daily_kwh", "HDD", hid).rename("r_hdd")

    cold = daily[daily["mean_temp"] < 15]
    f2 = _slope(cold, "daily_kwh", "mean_temp", hid).rename("beta_temp")

    winter_d = daily[daily["is_winter"]].copy()
    w_mean = winter_d.groupby(hid, observed=True)["daily_kwh"].mean()
    winter_d["temp_rank_pct"] = winter_d.groupby(hid, observed=True)["mean_temp"]\
                                        .rank(pct=True)
    cold5 = winter_d[winter_d["temp_rank_pct"] <= 0.05]
    f3 = (cold5.groupby(hid, observed=True)["daily_kwh"].mean() - w_mean)\
        .rename("delta_cold")

    winter_avg = winter_d.groupby(hid, observed=True)["daily_kwh"].mean()
    summer_avg = daily[daily["is_summer"]].groupby(hid, observed=True)["daily_kwh"].mean()
    f4 = (winter_avg / summer_avg.replace(0, np.nan)).rename("r_ws")

    summer_mid = hh[hh["is_summer"] & hh["hh_index"].between(20, 32)]
    f5 = summer_mid.groupby(hid, observed=True)["kwh"].quantile(0.05)\
        .rename("base_summer")

    n_total = hh.groupby(hid, observed=True)["kwh"].count()
    n_zero = hh[hh["kwh"] <= 0.005].groupby(hid, observed=True)["kwh"].count()
    f6 = (n_zero / n_total).fillna(0).rename("f_zero")

    w_hh = hh[hh["is_winter"]]
    w_total = w_hh.groupby(hid, observed=True)["kwh"].sum()
    am = w_hh[w_hh["hh_index"].between(13, 18)].groupby(hid, observed=True)["kwh"].sum()
    f7 = (am / w_total.replace(0, np.nan)).rename("i_ampk")
    pm = w_hh[w_hh["hh_index"].between(33, 40)].groupby(hid, observed=True)["kwh"].sum()
    f8 = (pm / w_total.replace(0, np.nan)).rename("i_pmpk")

    night_sum = hh[hh["hh_index"].between(1, 14)].groupby(hid, observed=True)["kwh"].sum()
    all_sum = hh.groupby(hid, observed=True)["kwh"].sum()
    f9 = (night_sum / all_sum.replace(0, np.nan)).rename("r_night")

    w_sorted = w_hh.sort_values([hid, "date", "hh_index"]).copy()
    w_sorted["kwh_diff"] = w_sorted.groupby(hid, observed=True)["kwh"].diff()
    f10 = w_sorted.groupby(hid, observed=True)["kwh_diff"].var().rename("sigma2_delta")

    w_p98 = w_hh.groupby(hid, observed=True)["kwh"].quantile(0.98)
    w_p50 = w_hh.groupby(hid, observed=True)["kwh"].quantile(0.50)
    f11 = (w_p98 / (w_p50 + 0.001)).rename("s_winter")

    hh_sorted = hh.sort_values([hid, "date", "hh_index"]).copy()
    hh_sorted["kwh_lag48"] = hh_sorted.groupby(hid, observed=True)["kwh"].shift(48)
    f12 = _pearson(hh_sorted.dropna(subset=["kwh_lag48"]),
                   "kwh", "kwh_lag48", hid).rename("rho_48")

    s_hh = hh[hh["is_summer"]]
    edge = s_hh[s_hh["hh_index"].isin([16, 42])].groupby(hid, observed=True)["kwh"].mean()
    mid = s_hh[s_hh["hh_index"].between(24, 28)].groupby(hid, observed=True)["kwh"].mean()
    f13 = (edge - mid).rename("d_midday")

    w_hh_temp = w_hh.dropna(subset=["temperature"])
    f14 = (_cov(w_hh_temp, "kwh", "temperature", hid).rename("cov_t_x")
           if len(w_hh_temp) else pd.Series(dtype=float, name="cov_t_x"))

    night_slots = hh[hh["hh_index"].between(2, 10)]
    w_p10 = night_slots[night_slots["is_winter"]].groupby(hid, observed=True)["kwh"]\
        .quantile(0.10)
    s_p10 = night_slots[night_slots["is_summer"]].groupby(hid, observed=True)["kwh"]\
        .quantile(0.10)
    f15 = (w_p10 - s_p10).rename("base_night")

    a1 = daily.groupby(hid, observed=True)["daily_kwh"].mean().rename("avg_daily_kwh")
    dstd = daily.groupby(hid, observed=True)["daily_kwh"].std()
    a2 = (dstd / a1.replace(0, np.nan)).rename("cv_daily_kwh")
    a3 = winter_avg.rename("winter_avg_kwh")
    a4 = summer_avg.rename("summer_avg_kwh")
    hdd_pos = daily[daily["HDD"] > 0]
    a5 = (hdd_pos.groupby(hid, observed=True)["daily_kwh"].sum()
          / hdd_pos.groupby(hid, observed=True)["HDD"].sum().replace(0, np.nan))\
        .rename("consumption_per_hdd")

    out = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                     f11, f12, f13, f14, f15, a1, a2, a3, a4, a5], axis=1)
    out.index.name = hid
    return out
