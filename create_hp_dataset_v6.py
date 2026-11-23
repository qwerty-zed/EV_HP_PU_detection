#!/usr/bin/env python3
"""
create_hp_dataset_v6.py
=======================
Synthetic heat-pump dataset: Low Carbon London (LCL) household load + Electrification of
Heat (EoH) heat-pump load. The join rules are set out in notebook 04_hp_linkage_lcl_eoh.

Two sets with an identical schema:
  training  LCL arm N gas homes  + EoH E.ON and Warmworks heat pumps  -> LCL_EoH_join_v6/
  transfer  LCL arm D gas homes  + EoH OVO heat pumps                 -> LCL_EoH_join_v6/transfer_armD_ovo/

What changed from v3
--------------------
- LCL is the UK Data Service release (study 7857), read from the notebook 01 caches, with
  hosts chosen from the appliance survey (gas space AND water heating) instead of a meter
  ratio.
- EoH is read from the notebook 02 cache: nulls dropped before differencing, only 30-minute
  differences kept, each half-hour labelled by its START. v3's step4 labelled by the end,
  which put heat-pump load 30 minutes late.
- Join window is calendar 2013 only, for every household.
- Each EoH day is used at most MAX_DAY_USES times per household (nearest-temperature
  candidate below the cap; nearest regardless if all are capped). Uncapped, nearest-day
  selection put 54% of a household's dates on a day used three or more times.
- Heat pumps are paired house->house and flat->flat.
- EoH days with 44-47 of 48 slots are filled by linear interpolation, not zero.
- Temperature is one shared London series in GMT, fetched from Open-Meteo. The daily
  mean_temp is the London daily mean for the date, not an average over the household's own
  half-hours (adopted after the first build failed the leak audit).

Unchanged from v3: 1x pairing, month +-1 and weekday-class candidate window, a London date
with no candidate is dropped from that positive (logged), the 20 household features, and
the temperature-only leak audit gate.
"""
import json
import os
import sys
import tempfile
import time
import urllib.request

import duckdb
import numpy as np
import pandas as pd

RANDOM_SEED = 42
JOIN_YEAR = 2013
MAX_DAY_USES = 2
MONTH_WINDOW = 1
MIN_MONTHS = 6
MIN_EOH_SLOTS = 44
COMPLETE_DAY_SLOTS = 46
MIN_COMPLETE_DAYS_2013 = 200
HOLDOUT_CONTRACTOR = "OVO"
MAX_POSITIVES = None          # None = every usable heat pump
LEAK_AUDIT_MAX = 0.55
APPLIANCE_TOL = 0.04          # gate 3, +-4 pp
LONDON = (51.5072, -0.1276)

# Paths are relative to this script, which sits in the repository root: the study data are in
# Data/ (EN, EoH, LCL), the notebooks write their caches to cache/, and the outputs go to a
# folder created next to this script.
DATA = os.path.dirname(os.path.abspath(__file__))
NB_CACHE = f"{DATA}/cache"
EOH_META = f"{DATA}/Data/EoH/9bbf3b4f-5b33-4652-b385-d69ba68004d0.csv"
DAILY_TEMP_CHECK = f"{NB_CACHE}/london_daily_temp.csv"   # optional cross-check, skipped if absent
OUT = f"{DATA}/LCL_EoH_join_v6"
OUT_TRANSFER = f"{OUT}/transfer_armD_ovo"
WEATHER_CACHE = f"{OUT}/london_weather_2013.parquet"
os.makedirs(OUT_TRANSFER, exist_ok=True)   # creates OUT as well

from household_features import features_from_long, FEATURE_COLS   # in this folder


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='6GB'")
    con.execute(f"SET temp_directory='{tempfile.gettempdir()}/duckdb_spill'")
    return con


def is_weekend(dates):
    return (pd.to_datetime(dates).dayofweek >= 5).astype(int)


# ──────────────────────────────────────────────────────────────────────
def step1_weather(con):
    """One shared London temperature series for 2013, in GMT."""
    print("\nSTEP 1: London weather, 2013, GMT (shared by every household)")
    if not os.path.exists(WEATHER_CACHE):
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={LONDON[0]}&longitude={LONDON[1]}"
               f"&start_date={JOIN_YEAR}-01-01&end_date={JOIN_YEAR}-12-31"
               "&hourly=temperature_2m&timezone=GMT")
        with urllib.request.urlopen(url, timeout=120) as r:
            payload = json.loads(r.read())
        hourly = pd.DataFrame({"time": pd.to_datetime(payload["hourly"]["time"]),
                               "temp": payload["hourly"]["temperature_2m"]})
        hourly.to_parquet(WEATHER_CACHE, index=False)
        print(f"  fetched {len(hourly):,} hourly values from Open-Meteo")
    hourly = pd.read_parquet(WEATHER_CACHE)
    assert len(hourly) == 8760 and hourly.temp.notna().all(), "incomplete weather fetch"
    hourly["date"] = hourly.time.dt.normalize()
    hourly["hour"] = hourly.time.dt.hour
    daily = hourly.groupby("date", as_index=False)["temp"].mean().rename(
        columns={"temp": "mean_temp"})
    # each hourly value covers two half-hour slots: hour h -> hh_index 2h+1, 2h+2
    hh = pd.concat([hourly.assign(hh_index=2 * hourly.hour + 1),
                    hourly.assign(hh_index=2 * hourly.hour + 2)])
    hh = hh[["date", "hh_index", "temp"]].rename(columns={"temp": "temperature"})
    con.register("london_daily", daily)
    con.register("london_hh", hh)
    if os.path.exists(DAILY_TEMP_CHECK):
        chk = pd.read_csv(DAILY_TEMP_CHECK, parse_dates=["date"]).merge(daily, on="date")
        mad = float((chk.temp_mean - chk.mean_temp).abs().mean())
        print(f"  {len(daily)} days | mean {daily.mean_temp.mean():.2f} °C | "
              f"mean abs diff vs daily cache {mad:.3f} °C over {len(chk)} days")
    else:
        mad = None
        print(f"  {len(daily)} days | mean {daily.mean_temp.mean():.2f} °C | no daily cache to check against")
    return daily, mad


def step2_lcl(con):
    """Qualifying LCL households (both arms) and their 2013 half-hourly rows."""
    print(f"\nSTEP 2: LCL households qualifying in {JOIN_YEAR}")
    con.execute(f"""
        CREATE TEMP TABLE hh_master AS
        SELECT m.arm, m.household_id AS lcl_id, m.heat_group, m.dwelling_type,
               m.has_electric_shower, m.has_portable_heater, m.suspected_ev,
               coalesce(d.n_complete, 0) AS n_complete_2013
        FROM '{NB_CACHE}/lcl_household_master.parquet' m
        LEFT JOIN (SELECT household_id,
                          count(*) FILTER (WHERE n_slots >= {COMPLETE_DAY_SLOTS}) AS n_complete
                   FROM '{NB_CACHE}/household_day.parquet'
                   WHERE year(date) = {JOIN_YEAR} GROUP BY 1) d USING (household_id)
        WHERE m.passes_filters AND coalesce(d.n_complete, 0) >= {MIN_COMPLETE_DAYS_2013}
    """)
    con.execute(f"""
        CREATE TEMP TABLE lcl AS
        SELECT x.household_id AS lcl_id, x.date,
               (2 * hour(x.period_start) + (minute(x.period_start) >= 30)::INT + 1)::TINYINT
                   AS hh_index,
               x.kwh
        FROM (SELECT household_id, period_start, date, kwh FROM '{NB_CACHE}/n_ext.parquet'
              UNION ALL
              SELECT household_id, period_start, date, kwh FROM '{NB_CACHE}/d_ext.parquet') x
        WHERE year(x.date) = {JOIN_YEAR}
          AND x.household_id IN (SELECT lcl_id FROM hh_master)
    """)
    s = con.execute("""SELECT arm, count(*) AS households,
                              count(*) FILTER (WHERE heat_group = 'Gas') AS gas
                       FROM hh_master GROUP BY 1 ORDER BY 1""").df()
    n = con.execute("SELECT count(*) FROM lcl").fetchone()[0]
    print(s.to_string(index=False))
    print(f"  {n:,} half-hourly rows in {JOIN_YEAR}")
    return con.execute("SELECT * FROM hh_master").df()


def step3_eoh(con):
    """Usable EoH days and properties, from the notebook 02 cache."""
    print("\nSTEP 3: EoH usable days and properties (cleaned cache, period start)")
    con.execute(f"""
        CREATE TEMP TABLE eoh_day AS
        SELECT property_id AS pid, date AS eoh_date, month::INT AS month,
               (dayofweek(date) IN (0, 6))::INT AS is_weekend,
               ext_mean AS site_temp, n_slots
        FROM '{NB_CACHE}/eoh_property_day.parquet'
        WHERE n_slots >= {MIN_EOH_SLOTS} AND ext_mean IS NOT NULL
    """)
    con.execute(f"""
        DELETE FROM eoh_day WHERE pid NOT IN (
            SELECT pid FROM eoh_day GROUP BY pid HAVING count(DISTINCT month) >= {MIN_MONTHS})
    """)
    props = con.execute(f"""
        SELECT DISTINCT e.pid, m.Delivery_Contractor AS contractor,
               CASE WHEN nullif(m.House_Type, 'null') IN ('Flat', 'Maisonette')
                    THEN 'Flat' ELSE 'House' END AS stratum,
               coalesce(nullif(m.House_Type, 'null'), 'Not recorded') AS eoh_dwelling_type,
               coalesce(nullif(m.HP_Installed, 'null'), 'Not recorded') AS hp_type
        FROM eoh_day e
        JOIN read_csv_auto('{EOH_META}', all_varchar=true) m ON m.Property_ID = e.pid
        ORDER BY e.pid
    """).df()
    days = con.execute("""SELECT pid, eoh_date, month, is_weekend, site_temp, n_slots
                          FROM eoh_day ORDER BY pid, eoh_date""").df()
    print(props.groupby(["contractor", "stratum"]).size().to_string())
    print(f"  {len(props)} properties | {len(days):,} usable property-days | "
          f"{(days.n_slots < 48).mean():.1%} of days need interpolation")
    return props, days


def step4_pairing(props, hosts_master, rng):
    """1x pairing within each set, house->house and flat->flat."""
    print("\nSTEP 4: Pairing (1x, dwelling-matched)")
    sets = {}
    for name, arm, mask in [("training", "N", props.contractor != HOLDOUT_CONTRACTOR),
                            ("transfer", "D", props.contractor == HOLDOUT_CONTRACTOR)]:
        p = props[mask].copy()
        if MAX_POSITIVES is not None and len(p) > MAX_POSITIVES:
            p = p.iloc[np.sort(rng.choice(len(p), MAX_POSITIVES, replace=False))]
        eligible = hosts_master[(hosts_master.arm == arm)
                                & (hosts_master.heat_group == "Gas")
                                & hosts_master.dwelling_type.isin(["House", "Flat"])]
        parts = []
        for stratum, grp in p.groupby("stratum"):
            pool = eligible[eligible.dwelling_type == stratum].lcl_id.sort_values().values
            if len(grp) > len(pool):
                raise SystemExit(f"{name}: {len(grp)} {stratum} heat pumps but only "
                                 f"{len(pool)} {stratum} hosts")
            grp = grp.sort_values("pid").copy()
            grp["lcl_id"] = rng.choice(pool, len(grp), replace=False)
            parts.append(grp)
        pairs = pd.concat(parts).sort_values("pid").reset_index(drop=True)
        pairs["household_id"] = "HP_" + pairs.lcl_id + "_" + pairs.pid
        pairs = pairs.merge(eligible[["lcl_id", "dwelling_type"]], on="lcl_id")
        sets[name] = {"arm": arm, "pairs": pairs,
                      "eligible": eligible}
        print(f"  {name}: {len(pairs)} positives on arm {arm} | "
              + ", ".join(f"{k} {v}" for k, v in pairs.stratum.value_counts().items()))
    return sets


def step5_select_days(pairs, lcl_dates, eoh_days, daily, rng):
    """Capped nearest-temperature EoH day for each host date (greedy, random date order)."""
    temp = daily.set_index("date")["mean_temp"]
    by_pid = {pid: g.reset_index(drop=True) for pid, g in eoh_days.groupby("pid")}
    rows, stats = [], []
    for r in pairs.itertuples():
        g = by_pid[r.pid]
        em, ew, et = g.month.values, g.is_weekend.values, g.site_temp.values
        uses = np.zeros(len(g), int)
        dates = lcl_dates[r.lcl_id]
        n_fallback = n_dropped = 0
        for i in rng.permutation(len(dates)):
            d = dates[i]
            lm, lw, lt = d.month, int(d.dayofweek >= 5), temp[d]
            md = np.abs(em - lm)
            ok = (np.minimum(md, 12 - md) <= MONTH_WINDOW) & (ew == lw)
            if not ok.any():
                n_dropped += 1
                continue
            gap = np.where(ok, np.abs(et - lt), np.inf)
            capped = np.where(uses < MAX_DAY_USES, gap, np.inf)
            if np.isfinite(capped).any():
                gap = capped
            else:
                n_fallback += 1
            j = int(np.argmin(gap))          # days sorted by date: ties go to the earliest
            uses[j] += 1
            rows.append((r.household_id, d, r.pid, g.eoh_date.iat[j], float(gap[j])))
        stats.append((r.household_id, n_fallback, n_dropped, int(uses.max())))
    sel = pd.DataFrame(rows, columns=["household_id", "date", "pid", "eoh_date", "temp_gap"])
    sel["eoh_date"] = pd.to_datetime(sel.eoh_date)
    st = pd.DataFrame(stats, columns=["household_id", "n_fallback_cap",
                                      "n_dropped_dates", "max_day_uses"])
    return sel, st


def eoh_slots_for(con, sel):
    """48 slots for every selected EoH day; days with 44-47 slots linearly interpolated."""
    days = sel[["pid", "eoh_date"]].drop_duplicates()
    con.register("sel_days", days)
    hh = con.execute(f"""
        SELECT h.property_id AS pid, CAST(h.ts AS DATE) AS eoh_date,
               (2 * hour(h.ts) + (minute(h.ts) >= 30)::INT + 1)::INT AS hh_index, h.kwh
        FROM '{NB_CACHE}/eoh_halfhourly.parquet' h
        JOIN sel_days s ON s.pid = h.property_id AND s.eoh_date = CAST(h.ts AS DATE)
        WHERE h.kwh IS NOT NULL
    """).df()
    con.unregister("sel_days")
    wide = hh.pivot_table(index=["pid", "eoh_date"], columns="hh_index", values="kwh")
    wide = wide.reindex(columns=range(1, 49))
    n_interp = int(wide.isna().any(axis=1).sum())
    wide = wide.interpolate(axis=1, limit_direction="both")
    long = wide.stack().rename("eoh_kwh").reset_index()
    long["eoh_date"] = pd.to_datetime(long.eoh_date)
    return long, n_interp


def step6_build(con, name, s, sel, out):
    """Half-hourly, daily and wide tables for one set."""
    print(f"\nSTEP 6 [{name}]: half-hourly, daily and wide tables")
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    pairs = s["pairs"]
    slots, n_interp = eoh_slots_for(con, sel)
    con.register("pairs_t", pairs[["household_id", "lcl_id", "pid"]])
    con.register("sel_t", sel[["household_id", "date", "pid", "eoh_date"]])
    con.register("eoh_slots", slots)
    con.register("arm_hh", s["arm_households"][["lcl_id"]])
    con.execute(f"""
        COPY (
            SELECT p.household_id, l.date, l.hh_index,
                   (l.kwh + e.eoh_kwh)::FLOAT AS kwh,
                   w.temperature::FLOAT AS temperature, 1::TINYINT AS label
            FROM sel_t t
            JOIN pairs_t p   ON p.household_id = t.household_id
            JOIN lcl l       ON l.lcl_id = p.lcl_id AND l.date = t.date
            JOIN eoh_slots e ON e.pid = t.pid AND e.eoh_date = t.eoh_date
                            AND e.hh_index = l.hh_index
            JOIN london_hh w ON w.date = l.date AND w.hh_index = l.hh_index
            UNION ALL
            SELECT l.lcl_id, l.date, l.hh_index, l.kwh::FLOAT,
                   w.temperature::FLOAT, 0::TINYINT
            FROM lcl l
            JOIN london_hh w ON w.date = l.date AND w.hh_index = l.hh_index
            WHERE l.lcl_id IN (SELECT lcl_id FROM arm_hh)
              AND l.lcl_id NOT IN (SELECT lcl_id FROM pairs_t)
        ) TO '{out}/hp_dataset_hh30.parquet' (FORMAT PARQUET)
    """)
    for t in ("pairs_t", "sel_t", "eoh_slots", "arm_hh"):
        con.unregister(t)
    # mean_temp is the London daily mean for the date, identical for every household. v3
    # averaged the half-hourly temperature over the slots a household happened to have, so
    # meter gaps made it household-specific and the leak audit could read them (0.564).
    con.execute(f"""
        COPY (
            SELECT h.household_id, h.date, min(h.label) AS label,
                   sum(h.kwh) AS daily_kwh, any_value(w.mean_temp) AS mean_temp,
                   month(h.date)::TINYINT AS month,
                   CASE WHEN month(h.date) IN (12,1,2) THEN 'winter'
                        WHEN month(h.date) IN (3,4,5)  THEN 'spring'
                        WHEN month(h.date) IN (6,7,8)  THEN 'summer'
                        ELSE 'autumn' END AS season
            FROM '{out}/hp_dataset_hh30.parquet' h
            JOIN london_daily w ON CAST(w.date AS DATE) = h.date
            GROUP BY h.household_id, h.date
        ) TO '{out}/hp_dataset.parquet' (FORMAT PARQUET)
    """)
    piv = ",\n ".join(f"max(CASE WHEN hh_index={i+1} THEN kwh END)::FLOAT AS h{i:02d}"
                      for i in range(48))
    con.execute(f"""
        COPY (SELECT household_id, date, min(label) AS label, {piv}
              FROM '{out}/hp_dataset_hh30.parquet' GROUP BY household_id, date)
        TO '{out}/lcl_wide_cache.parquet' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*), count(DISTINCT household_id) "
                    f"FROM '{out}/hp_dataset_hh30.parquet'").fetchone()
    print(f"  {n[0]:,} half-hourly rows | {n[1]:,} households | "
          f"{n_interp} selected EoH days interpolated ({time.time()-t0:.0f}s)")
    return n_interp


def step7_features(name, s, stats, out):
    """The 20 household features plus survey and pairing metadata."""
    print(f"\nSTEP 7 [{name}]: household features")
    t0 = time.time()
    con = connect()
    ids = con.execute(f"SELECT DISTINCT household_id FROM '{out}/hp_dataset.parquet' "
                      "ORDER BY 1").df()["household_id"]
    parts, CH = [], 400
    for i in range(0, len(ids), CH):
        batch = tuple(ids[i:i + CH])
        hh = con.execute(f"""
            SELECT household_id, date, hh_index, kwh, temperature,
                   month(date)::TINYINT AS month
            FROM '{out}/hp_dataset_hh30.parquet' WHERE household_id IN {batch}
            ORDER BY household_id, date, hh_index""").df()   # fixed order, so sums repeat exactly
        daily = con.execute(f"""
            SELECT household_id, date, daily_kwh, mean_temp, month
            FROM '{out}/hp_dataset.parquet' WHERE household_id IN {batch}
            ORDER BY household_id, date""").df()
        hh["date"] = pd.to_datetime(hh["date"]); daily["date"] = pd.to_datetime(daily["date"])
        parts.append(features_from_long(daily, hh))
        print(f"    {min(i + CH, len(ids)):>5}/{len(ids)} households", end="\r")
    feats = pd.concat(parts)
    # Sorted, so hp_metadata has the same row order on every build: notebook 06's test split
    # and L/U draws pick rows by position.
    labels = con.execute(f"""SELECT household_id, min(label) AS label, count(*) AS n_days
                             FROM '{out}/hp_dataset.parquet' GROUP BY 1 ORDER BY 1""").df()
    con.close()
    pairs = s["pairs"].rename(columns={"pid": "eoh_id", "contractor": "eoh_contractor"})
    meta = labels.merge(feats.reset_index(), on="household_id", how="left")
    meta["lcl_id"] = meta.household_id.where(meta.label == 0)
    meta = meta.merge(pairs[["household_id", "lcl_id", "eoh_id", "eoh_contractor",
                             "eoh_dwelling_type", "hp_type"]],
                      on="household_id", how="left", suffixes=("", "_p"))
    meta["lcl_id"] = meta.lcl_id.fillna(meta.lcl_id_p)
    meta = meta.drop(columns="lcl_id_p")
    meta = meta.merge(s["arm_households"][["lcl_id", "heat_group", "dwelling_type",
                                           "has_electric_shower", "has_portable_heater"]],
                      on="lcl_id", how="left")
    meta = meta.merge(stats, on="household_id", how="left")
    meta.to_parquet(f"{out}/hp_metadata.parquet", index=False)
    print(f"\n  {len(meta):,} households × {len(FEATURE_COLS)} features "
          f"({time.time()-t0:.0f}s)")
    return meta


def step8_leak_audit(name, meta, out):
    """Gate 1: pure temperature statistics must not predict the label."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    con = connect()
    d = con.execute(f"""SELECT household_id, date, mean_temp, month
                        FROM '{out}/hp_dataset.parquet' ORDER BY household_id, date""").df()
    con.close()
    d["dt"] = d.groupby("household_id")["mean_temp"].diff().abs()
    s = d.groupby("household_id").agg(t_mean=("mean_temp", "mean"),
        t_std=("mean_temp", "std"), t_min=("mean_temp", "min"),
        t_max=("mean_temp", "max"), t_rough=("dt", "mean"))
    s["t_wm_std"] = (d.groupby(["household_id", "month"])["mean_temp"].std()
                     .groupby("household_id").mean())
    s = s.reindex(meta.set_index("household_id").index).fillna(0)
    y = meta["label"].values
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1,
                      random_state=RANDOM_SEED, scale_pos_weight=(1 - y.mean()) / y.mean())
    auc = float(cross_val_score(m, s.values, y, cv=StratifiedKFold(
        5, shuffle=True, random_state=RANDOM_SEED), scoring="roc_auc").mean())
    result = {"temperature_only_auc": auc, "threshold": LEAK_AUDIT_MAX,
              "passed": bool(auc <= LEAK_AUDIT_MAX), "v2_reference_auc": 1.000,
              "v3_reference_auc": 0.495}
    json.dump(result, open(f"{out}/leak_audit.json", "w"), indent=2)
    print(f"\nSTEP 8 [{name}]: leak audit: temperature-only ROC-AUC {auc:.4f} "
          f"({'PASS' if result['passed'] else 'FAIL'} at ≤ {LEAK_AUDIT_MAX})")
    return result


def gates_and_manifest(name, s, sel, meta, leak, n_interp, mad, out):
    """Gates 1-5 for one set, plus its manifest and summary."""
    pos, neg = meta[meta.label == 1], meta[meta.label == 0]
    g = {}
    g["1_leak_audit"] = leak["passed"]
    g["2_positives_on_gas"] = bool((pos.heat_group == "Gas").all())
    # gate 3: positives' appliance rates vs the eligible pool, weighted by house/flat mix
    elig = s["eligible"]
    mix = s["pairs"].stratum.value_counts(normalize=True)
    rates = {}
    for col in ("has_electric_shower", "has_portable_heater"):
        pool = sum(mix.get(st, 0) * elig.loc[elig.dwelling_type == st, col].mean()
                   for st in ("House", "Flat"))
        rates[col] = {"positives": float(pos[col].mean()), "pool": float(pool)}
    g3 = all(abs(v["positives"] - v["pool"]) <= APPLIANCE_TOL for v in rates.values())
    if name == "training":
        g["3_appliances"] = bool(g3)
        hm = s["arm_households"]
        n_el = int((hm.heat_group == "Electric only").sum())
        n_mx = int((hm.heat_group == "Mixed").sum())
        g["4_confounders"] = bool((neg.heat_group == "Electric only").sum() == n_el
                                  and (neg.heat_group == "Mixed").sum() == n_mx)
    gap = sel.temp_gap
    g["5_matching"] = bool(gap.median() < 0.5)
    by_month = sel.assign(m=pd.to_datetime(sel.date).dt.month).groupby("m").temp_gap
    manifest = {
        "set": name, "arm": s["arm"], "join_year": JOIN_YEAR, "seed": RANDOM_SEED,
        "holdout_contractor": HOLDOUT_CONTRACTOR, "max_day_uses": MAX_DAY_USES,
        "month_window": MONTH_WINDOW,
        "sources": {"lcl": [f"{NB_CACHE}/n_ext.parquet", f"{NB_CACHE}/d_ext.parquet",
                            f"{NB_CACHE}/lcl_household_master.parquet",
                            f"{NB_CACHE}/household_day.parquet"],
                    "eoh": [f"{NB_CACHE}/eoh_halfhourly.parquet",
                            f"{NB_CACHE}/eoh_property_day.parquet", EOH_META],
                    "weather": f"Open-Meteo archive, hourly temperature_2m, {LONDON}, GMT",
                    "weather_check_mad_c": mad},
        "households": int(len(meta)), "positives": int(len(pos)),
        "prevalence": float(len(pos) / len(meta)),
        "positives_by_stratum": s["pairs"].stratum.value_counts().to_dict(),
        "positives_by_hp_type": s["pairs"].hp_type.value_counts().to_dict(),
        "positives_by_contractor": s["pairs"].contractor.value_counts().to_dict(),
        "households_by_heat_group_label": {f"{k[0]}|{k[1]}": int(v) for k, v in
            meta.groupby(["heat_group", "label"], dropna=False).size().items()},
        "matching_gap_c": {"median": float(gap.median()), "p90": float(gap.quantile(.9)),
                           "pct_over_1c": float((gap > 1).mean() * 100),
                           "median_by_month": by_month.median().round(3).to_dict(),
                           "pct_over_1c_by_month": (by_month.apply(lambda x: (x > 1).mean() * 100)
                                                    .round(1).to_dict())},
        "reuse": {"households_with_cap_fallback": int((pos.n_fallback_cap > 0).sum()),
                  "cap_fallback_dates": int(pos.n_fallback_cap.sum()),
                  "max_day_uses_max": int(pos.max_day_uses.max())},
        "dropped_dates": {"total": int(pos.n_dropped_dates.sum()),
                          "households_affected": int((pos.n_dropped_dates > 0).sum()),
                          "pct_of_positive_dates": float(pos.n_dropped_dates.sum()
                              / (pos.n_dropped_dates.sum() + len(sel)) * 100)},
        "eoh_days_interpolated": n_interp,
        "appliance_rates": rates,
        "leak_audit_auc": leak["temperature_only_auc"],
        "gates": g,
    }
    json.dump(manifest, open(f"{out}/join_manifest.json", "w"), indent=2, default=str)
    print(f"\n── {name.upper()} SUMMARY ──")
    print(f"  households {len(meta):,} | positives {len(pos)} ({len(pos)/len(meta):.1%}) | "
          + ", ".join(f"{k} {v}" for k, v in manifest["positives_by_stratum"].items()))
    print(meta.groupby(["heat_group", "label"], dropna=False).size().unstack(fill_value=0)
          .to_string())
    print(f"  days: {len(sel):,} positive household-days | dropped {manifest['dropped_dates']['total']}"
          f" | cap fallbacks {manifest['reuse']['cap_fallback_dates']} on "
          f"{manifest['reuse']['households_with_cap_fallback']} households")
    print(f"  matching gap: median {gap.median():.3f} °C, p90 {gap.quantile(.9):.2f} °C, "
          f"{(gap > 1).mean():.1%} > 1 °C | leak-audit AUC {leak['temperature_only_auc']:.4f}")
    print(f"  appliance rates (positives vs pool): " + "; ".join(
          f"{k} {v['positives']:.1%} vs {v['pool']:.1%}" for k, v in rates.items()))
    print("  gates: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in g.items()))
    return g


def main():
    t0 = time.time()
    os.makedirs(OUT_TRANSFER, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    con = connect()
    print("=" * 70 + "\nLCL_EoH_join_v6: survey hosts, 2013, capped temperature matching\n"
          + "=" * 70)
    daily, mad = step1_weather(con)
    master = step2_lcl(con)
    props, eoh_days = step3_eoh(con)
    sets = step4_pairing(props, master, rng)

    host_ids = pd.concat([s["pairs"].lcl_id for s in sets.values()]).unique()
    lcl_dates = (con.execute("SELECT DISTINCT lcl_id, date FROM lcl WHERE lcl_id IN "
                             f"{tuple(host_ids)} ORDER BY lcl_id, date").df()
                 .assign(date=lambda d: pd.to_datetime(d.date))
                 .groupby("lcl_id").date.apply(lambda x: list(x)).to_dict())
    daily = daily.assign(date=pd.to_datetime(daily.date))

    all_gates = {}
    for name, out in [("training", OUT), ("transfer", OUT_TRANSFER)]:
        s = sets[name]
        s["arm_households"] = master[master.arm == s["arm"]]
        print(f"\nSTEP 5 [{name}]: capped nearest-temperature day selection")
        t1 = time.time()
        sel, stats = step5_select_days(s["pairs"], lcl_dates, eoh_days, daily, rng)
        print(f"  {len(sel):,} matched household-days ({time.time()-t1:.0f}s)")
        n_interp = step6_build(con, name, s, sel, out)
        meta = step7_features(name, s, stats, out)
        leak = step8_leak_audit(name, meta, out)
        all_gates[name] = gates_and_manifest(name, s, sel, meta, leak, n_interp, mad, out)
        s["meta"] = meta
    con.close()

    # gate 6: hold-out integrity across the two sets
    tr, te = sets["training"], sets["transfer"]
    integrity = {
        "no_ovo_in_training": bool((tr["pairs"].contractor != HOLDOUT_CONTRACTOR).all()),
        "transfer_positives_all_ovo": bool((te["pairs"].contractor == HOLDOUT_CONTRACTOR).all()),
        "no_household_in_both": len(set(tr["meta"].household_id)
                                    & set(te["meta"].household_id)) == 0,
        "no_arm_d_in_training": bool(tr["meta"].lcl_id.str.startswith("N").all()),
    }
    print("\nGATE 6: hold-out integrity: " + ", ".join(
          f"{k} {'PASS' if v else 'FAIL'}" for k, v in integrity.items()))
    for out in (OUT, OUT_TRANSFER):
        m = json.load(open(f"{out}/join_manifest.json"))
        m["gates"]["6_holdout_integrity"] = integrity
        json.dump(m, open(f"{out}/join_manifest.json", "w"), indent=2, default=str)

    failed = [f"{n}:{k}" for n, g in all_gates.items() for k, v in g.items() if not v]
    failed += [f"integrity:{k}" for k, v in integrity.items() if not v]
    print(f"\nBuild finished in {time.time()-t0:.0f}s → {OUT}")
    if failed:
        raise SystemExit("GATES FAILED: dataset not usable: " + ", ".join(failed))
    print("All gates passed (gate 7: gates 1, 2 and 5 applied to the transfer set).")


if __name__ == "__main__":
    main()
