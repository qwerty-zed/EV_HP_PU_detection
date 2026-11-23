#!/usr/bin/env python3
"""
create_ev_dataset_v2.py
=======================
Synthetic EV dataset: Low Carbon London (LCL) arm N household load + Electric Nation (EN)
home-charger load. The join rules are set out in notebook 05_ev_linkage_lcl_en.

  positives  one 21-day single-regime EN segment with recorded charging, per charger and
             regime, added to a random arm-N host at a random date on the same weekday
  negatives  21-day windows of other arm-N households (donors), only as many as the
             experiments use: 3,000 for training and 9 per test positive

What changed from v1 (LCL_EN_join)
----------------------------------
- Four EN regimes (T1 control, T1 demand management, T2, T3) in single-regime 21-day segments
  instead of one charger-year verbatim. The unmanaged (no-group) regime is excluded: 84% of its
  days come before a charger joins any trial group, GreenFlux includes a pre-first-charge idle
  period, and its windows are empty about four times as often as the managed regimes'. 21 days is
  the longest whole-week length that keeps T1 control usable (28 segments; 8 at 28 days).
- EN is read from the notebook 03 GMT caches; contiguity is on GMT dates, so the BST dates are
  kept and T3 runs stay contiguous.
- A positive must contain at least one charging day; its start is drawn among such windows.
- No month matching. The host date is random, on the same weekday as the EN start, and each EV
  half-hour is shifted by the UK clock-offset difference (0 or +-1 h) so it keeps local time.
- LCL is the UK Data Service release (study 7857), read from the notebook 01 caches.
- Split by charger and by household before pairing; donors split in proportion to need.
- Negative windows are drawn the way positives are placed: a weekday from the positives'
  start-weekday mix, then a uniform valid start on that weekday, with no overlap within a donor
  (adopted after the first build failed the metadata audit at 0.559: a fixed
  segment-length grid over each donor's record bunched negative start weekdays).
"""
import json
import os
import tempfile
import time

import duckdb
import numpy as np
import pandas as pd

RANDOM_SEED = 42
SEGMENT_DAYS = 21
COMPLETE_DAY_SLOTS = 46
TEST_FRACTION = 0.30          # of chargers
TRAIN_NEGATIVES = 3_000       # size of the PU unlabelled pool
TEST_NEG_PER_POS = 9          # exactly 10% test prevalence
REGIMES = ["T1 control", "T1 demand management", "T2 priority-charging app",
           "T3 time-of-use tariff"]
POWER_CLASS_KW = 5.0          # 95th-percentile session peak below this = 3.5 kW class
AUDIT_MAX = 0.55
MAX_DROPPED_SHARE = 0.01
EVENING = range(33, 45)       # 16:00-22:00, hh_index of period start
OVERNIGHT = list(range(45, 49)) + list(range(1, 15))   # 22:00-07:00

# Paths are relative to this script, which sits in the repository root: the study data are in
# Data/ (EN, EoH, LCL), the notebooks write their caches to cache/, and the outputs go to a
# folder created next to this script.
DATA = os.path.dirname(os.path.abspath(__file__))
NB_CACHE = f"{DATA}/cache"
OUT = f"{DATA}/LCL_EN_join_v2"
os.makedirs(OUT, exist_ok=True)


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='6GB'")
    con.execute(f"SET temp_directory='{tempfile.gettempdir()}/duckdb_spill'")
    return con


def uk_offset(ts):
    """UK clock offset from GMT (0 or 1 hour) at GMT timestamps, as a timedelta64 array."""
    gmt = pd.DatetimeIndex(ts)
    local = gmt.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
    return (local - gmt).to_numpy()


def window_days(con, wins, name):
    """Register one row per (segment, household, date) covered by each window."""
    days = pd.DataFrame({
        "segment_id": np.repeat(wins.segment_id.values, SEGMENT_DAYS),
        "household_id": np.repeat(wins.household_id.values, SEGMENT_DAYS),
        "date": (np.repeat(wins.window_start.values.astype("datetime64[D]"), SEGMENT_DAYS)
                 + np.tile(np.arange(SEGMENT_DAYS), len(wins)).astype("timedelta64[D]")),
    })
    days["date"] = pd.to_datetime(days.date)
    con.register(name, days)


# ──────────────────────────────────────────────────────────────────────
def step1_ev_segments(con, rng):
    """One single-regime segment with charging, per charger and regime."""
    print(f"\nSTEP 1: EN segments ({SEGMENT_DAYS} contiguous GMT days, one regime, at least one charging day)")
    con.execute(f"""
        CREATE TEMP TABLE cd AS
        SELECT provider || ':' || charger_id AS charger, provider, charger_id,
               CAST(date AS DATE) AS d, phase AS regime,
               coalesce(charged, false)::INT AS c, n_sessions, kwh
        FROM '{NB_CACHE}/en_charger_day.parquet'
        WHERE phase IN {tuple(REGIMES)}
    """)
    # gaps-and-islands: a run is consecutive dates in one regime
    con.execute("""
        CREATE TEMP TABLE cd_run AS
        SELECT *, (d - DATE '2016-01-01')
                  - row_number() OVER (PARTITION BY charger, regime ORDER BY d) AS run
        FROM cd
    """)
    win = con.execute(f"""
        WITH w AS (
            SELECT charger, regime, d AS en_start,
                   sum(c) OVER (PARTITION BY charger, regime, run ORDER BY d
                                ROWS BETWEEN CURRENT ROW AND {SEGMENT_DAYS - 1} FOLLOWING) AS n_c,
                   count(*) OVER (PARTITION BY charger, regime, run ORDER BY d
                                  ROWS BETWEEN CURRENT ROW AND {SEGMENT_DAYS - 1} FOLLOWING) AS n_d
            FROM cd_run)
        SELECT charger, regime, en_start, n_c FROM w WHERE n_d = {SEGMENT_DAYS}
        ORDER BY charger, regime, en_start
    """).df()
    supply = win.groupby("regime").charger.nunique()
    ok = win[win.n_c > 0]
    picks = []
    for (ch, rg), g in ok.groupby(["charger", "regime"], sort=True):
        picks.append(g.iloc[rng.integers(len(g))])
    seg = pd.DataFrame(picks).reset_index(drop=True)
    seg["en_start"] = pd.to_datetime(seg.en_start)
    seg = seg.drop(columns="n_c")

    # segment statistics from the charger-day spine
    con.register("seg_t", seg[["charger", "regime", "en_start"]])
    stats = con.execute(f"""
        SELECT s.charger, s.regime, sum(c.c) AS n_charge_days,
               sum(c.n_sessions)::INT AS n_sessions, sum(c.kwh) AS en_kwh_sessions
        FROM seg_t s JOIN cd c ON c.charger = s.charger AND c.regime = s.regime
             AND c.d >= CAST(s.en_start AS DATE)
             AND c.d < CAST(s.en_start AS DATE) + {SEGMENT_DAYS}
        GROUP BY 1, 2
    """).df()
    con.unregister("seg_t")
    seg = seg.merge(stats, on=["charger", "regime"])
    first = con.execute("SELECT DISTINCT charger, provider, charger_id FROM cd").df()
    seg = seg.merge(first, on="charger")

    # charge-power class: 95th-percentile session peak below 5 kW = 3.5 kW class
    pc = con.execute(f"""
        SELECT provider || ':' || charger_id AS charger,
               quantile_cont(peak_kw, 0.95) AS p95_peak_kw
        FROM '{NB_CACHE}/en_sessions.parquet' GROUP BY 1
    """).df()
    pc["power_class"] = np.where(pc.p95_peak_kw < POWER_CLASS_KW, "3.5 kW class", "7 kW class")
    seg = seg.merge(pc[["charger", "power_class"]], on="charger", how="left")

    table = pd.DataFrame({"able_to_supply": supply,
                          "positives_with_charging": seg.groupby("regime").size()}
                         ).reindex(REGIMES).fillna(0).astype(int)
    print(table.to_string())
    print(f"  {len(seg):,} positive segments from {seg.charger.nunique()} chargers "
          f"({len(win.groupby(['charger', 'regime'])) - len(seg)} charger-regimes never charge "
          f"in any {SEGMENT_DAYS}-day window)")
    return seg, table


def step2_lcl(con):
    """Qualifying arm-N households and their valid window starts."""
    print(f"\nSTEP 2: LCL arm-N households and valid windows ({SEGMENT_DAYS} consecutive complete days)")
    master = con.execute(f"""
        SELECT household_id, suspected_ev, heat_group, dwelling_type
        FROM '{NB_CACHE}/lcl_household_master.parquet'
        WHERE arm = 'N' AND passes_filters ORDER BY household_id
    """).df()
    con.register("master_t", master[["household_id"]])
    runs = con.execute(f"""
        WITH d AS (
            SELECT household_id, date AS d FROM '{NB_CACHE}/household_day.parquet'
            WHERE n_slots >= {COMPLETE_DAY_SLOTS}
              AND household_id IN (SELECT household_id FROM master_t)),
        g AS (SELECT *, (d - DATE '2011-01-01')
                        - row_number() OVER (PARTITION BY household_id ORDER BY d) AS grp FROM d)
        SELECT household_id, min(d) AS run_start, count(*) AS n
        FROM g GROUP BY household_id, grp HAVING count(*) >= {SEGMENT_DAYS}
        ORDER BY household_id, run_start
    """).df()
    con.unregister("master_t")
    runs["run_start"] = pd.to_datetime(runs.run_start)
    with_win = set(runs.household_id)
    n_tiles = int((runs.n // SEGMENT_DAYS).sum())
    print(f"  {len(master):,} households pass the quality filters; {len(with_win):,} have a valid "
          f"window; {n_tiles:,} non-overlapping windows")
    master = master[master.household_id.isin(with_win)].reset_index(drop=True)
    return master, runs


def step3_split(seg, master, rng):
    """Charger split 70/30, then one role per household; donors split in proportion to need."""
    print("\nSTEP 3: Split by charger and by household (before pairing)")
    chargers = np.array(sorted(seg.charger.unique()))
    test_ch = set(rng.choice(chargers, int(round(TEST_FRACTION * len(chargers))), replace=False))
    seg["split"] = np.where(seg.charger.isin(test_ch), "test", "train")
    p_train, p_test = int((seg.split == "train").sum()), int((seg.split == "test").sum())
    need = {"train": TRAIN_NEGATIVES, "test": TEST_NEG_PER_POS * p_test}
    n_donors = len(master) - p_train - p_test
    d_test = int(round(n_donors * need["test"] / (need["train"] + need["test"])))
    d_train = n_donors - d_test
    order = rng.permutation(len(master))
    sides = {"train": master.iloc[order[:p_train + d_train]].household_id.values,
             "test": master.iloc[order[p_train + d_train:]].household_id.values}
    print(f"  chargers: {len(chargers) - len(test_ch)} train / {len(test_ch)} test | "
          f"positive segments {p_train} / {p_test}")
    print(f"  households: {len(sides['train'])} train side / {len(sides['test'])} test side | "
          f"donors {d_train} / {d_test} | negatives needed {need['train']} / {need['test']}")
    return seg, sides, need


def valid_starts(runs):
    """Every valid window start date per household."""
    out = {}
    for hid, g in runs.groupby("household_id"):
        out[hid] = pd.DatetimeIndex(np.concatenate([
            pd.date_range(r.run_start, periods=r.n - SEGMENT_DAYS + 1, freq="D").values
            for r in g.itertuples()]))
    return out


def step4_place(seg, sides, starts, rng):
    """Each segment on a random host of its side, starting on the EN start's weekday."""
    print("\nSTEP 4: Placement: random host, random date, same weekday")
    seg = seg.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    hosts, host_starts, redraws = [], [], 0
    pools = {k: list(rng.permutation(v)) for k, v in sides.items()}
    for r in seg.itertuples():
        pool = pools[r.split]
        wd = r.en_start.dayofweek
        for i, hid in enumerate(pool):
            cand = starts[hid][starts[hid].dayofweek == wd]
            if len(cand):
                break
            redraws += 1
        else:
            raise SystemExit(f"no host with a {wd} window left on the {r.split} side")
        pool.pop(i)
        hosts.append(hid)
        host_starts.append(cand[rng.integers(len(cand))])
    seg["household_id"] = hosts
    seg["window_start"] = pd.DatetimeIndex(host_starts)
    seg["segment_id"] = [f"EV{i:04d}" for i in range(len(seg))]
    host_set = set(hosts)
    donors = {k: np.array([h for h in sides[k] if h not in host_set]) for k in sides}
    print(f"  {len(seg)} hosts | {redraws} host redraws for a weekday | donors "
          f"{len(donors['train'])} train / {len(donors['test'])} test")
    return seg, donors


def add_window(starts, chosen, pos_wd, rng, tries=50):
    """One negative start drawn like a positive's: a weekday from the positives' mix, then a
    uniform valid start on that weekday that does not overlap the donor's other windows."""
    for _ in range(tries):
        c = starts[starts.dayofweek == pos_wd[rng.integers(len(pos_wd))]]
        if chosen:
            gap = np.abs((c.values[:, None] - np.array(chosen)[None, :]) / np.timedelta64(1, "D"))
            c = c[gap.min(axis=1) >= SEGMENT_DAYS]
        if len(c):
            chosen.append(c[rng.integers(len(c))].to_datetime64())
            return True
    return False


def step5_negatives(donors, need, starts, pos_wd, rng):
    """Negative windows drawn like positives; a donor's shortfall moves to donors with room."""
    print("\nSTEP 5: Negative windows (only those the experiments use), drawn like positives")
    rows = []
    for side in ("train", "test"):
        ids = donors[side]
        base, extra = divmod(need[side], len(ids))
        quota = np.full(len(ids), base)
        quota[rng.choice(len(ids), extra, replace=False)] += 1
        chosen = {hid: [] for hid in ids}
        short = 0
        for hid, q in zip(ids, quota):
            for _ in range(q):
                if not add_window(starts[hid], chosen[hid], pos_wd, rng):
                    short += 1
        full, moved = set(), short
        while short:
            room = [h for h in ids if h not in full]
            if not room:
                raise SystemExit(f"{side}: donors cannot supply {need[side]} windows")
            hid = room[rng.integers(len(room))]
            if add_window(starts[hid], chosen[hid], pos_wd, rng):
                short -= 1
            else:
                full.add(hid)
        rows += [(hid, s, side) for hid in ids for s in chosen[hid]]
        print(f"  {side}: {need[side]:,} windows from {len(ids):,} donors "
              f"({need[side] / len(ids):.2f} per donor; {moved} moved from donors without room)")
    neg = pd.DataFrame(rows, columns=["household_id", "window_start", "split"])
    neg["window_start"] = pd.to_datetime(neg.window_start)
    neg["segment_id"] = [f"NE{i:05d}" for i in range(len(neg))]
    return neg


def step6_ev_slots(con, seg):
    """EN half-hours of each segment, moved onto the host date with the clock-offset shift."""
    print("\nSTEP 6: EV half-hours, shifted to keep local clock time")
    con.register("seg_t", seg[["segment_id", "provider", "charger_id", "en_start", "window_start"]])
    ev = con.execute(f"""
        SELECT s.segment_id, s.en_start, s.window_start, h.ts, h.kwh
        FROM seg_t s JOIN '{NB_CACHE}/en_halfhourly_gmt.parquet' h
          ON h.provider = s.provider AND h.charger_id = s.charger_id
         AND h.ts >= s.en_start AND h.ts < s.en_start + INTERVAL {SEGMENT_DAYS} DAY
    """).df()
    con.unregister("seg_t")
    raw = ev.ts - ev.en_start + ev.window_start            # same GMT position on the host
    shift = uk_offset(ev.ts) - uk_offset(raw)               # source offset minus host offset
    ev["host_ts"] = raw + shift
    ev["shift_h"] = shift / np.timedelta64(1, "h")
    ev["date"] = ev.host_ts.dt.normalize()
    ev["hh_index"] = (2 * ev.host_ts.dt.hour + (ev.host_ts.dt.minute >= 30) + 1).astype(int)
    end = ev.window_start + pd.Timedelta(days=SEGMENT_DAYS)
    ev["edge_drop"] = (ev.host_ts < ev.window_start) | (ev.host_ts >= end)
    # weekday check on the local clock: source local day vs host local day
    src_local = ev.ts + uk_offset(ev.ts)
    host_local = ev.host_ts + uk_offset(ev.host_ts)
    ev["weekday_ok"] = src_local.dt.dayofweek.values == host_local.dt.dayofweek.values
    print(f"  {len(ev):,} EV half-hours | {(ev.shift_h != 0).mean():.1%} shifted by an hour | "
          f"{ev.edge_drop.sum()} past the window edge")
    return ev


def step7_build(con, seg, neg, ev):
    """Half-hourly, daily and wide tables; host load plus EV load for positives."""
    print("\nSTEP 7: Half-hourly, daily and wide tables")
    t0 = time.time()
    wins = pd.concat([seg[["segment_id", "household_id", "window_start"]].assign(label=1),
                      neg[["segment_id", "household_id", "window_start"]].assign(label=0)])
    con.register("wins", wins)
    window_days(con, wins, "wdays")
    keep = ev[~ev.edge_drop].groupby(["segment_id", "date", "hh_index"], as_index=False).kwh.sum()
    con.register("ev_t", keep.rename(columns={"kwh": "ev_kwh"}))
    con.execute(f"""
        CREATE TEMP TABLE host AS
        SELECT w.segment_id, w.household_id, l.label, x.date,
               (2 * hour(x.period_start) + (minute(x.period_start) >= 30)::INT + 1)::TINYINT
                   AS hh_index,
               x.kwh AS host_kwh
        FROM wdays w
        JOIN wins l USING (segment_id)
        JOIN '{NB_CACHE}/n_ext.parquet' x
          ON x.household_id = w.household_id AND x.date = CAST(w.date AS DATE)
    """)
    lost = con.execute("""
        SELECT e.segment_id, sum(e.ev_kwh) AS kwh_lost_missing_host
        FROM ev_t e LEFT JOIN host h ON h.segment_id = e.segment_id
             AND h.date = CAST(e.date AS DATE) AND h.hh_index = e.hh_index
        WHERE h.segment_id IS NULL GROUP BY 1
    """).df()
    con.execute(f"""
        COPY (
            SELECT h.segment_id, h.household_id, h.date, h.hh_index,
                   (h.host_kwh + coalesce(e.ev_kwh, 0))::FLOAT AS kwh, h.label::TINYINT AS label
            FROM host h LEFT JOIN ev_t e ON e.segment_id = h.segment_id
                 AND CAST(e.date AS DATE) = h.date AND e.hh_index = h.hh_index
        ) TO '{OUT}/ev_dataset_hh30.parquet' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT h.segment_id, any_value(h.household_id) AS household_id, h.date,
                   (h.date - CAST(any_value(w.window_start) AS DATE) + 1)::TINYINT AS day_index,
                   dayname(h.date) AS weekday, (dayofweek(h.date) IN (0, 6)) AS is_weekend,
                   sum(h.kwh) AS daily_kwh, count(*)::TINYINT AS n_slots, min(h.label) AS label
            FROM '{OUT}/ev_dataset_hh30.parquet' h JOIN wins w USING (segment_id)
            GROUP BY h.segment_id, h.date
        ) TO '{OUT}/ev_dataset.parquet' (FORMAT PARQUET)
    """)
    piv = ",\n ".join(f"max(CASE WHEN hh_index={i+1} THEN kwh END)::FLOAT AS h{i:02d}"
                      for i in range(48))
    con.execute(f"""
        COPY (SELECT segment_id, date, min(label) AS label, {piv}
              FROM '{OUT}/ev_dataset_hh30.parquet' GROUP BY segment_id, date)
        TO '{OUT}/ev_wide_cache.parquet' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*), count(DISTINCT segment_id) "
                    f"FROM '{OUT}/ev_dataset_hh30.parquet'").fetchone()
    print(f"  {n[0]:,} half-hourly rows | {n[1]:,} segments ({time.time() - t0:.0f}s)")
    return lost


def step8_metadata(con, seg, neg, ev, lost, master):
    """One row per segment."""
    days = con.execute(f"""
        SELECT segment_id, count(*) FILTER (WHERE n_slots >= {COMPLETE_DAY_SLOTS}) AS n_complete_days,
               1 - sum(n_slots) / ({SEGMENT_DAYS} * 48.0) AS missing_share
        FROM '{OUT}/ev_dataset.parquet' GROUP BY 1
    """).df()
    added = ev.groupby("segment_id").agg(ev_kwh_source=("kwh", "sum"),
                                         n_shifted=("shift_h", lambda s: int((s != 0).sum())))
    edge = ev[ev.edge_drop].groupby("segment_id").kwh.sum().rename("kwh_lost_edge").to_frame()
    pos = (seg.set_index("segment_id").join(added).join(edge)
           .join(lost.set_index("segment_id")).reset_index())
    pos[["ev_kwh_source", "kwh_lost_edge", "kwh_lost_missing_host"]] = \
        pos[["ev_kwh_source", "kwh_lost_edge", "kwh_lost_missing_host"]].fillna(0.0)
    pos["kwh_dropped"] = pos.kwh_lost_edge + pos.kwh_lost_missing_host
    pos["ev_kwh_added"] = pos.ev_kwh_source - pos.kwh_dropped
    pos["label"], pos["role"] = 1, pos.split + "_host"
    neg = neg.assign(label=0, role=neg.split + "_donor")
    meta = pd.concat([pos, neg], ignore_index=True)
    meta["start_weekday"] = meta.window_start.dt.day_name()
    meta["start_month"] = meta.window_start.dt.month
    meta = meta.merge(days, on="segment_id").merge(master, on="household_id", how="left")
    cols = ["segment_id", "household_id", "label", "split", "role", "window_start",
            "start_weekday", "start_month", "n_complete_days", "missing_share", "suspected_ev",
            "heat_group", "dwelling_type", "charger", "charger_id", "provider", "regime",
            "power_class", "en_start", "ev_kwh_source", "ev_kwh_added", "kwh_dropped",
            "kwh_lost_edge", "kwh_lost_missing_host", "n_shifted", "n_charge_days",
            "n_sessions"]
    meta = meta[cols].sort_values("segment_id").reset_index(drop=True)
    meta.to_parquet(f"{OUT}/ev_metadata.parquet", index=False)
    print(f"\nSTEP 8: metadata: {len(meta):,} segments "
          f"({int(meta.label.sum())} positive, {int((meta.label == 0).sum())} negative)")
    return meta


def cv_auc(X, y):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1,
                      random_state=RANDOM_SEED, scale_pos_weight=(1 - y.mean()) / y.mean())
    return float(cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True,
                 random_state=RANDOM_SEED), scoring="roc_auc").mean())


def step9_audits(con, meta):
    """Gate 1 (metadata only) and gate 2 (hosts' own load before the EV)."""
    print("\nSTEP 9: Leakage audits")
    y = meta.label.values
    md = pd.DataFrame({"start_month": meta.start_month,
                       "start_year": meta.window_start.dt.year,
                       "start_weekday": meta.window_start.dt.dayofweek,
                       "n_complete_days": meta.n_complete_days,
                       "missing_share": meta.missing_share})
    auc_meta = cv_auc(md.values, y)
    window_days(con, meta[["segment_id", "household_id", "window_start"]], "m_days")
    own = con.execute(f"""
        WITH x AS (
            SELECT m.segment_id, n.date, n.kwh,
                   (2 * hour(n.period_start) + (minute(n.period_start) >= 30)::INT + 1) AS hh
            FROM m_days m JOIN '{NB_CACHE}/n_ext.parquet' n
              ON n.household_id = m.household_id AND n.date = CAST(m.date AS DATE)),
        d AS (SELECT segment_id, date, sum(kwh) AS k, dayofweek(date) IN (0, 6) AS we
              FROM x GROUP BY 1, 2)
        SELECT s.segment_id, s.mean_daily, s.wd / nullif(s.we, 0) AS wd_we_ratio,
               q.evening, q.overnight, q.p95
        FROM (SELECT segment_id, avg(k) AS mean_daily,
                     avg(k) FILTER (WHERE NOT we) AS wd, avg(k) FILTER (WHERE we) AS we
              FROM d GROUP BY 1) s
        JOIN (SELECT segment_id,
                     sum(kwh) FILTER (WHERE hh BETWEEN {EVENING.start} AND {EVENING.stop - 1})
                         / sum(kwh) AS evening,
                     sum(kwh) FILTER (WHERE hh IN {tuple(OVERNIGHT)}) / sum(kwh) AS overnight,
                     quantile_cont(kwh, 0.95) AS p95
              FROM x GROUP BY 1) q USING (segment_id)
    """).df()
    con.unregister("m_days")
    own = meta[["segment_id"]].merge(own, on="segment_id", how="left").fillna(0)
    auc_host = cv_auc(own.drop(columns="segment_id").values, y)
    res = {"metadata_only_auc": auc_meta, "host_own_load_auc": auc_host,
           "threshold": AUDIT_MAX,
           "metadata_features": list(md.columns),
           "host_features": ["mean_daily_kwh", "weekday_weekend_ratio", "evening_share",
                             "overnight_share", "p95_half_hour_kwh"],
           "passed": bool(auc_meta <= AUDIT_MAX and auc_host <= AUDIT_MAX)}
    json.dump(res, open(f"{OUT}/leak_audit.json", "w"), indent=2)
    print(f"  metadata-only ROC-AUC {auc_meta:.4f} | host-own-load ROC-AUC {auc_host:.4f} "
          f"(gate ≤ {AUDIT_MAX})")
    return res


def gates_and_manifest(seg, meta, ev, audit, supply, need, t_build):
    pos, neg = meta[meta.label == 1], meta[meta.label == 0]
    g = {}
    g["1_metadata_audit"] = audit["metadata_only_auc"] <= AUDIT_MAX
    g["2_host_audit"] = audit["host_own_load_auc"] <= AUDIT_MAX
    g["3_weekday_alignment"] = bool(ev.loc[~ev.edge_drop, "weekday_ok"].all())
    ch_sides = pos.groupby("charger").split.nunique()
    hh_sides = meta.groupby("household_id").split.nunique()
    hh_roles = meta.groupby("household_id").role.nunique()
    host_ids = set(pos.household_id)
    overlaps = 0
    for _, w in neg.sort_values("window_start").groupby("household_id").window_start:
        overlaps += int((w.diff().dt.days.dropna() < SEGMENT_DAYS).sum())
    integrity = {
        "positives_one_regime_full_length_with_charging": bool(pos.regime.isin(REGIMES).all()
                                                       and (pos.n_charge_days >= 1).all()),
        "no_charger_on_both_sides": bool((ch_sides == 1).all()),
        "no_household_on_both_sides": bool((hh_sides == 1).all()),
        "one_role_per_household": bool((hh_roles == 1).all()),
        "hosts_contribute_no_negatives": not bool(set(neg.household_id) & host_ids),
        "one_segment_per_host": bool(pos.household_id.is_unique),
        "no_overlapping_donor_windows": overlaps == 0,
    }
    g["4_integrity"] = all(integrity.values())
    dropped_share = float(pos.kwh_dropped.sum() / pos.ev_kwh_source.sum())
    g["5_energy"] = dropped_share <= MAX_DROPPED_SHARE
    t = meta[meta.split == "test"]
    test_prev = float(t.label.mean())
    g["6_test_prevalence"] = int(t.label.sum()) * TEST_NEG_PER_POS == int((t.label == 0).sum())
    reuse = neg.groupby(["split", "household_id"]).size().groupby("split").agg(["mean", "max"])
    manifest = {
        "segment_days": SEGMENT_DAYS, "seed": RANDOM_SEED, "test_fraction_chargers": TEST_FRACTION,
        "sources": {"en": [f"{NB_CACHE}/en_charger_day.parquet",
                           f"{NB_CACHE}/en_halfhourly_gmt.parquet",
                           f"{NB_CACHE}/en_sessions.parquet"],
                    "lcl": [f"{NB_CACHE}/n_ext.parquet", f"{NB_CACHE}/household_day.parquet",
                            f"{NB_CACHE}/lcl_household_master.parquet"]},
        "segments": int(len(meta)), "positives": int(len(pos)), "negatives": int(len(neg)),
        "supply_by_regime": supply.to_dict(orient="index"),
        "positives_by_regime_split": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                      pos.groupby(["regime", "split"]).size().items()},
        "positives_by_provider": pos.provider.value_counts().to_dict(),
        "positives_by_power_class": pos.power_class.value_counts(dropna=False).to_dict(),
        "chargers": {"total": int(pos.charger.nunique()),
                     "train": int(pos.loc[pos.split == "train", "charger"].nunique()),
                     "test": int(pos.loc[pos.split == "test", "charger"].nunique())},
        "split": {s: {"positives": int((meta[meta.split == s].label == 1).sum()),
                      "negatives": int((meta[meta.split == s].label == 0).sum()),
                      "hosts": int(pos[pos.split == s].household_id.nunique()),
                      "donors": int(neg[neg.split == s].household_id.nunique())}
                  for s in ("train", "test")},
        "negatives_needed": need, "test_prevalence": test_prev,
        "donor_reuse": {s: {"mean": float(reuse.loc[s, "mean"]), "max": int(reuse.loc[s, "max"])}
                        for s in reuse.index},
        "energy": {"ev_kwh_source": float(pos.ev_kwh_source.sum()),
                   "kwh_lost_edge": float(pos.kwh_lost_edge.sum()),
                   "kwh_lost_missing_host": float(pos.kwh_lost_missing_host.sum()),
                   "dropped_share": dropped_share,
                   "half_hours_shifted_share": float((ev.shift_h != 0).mean())},
        "suspected_ev_hosts": int(pos.suspected_ev.fillna(False).sum()),
        "suspected_ev_donors": int(neg.drop_duplicates("household_id").suspected_ev
                                   .fillna(False).sum()),
        "audits": audit, "integrity": integrity,
        "gates": {k: bool(v) for k, v in g.items()}, "build_seconds": round(t_build),
    }
    json.dump(manifest, open(f"{OUT}/join_manifest.json", "w"), indent=2, default=str)

    print("\n── SUMMARY ──")
    print(f"  positives {len(pos):,} from {manifest['chargers']['total']} chargers | "
          f"negatives {len(neg):,}")
    print(pos.groupby(["regime", "split"]).size().unstack().reindex(REGIMES).to_string())
    for s in ("train", "test"):
        v = manifest["split"][s]
        print(f"  {s}: {v['positives']} positives, {v['negatives']} negatives, "
              f"{v['hosts']} hosts, {v['donors']} donors "
              f"(reuse {manifest['donor_reuse'][s]['mean']:.2f}, max {manifest['donor_reuse'][s]['max']})")
    print(f"  test prevalence {test_prev:.1%} | EV energy dropped {dropped_share:.3%} "
          f"(edge {pos.kwh_lost_edge.sum():.0f} kWh, missing host {pos.kwh_lost_missing_host.sum():.0f}"
          f" kWh, of {pos.ev_kwh_source.sum():,.0f})")
    print("  integrity: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in integrity.items()))
    print("  gates: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in g.items()))
    return g


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    con = connect()
    print("=" * 70 + f"\nLCL_EN_join_v2: {SEGMENT_DAYS}-day single-regime EV segments on arm-N hosts\n"
          + "=" * 70)
    seg, supply = step1_ev_segments(con, rng)
    master, runs = step2_lcl(con)
    seg, sides, need = step3_split(seg, master, rng)
    starts = valid_starts(runs)
    seg, donors = step4_place(seg, sides, starts, rng)
    neg = step5_negatives(donors, need, starts, seg.en_start.dt.dayofweek.values, rng)
    ev = step6_ev_slots(con, seg)
    lost = step7_build(con, seg, neg, ev)
    meta = step8_metadata(con, seg, neg, ev, lost, master)
    audit = step9_audits(con, meta)
    con.close()
    g = gates_and_manifest(seg, meta, ev, audit, supply, need, time.time() - t0)
    print(f"\nBuild finished in {time.time() - t0:.0f}s → {OUT}")
    failed = [k for k, v in g.items() if not v]
    if failed:
        raise SystemExit("GATES FAILED: dataset not usable: " + ", ".join(failed))
    print("All gates passed.")


if __name__ == "__main__":
    main()
