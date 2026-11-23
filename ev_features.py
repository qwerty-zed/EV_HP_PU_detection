"""The 20 EV segment features: computation and definitions, for 07_ev_pu_classifier.

One function, `features_from_wide`, computes every feature for every 21-day segment at once
from the wide table of the EV join (`ev_wide_cache.parquet`: segment_id, date, h00..h47), as a
(segments x 21 days x 48 half-hours) array. Nothing loops over segments or sessions, so the
whole dataset takes seconds. The features use the meter alone (no label, regime or charger
field), so the same code scores any 21-day window of a GB smart meter.

Four groups of five:
- Level and variability.
- Time of day and week: shares of kWh, named <period>_share, plus the weekend/weekday ratio.
- High-power load: day-level features are aggregated as the mean of the segment's 3 highest days
  (one per week).
- Suspected charging events: day-level runs above the household's reference profile.

`FEATURES` describes each feature: its group and definition, as the tables show them.

Conventions (per 21-day segment, GMT): half-hour h = 1..48 covers [(h-1)/2, h/2) hours, and
high-power and event features take the mean of the segment's 3 highest days.
"""
import warnings

import numpy as np
import pandas as pd

SEGMENT_DAYS = 21          # 3 whole weeks
HH_COLS = [f"h{i:02d}" for i in range(48)]
EPS = 1e-3
TOP_K = SEGMENT_DAYS // 7   # one day per week of the segment
HI_KWH = 1.25        # high-power half-hour (2.5 kW)
REF_PCTL = 30        # event detector: per-slot reference profile
EVENT_KWH = 1.2      # event detector: kWh above the reference (2.4 kW)
EVENT_MIN_HH = 2     # event detector: consecutive half-hours

GROUPS = ["Level and variability", "Time of day and week", "High-power load",
          "Suspected charging events"]

FEATURES = [
    # ── level and variability ──
    dict(name="avg_daily_kwh", group="Level and variability",
         definition="Mean daily kWh"),
    dict(name="cv_daily_kwh", group="Level and variability",
         definition="Standard deviation of daily kWh divided by its mean"),
    dict(name="sigma2_delta", group="Level and variability",
         definition="Variance of the change between consecutive half-hourly readings"),
    dict(name="rho_48", group="Level and variability",
         definition="Correlation of each half-hourly reading with the same half-hour on the "
                    "previous day"),
    dict(name="s_peak", group="Level and variability",
         definition="98th percentile of half-hourly kWh divided by the median (plus 0.001 kWh)"),
    # ── time of day and week ──
    dict(name="night_share", group="Time of day and week",
         definition="Share of kWh used 00:00–07:00"),
    dict(name="am_peak_share", group="Time of day and week",
         definition="Share of kWh used 06:00–09:00"),
    dict(name="pm_peak_share", group="Time of day and week",
         definition="Share of kWh used 16:00–20:00"),
    dict(name="late_evening_share", group="Time of day and week",
         definition="Share of kWh used 20:00–24:00"),
    dict(name="weekend_ratio", group="Time of day and week",
         definition="Mean weekend daily kWh divided by mean weekday daily kWh"),
    # ── high-power load (top-3-day mean) ──
    dict(name="q99_hh", group="High-power load",
         definition="99th percentile of half-hourly kWh"),
    dict(name="high_energy_share", group="High-power load",
         definition="Share of kWh in half-hours above 1.25 kWh (2.5 kW)"),
    dict(name="longest_run_1p25", group="High-power load",
         definition="Longest run of half-hours above 1.25 kWh in a day, mean of the 3 highest days"),
    dict(name="max_step_up", group="High-power load",
         definition="Largest rise between consecutive half-hours in a day, mean of the 3 highest days"),
    dict(name="plateau_flatness", group="High-power load",
         definition="Mean of a day's half-hours above 1.25 kWh divided by their maximum "
                    "(0 when none), mean of the 3 highest days"),
    # ── suspected charging events (day-level) ──
    dict(name="event_rate_weekday", group="Suspected charging events",
         definition="Share of weekdays with a suspected charging event"),
    dict(name="event_rate_weekend", group="Suspected charging events",
         definition="Share of weekend days with a suspected charging event"),
    dict(name="event_kwh_per_week", group="Suspected charging events",
         definition="Energy above the reference profile inside events, per week"),
    dict(name="event_hours_median", group="Suspected charging events",
         definition="Median over event days of the day's longest event, in hours"),
    dict(name="onset_concentration", group="Suspected charging events",
         definition="How tightly event start times cluster at one time of day "
                    "(0 = spread out, 1 = always the same half-hour)"),
]
assert len(FEATURES) == 20
FEATURE_NAMES = [f["name"] for f in FEATURES]

DETECTOR_NOTE = (
    "Event detector: ref_h is the 30th percentile of half-hour h over the segment's 21 days. "
    "A half-hour is hot when x_{d,h} - ref_h >= 1.2 kWh; an event is a run of at least 2 hot "
    "half-hours within one day.")


def _runs(mask):
    """Length of the run of True each element belongs to, along the last axis (0 if False)."""
    fwd = np.zeros(mask.shape, np.int16)
    bwd = np.zeros(mask.shape, np.int16)
    n = mask.shape[-1]
    for h in range(n):
        prev = fwd[..., h - 1] if h else 0
        fwd[..., h] = np.where(mask[..., h], prev + 1, 0)
    for h in range(n - 1, -1, -1):
        nxt = bwd[..., h + 1] if h < n - 1 else 0
        bwd[..., h] = np.where(mask[..., h], nxt + 1, 0)
    return np.where(mask, fwd + bwd - 1, 0)


def _top_mean(v, k=TOP_K):
    """Mean of the k largest values along axis 1 (NaN ignored)."""
    s = np.sort(np.where(np.isnan(v), -np.inf, v), axis=1)[:, -k:]
    s = np.where(np.isinf(s), np.nan, s)
    return np.nanmean(s, axis=1)


def _row_corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    n = ok.sum(1)
    ma, mb = a.sum(1) / n, b.sum(1) / n
    da, db = np.where(ok, a - ma[:, None], 0), np.where(ok, b - mb[:, None], 0)
    return (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))


def to_array(wide):
    """(segments x 21 x 48) array, segment ids and weekend flags from the wide table."""
    wide = wide.sort_values(["segment_id", "date"])
    ids, counts = np.unique(wide.segment_id.to_numpy(), return_counts=True)
    assert (counts == SEGMENT_DAYS).all(), f"every segment must have {SEGMENT_DAYS} days"
    X = wide[HH_COLS].to_numpy(np.float64).reshape(len(ids), SEGMENT_DAYS, 48)
    we = (pd.to_datetime(wide.date).dt.dayofweek.to_numpy() >= 5).reshape(len(ids), SEGMENT_DAYS)
    return X, ids, we


def features_from_wide(wide):
    """All 20 features, one row per segment_id. Undefined values (e.g. a ratio over a window
    that reads zero throughout) are NaN, which XGBoost routes natively."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = _features(wide)
    return out.replace([np.inf, -np.inf], np.nan)


def _features(wide):
    X, ids, we = to_array(wide)
    n = len(ids)
    flat = X.reshape(n, -1)
    E = np.nansum(X, axis=2)                                  # daily kWh
    total = np.nansum(flat, axis=1)
    share = lambda a, b: np.nansum(X[:, :, a - 1:b], axis=(1, 2)) / total   # hh_index a..b
    F = {}

    # level and variability
    F["avg_daily_kwh"] = E.mean(1)
    F["cv_daily_kwh"] = E.std(1, ddof=1) / E.mean(1)
    F["sigma2_delta"] = np.nanvar(np.diff(flat, axis=1), axis=1, ddof=1)
    F["rho_48"] = _row_corr(flat[:, 48:], flat[:, :-48])
    q = np.nanpercentile(flat, [50, 98, 99], axis=1)
    F["s_peak"] = q[1] / (q[0] + 0.001)

    # time of day and week
    F["night_share"] = share(1, 14)
    F["am_peak_share"] = share(13, 18)
    F["pm_peak_share"] = share(33, 40)
    F["late_evening_share"] = share(41, 48)
    F["weekend_ratio"] = (np.nanmean(np.where(we, E, np.nan), 1)
                          / np.nanmean(np.where(~we, E, np.nan), 1))

    # high-power load
    hi = np.nan_to_num(X, nan=0.0) > HI_KWH
    F["q99_hh"] = q[2]
    F["high_energy_share"] = np.nansum(np.where(hi, X, 0), axis=(1, 2)) / total
    F["longest_run_1p25"] = _top_mean(_runs(hi).max(2).astype(float))
    F["max_step_up"] = _top_mean(np.nanmax(np.diff(X, axis=2), axis=2))
    cnt = hi.sum(2)
    flatness = np.where(cnt > 0, np.where(hi, X, 0).sum(2) / np.maximum(cnt, 1)
                        / np.maximum(np.where(hi, X, 0).max(2), EPS), 0.0)
    F["plateau_flatness"] = _top_mean(flatness)

    # suspected charging events
    ref = np.nanpercentile(X, REF_PCTL, axis=1)               # (n, 48)
    excess = X - ref[:, None, :]
    hot = np.nan_to_num(excess, nan=-1.0) >= EVENT_KWH
    run = _runs(hot)
    in_event = run >= EVENT_MIN_HH
    event_day = in_event.any(2)
    F["event_rate_weekday"] = np.where(~we, event_day, 0).sum(1) / (~we).sum(1)
    F["event_rate_weekend"] = np.where(we, event_day, 0).sum(1) / we.sum(1)
    F["event_kwh_per_week"] = np.where(in_event, excess, 0).sum((1, 2)) / (SEGMENT_DAYS / 7)
    longest = np.where(event_day, run.max(2) / 2.0, np.nan)
    with warnings.catch_warnings():                           # all-NaN rows: no event days
        warnings.simplefilter("ignore", RuntimeWarning)
        F["event_hours_median"] = np.nanmedian(longest, 1)
    prev = np.concatenate([np.zeros((n, SEGMENT_DAYS, 1), bool), in_event[:, :, :-1]], axis=2)
    start = in_event & ~prev
    ang = 2 * np.pi * np.arange(48) / 48
    k = start.sum((1, 2))
    c = (start * np.cos(ang)).sum((1, 2)) / np.maximum(k, 1)
    s = (start * np.sin(ang)).sum((1, 2)) / np.maximum(k, 1)
    F["onset_concentration"] = np.where(k >= 2, np.hypot(c, s), np.nan)

    out = pd.DataFrame(F, index=pd.Index(ids, name="segment_id"))
    return out[FEATURE_NAMES]
