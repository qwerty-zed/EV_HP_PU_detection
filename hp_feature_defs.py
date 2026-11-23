"""Definitions of the 20 household heat-pump features, for the notebook tables.

The features themselves are computed by household_features.py (`features_from_long`). This
module only describes them: one entry per feature with its group and a plain-language
definition, which are what the tables show.

Conventions (household level, calendar 2013, GMT): winter is December to February and summer
June to August; heating degree-days use a 15.5 °C base on the London daily mean temperature;
half-hour h = 1..48 covers [(h-1)/2, h/2) hours, so h = 13 is 06:00-06:30.
"""

# household_features.py and hp_metadata.parquet keep the old column names, because older deployment
# models read them. Notebook 06 applies this on load; apply it too before scoring with the model.
SHARE_RENAME = {"r_night": "night_share", "i_ampk": "am_peak_share", "i_pmpk": "pm_peak_share"}

GROUPS = ["Temperature response", "Seasonal level", "Time of day", "Variability"]

FEATURES = [
    # ── temperature response ──
    dict(name="r_hdd", group="Temperature response",
         definition="Pearson correlation of daily kWh with heating degree-days (base 15.5 °C), "
                    "over all days"),
    dict(name="beta_temp", group="Temperature response",
         definition="Least-squares slope of daily kWh on daily mean temperature, over days "
                    "below 15 °C (kWh per °C; negative when use rises in the cold)"),
    dict(name="delta_cold", group="Temperature response",
         definition="Mean daily kWh on the household's coldest 5% of winter days minus its "
                    "winter mean daily kWh"),
    dict(name="cov_t_x", group="Temperature response",
         definition="Covariance of half-hourly kWh with half-hourly temperature, over winter "
                    "readings (population covariance)"),
    dict(name="consumption_per_hdd", group="Temperature response",
         definition="Total kWh on days with any heating demand divided by total heating "
                    "degree-days"),
    # ── seasonal level ──
    dict(name="r_ws", group="Seasonal level",
         definition="Winter mean daily kWh divided by summer mean daily kWh"),
    dict(name="avg_daily_kwh", group="Seasonal level",
         definition="Mean daily kWh over all days"),
    dict(name="winter_avg_kwh", group="Seasonal level",
         definition="Mean daily kWh over winter days (Dec, Jan, Feb)"),
    dict(name="summer_avg_kwh", group="Seasonal level",
         definition="Mean daily kWh over summer days (Jun, Jul, Aug)"),
    dict(name="base_night", group="Seasonal level",
         definition="10th percentile of half-hourly kWh at 00:30–05:00 in winter minus the same "
                    "in summer"),
    # ── time of day ──
    dict(name="am_peak_share", group="Time of day",
         definition="Share of winter kWh used 06:00–09:00"),
    dict(name="pm_peak_share", group="Time of day",
         definition="Share of winter kWh used 16:00–20:00"),
    dict(name="night_share", group="Time of day",
         definition="Share of all kWh used 00:00–07:00, over the whole year"),
    dict(name="d_midday", group="Time of day",
         definition="Summer mean kWh in the half-hours 07:30–08:00 and 20:30–21:00 minus summer "
                    "mean kWh 11:30–14:00"),
    dict(name="base_summer", group="Time of day",
         definition="5th percentile of half-hourly kWh at 09:30–16:00 in summer"),
    # ── variability ──
    dict(name="sigma2_delta", group="Variability",
         definition="Variance of the change between consecutive winter half-hourly readings"),
    dict(name="s_winter", group="Variability",
         definition="98th percentile of winter half-hourly kWh divided by the winter median "
                    "(plus 0.001 kWh)"),
    dict(name="rho_48", group="Variability",
         definition="Correlation of each half-hourly reading with the reading 48 places earlier "
                    "(the same half-hour on the previous day), over the whole year"),
    dict(name="cv_daily_kwh", group="Variability",
         definition="Standard deviation of daily kWh divided by its mean"),
    dict(name="f_zero", group="Variability",
         definition="Share of half-hourly readings at or below 0.005 kWh"),
]

assert len(FEATURES) == 20
