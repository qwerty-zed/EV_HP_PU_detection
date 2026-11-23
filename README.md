# Detecting heat pumps and EV charging in GB smart meter data with positive-unlabelled learning

This repository holds the code for an MSc Energy Systems dissertation (University of Oxford). It tests whether domestic heat pumps (HPs) and home electric vehicle (EV) charging can be detected from half-hourly household electricity consumption, plus outdoor temperature for HPs, when the available labels are incomplete.

GB records of HPs and EV charger installations cover only part of the installed base, so a detector has to learn from a set of known positives and an unlabelled pool that hides further positives. This is the positive-unlabelled (PU) setting.

The code:

- builds two labelled GB datasets by superposition: the metered load of a real HP (Electrification of Heat) or
  home charger (Electric Nation) is added, half-hour by half-hour, to the load of a real Low Carbon London
  household;
- trains a PU-bagged XGBoost classifier, with a label-free threshold, and compares it with a naive supervised
  classifier and an oracle classifier, while 0–10% of the positives are hidden in the training pool;
- compares the EV classifier with a rule-based charging detector;
- applies the classifiers, unchanged, to households they have never seen.

**Headline results**, at settings close to current GB uptake (F1 restated at a realistic prevalence):

| Technology | Setting | PU-bagged XGBoost | Naive XGBoost | Rule-based detector |
|---|---|---|---|---|
| Heat pumps (household, one year) | 2% contamination, 2% prevalence | F1 0.675 ± 0.048 | 0.516 | - |
| EV charging (21-day segment) | 5% contamination, 5% prevalence | F1 0.753 ± 0.022 | 0.689 | 0.388 |

± is the spread over five runs. All results are on constructed data, not on real owners.

## Repository contents

| File | Purpose |
|---|---|
| `01_lcl_analysis.ipynb` | Low Carbon London: data loading, quality filters, appliance survey, descriptive tables and figures |
| `02_eoh_analysis.ipynb` | Electrification of Heat: half-hourly heat-pump energy, temperature response, descriptive tables and figures |
| `03_en_analysis.ipynb` | Electric Nation: charging sessions, regimes, charging behaviour, descriptive tables and figures |
| `04_hp_linkage_lcl_eoh.ipynb` | Builds (via `create_hp_dataset_v6.py`) and describes the constructed heat-pump dataset |
| `05_ev_linkage_lcl_en.ipynb` | Builds (via `create_ev_dataset_v2.py`) and describes the constructed EV dataset |
| `06_hp_pu_classifier.ipynb` | Heat-pump PU classifier: contamination experiment, results, feature importance, exported models |
| `07_ev_pu_classifier.ipynb` | EV PU classifier: contamination experiment, rule-based comparison, feature importance, exported models |
| `08_hp_transfer_test.ipynb` | Heat-pump classifiers applied to an unseen population, with prevalence estimates |
| `09_ev_detector_on_hp.ipynb` | EV classifiers applied to unseen households with and without a heat pump (false alarms) |
| `create_hp_dataset_v6.py` | Builds the heat-pump dataset (training and transfer sets) and runs its leak audit |
| `create_ev_dataset_v2.py` | Builds the EV dataset (21-day segments) and runs its leakage audits |
| `pu_common.py` | Shared PU learning code: PU bagging, naive and oracle baselines, label-free threshold, model export |
| `household_features.py` | Computes the 20 household heat-pump features |
| `hp_feature_defs.py` | Definitions of the heat-pump features, for the tables |
| `ev_features.py` | Computes and defines the 20 EV segment features |

## Data

The study data are **not included** in this repository and must be downloaded from their providers under their
own licences. Place them in a `Data/` folder in the repository root (this folder is git-ignored).

| Folder | Dataset | Source |
|---|---|---|
| `Data/LCL/` | J. R. Schofield, R. Carmichael, S. Tindemans, M. Bilton, W. Matt and G. Strbac, *Low Carbon London Project: Data from the Dynamic Time-of-Use Electricity Pricing Trial, 2013*, UK Data Service, 2016. [doi:10.5255/UKDA-SN-7857-2](https://doi.org/10.5255/UKDA-SN-7857-2) | [UK Data Service study 7857](https://datacatalogue.ukdataservice.ac.uk/studies/study/7857#doi) |
| `Data/EoH/` | Energy Systems Catapult, *Electrification of Heat Demonstration Project: Heat Pump Performance Cleansed Data, 2020-2023*, UK Data Service, 2024. [doi:10.5255/UKDA-SN-9050-2](https://doi.org/10.5255/UKDA-SN-9050-2) | [UK Data Service study 9050](https://datacatalogue.ukdataservice.ac.uk/studies/study/9050#details) |
| `Data/EN/` | National Grid Electricity Distribution, *Electric Nation* | [NGED Connected Data Portal](https://connecteddata.nationalgrid.co.uk/dataset/electric-nation) |

The code expects these files:

```
Data/
├── LCL/csv/data_collection/data_tables/
│   ├── consumption_n.csv          # arm N (standard tariff), half-hourly kWh
│   ├── consumption_d.csv          # arm D (dynamic time-of-use tariff), half-hourly kWh
│   ├── survey_answers.csv         # appliance survey
│   └── survey_questions.csv       # note this is out of order by 1 compared to the answers
├── EoH/
│   ├── eoh_cleaned_half_hourly.csv                     # half-hourly heat-pump readings
│   └── 9bbf3b4f-5b33-4652-b385-d69ba68004d0.csv        # installation record
└── EN/
    ├── CrowdChargeMeterValues.csv
    ├── GreenFlux15Minute.csv
    └── customer_survey_results.csv
```

**Weather.** `create_hp_dataset_v6.py` downloads London hourly temperature for 2013 from the
[Open-Meteo](https://open-meteo.com/) historical weather API the first time it runs, so that step needs an internet
connection. Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Requirements

The code was run with Python 3.12 and these packages:

| Package | Version |
|---|---|
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| duckdb | 1.5.3 |
| pyarrow | 24.0.0 |
| xgboost | 1.7.6 |
| scikit-learn | 1.5.1 |
| shap | 0.49.1 |
| joblib | 1.4.2 |
| matplotlib | 3.9.2 |

The dataset builds use DuckDB with a 6 GB memory limit, and spill to a temporary folder beyond that.

## How to run

Start Jupyter in the repository root (all paths are relative to it) and **run the notebooks 01 to 09 in order**.

To work on one technology only, run notebook 01, which both need, and then that technology's notebooks in order:

| Technology | Notebooks |
|---|---|
| Heat pumps | 01, then the even notebooks: 02, 04, 06, 08 |
| EV charging | 01, then the odd notebooks: 03, 05, 07, 09 |

Notebook 09 applies the EV classifiers to heat-pump households, so it also needs the heat-pump dataset built by
notebooks 02 and 04. Without them, the EV notebooks run through notebook 07.

- Notebooks 01–03 write cached tables to `cache/`, which the dataset builds read.
- Notebooks 04 and 05 each start by running their build script (`create_hp_dataset_v6.py`,
  `create_ev_dataset_v2.py`) if its outputs are not there yet, and show its log. Set `REBUILD = True` in that cell
  to build again. The first run of notebook 04 needs an internet connection for the weather download.
- Notebooks 06 and 07 export the trained classifiers to `HP_classifier/` and `EV_classifier/`, which notebooks
  08 and 09 apply. Notebook 08 also reads a table written by 06, and notebook 09 one written by 07.

Random seeds are fixed (test split seed 42; runs use seeds 0–4), so reruns reproduce the reported results.

## Outputs

| Folder | Contents | In git |
|---|---|---|
| `cache/` | Cached tables from notebooks 01–03 | No (derived from licensed data) |
| `LCL_EoH_join_v6/` | Constructed heat-pump dataset (training set and `transfer_armD_ovo/`), build log, leak audit | No |
| `LCL_EN_join_v2/` | Constructed EV dataset, build log, leakage audits | No |
| `HP_classifier/` | Exported heat-pump classifiers `hp_pu_classifier_alpha2_seed0.json` to `seed4.json`, from notebook 06 | No (about 40–55 MB each) |
| `EV_classifier/` | Exported EV classifiers `ev_pu_classifier_alpha5_seed0.json` to `seed4.json`, from notebook 07 | No (about 40–70 MB each) |
| `figures/` | Figures, as PNG | Yes |
| `tables/` | Tables, as CSV | Yes |

Figures and tables are named by notebook:

| Prefix | Notebook | Figures | Tables | Supplementary tables |
|---|---|---|---|---|
| `LCL` | 01 | `LCL1_…` | `LCL_T1_…` | `LCL_S_…` |
| `EoH` | 02 | `EoH1_…` | `EoH_T1_…` | |
| `EN` | 03 | `EN1_…` | `EN_T1_…` | `EN_S_…` |
| `HP_J` | 04 (heat-pump join) | `HP_J1_…` | `HP_J_T1_…` | |
| `EV_J` | 05 (EV join) | `EV_J1_…` | `EV_J_T1_…` | `EV_J_S_…` |
| `HP_PU` | 06 (heat-pump PU classifier) | `HP_PU1_…` | `HP_PU_T1_…` | `HP_PU_S_…` |
| `EV_PU` | 07 (EV PU classifier) | `EV_PU1_…` | `EV_PU_T1_…` | `EV_PU_S_…` |
| `HP_T` | 08 (heat-pump transfer test) | `HP_T1_…` | `HP_T_T1_…` | |
| `EV_T` | 09 (EV classifier on unseen households) | `EV_T1_…` | `EV_T_T1_…` | |

## Method in brief

- **Unit.** Heat pumps: a household over calendar 2013. EV charging: a 21-day segment of a household's load.
- **Features.** 20 engineered features per technology: for heat pumps, temperature response, seasonal level, time of
  day and variability; for EVs, level and variability, time of day and week, high-power load and suspected charging
  events.
- **Methods.** PU-bagged XGBoost (25 bags), against a naive XGBoost classifier that treats the unlabelled pool as
  negative and an oracle XGBoost classifier trained on true negatives.
- **Threshold.** Chosen without labels by a PU-adapted F1 criterion on five-fold cross-fitted scores.
- **Contamination.** 0, 2, 5 and 10% of the unlabelled pool are hidden positives, each with five seeds.
- **Evaluation.** A fixed held-out test set; precision and F1 are also restated at 5% and 2% prevalence from recall
  and the false-positive rate.

## Limitations

- Every positive is constructed by adding real HP or charger load to a different real household; detection has not
  been tested on real owners.
- The source data come from different periods (LCL 2011–14, Electric Nation 2017–18, EoH 2020–23).
- Low Carbon London excluded Economy 7 homes, so storage heating, a likely confounder, is absent.
- EV detection is of home charging within a 21-day window, not of EV ownership.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The licence covers the code only. The study data and
anything derived from them, including `cache/` and the constructed datasets, remain under the data providers'
terms.

## Acknowledgements

The work uses data from the Low Carbon London project (UK Data Service), the Electrification of Heat Demonstration Project (Energy Systems Catapult, UK Data Service), the Electric Nation project (National Grid Electricity Distribution) and Open-Meteo.
