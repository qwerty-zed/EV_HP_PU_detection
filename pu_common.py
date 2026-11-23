"""Shared positive-unlabelled (PU) learning code for the thesis notebooks.

Used by notebooks 06 to 09. Three methods are compared:

- PU-bagged XGBoost (Mordelet & Vert 2014): the main method.
- Naive XGBoost: the unlabelled pool treated as negative. The baseline.
- Oracle XGBoost: labelled positives against TRUE negatives. The ceiling.

Everything that picks a hyperparameter or a threshold for the PU method uses only labelled
positives and unlabelled households, never a true label. Scores are cross-fitted: each
training household is scored by the fold model that did not see it, and a new household is
scored by the mean of the fold models, so a threshold chosen on out-of-fold scores carries over
without refitting.

Folds are stratified by the label. When `groups` is given (EV: charger for positives, household
for negatives), folds are also grouped, so no charger or household is split across a fold's
training and scoring sides. Without `groups` the behaviour is unchanged (HP).
"""
import base64
import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import Booster, XGBClassifier

N_FOLDS = 5
N_BAGS = 25
BAG_GRID = [dict(max_depth=d, u_ratio=k) for d in (3, 5) for k in (1, 3)]
XGB_BASE = dict(n_estimators=250, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=2, reg_lambda=1.0, n_jobs=1, verbosity=0)
SUPERVISED_DEPTH = 4     # naive and oracle XGBoost, as in the pilot


# ── thresholds and the prevalence estimate ─────────────────────────────────────────
def best_f1_threshold(y, s):
    """Threshold maximising F1 of scores s against labels y."""
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(th[i]), float(f1[i])


def pu_f1_threshold(s_lab, oof, min_recall=0.2):
    """Label-free threshold maximising an ESTIMATED F1 on the unlabelled pool.

    For threshold t, recall r(t) is measured on the labelled positives and the unlabelled flag
    rate is f(t) = pi*r(t) + (1-pi)*FPR(t) >= pi*r(t). So pi_hat = min_t f(t)/r(t) bounds the
    hidden prevalence from above, tightly where FPR -> 0 (Blanchard et al. 2010), restricted to
    r >= min_recall to avoid the noisy tail. Then precision(t) = pi_hat*r/f and
    F1_hat = 2pr/(p+r).

    The unlabelled flag count is add-one smoothed: without it, a tail where the strongest
    labelled positives outrank every unlabelled household gives pi_hat = 0 and a rule that
    flags nobody (pilot, alpha <= 1%). Disclosed as a design choice.

    Returns (threshold, pi_hat, estimated F1 at the threshold).
    """
    pos = np.sort(oof[s_lab == 1])
    u = np.sort(oof[s_lab == 0])
    cand = np.unique(pos)
    r = 1 - np.searchsorted(pos, cand, side="left") / len(pos)
    f = 1 - np.searchsorted(u, cand, side="left") / len(u)
    ok = r >= min_recall
    f_bound = (f * len(u) + 1) / len(u)
    pi_hat = float(np.min(np.where(ok, f_bound / np.maximum(r, 1e-12), np.inf)))
    pi_use = max(pi_hat, 1.0 / len(u))
    prec = np.where(f > 0, np.clip(pi_use * r / np.maximum(f, 1e-12), 0, 1), 1.0)
    f1 = 2 * prec * r / np.maximum(prec + r, 1e-12)
    i = int(np.argmax(f1))
    return float(cand[i]), pi_hat, float(f1[i])


# ── metrics ────────────────────────────────────────────────────────────────────
def score_row(y, s, thr):
    yh = (s >= thr).astype(int)
    neg = y == 0
    return {"threshold": thr,
            "ROC_AUC": roc_auc_score(y, s), "PR_AUC": average_precision_score(y, s),
            "F1": f1_score(y, yh, zero_division=0),
            "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "FPR": float(yh[neg].mean()) if neg.any() else np.nan,
            "flag_rate": float(yh.mean()),
            "F1_best_possible": best_f1_threshold(y, s)[1]}


def restate(recall, fpr, prevalence):
    """Precision and F1 a classifier with this recall and FPR would have at another prevalence."""
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = 2 * p * recall / (p + recall) if p + recall > 0 else 0.0
    return p, f1


# ── folds ──────────────────────────────────────────────────────────────────────
def make_folds(X, y, seed, groups=None):
    """Stratified folds; also grouped when `groups` is given."""
    if groups is None:
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
                .split(X, y, groups))


# ── models ─────────────────────────────────────────────────────────────────────
class PUBaggingXGB:
    """Mordelet & Vert PU bagging with an XGBoost base learner: each bag trains on all labelled
    positives against a bootstrap sample of u_ratio x |L| unlabelled households. The score is
    the mean margin over bags."""

    def __init__(self, max_depth, u_ratio, n_bags=N_BAGS, seed=0):
        self.max_depth, self.u_ratio, self.n_bags, self.seed = max_depth, u_ratio, n_bags, seed

    def fit(self, X, s):
        rng = np.random.RandomState(self.seed)
        P, U = np.flatnonzero(s == 1), np.flatnonzero(s == 0)
        k = min(len(U), int(self.u_ratio * len(P)))
        self.models = []
        for b in range(self.n_bags):
            ub = rng.choice(U, k, replace=True)
            idx = np.r_[P, ub]
            m = XGBClassifier(**XGB_BASE, max_depth=self.max_depth,
                              scale_pos_weight=k / len(P), random_state=self.seed + b)
            m.fit(X[idx], np.r_[np.ones(len(P)), np.zeros(k)])
            self.models.append(m)
        return self

    def decision_function(self, X):
        return np.mean([m.predict(X, output_margin=True) for m in self.models], axis=0)


def _bag_fold(Xtr, s_tr, Xev, Xte, params, seed):
    m = PUBaggingXGB(**params, seed=seed).fit(Xtr, s_tr)
    return m.decision_function(Xev), m.decision_function(Xte), m


def _supervised(X, y, seed):
    return XGBClassifier(**{**XGB_BASE, "n_jobs": -1}, max_depth=SUPERVISED_DEPTH,
                         scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                         random_state=seed).fit(X, y)


def _crossfit_supervised(X, y, Xte, seed, groups=None):
    folds = make_folds(X, y, seed, groups)
    oof, te = np.zeros(len(X)), np.zeros(len(Xte))
    for tr, ev in folds:
        m = _supervised(X[tr], y[tr], seed)
        oof[ev] = m.predict(X[ev], output_margin=True)
        te += m.predict(Xte, output_margin=True) / N_FOLDS
    return oof, te


def run_pu_bagging(X, s, Xte, seed, n_jobs=-1, groups=None):
    """Cross-fitted PU bagging over BAG_GRID. The configuration and the threshold are both
    chosen by the label-free PU-F1 criterion on out-of-fold scores.

    Returns a dict with the chosen params, threshold, pi_hat, the out-of-fold and test scores
    and the fold models (5 folds x N_BAGS boosters)."""
    folds = make_folds(X, s, seed, groups)
    jobs = [(gi, fi) for gi in range(len(BAG_GRID)) for fi in range(N_FOLDS)]
    outs = Parallel(n_jobs=n_jobs)(delayed(_bag_fold)(
        X[folds[fi][0]], s[folds[fi][0]], X[folds[fi][1]], Xte, BAG_GRID[gi], seed)
        for gi, fi in jobs)
    oof = np.zeros((len(BAG_GRID), len(X)))
    te = np.zeros((len(BAG_GRID), len(Xte)))
    fold_models = {gi: [] for gi in range(len(BAG_GRID))}
    for (gi, fi), (o, t, m) in zip(jobs, outs):
        oof[gi, folds[fi][1]] = o
        te[gi] += t / N_FOLDS
        fold_models[gi].append(m)
    sel = [pu_f1_threshold(s, oof[gi]) for gi in range(len(BAG_GRID))]
    gi = int(np.argmax([f1_hat for _, _, f1_hat in sel]))
    thr, pi_hat, f1_hat = sel[gi]
    return {"params": BAG_GRID[gi], "threshold": thr, "pi_hat": pi_hat, "f1_hat": f1_hat,
            "oof": oof[gi], "test": te[gi], "models": fold_models[gi]}


def run_one(X_train, y_train, s_train, X_test, y_test, seed, n_jobs=-1, keep=False,
            groups=None):
    """One (alpha, seed) run of all three methods.

    X_train holds the labelled positives and the unlabelled pool U. s_train is 1 for labelled
    positives and 0 for U. y_train is the truth (1 for labelled and hidden positives), used only
    by the oracle and for reporting. Returns (rows, kept) where kept holds the PU-bagging
    output when keep=True. `groups` (optional) groups the cross-fitting folds."""
    rows = []
    pu = run_pu_bagging(X_train, s_train, X_test, seed, n_jobs=n_jobs, groups=groups)
    rows.append({"method": "PU-bagged XGBoost", **score_row(y_test, pu["test"], pu["threshold"]),
                 "pi_hat": pu["pi_hat"], "max_depth": pu["params"]["max_depth"],
                 "u_ratio": pu["params"]["u_ratio"]})

    oof_n, te_n = _crossfit_supervised(X_train, s_train, X_test, seed, groups)
    thr_n = best_f1_threshold(s_train, oof_n)[0]
    rows.append({"method": "Naive XGBoost", **score_row(y_test, te_n, thr_n)})

    clean = ~((s_train == 0) & (y_train == 1))       # hidden positives removed
    oof_o, te_o = _crossfit_supervised(X_train[clean], y_train[clean], X_test, seed,
                                       None if groups is None else groups[clean])
    thr_o = best_f1_threshold(y_train[clean], oof_o)[0]
    rows.append({"method": "Oracle XGBoost", **score_row(y_test, te_o, thr_o)})
    if keep:
        pu = {**pu, "naive_test": te_n, "naive_threshold": thr_n}
    return rows, (pu if keep else None)


# ── export and reload ──────────────────────────────────────────────────────────
def export_ensemble(path, fold_models, features, threshold, meta):
    """Write a cross-fitted PU-bagging ensemble to one JSON file.

    Each booster is stored in XGBoost's binary UBJSON form, base64-encoded. The score of a
    household is the mean margin over every booster (equal bags per fold, so this equals the
    mean of the fold-model scores), and `threshold` applies to that score."""
    boosters = []
    for fm in fold_models:
        for m in fm.models:
            raw = m.get_booster().save_raw(raw_format="ubj")
            boosters.append(base64.b64encode(bytes(raw)).decode("ascii"))
    doc = {"format": "pu_bagged_xgboost_crossfitted_v1",
           "score": "mean XGBoost margin over all boosters; flag when score >= threshold",
           "features": list(features), "threshold": float(threshold),
           "n_boosters": len(boosters), "xgboost_booster_format": "ubj, base64",
           **meta, "boosters": boosters}
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return doc


class ExportedPUModel:
    """Reload an exported ensemble and score a feature table with the same columns."""

    def __init__(self, path):
        with open(path) as fh:
            doc = json.load(fh)
        self.meta = {k: v for k, v in doc.items() if k != "boosters"}
        self.features, self.threshold = doc["features"], doc["threshold"]
        self.boosters = []
        for b64 in doc["boosters"]:
            b = Booster()
            b.load_model(bytearray(base64.b64decode(b64)))
            self.boosters.append(b)

    def decision_function(self, df):
        from xgboost import DMatrix
        d = DMatrix(df[self.features].to_numpy(dtype=float), missing=np.nan)
        return np.mean([b.predict(d, output_margin=True) for b in self.boosters], axis=0)

    def predict(self, df):
        return (self.decision_function(df) >= self.threshold).astype(int)
