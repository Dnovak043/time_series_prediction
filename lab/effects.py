# -*- coding: utf-8 -*-
"""
Effects of features on models: out-of-sample evaluation and ablation.

Validation is time-respecting: walk-forward over days (train on earlier
days, test on the next block of days). A single-day panel falls back to a
chronological split inside the day. Scaling is fit on the train fold only.

    evaluate(panel, features, target, model="ridge")   one feature set
    ablate(panel, {"name": [features]}, target)         compare sets
    drop_one(panel, features, target)                   leave-one-feature-out
    permutation_importance(...)                         on the last fold
    feature_sets(...)                                   sets from metric/causal results
                                                        + size-matched random controls

task="regression"  target as-is: R2 (OOS), IC, rank IC, hit rate on moves
task="direction"   sign(target) in {-1,0,+1} (the CLS classes): accuracy,
                   balanced accuracy, log loss, and the majority-class baseline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import data as _data

MODELS = ("ridge", "logit", "hgb", "rf")


def make_model(model, task: str):
    if not isinstance(model, str):
        return model() if callable(model) and not hasattr(model, "fit") else model
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    reg = task == "regression"
    if model == "ridge" or (model == "logit" and reg):
        from sklearn.linear_model import RidgeCV
        est = RidgeCV(alphas=np.logspace(-3, 3, 7))
    elif model == "logit":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=500)
    elif model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        est = (HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier)(
            max_iter=200, learning_rate=0.05, early_stopping=True, random_state=0)
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        est = (RandomForestRegressor if reg else RandomForestClassifier)(
            n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=0)
    else:
        raise ValueError(f"model must be one of {MODELS} or an sklearn estimator")
    return make_pipeline(StandardScaler(), est)


def folds(panel: pd.DataFrame, n_splits: int = 3, min_train_days: int | None = None):
    """Yield (train_idx, test_idx) walk-forward over sorted days."""
    days = sorted(panel["day"].unique())
    if len(days) == 1:
        n = len(panel)
        for f in range(n_splits):
            cut = int(n * (0.5 + 0.5 * f / n_splits))
            end = int(n * (0.5 + 0.5 * (f + 1) / n_splits))
            yield np.arange(cut), np.arange(cut, end)
        return
    m = min_train_days or max(1, len(days) // 2)
    n_splits = min(n_splits, len(days) - m)
    rest = days[m:]
    chunks = np.array_split(np.arange(len(rest)), n_splits)
    for ch in chunks:
        first = rest[ch[0]]
        test_days = [rest[i] for i in ch]
        tr = np.flatnonzero((panel["day"] < first).to_numpy())
        te = np.flatnonzero(panel["day"].isin(test_days).to_numpy())
        yield tr, te


def _design(panel, features, target, lags):
    lags = list(lags)
    extra = [l for l in lags if l > 0]
    if extra:
        L = _data.lagged(panel.reset_index(drop=True), features, extra,
                         keep=[*features, target])
        cols = list(features) + [f"{c}_L{k}" for k in extra for c in features]
        if 0 not in lags:
            cols = cols[len(features):]
        P = L
    else:
        P, cols = panel, list(features)
    P = _data.clean(P, cols + [target])
    return P, cols


def _metrics(task, y, pred, proba=None, classes=None, y_train=None):
    from sklearn import metrics as M
    if task == "regression":
        mv = y != 0
        return dict(
            r2=1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y_train)) ** 2),
            IC=np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else np.nan,
            rank_IC=pd.Series(y).corr(pd.Series(pred), method="spearman"),
            hit_rate=np.mean(np.sign(pred[mv]) == np.sign(y[mv])) if mv.any() else np.nan,
        )
    maj = pd.Series(y_train).mode()[0]
    out = dict(acc=M.accuracy_score(y, pred),
               bal_acc=M.balanced_accuracy_score(y, pred),
               baseline_acc=np.mean(y == maj))
    if proba is not None:
        out["log_loss"] = M.log_loss(y, proba, labels=classes)
    return out


@dataclass
class Result:
    name: str
    features: list[str]
    per_fold: pd.DataFrame
    predictions: pd.DataFrame = field(repr=False)
    model: object = field(repr=False, default=None)

    @property
    def mean(self) -> pd.Series:
        return self.per_fold.mean(numeric_only=True)


def evaluate(panel: pd.DataFrame, features: Sequence[str], target: str,
             model="ridge", task: str = "regression", lags: Sequence[int] = (0,),
             n_splits: int = 3, max_train_rows: int | None = 1_000_000,
             name: str = "") -> Result:
    features = list(features)
    P, cols = _design(panel, features, target, lags)
    y_all = P[target].to_numpy(float)
    if task == "direction":
        y_all = np.sign(y_all).astype(int)
    rows, preds, est = [], [], None
    for f, (tr, te) in enumerate(folds(P, n_splits)):
        if max_train_rows and len(tr) > max_train_rows:
            tr = tr[-max_train_rows:]  # most recent rows
        est = make_model(model, task)
        Xtr, Xte = P[cols].to_numpy(float)[tr], P[cols].to_numpy(float)[te]
        est.fit(Xtr, y_all[tr])
        pred = est.predict(Xte)
        proba = est.predict_proba(Xte) if task == "direction" and hasattr(est, "predict_proba") else None
        m = _metrics(task, y_all[te], pred, proba,
                     getattr(est, "classes_", None), y_all[tr])
        m.update(fold=f, n_train=len(tr), n_test=len(te),
                 test_days=",".join(sorted(P["day"].iloc[te].unique())))
        rows.append(m)
        preds.append(pd.DataFrame(dict(day=P["day"].iloc[te].to_numpy(),
                                       y=y_all[te], pred=pred, fold=f)))
    return Result(name or f"{len(features)} features", features, pd.DataFrame(rows),
                  pd.concat(preds, ignore_index=True), est)


def ablate(panel: pd.DataFrame, feature_sets: dict[str, Sequence[str]], target: str,
           reference: str | None = None, verbose: bool = True, **kw) -> pd.DataFrame:
    """
    Evaluate each named feature set with identical folds/model. Returns one row
    per set (mean over folds, plus `<metric>_sd`); if `reference` names a set,
    `Δ<metric>` columns give the difference to it.
    """
    res = {}
    for name, feats in feature_sets.items():
        if not feats:
            continue
        if verbose:
            print(f"· {name} ({len(feats)})", flush=True)
        res[name] = evaluate(panel, feats, target, name=name, **kw)
    rows = []
    for name, r in res.items():
        num = r.per_fold.drop(columns=["fold", "n_train", "n_test"], errors="ignore")\
                        .select_dtypes("number")
        row = {"set": name, "n_features": len(r.features)}
        row.update(num.mean().to_dict())
        row.update({f"{k}_sd": v for k, v in num.std().to_dict().items()})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("set")
    if reference and reference in out.index:
        for k in [c for c in out.columns if not c.endswith("_sd") and c != "n_features"]:
            out[f"Δ{k}"] = out[k] - out.loc[reference, k]
    out.attrs["results"] = res
    return out


def drop_one(panel, features: Sequence[str], target: str, **kw) -> pd.DataFrame:
    sets = {"all": list(features)}
    sets.update({f"−{f}": [g for g in features if g != f] for f in features})
    return ablate(panel, sets, target, reference="all", **kw)


def permutation_importance(panel, features: Sequence[str], target: str, model="ridge",
                           task="regression", n_repeats: int = 5, n_splits: int = 3,
                           seed: int = 0) -> pd.DataFrame:
    """Score drop when one feature is shuffled on the LAST walk-forward fold
    (score = R2 for regression, balanced accuracy for direction)."""
    features = list(features)
    P, cols = _design(panel, features, target, (0,))
    y = P[target].to_numpy(float)
    if task == "direction":
        y = np.sign(y).astype(int)
    tr, te = list(folds(P, n_splits))[-1]
    X = P[cols].to_numpy(float)
    est = make_model(model, task).fit(X[tr], y[tr])

    def score(Xt):
        m = _metrics(task, y[te], est.predict(Xt), y_train=y[tr])
        return m["r2"] if task == "regression" else m["bal_acc"]

    base = score(X[te])
    rng = np.random.default_rng(seed)
    rows = []
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - score(Xp))
        rows.append(dict(feature=c, importance=np.mean(drops), sd=np.std(drops)))
    return pd.DataFrame(rows).sort_values("importance", ascending=False).set_index("feature")


def feature_sets(features: Sequence[str], *, pc_graph=None, target: str | None = None,
                 pcmci_links: pd.DataFrame | None = None, pcmci_effect: str | None = None,
                 table: pd.DataFrame | None = None, top_k: int = 5,
                 granger_alpha: float = 0.01, n_random: int = 3,
                 seed: int = 0) -> dict[str, list[str]]:
    """
    Candidate feature sets from earlier analysis, plus controls:
        all, PC parents / Markov blanket of target, PCMCI lagged parents,
        Granger-significant, top-k by MI / |IC|, and `n_random` random sets the
        size of the PC-parent set (does causal selection beat chance?).
    """
    from . import causal
    features = list(features)
    sets = {"all": features}
    if pc_graph is not None and target:
        sets["pc_parents"] = [f for f in pc_graph.parents(target, include_undirected=True)
                              if f in features]
        sets["pc_blanket"] = [f for f in pc_graph.markov_blanket(target) if f in features]
    if pcmci_links is not None and pcmci_effect:
        sets["pcmci_parents"] = [f for f in causal.lagged_parents(pcmci_links, pcmci_effect)
                                 if f in features]
    if table is not None:
        if "granger_p" in table:
            sets["granger_sig"] = table.index[table.granger_p < granger_alpha].tolist()
        if "MI" in table:
            sets[f"top{top_k}_MI"] = table.MI.nlargest(top_k).index.tolist()
        if "IC" in table:
            sets[f"top{top_k}_|IC|"] = table.IC.abs().nlargest(top_k).index.tolist()
    k = len(sets.get("pc_parents") or []) or top_k
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        sets[f"random{k}_{r}"] = rng.choice(features, size=min(k, len(features)),
                                            replace=False).tolist()
    return {n: s for n, s in sets.items() if s}
