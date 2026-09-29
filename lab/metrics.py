# -*- coding: utf-8 -*-
"""
Dependence metrics among features and toward targets.

All lagged quantities are computed within days (see data.lagged). Tidy
DataFrames are returned; `to_matrix` pivots a (cause, effect, value) table.

    corr_matrix        Pearson / Spearman / Kendall
    mi_matrix          kNN mutual information (nats), symmetric
    xcorr              corr(x_{t-k}, y_t) for k in [-L, L]
    granger            pooled OLS Granger F-tests, pairwise or conditional
    transfer_entropy   discrete TE on symbol streams (bits) + surrogate p-values
    vif                variance inflation factors (multicollinearity)
    target_table       one row per feature: IC, rank IC, MI, Granger p, best lag
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import data as _data


def to_matrix(tidy: pd.DataFrame, value: str, rows="cause", cols="effect") -> pd.DataFrame:
    order = list(dict.fromkeys(tidy[rows].tolist() + tidy[cols].tolist()))
    return tidy.pivot(index=rows, columns=cols, values=value).reindex(index=order, columns=order)


# -- contemporaneous ------------------------------------------------------------

def corr_matrix(panel: pd.DataFrame, cols: Sequence[str], method: str = "pearson") -> pd.DataFrame:
    return _data.clean(panel, cols)[list(cols)].corr(method=method)


def mi_matrix(panel: pd.DataFrame, cols: Sequence[str], max_rows: int = 20000,
              n_neighbors: int = 3, seed: int = 0) -> pd.DataFrame:
    """Symmetrized kNN MI (sklearn), on a random row sample of size max_rows."""
    from sklearn.feature_selection import mutual_info_regression

    cols = list(cols)
    df = _data.clean(panel, cols)[cols]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    X = _data.standardize(df, cols)[cols].to_numpy()
    X = X + 1e-10 * np.random.default_rng(seed).standard_normal(X.shape)  # break ties
    M = np.zeros((len(cols), len(cols)))
    for j in range(len(cols)):
        M[:, j] = mutual_info_regression(X, X[:, j], n_neighbors=n_neighbors,
                                         random_state=seed)
    M = (M + M.T) / 2
    return pd.DataFrame(M, index=cols, columns=cols)


def vif(panel: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), R^2 of col j regressed on the rest."""
    X = _data.standardize(_data.clean(panel, cols), cols)[list(cols)].to_numpy()
    out = {}
    for j, c in enumerate(cols):
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        r2 = 1 - np.var(X[:, j] - others @ beta) / np.var(X[:, j])
        out[c] = 1 / max(1 - r2, 1e-12)
    return pd.Series(out, name="VIF").sort_values(ascending=False)


# -- lead / lag -------------------------------------------------------------------

def xcorr(panel: pd.DataFrame, x: str, y: str, max_lag: int = 10,
          method: str = "pearson") -> pd.Series:
    """corr(x_{t-k}, y_t); k>0 means x leads y. Within-day shifts."""
    g = panel.groupby("day")
    out = {}
    for k in range(-max_lag, max_lag + 1):
        xs = g[x].shift(k)
        m = xs.notna() & panel[y].notna() & np.isfinite(xs) & np.isfinite(panel[y])
        a, b = xs[m], panel[y][m]
        out[k] = (stats.spearmanr(a, b)[0] if method == "spearman"
                  else np.corrcoef(a, b)[0, 1])
    return pd.Series(out, name=f"{x}->{y}").rename_axis("lag")


def _lag_block(panel, cols, max_lag):
    """(Y_now [n×p], LAGS {col: [n×max_lag]}) aligned, NaN rows dropped."""
    lag = _data.lagged(panel, cols, range(1, max_lag + 1), keep=cols)
    Y = lag[list(cols)].to_numpy(float)
    L = {c: lag[[f"{c}_L{k}" for k in range(1, max_lag + 1)]].to_numpy(float) for c in cols}
    return Y, L


def _rss(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    return float(r @ r), X1.shape[1]


def granger(panel: pd.DataFrame, cols: Sequence[str], max_lag: int = 5,
            conditional: bool = False, max_rows: int | None = 500_000) -> pd.DataFrame:
    """
    Granger F-test for every ordered pair (cause -> effect), pooled over days.

    pairwise     restricted = effect's own lags; unrestricted adds cause's lags
    conditional  restricted = lags of *all other* variables; tests whether the
                 cause helps beyond everything else in `cols` (a VAR Granger test)
    """
    cols = list(cols)
    P = _data.subsample(_data.clean(panel, cols), max_rows)
    Y, L = _lag_block(P, cols, max_lag)
    n = len(Y)
    rows = []
    for j, eff in enumerate(cols):
        y = Y[:, j]
        for cause in cols:
            if cause == eff:
                continue
            base = [L[eff]] + ([L[c] for c in cols if c not in (eff, cause)] if conditional else [])
            Xr = np.hstack(base)
            rss_r, _ = _rss(y, Xr)
            rss_u, k_u = _rss(y, np.hstack([Xr, L[cause]]))
            q = max_lag
            F = ((rss_r - rss_u) / q) / (rss_u / (n - k_u))
            rows.append(dict(cause=cause, effect=eff, F=F,
                             p=float(stats.f.sf(F, q, n - k_u)),
                             delta_r2=(rss_r - rss_u) / float(((y - y.mean()) ** 2).sum())))
    return pd.DataFrame(rows)


# -- information theory on symbols ------------------------------------------------

def _hist_codes(s: np.ndarray, k: int, base: int) -> np.ndarray:
    """Encode the k-history ending at t-1 as one int, for t = k..n-1."""
    n = len(s)
    code = np.zeros(n - k, dtype=np.int64)
    for i in range(1, k + 1):
        code = code * base + s[k - i:n - i]
    return code


def _te_day(x, y, k, l, base):
    h = max(k, l)
    y_now = y[h:]
    yh = _hist_codes(y, k, base)[h - k:]
    xh = _hist_codes(x, l, base)[h - l:]
    return y_now, yh, xh


def _te_from_codes(y_now, yh, xh, base, k, l):
    by, bh, bx = base, base ** k, base ** l
    j = (y_now * bh + yh) * bx + xh
    c_yyx = np.bincount(j, minlength=by * bh * bx).reshape(by, bh, bx).astype(float)
    n = c_yyx.sum()
    c_yx = c_yyx.sum(0)          # (yh, xh)
    c_yy = c_yyx.sum(2)          # (y_now, yh)
    c_h = c_yyx.sum((0, 2))      # (yh,)
    nz = c_yyx > 0
    # TE = sum p(y,yh,xh) log[ p(y|yh,xh) / p(y|yh) ]
    num = c_yyx * c_h[None, :, None]
    den = c_yx[None, :, :] * c_yy[:, :, None]
    return float((c_yyx[nz] * np.log2(num[nz] / den[nz])).sum() / n)


def transfer_entropy(sym: pd.DataFrame, cols: Sequence[str] | None = None,
                     k: int = 1, l: int = 1, n_surrogates: int = 0,
                     seed: int = 0) -> pd.DataFrame:
    """
    Plug-in transfer entropy X -> Y (bits) for every ordered pair of symbol
    columns in `sym` (output of data.encode_symbols). Histories of length k
    (target) and l (source); days are never bridged.

    n_surrogates > 0 adds a p-value and a bias-corrected TE (TE minus the mean
    surrogate): each surrogate circularly shifts X by a random offset within
    each day, which keeps X's autocorrelation but breaks its alignment with Y.
    """
    cols = list(cols) if cols is not None else [c for c in sym.columns if c.endswith("_sym")]
    base = int(sym[cols].to_numpy().max()) + 1
    rng = np.random.default_rng(seed)
    days = [g for _, g in sym.groupby("day", sort=False)]
    rows = []
    for ycol in cols:
        for xcol in cols:
            if xcol == ycol:
                continue

            def codes(shift=False):
                parts = []
                for g in days:
                    x = g[xcol].to_numpy(np.int64)
                    y = g[ycol].to_numpy(np.int64)
                    if len(x) <= max(k, l) + 1:
                        continue
                    if shift:
                        x = np.roll(x, rng.integers(max(k, l) + 1, len(x)))
                    parts.append(_te_day(x, y, k, l, base))
                return [np.concatenate(z) for z in zip(*parts)]

            te = _te_from_codes(*codes(), base, k, l)
            row = dict(cause=xcol.removesuffix("_sym"), effect=ycol.removesuffix("_sym"), TE=te)
            if n_surrogates:
                sur = np.array([_te_from_codes(*codes(True), base, k, l)
                                for _ in range(n_surrogates)])
                row.update(TE_corrected=te - sur.mean(),
                           p=(1 + (sur >= te).sum()) / (1 + n_surrogates))
            rows.append(row)
    return pd.DataFrame(rows)


# -- one-stop feature → target summary ----------------------------------------------

def target_table(panel: pd.DataFrame, features: Sequence[str], target: str,
                 granger_effect: str | None = "log_mid_ret", max_lag: int = 5,
                 mi_rows: int = 20000) -> pd.DataFrame:
    """
    Per feature: IC (Pearson with target), rank IC, MI with target, the lag
    with the largest |xcorr| vs the target, and the pairwise Granger p-value
    toward `granger_effect` (a *contemporaneous* return — Granger on a forward
    return would double-count the lead).
    """
    features = list(features)
    cols = features + [target]
    P = _data.clean(panel, cols)
    ic = P[features].corrwith(P[target])
    ric = P[features].corrwith(P[target], method="spearman")
    mi = mi_matrix(P, cols, max_rows=mi_rows)[target].drop(target)
    best = {}
    for f in features:
        s = xcorr(P, f, target, max_lag=max_lag)
        best[f] = int(s.abs().idxmax())
    out = pd.DataFrame(dict(IC=ic, rank_IC=ric, MI=mi, best_lag=pd.Series(best)))
    if granger_effect and granger_effect in panel.columns:
        gcols = list(dict.fromkeys(features + [granger_effect]))
        G = granger(panel, gcols, max_lag=max_lag)
        G = G[G.effect == granger_effect].set_index("cause")
        out["granger_p"] = G["p"]
        out["granger_dR2"] = G["delta_r2"]
    return out.sort_values("MI", ascending=False)
