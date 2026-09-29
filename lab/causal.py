# -*- coding: utf-8 -*-
"""
Causal discovery over feature panels.

Dependency-free implementations (partial-correlation / Fisher-z tests, so
they assume roughly linear-Gaussian dependence — cross-check with the
nonlinear metrics in lab.metrics):

    pc(panel, cols, sinks=[target])   contemporaneous CPDAG (PC-stable + Meek)
    pcmci(panel, cols, tau_max)       time-lagged links X_{t-τ} -> Y_t
                                      (Runge et al. 2019: PC1 + MCI)

Optional backends (pip install -r lab/requirements-lab.txt):

    tigramite_pcmci(...)   the reference PCMCI / PCMCI+ with nonlinear tests
    var_lingam(...)        VAR-LiNGAM (non-Gaussian, gives contemporaneous order)

Both of our own methods work off one correlation matrix, so after it is
built the cost no longer depends on the row count — full months are fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import data as _data


# -- CI test ---------------------------------------------------------------------

class ParCorr:
    """Gaussian CI test from a correlation matrix: Fisher-z on partial corr."""

    def __init__(self, C: np.ndarray, n: int):
        self.C, self.n = C, n

    def test(self, i: int, j: int, S: Sequence[int] = ()) -> tuple[float, float]:
        idx = [i, j, *S]
        P = np.linalg.pinv(self.C[np.ix_(idx, idx)])
        r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        r = float(np.clip(r, -0.999999, 0.999999))
        dof = self.n - len(S) - 3
        z = np.sqrt(max(dof, 1)) * np.arctanh(r)
        return r, float(2 * stats.norm.sf(abs(z)))


def _corr(panel, cols, max_rows=None):
    P = _data.subsample(_data.clean(panel, cols), max_rows)
    X = P[list(cols)].to_numpy(float)
    return np.corrcoef(X, rowvar=False), len(X)


# -- PC (contemporaneous) ------------------------------------------------------------

@dataclass
class Graph:
    """CPDAG: A[i,j]=1 & A[j,i]=0 -> i→j ; both 1 -> undirected i—j."""
    nodes: list[str]
    A: np.ndarray
    strength: np.ndarray                      # |partial corr| at removal / final
    sepsets: dict = field(default_factory=dict, repr=False)

    def edges(self) -> pd.DataFrame:
        rows, n = [], len(self.nodes)
        for i in range(n):
            for j in range(n):
                if self.A[i, j] and not self.A[j, i]:
                    rows.append((self.nodes[i], self.nodes[j], "->", self.strength[i, j]))
                elif self.A[i, j] and self.A[j, i] and i < j:
                    rows.append((self.nodes[i], self.nodes[j], "--", self.strength[i, j]))
        return pd.DataFrame(rows, columns=["source", "target", "type", "strength"])

    def parents(self, node: str, include_undirected: bool = False) -> list[str]:
        j = self.nodes.index(node)
        return [self.nodes[i] for i in range(len(self.nodes))
                if self.A[i, j] and (include_undirected or not self.A[j, i])]

    def neighbors(self, node: str) -> list[str]:
        j = self.nodes.index(node)
        return [self.nodes[i] for i in range(len(self.nodes)) if self.A[i, j] or self.A[j, i]]

    def markov_blanket(self, node: str) -> list[str]:
        j = self.nodes.index(node)
        n = len(self.nodes)
        mb = set(self.neighbors(node))
        children = [c for c in range(n) if self.A[j, c] and not self.A[c, j]]
        for c in children:
            mb |= {self.nodes[p] for p in range(n) if self.A[p, c] and p != j}
        return sorted(mb)


def pc(panel: pd.DataFrame, cols: Sequence[str], alpha: float = 0.01,
       max_cond: int = 3, sinks: Sequence[str] = (), max_rows: int | None = None) -> Graph:
    """
    PC-stable with Fisher-z tests.

    sinks  background knowledge: nodes that cannot cause anything in `cols`
           (e.g. a forward-return target). Edges touching them are oriented
           into them, so graph.parents(target) = its direct causes.
    Note:  with ~1e5+ rows almost every dependence is "significant"; lower
           alpha (1e-4 .. 1e-8) or read `strength` rather than presence.
    """
    cols = list(cols)
    C, n = _corr(panel, cols, max_rows)
    t = ParCorr(C, n)
    p = len(cols)
    A = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
    S = np.abs(C.copy())
    sep = {}
    level = 0
    while level <= max_cond:
        adj = {i: [k for k in range(p) if A[i, k]] for i in range(p)}  # stable
        any_tested = False
        for i in range(p):
            for j in range(p):
                if i >= j or not A[i, j]:
                    continue
                for a, b in ((i, j), (j, i)):
                    cand = [k for k in adj[a] if k != b]
                    if len(cand) < level:
                        continue
                    any_tested = True
                    done = False
                    for Z in combinations(cand, level):
                        r, pv = t.test(i, j, Z)
                        if pv > alpha:
                            A[i, j] = A[j, i] = 0
                            sep[(i, j)] = sep[(j, i)] = set(Z)
                            S[i, j] = S[j, i] = abs(r)
                            done = True
                            break
                        S[i, j] = S[j, i] = min(S[i, j], abs(r))
                    if done:
                        break
        if not any_tested:
            break
        level += 1

    sink_idx = {cols.index(s) for s in sinks if s in cols}
    for s in sink_idx:
        for k in range(p):
            if A[k, s]:
                A[s, k] = 0
    # v-structures i -> k <- j for non-adjacent i, j with k not in sepset
    for k in range(p):
        nb = [i for i in range(p) if A[i, k] or A[k, i]]
        for i, j in combinations(nb, 2):
            if A[i, j] or A[j, i] or k in sep.get((i, j), set()):
                continue
            if i in sink_idx or j in sink_idx:  # sinks never point anywhere
                continue
            # only orient still-undirected ends (conflicting v-structures keep
            # the edge rather than deleting it)
            if A[i, k]:
                A[k, i] = 0
            if A[j, k]:
                A[k, j] = 0
    _meek(A)
    return Graph(cols, A, S, sep)


def _meek(A):
    p = len(A)
    und = lambda i, j: A[i, j] and A[j, i]
    dirc = lambda i, j: A[i, j] and not A[j, i]
    adj = lambda i, j: A[i, j] or A[j, i]
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if not und(i, j):
                    continue
                # R1: k -> i — j, k not adj j  =>  i -> j
                r1 = any(dirc(k, i) and not adj(k, j) for k in range(p) if k != j)
                # R2: i -> k -> j  =>  i -> j
                r2 = any(dirc(i, k) and dirc(k, j) for k in range(p))
                # R3: i — k1 -> j, i — k2 -> j, k1 not adj k2  =>  i -> j
                ks = [k for k in range(p) if und(i, k) and dirc(k, j)]
                r3 = any(not adj(a, b) for a, b in combinations(ks, 2))
                if r1 or r2 or r3:
                    A[j, i] = 0
                    changed = True


# -- PCMCI (time-lagged) ---------------------------------------------------------------

def pcmci(panel: pd.DataFrame, cols: Sequence[str], tau_max: int = 5,
          alpha_pc: float = 0.05, alpha: float = 0.01, max_conds: int | None = None,
          fdr: bool = True, max_rows: int | None = None, verbose: bool = False) -> pd.DataFrame:
    """
    PCMCI with ParCorr (lagged links only; contemporaneous structure -> pc()).

    Stage 1 (PC1): for each Y, prune candidate parents X_{t-τ}, τ=1..tau_max,
                   conditioning on the strongest remaining candidates.
    Stage 2 (MCI): test X_{t-τ} ⊥ Y_t | Pa(Y) \\ {X_{t-τ}}, Pa(X)_{shifted by τ}.

    Returns one row per (cause, lag, effect): partial corr `val`, `p`,
    `q` (Benjamini-Hochberg, if fdr) and `significant` at `alpha`.
    Self-links (autodependence) are included.
    """
    cols = list(cols)
    L = 2 * tau_max  # MCI needs parents of X up to tau + tau_max back
    lag = _data.lagged(_data.subsample(_data.clean(panel, cols), max_rows),
                       cols, range(1, L + 1), keep=cols)
    names = cols + [f"{c}_L{k}" for k in range(1, L + 1) for c in cols]
    X = lag[names].to_numpy(float)
    C = np.corrcoef(X, rowvar=False)
    t = ParCorr(C, len(X))
    p = len(cols)
    node = lambda v, tau: v if tau == 0 else p * tau + v  # column index

    # PC1
    parents = {}
    for j in range(p):
        cand = [(i, tau) for tau in range(1, tau_max + 1) for i in range(p)]
        strength = {c: np.inf for c in cand}
        k = 0
        while len(cand) > k and (max_conds is None or k <= max_conds):
            ranked = sorted(cand, key=lambda c: -strength[c])
            keep = []
            for c in ranked:
                Z = [z for z in ranked if z != c][:k]
                r, pv = t.test(node(*c), node(j, 0), [node(*z) for z in Z])
                strength[c] = min(strength[c], abs(r))
                if pv <= alpha_pc:
                    keep.append(c)
            cand = keep
            k += 1
        parents[j] = sorted(cand, key=lambda c: -strength[c])
        if verbose:
            print(f"PC1 {cols[j]}: {[(cols[i], -tau) for i, tau in parents[j]]}")

    # MCI
    rows = []
    for j in range(p):
        for tau in range(1, tau_max + 1):
            for i in range(p):
                Z = [z for z in parents[j] if z != (i, tau)]
                Z += [(a, s + tau) for a, s in parents[i] if s + tau <= L]
                Z = list(dict.fromkeys(z for z in Z if z != (i, tau)))
                r, pv = t.test(node(i, tau), node(j, 0), [node(*z) for z in Z])
                rows.append(dict(cause=cols[i], lag=tau, effect=cols[j], val=r, p=pv))
    out = pd.DataFrame(rows)
    out["q"] = _bh(out["p"].to_numpy()) if fdr else out["p"]
    out["significant"] = out["q"] <= alpha
    return out.sort_values("val", key=np.abs, ascending=False).reset_index(drop=True)


def _bh(p):
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def lagged_parents(links: pd.DataFrame, effect: str, self_links: bool = False) -> list[str]:
    """Distinct significant causes of `effect` from a pcmci() table."""
    m = links.significant & (links.effect == effect)
    if not self_links:
        m &= links.cause != effect
    return list(dict.fromkeys(links.loc[m, "cause"]))


def summary_graph(links: pd.DataFrame) -> pd.DataFrame:
    """Collapse lags: strongest significant |val| per (cause, effect) + its lag."""
    s = links[links.significant]
    if s.empty:
        return pd.DataFrame(columns=["cause", "effect", "val", "lag"])
    idx = s.groupby(["cause", "effect"])["val"].apply(lambda v: v.abs().idxmax())
    return s.loc[idx.values, ["cause", "effect", "val", "lag"]].reset_index(drop=True)


# -- optional backends -----------------------------------------------------------

def _need(mod, pkg):
    try:
        return __import__(mod, fromlist=["_"])
    except ImportError as e:
        raise ImportError(f"{pkg} not installed: pip install -r lab/requirements-lab.txt") from e


def tigramite_pcmci(panel: pd.DataFrame, cols: Sequence[str], tau_max: int = 5,
                    test: str = "parcorr", pc_alpha: float = 0.05, alpha: float = 0.01,
                    plus: bool = False, max_rows: int | None = 200_000):
    """Reference PCMCI(+). test: parcorr | gpdc | cmiknn (nonlinear ones are slow — subsample).
    Days are separated with NaN rows + missing_flag so no lag crosses a day."""
    pp = _need("tigramite.data_processing", "tigramite")
    tp = _need("tigramite.pcmci", "tigramite")
    tests = {"parcorr": ("tigramite.independence_tests.parcorr", "ParCorr"),
             "gpdc": ("tigramite.independence_tests.gpdc", "GPDC"),
             "cmiknn": ("tigramite.independence_tests.cmiknn", "CMIknn")}
    mod, cls = tests[test]
    Test = getattr(_need(mod, "tigramite"), cls)
    P = _data.subsample(_data.clean(panel, cols), max_rows)
    blocks = []
    for _, g in P.groupby("day", sort=False):
        blocks.append(g[list(cols)].to_numpy(float))
        blocks.append(np.full((tau_max, len(cols)), 999.0))
    df = pp.DataFrame(np.vstack(blocks[:-1]), var_names=list(cols), missing_flag=999.0)
    m = tp.PCMCI(dataframe=df, cond_ind_test=Test())
    if plus:
        return m.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
    res = m.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha)
    return res


def var_lingam(panel: pd.DataFrame, cols: Sequence[str], lags: int = 1,
               max_rows: int | None = 200_000):
    """VAR-LiNGAM (lingam package). Returns the fitted model: .adjacency_matrices_
    [0] = contemporaneous B0 (row=effect, col=cause), [k] = lag-k matrices."""
    lingam = _need("lingam", "lingam")
    P = _data.subsample(_data.clean(panel, cols), max_rows)
    m = lingam.VARLiNGAM(lags=lags)
    m.fit(_data.standardize(P, cols)[list(cols)].to_numpy())
    return m
