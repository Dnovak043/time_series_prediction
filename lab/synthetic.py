# -*- coding: utf-8 -*-
"""
Synthetic panels with a KNOWN causal graph — run a method here first and
check it recovers the truth before believing it on market data.

    panel, truth = synthetic.var_panel(seed=0)
    truth  # DataFrame(cause, lag, effect, coef); lag 0 = contemporaneous
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LINKS = [
    # cause, lag, effect, coef
    ("a", 1, "a", 0.5), ("b", 1, "b", 0.4), ("c", 1, "c", 0.3), ("d", 1, "d", 0.2),
    ("a", 1, "b", 0.3),          # a leads b
    ("b", 2, "c", -0.25),        # b leads c at lag 2
    ("a", 0, "d", 0.4),          # contemporaneous a -> d
    ("c", 0, "d", 0.3),          # v-structure a -> d <- c
    ("d", 1, "y", 0.35), ("b", 1, "y", -0.2),  # y's lagged parents
]


def var_panel(links=DEFAULT_LINKS, n_days: int = 4, n_per_day: int = 5000,
              noise: str = "gaussian", nonlinear: bool = False, seed: int = 0,
              forward_target: bool = True):
    """
    Structural VAR simulated day by day (each day restarts — like the real
    panels). Variables are simulated in topological order of their lag-0
    links. If forward_target, adds `y_fwd_1` = y_{t+1} (within day), matching
    the forward-return targets of real panels.
    noise: gaussian | laplace (non-Gaussian — what LiNGAM needs)
    """
    rng = np.random.default_rng(seed)
    links = pd.DataFrame(links, columns=["cause", "lag", "effect", "coef"])
    names = list(dict.fromkeys(links.cause.tolist() + links.effect.tolist()))
    order = _topo(names, links[links.lag == 0])
    tmax = int(links.lag.max())
    f = np.tanh if nonlinear else (lambda v: v)
    days = []
    for d in range(n_days):
        T = n_per_day + tmax + 50
        X = {v: np.zeros(T) for v in names}
        eps = {v: (rng.laplace(size=T) if noise == "laplace" else rng.standard_normal(T))
               for v in names}
        for t in range(tmax, T):
            for v in order:
                acc = eps[v][t]
                for _, r in links[links.effect == v].iterrows():
                    acc += r.coef * f(X[r.cause][t - r.lag])
                X[v][t] = acc
        df = pd.DataFrame({v: X[v][-n_per_day:] for v in names})
        if forward_target and "y" in names:
            df["y_fwd_1"] = df["y"].shift(-1)
        df.insert(0, "day", f"2099{d + 1:04d}")
        days.append(df.dropna())
    panel = pd.concat(days, ignore_index=True)
    feats = [v for v in names if v != "y"]
    panel.attrs.update(features=feats, targets=["y_fwd_1"] if forward_target else [])
    return panel, links


def _topo(names, contemp):
    order, left = [], list(names)
    while left:
        for v in left:
            if not any((contemp.effect == v) & contemp.cause.isin(left) & (contemp.cause != v)):
                order.append(v)
                left.remove(v)
                break
        else:
            raise ValueError("lag-0 links contain a cycle")
    return order
