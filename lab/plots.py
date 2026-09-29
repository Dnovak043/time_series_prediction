# -*- coding: utf-8 -*-
"""
Plots for lab results (matplotlib; every function returns the Figure).

Colour jobs: diverging blue↔red with a gray midpoint for signed values
(correlations, partial corrs, Δ-metrics); a single-hue blue ramp for
magnitudes (MI, TE, −log10 p). Cells carry their value as text, so colour
is never the only channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("lab_div", [BLUE, NEUTRAL, RED])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "lab_seq", ["#f5f9fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def heatmap(M: pd.DataFrame, title: str = "", signed: bool | None = None,
            annotate: bool = True, fmt: str = ".2f", vmax: float | None = None, ax=None):
    """Signed values -> diverging map centred on 0; otherwise sequential.
    A square matrix with matching labels has its diagonal blanked."""
    A = M.to_numpy(float).copy()
    if M.shape[0] == M.shape[1] and list(M.index) == list(M.columns):
        np.fill_diagonal(A, np.nan)  # self-dependence would swamp the scale
    if signed is None:
        signed = np.nanmin(A) < 0
    if ax is None:
        n = max(M.shape)
        _, ax = plt.subplots(figsize=(0.55 * n + 2.5, 0.5 * n + 1.5))
    _style(ax)
    vmax = vmax or np.nanmax(np.abs(A)) or 1
    im = ax.imshow(A, cmap=DIVERGING if signed else SEQUENTIAL,
                   vmin=-vmax if signed else 0, vmax=vmax)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    if annotate and max(M.shape) <= 25:
        for (i, j), v in np.ndenumerate(A):
            if np.isfinite(v):
                light = abs(v) / vmax > 0.6
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="white" if light else INK)
    ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.figure.tight_layout()
    return ax.figure


def xcorr_plot(series: dict[str, pd.Series] | pd.Series, title: str = "", ax=None):
    """Lead-lag curves; k>0 = feature leads."""
    if isinstance(series, pd.Series):
        series = {series.name: series}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    _style(ax)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for (name, s), c in zip(series.items(), colors):
        ax.plot(s.index, s.values, lw=2, color=c, label=name, marker="o", ms=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("lag k  (corr(x[t−k], y[t]);  k>0: x leads)", fontsize=8, color=INK)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.figure.tight_layout()
    return ax.figure


def graph(edges: pd.DataFrame, source="source", target="target", weight="strength",
          directed_col: str | None = "type", label: str | None = None,
          highlight: str | None = None, title: str = "", ax=None, seed: int = 0):
    """
    Causal graph from an edge table (pc().edges(), summary_graph(), or any
    frame with source/target columns). Edge width ∝ |weight|, colour = sign
    when weights are signed. `highlight` rings one node (e.g. the target).
    """
    import networkx as nx
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        w = float(r[weight]) if weight in r else 1.0
        und = directed_col in r and r[directed_col] == "--" if directed_col else False
        G.add_edge(r[source], r[target], w=w, und=und,
                   lab=(f"{r[label]}" if label and label in r else ""))
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.axis("off")
    if not G.number_of_nodes():
        ax.text(0.5, 0.5, "no edges", ha="center", color=MUTED)
        return ax.figure
    pos = nx.circular_layout(G) if G.number_of_nodes() <= 12 else nx.spring_layout(G, seed=seed)
    wmax = max(abs(d["w"]) for *_, d in G.edges(data=True)) or 1
    signed = any(d["w"] < 0 for *_, d in G.edges(data=True))
    node_col = [RED if n == highlight else "#cde2fb" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_col, node_size=1400,
                           edgecolors=MUTED, linewidths=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color=INK)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax, width=0.8 + 4 * abs(d["w"]) / wmax,
            edge_color=RED if d["w"] < 0 else (BLUE if signed else MUTED), alpha=0.85,
            arrows=not d["und"], arrowsize=14, node_size=1400,
            connectionstyle="arc3,rad=0.08")
    labels = {(u, v): d["lab"] for u, v, d in G.edges(data=True) if d["lab"]}
    if labels:
        nx.draw_networkx_edge_labels(G, pos, labels, ax=ax, font_size=6, font_color=MUTED)
    key = "blue +, red −; " if signed else ""
    ax.set_title(f"{title}   ({key}width ∝ |{weight}|; line = undirected)", fontsize=10,
                 color=INK, loc="left")
    return ax.figure


def ablation_bars(table: pd.DataFrame, metric: str, title: str = "", ax=None,
                  reference: str | None = "all"):
    """Horizontal bars of one metric per feature set, ±1 sd across folds."""
    t = table.sort_values(metric)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.2))
    _style(ax)
    colors = [RED if (reference and i == reference) else BLUE for i in t.index]
    ax.barh(t.index, t[metric], xerr=t.get(f"{metric}_sd"), color=colors, height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1))
    if reference and reference in t.index:
        ax.axvline(t.loc[reference, metric], color=MUTED, ls=":", lw=1)
    ax.set_xlabel(metric, fontsize=8, color=INK)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title or f"{metric} by feature set (reference in red)", fontsize=10,
                 color=INK, loc="left")
    ax.figure.tight_layout()
    return ax.figure


def lag_heatmap(links: pd.DataFrame, effect: str, title: str = "", ax=None):
    """PCMCI val for causes × lags into one effect; non-significant cells blank."""
    s = links[links.effect == effect].copy()
    s.loc[~s.significant, "val"] = np.nan
    M = s.pivot(index="cause", columns="lag", values="val")
    return heatmap(M, title or f"PCMCI links into {effect} (blank = n.s.)",
                   signed=True, ax=ax)
