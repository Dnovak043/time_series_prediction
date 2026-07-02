# -*- coding: utf-8 -*-
"""
Load SEQ_DISTR_* and CLS_DISTR_* pickle files from the repo root and save
one PNG per (file, sequence-length) combination into outputs/.

Run from any directory:
    python TrainingDistributions/plot_saved_distributions.py
"""
import os
import sys
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_distributions import plot_distributions_comparison_preserve_order

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT      = os.path.join(ROOT, 'outputs')
SEQ_LENS = [1, 2]
MIN_PROB = 1e-6

os.makedirs(OUT, exist_ok=True)


def _parse_name(basename):
    """
    'SEQ_DISTR_NVDA_bivariate_log_mid-ofi_L1_n_norm_202504'
    → (predicted='log_mid', predictor='ofi_L1_n_norm', period='202504')
    """
    parts = basename.split('_', 4)          # split off prefix tokens
    tail  = parts[4]                        # 'log_mid-ofi_L1_n_norm_202504'
    combo, period = tail.rsplit('_', 1)     # period is rightmost token
    predicted, predictor = combo.split('-', 1)
    return predicted, predictor, period


def _seq_label(seq):
    return "(" + ",".join(str(x) for x in seq) + ")"


def _filter_length(dist, lens, min_prob=0.0):
    return [e for e in dist if len(e[0]) in lens and e[1] > min_prob]


# ---------------------------------------------------------------------------
# SEQ_DISTR  —  marginal sequence probability distribution
# ---------------------------------------------------------------------------
def plot_seq_distr(fpath):
    name = os.path.basename(fpath)
    predicted, predictor, period = _parse_name(name)

    distrsall, _ = pickle.load(open(fpath, 'rb'))

    for seq_len in SEQ_LENS:
        filtered = _filter_length(distrsall, [seq_len], MIN_PROB)
        if not filtered:
            continue

        n = len(filtered)
        fig, ax, _, _ = plot_distributions_comparison_preserve_order(
            distributions=[filtered],
            names=[period],
            colors=["steelblue"],
            reference_index=0,
            top_n=n,
            figsize=(max(12, n * 0.35), 6),
            bar_group_width=0.6,
            title=f"{predicted}  ←  {predictor}   |   sequence length {seq_len}",
            xlabel="Sequence",
            ylabel="Probability",
            rotation=90,
            edgecolor="white",
            linewidth=0.1,
            alpha=0.88,
        )

        out = os.path.join(OUT, f"{name}_len{seq_len}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    saved {out}")


# ---------------------------------------------------------------------------
# CLS_DISTR  —  conditional class probability per subsequence context
#
# estimate_subsequence_class_probabilities stores classes via list[terminal_class]
# with terminal_class in {-1, 0, 1} and num_classes=3:
#   index 0  →  class  0  (flat / neutral)     [classes.iloc[...] == 0]
#   index 1  →  class +1  (up)                 [classes.iloc[...] == 1]
#   index 2  →  class -1  (down)               [classes.iloc[...] == -1, Python -1 → last slot]
# ---------------------------------------------------------------------------
_CLASS_LABELS = ['0 (flat)', '+1 (up)', '-1 (down)']
_CLASS_COLORS = ['gray',     'green',   'red'       ]


def plot_cls_distr(fpath):
    name = os.path.basename(fpath)
    predicted, predictor, period = _parse_name(name)

    cls_distr = pickle.load(open(fpath, 'rb'))

    for seq_len in SEQ_LENS:
        rows = [e for e in cls_distr if len(e[0]) == seq_len and e[3] > 0]
        if not rows:
            continue

        seqs   = [e[0] for e in rows]
        probs  = np.array([e[2] for e in rows])   # (N, 3)
        labels = [_seq_label(s) for s in seqs]

        n = len(seqs)
        x = np.arange(n)
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 6))
        for i, (lbl, col) in enumerate(zip(_CLASS_LABELS, _CLASS_COLORS)):
            ax.bar(x + (i - 1) * w, probs[:, i], width=w,
                   color=col, alpha=0.85, label=lbl)

        tick_fs = max(5, 9 - n // 25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fs)
        ax.set_ylabel("Class Probability", fontsize=12)
        ax.set_xlabel("Sequence", fontsize=12)
        ax.set_title(
            f"Class distribution:  {predicted}  ←  {predictor}   |   sequence length {seq_len}",
            fontsize=13,
        )
        ax.legend(loc="upper right")
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        out = os.path.join(OUT, f"{name}_len{seq_len}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    saved {out}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== SEQ_DISTR ===")
    for fpath in sorted(glob.glob(os.path.join(ROOT, 'SEQ_DISTR_*'))):
        print(f"  {os.path.basename(fpath)}")
        try:
            plot_seq_distr(fpath)
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\n=== CLS_DISTR ===")
    for fpath in sorted(glob.glob(os.path.join(ROOT, 'CLS_DISTR_*'))):
        print(f"  {os.path.basename(fpath)}")
        try:
            plot_cls_distr(fpath)
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nDone.")
