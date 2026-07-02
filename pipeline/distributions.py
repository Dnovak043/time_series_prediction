# -*- coding: utf-8 -*-
"""
DistributionBuilder: featurized day -> per-day sequence & class counts.

Pure functions of (day_df, predicted, predictor) plus the encode/distribution
configs — no I/O. The math is the original code, called with the exact same
arguments the legacy driver used (get_distribution_by_ts and the module-level
loop in process_distributions.py), so results are identical for the default
config; the difference is that every previously hardcoded argument now comes
from the config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DistributionConfig, EncodeConfig


class DistributionBuilder:
    def __init__(self, encode_cfg: EncodeConfig, dist_cfg: DistributionConfig,
                 device=None):
        """device: torch device for the counting histograms (None = auto
        cuda->mps->cpu). The counts are integer-exact on every device; the
        parallel runner passes "cpu" so N day-workers don't each open a CUDA
        context for sub-second histogram work."""
        self.encode_cfg = encode_cfg
        self.dist_cfg = dist_cfg
        self.device = device

    # -- encoding ---------------------------------------------------------------
    def encode_bivariate(self, day_df: pd.DataFrame, predictor: str):
        """
        Z-encode (predicted, predictor) and combine the two n_symbols-ary
        streams into one n_symbols^2-ary symbol series.

        Returns (ts, z_series12): ts is a working copy of day_df with the
        encoding (and later class) columns added; z_series12 is the combined
        int Series. The copy keeps repeated calls on a cached day independent.
        """
        from process_distributions import z_encoding

        e = self.encode_cfg
        predicted = self.dist_cfg.predicted

        ts = day_df.copy()
        z_encoding(ts, predictor, e.n_symbols, e.alpha,
                   min_periods=e.min_periods,
                   fill_mode=e.fill_mode, bins_mode=e.bins_mode)
        z_encoding(ts, predicted, e.n_symbols, e.alpha,
                   min_periods=e.min_periods,
                   fill_mode=e.fill_mode, bins_mode=e.bins_mode)

        predicted_series = ts[predicted + "_sym"]
        predictor_series = ts[predictor + "_sym"]

        # mixed-radix combination, verbatim from the original driver
        z_series12 = pd.concat([predicted_series, predictor_series], axis=1)
        ni = np.asarray([e.n_symbols, e.n_symbols])
        weights = np.concatenate(([1], np.cumprod(ni[:-1])))
        z_series12 = (z_series12 * weights).sum(axis=1)
        return ts, z_series12.astype(int)

    # -- per-day counts -----------------------------------------------------------
    def sequence_counts(self, z_series12: pd.Series):
        """Observed subsequence counts for one day (SEQ distribution input)."""
        # torch unfold+unique counting; exact-equality contract vs the
        # pure-Python original is covered by test_subsequence_torch.py
        from subsequence_torch import estimate_observed_subsequence_counts_torch

        c = self.dist_cfg
        bi_series = list(z_series12)
        all_subsequences, counts = estimate_observed_subsequence_counts_torch(
            seq=bi_series,
            max_subsequence_length=c.max_seq_length,
            sample_size=c.sample_size,
            sample_after_length=c.sample_after_length,
            random_state=c.random_state,
            sort="lexicographic",
            include_prob=True,
            device=self.device,
        )
        return all_subsequences, counts

    def class_counts(self, ts: pd.DataFrame, z_series12: pd.Series):
        """Class-conditional counts for one day (CLS distribution input)."""
        from process_distributions import add_class_label
        # torch unfold+bincount histogram (PR #1); verified identical to the
        # pure-Python estimate_subsequence_class_probabilities
        from subsequence_torch import (
            estimate_subsequence_class_probabilities_torch,
        )

        c = self.dist_cfg
        cls_ts = add_class_label(ts, c.class_name,
                                 theta=c.class_theta or None)
        cl_distributions = estimate_subsequence_class_probabilities_torch(
            z_series12, cls_ts,
            num_classes=c.num_classes,
            max_subsequence_length=c.max_seq_length,
            device=self.device,
        )
        return cl_distributions[1]


# -- output formatting (verbatim from the legacy driver) -------------------------
def format_counts(counts):
    """Nested per-length counts -> ([ [seq, prob], ... ], [seq, ...])."""
    cnts = [[np.array(t[0]).astype(int).tolist(), t[1], t[2]]
            for sublist in counts for t in sublist]
    distrs = [[c[0], c[1] / c[2]] for c in cnts]
    samples = [s[0] for s in cnts]
    return distrs, samples


def flatten_class_distributions(all_cls_distr):
    return [item for sublist in all_cls_distr for item in sublist]
