# -*- coding: utf-8 -*-
"""Vectorized replacements for per-row Python loops in the featurization
stage (ARCHITECTURE_AND_PERFORMANCE.md bottleneck #3).

Pure numpy, designed to be BIT-IDENTICAL to the originals: same element
values, same per-window summation order (numpy pairwise reduction over the
same logical sequence), just without a CPython callback per window. Verified
by tests/test_fast_ops.py (exact equality, no tolerance) and end-to-end by
tests/verify_against_baseline.py.

Why these stay on CPU: once vectorized, both operations are single
memory-bandwidth-bound passes over the day's events; host<->device transfer
would cost more than it saves. The GPU work in this repo goes where the
FLOPs are — subsequence histograms (subsequence_torch.py) and model
training (LearningKraus.py / pipeline train-all).
"""
import numpy as np
import pandas as pd


def rolling_rms(series: pd.Series, window: int) -> pd.Series:
    """Vectorized equivalent of

        series.rolling(window, min_periods=window)
              .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True)

    The sliding windows are views over the squared array, and np.mean
    reduces each row over the same logical element order the original
    lambda saw, so the floats match bit-for-bit.
    """
    v = series.to_numpy(dtype=np.float64, copy=False)
    out = np.full(v.shape, np.nan)
    if window is not None and 0 < window <= v.size:
        xx = v * v
        win = np.lib.stride_tricks.sliding_window_view(xx, window)
        out[window - 1:] = np.sqrt(np.mean(win, axis=-1))
    return pd.Series(out, index=series.index)


def carry_last_nonzero(a: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of

        last = 0
        for i, val in enumerate(a):
            if val == 0:
                a[i] = last
            else:
                last = val

    i.e. zeros take the most recent nonzero value; zeros before the first
    nonzero stay zero. Returns a new array with the input's dtype.
    """
    a = np.asarray(a)
    if a.size == 0:
        return a.copy()
    idx = np.where(a != 0, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    # positions before the first nonzero resolve to a[0]; if a[0] == 0 the
    # carried value is 0, exactly matching the loop's `last = 0` start
    return a[idx]
