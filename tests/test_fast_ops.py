# -*- coding: utf-8 -*-
"""
EXACT-equality tests for TrainingDistributions/fast_ops.py.

The vectorized ops claim to be bit-identical to the original per-row Python
implementations they replaced. This test re-implements those originals
verbatim and compares with NO tolerance, across many shapes/edge cases and
seeds, on data shaped like real log-returns (including NaN warmups and
zero runs).

Run:  .env/bin/python tests/test_fast_ops.py           (seconds)

The end-to-end proof remains tests/verify_against_baseline.py — this test
just localizes any disagreement to the exact op and case.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TrainingDistributions"))

from fast_ops import carry_last_nonzero, rolling_rms  # noqa: E402

failures = []


def check(name, ok):
    print(("  PASS  " if ok else "  FAIL  ") + name)
    if not ok:
        failures.append(name)


# -- originals, verbatim ------------------------------------------------------
def rolling_rms_original(series: pd.Series, window: int) -> pd.Series:
    return (series.rolling(window, min_periods=window)
                  .apply(lambda x: np.sqrt(np.mean(x * x)), raw=True))


def carry_last_nonzero_original(a: np.ndarray) -> np.ndarray:
    s = a.copy()
    last = 0
    for i, val in enumerate(s):
        if val == 0:
            s[i] = last
        else:
            last = val
    return s


# -- rolling_rms --------------------------------------------------------------
def series_like_returns(rng, n):
    """log-return-shaped: tiny floats, a NaN warmup, some exact zeros."""
    v = rng.standard_normal(n) * 1e-4
    v[rng.random(n) < 0.3] = 0.0            # flat mid: exact zero returns
    if n:
        v[0] = np.nan                        # diff() warmup
    return pd.Series(v)


rng = np.random.default_rng(0)
for n in (0, 1, 5, 299, 300, 301, 5000, 200_000):
    for W in (1, 2, 30, 300, 5000):
        s = series_like_returns(rng, n)
        a = rolling_rms_original(s, W)
        b = rolling_rms(s, W)
        same = (len(a) == len(b)
                and np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
        check(f"rolling_rms n={n} W={W}", same)

# a case with NaNs scattered mid-series (halted trading etc.)
s = series_like_returns(rng, 10_000)
s[rng.random(10_000) < 0.01] = np.nan
a, b = rolling_rms_original(s, 300), rolling_rms(s, 300)
check("rolling_rms scattered NaNs",
      np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))

# -- carry_last_nonzero -------------------------------------------------------
cases = {
    "empty": np.array([], dtype=np.int8),
    "all zero": np.zeros(50, dtype=np.int8),
    "leading zeros": np.array([0, 0, 0, 1, 0, -1, 0, 0], dtype=np.int8),
    "starts nonzero": np.array([1, 0, 0, -1, 0], dtype=np.int8),
    "no zeros": np.array([1, -1, 1, 1, -1], dtype=np.int8),
    "single zero": np.array([0], dtype=np.int8),
    "single nonzero": np.array([-1], dtype=np.int8),
}
for name, a in cases.items():
    got = carry_last_nonzero(a)
    want = carry_last_nonzero_original(a)
    check(f"carry_last_nonzero {name}",
          np.array_equal(got, want) and got.dtype == want.dtype)

for seed in range(20):
    r = np.random.default_rng(seed)
    a = r.choice(np.array([-1, 0, 0, 0, 1], dtype=np.int8), size=10_000)
    check(f"carry_last_nonzero random seed={seed}",
          np.array_equal(carry_last_nonzero(a), carry_last_nonzero_original(a)))

# -- timing note (informational, not asserted) --------------------------------
import time  # noqa: E402
s = series_like_returns(rng, 2_000_000)
t0 = time.time(); rolling_rms(s, 300); t_fast = time.time() - t0
print(f"\n  info: vectorized rolling_rms on 2M events: {t_fast:.2f}s "
      "(original lambda takes minutes at this size)")

if failures:
    print(f"\n{len(failures)} FAILURES"); sys.exit(1)
print("\nALL FAST-OPS TESTS PASSED (exact equality, zero tolerance)")
