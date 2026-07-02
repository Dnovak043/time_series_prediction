# -*- coding: utf-8 -*-
"""
EXACT-equality test for estimate_observed_subsequence_counts_torch
(the SEQ-distribution counterpart of the verified class-histogram).

Compares against the UNTOUCHED pure-Python original in
read_databento_new.estimate_observed_subsequence_counts (that file was not
modified by this change), across stream sizes, alphabets, sampling
configurations, sort modes, and input types — with zero tolerance: integer
counts must match exactly and probabilities bit-for-bit (same int/int
division). One case additionally requires the pickled bytes of both results
to be identical.

Run:  .env/bin/python tests/test_seq_counts_torch.py        (seconds)
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TrainingDistributions"))

from read_databento_new import estimate_observed_subsequence_counts  # noqa: E402
from subsequence_torch import (  # noqa: E402
    estimate_observed_subsequence_counts_torch,
    pick_device,
)

failures = []


def deep_eq(a, b, path="root"):
    """Exact recursive equality; floats compared with ==, no tolerance."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = deep_eq(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if not (a == b):
        return f"{path}: {a!r} != {b!r}"
    return None


def check(label, ref, new, expect_same_bytes=False):
    err = deep_eq(ref, new)
    ok = err is None
    if ok and expect_same_bytes:
        ok = pickle.dumps(ref) == pickle.dumps(new)
        err = "pickle bytes differ (value types?)" if not ok else None
    print(("  PASS  " if ok else f"  FAIL  ") + label + ("" if ok else f" — {err}"))
    if not ok:
        failures.append(label)


def run_case(label, seq, device, **kw):
    ref = estimate_observed_subsequence_counts(seq, **kw)
    new = estimate_observed_subsequence_counts_torch(seq, device=device, **kw)
    check(f"[{device}] {label}", ref, new,
          expect_same_bytes=isinstance(seq, list)
          and (not seq or isinstance(seq[0], int)))


def main():
    devices = ["cpu"]
    auto = pick_device()
    if str(auto) != "cpu":
        devices.append(str(auto))
    print(f"torch {torch.__version__}; devices: {devices}")

    for device in devices:
        rng = np.random.default_rng(1)
        streams = {
            "empty": [],
            "single": [3],
            "shorter than max_len": [1, 2, 3],
            "day-shaped 16-symbol": [int(x) for x in rng.integers(0, 16, 50_000)],
            "binary": [int(x) for x in rng.integers(0, 2, 5_000)],
            "sparse alphabet": [int(x) for x in
                                rng.choice([0, 7, 15], size=3_000)],
        }
        for name, seq in streams.items():
            run_case(f"{name}, defaults (driver args)", seq, device,
                     max_subsequence_length=6, sample_size=1.0,
                     sample_after_length=30, random_state=42,
                     sort="lexicographic", include_prob=True)

        base = streams["day-shaped 16-symbol"]
        run_case("subsampled support (0.5 after len 2, seed 42)", base, device,
                 max_subsequence_length=6, sample_size=0.5,
                 sample_after_length=2, random_state=42,
                 sort="lexicographic", include_prob=True)
        run_case("subsampled support (0.25, seed 7)", base, device,
                 max_subsequence_length=5, sample_size=0.25,
                 sample_after_length=1, random_state=7,
                 sort="lexicographic", include_prob=True)
        run_case("sample_size 0.0", base, device,
                 max_subsequence_length=4, sample_size=0.0,
                 sample_after_length=2, random_state=0,
                 sort="lexicographic", include_prob=True)
        run_case("count_desc sort", base, device,
                 max_subsequence_length=4, sample_size=1.0,
                 sample_after_length=30, random_state=42,
                 sort="count_desc", include_prob=True)
        run_case("sort='none' (falls back to original)", base, device,
                 max_subsequence_length=3, sample_size=1.0,
                 sample_after_length=30, random_state=42,
                 sort="none", include_prob=True)
        run_case("include_prob=False", base, device,
                 max_subsequence_length=4, sample_size=1.0,
                 sample_after_length=30, random_state=42,
                 sort="lexicographic", include_prob=False)
        # input types the call sites actually pass
        run_case("np.int64 list (pipeline call site)",
                 list(pd.Series(base).astype(int)), device,
                 max_subsequence_length=6, sample_size=1.0,
                 sample_after_length=30, random_state=42,
                 sort="lexicographic", include_prob=True)

    # timing, informational
    seq = [int(x) for x in np.random.default_rng(2).integers(0, 16, 300_000)]
    t0 = time.time()
    estimate_observed_subsequence_counts(
        seq, 6, sample_size=1.0, sample_after_length=30,
        random_state=42, sort="lexicographic", include_prob=True)
    t_ref = time.time() - t0
    t0 = time.time()
    estimate_observed_subsequence_counts_torch(
        seq, 6, sample_size=1.0, sample_after_length=30,
        random_state=42, sort="lexicographic", include_prob=True)
    t_new = time.time() - t0
    print(f"\n  info: 300k-symbol stream, lengths 1-6: "
          f"python {t_ref:.2f}s -> torch {t_new:.2f}s")

    if failures:
        print(f"\n{len(failures)} FAILURES")
        sys.exit(1)
    print("\nALL SEQ-COUNT TESTS PASSED (exact equality, zero tolerance)")


if __name__ == "__main__":
    main()
