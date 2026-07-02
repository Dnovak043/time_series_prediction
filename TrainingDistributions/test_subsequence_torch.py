# -*- coding: utf-8 -*-
"""
Correctness verification for subsequence_torch (ARCHITECTURE_AND_PERFORMANCE.md 7.1).

Compares estimate_subsequence_class_probabilities_torch element-for-element
against the pure-Python reference estimate_subsequence_class_probabilities
in process_distributions.py, on inputs shaped exactly like the batch
driver's call site (int64 pd.Series symbols 0..15, int8 class labels in
{-1,0,1}, num_classes=3, max_subsequence_length=6), plus edge cases.

Run:  .env/bin/python TrainingDistributions/test_subsequence_torch.py

The reference implementation is defined inline below as a verbatim copy of
process_distributions.py's function — importing process_distributions would
execute the whole batch pipeline (it has no __main__ guard).
"""
from collections import defaultdict
from itertools import product
import time

import numpy as np
import pandas as pd
import torch

from subsequence_torch import (
    estimate_subsequence_class_probabilities_torch,
    pick_device,
)


# --- verbatim copy of process_distributions.estimate_subsequence_class_probabilities
def estimate_subsequence_class_probabilities(seq, classes, num_classes, max_subsequence_length, non_zero_only = True):
    """
    Same as estimate_subsequence_class_distributions, but also returns
    empirical conditional class probabilities.
    """
    if len(seq) != len(classes):
        raise ValueError("seq and classes must have the same length.")

    unique_symbols = sorted(set(seq))
    all_subsequences = []
    distributions = []

    for length in range(1, max_subsequence_length + 1):
        possible_subseqs = list(product(unique_symbols, repeat=length))
        all_subsequences.append(possible_subseqs)

        subseq_class_counts = defaultdict(lambda: [0] * num_classes)

        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            terminal_class = classes.iloc[i + length - 1]
            subseq_class_counts[subseq][terminal_class] += 1

        length_distributions = []
        for subseq in possible_subseqs:
            class_counts = subseq_class_counts.get(subseq, [0] * num_classes)
            total_occurrences = sum(class_counts)

            if total_occurrences > 0:
                class_probs = [c / total_occurrences for c in class_counts]
            else:
                class_probs = [0.0] * num_classes

            if  not non_zero_only or total_occurrences > 0 :
                length_distributions.append(
                    [subseq, class_counts, class_probs, total_occurrences]
                )
        if len(length_distributions)>0:
            distributions.append(length_distributions)

    return all_subsequences, distributions
# --- end verbatim copy


def compare_distributions(ref_dist, new_dist, label):
    """Exact structural comparison: same lengths, same order, identical
    tuples/counts/totals, bit-identical probabilities."""
    assert len(ref_dist) == len(new_dist), (
        f"[{label}] #lengths differ: ref {len(ref_dist)} vs new {len(new_dist)}"
    )
    for li, (ref_rows, new_rows) in enumerate(zip(ref_dist, new_dist)):
        assert len(ref_rows) == len(new_rows), (
            f"[{label}] length-block {li}: {len(ref_rows)} vs {len(new_rows)} rows"
        )
        for ri, (r, m) in enumerate(zip(ref_rows, new_rows)):
            r_seq, r_counts, r_probs, r_total = r
            m_seq, m_counts, m_probs, m_total = m
            assert tuple(int(x) for x in r_seq) == tuple(m_seq), (
                f"[{label}] block {li} row {ri}: subseq {r_seq} vs {m_seq}"
            )
            assert [int(c) for c in r_counts] == m_counts, (
                f"[{label}] block {li} row {ri} {r_seq}: counts {r_counts} vs {m_counts}"
            )
            assert int(r_total) == m_total, (
                f"[{label}] block {li} row {ri} {r_seq}: total {r_total} vs {m_total}"
            )
            # bit-identical float equality, deliberately not almost-equal
            assert r_probs == m_probs, (
                f"[{label}] block {li} row {ri} {r_seq}: probs {r_probs} vs {m_probs}"
            )


def make_case(n, alphabet, seed, p_class=(0.3, 0.4, 0.3)):
    """pd.Series pair shaped like the driver call site: int64 symbol series
    with a DatetimeIndex, int8 class series with labels in {-1, 0, 1}."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-04-01 09:30", periods=n, freq="s", tz="UTC")
    seq = pd.Series(rng.choice(alphabet, size=n).astype(np.int64), index=idx)
    classes = pd.Series(
        rng.choice([0, 1, -1], size=n, p=list(p_class)).astype(np.int8), index=idx
    )
    return seq, classes


def run_case(label, seq, classes, num_classes, max_len, non_zero_only, device):
    ref = estimate_subsequence_class_probabilities(
        seq, classes, num_classes, max_len, non_zero_only=non_zero_only
    )
    new = estimate_subsequence_class_probabilities_torch(
        seq, classes, num_classes, max_len,
        non_zero_only=non_zero_only, device=device,
    )
    compare_distributions(ref[1], new[1], label)
    # new[0] must list exactly the subsequences whose rows were emitted
    emitted = [[row[0] for row in rows] for rows in new[1]]
    non_empty = [s for s in new[0] if len(s) > 0]
    assert emitted == non_empty, f"[{label}] all_subsequences inconsistent with rows"
    n_rows = sum(len(rows) for rows in new[1])
    print(f"  PASS  {label}  ({n_rows} rows identical)")


def main():
    devices = ["cpu"]
    auto = pick_device()
    if auto.type != "cpu":
        devices.append(str(auto))
    print(f"torch {torch.__version__}; verifying on devices: {devices}")

    for device in devices:
        print(f"--- device={device} ---")
        # 1. call-site shape: full 16-symbol bivariate alphabet, len 6
        seq, cls = make_case(3000, list(range(16)), seed=1)
        run_case("bivariate n=3000 A=16 L=6", seq, cls, 3, 6, True, device)

        # 2. alphabet with gaps (unobserved symbols, non-contiguous values)
        seq, cls = make_case(800, [0, 2, 5, 11, 15], seed=2)
        run_case("gapped alphabet", seq, cls, 3, 6, True, device)

        # 3. sequence shorter than max length (empty window blocks)
        seq, cls = make_case(4, list(range(16)), seed=3)
        run_case("n=4 < max_len=6", seq, cls, 3, 6, True, device)

        # 4. non_zero_only=False (zero rows for unobserved combinations)
        seq, cls = make_case(60, [0, 1, 2, 3], seed=4)
        run_case("non_zero_only=False", seq, cls, 3, 4, False, device)

        # 5. constant symbol series
        seq, cls = make_case(200, [7], seed=5)
        run_case("constant series", seq, cls, 3, 6, True, device)

        # 6. single class present, num_classes=3
        seq, cls = make_case(500, list(range(16)), seed=6, p_class=(0.0, 1.0, 0.0))
        run_case("single observed class", seq, cls, 3, 6, True, device)

        # 7. plain python lists + classes as Series (reference needs .iloc)
        rng = np.random.default_rng(7)
        seq = pd.Series(rng.integers(0, 16, 1000).astype(np.int64))
        cls = pd.Series(rng.choice([-1, 0, 1], 1000).astype(np.int8))
        run_case("range-indexed Series", seq, cls, 3, 6, True, device)

    # timing on a realistic day-sized series (~23400 1s samples)
    print("--- timing (n=23400, A=16, L=6, non_zero_only=True) ---")
    seq, cls = make_case(23400, list(range(16)), seed=42)
    t0 = time.perf_counter()
    ref = estimate_subsequence_class_probabilities(seq, cls, 3, 6)
    t_ref = time.perf_counter() - t0
    for device in devices:
        estimate_subsequence_class_probabilities_torch(seq, cls, 3, 6, device=device)
        t0 = time.perf_counter()
        new = estimate_subsequence_class_probabilities_torch(seq, cls, 3, 6, device=device)
        t_new = time.perf_counter() - t0
        compare_distributions(ref[1], new[1], f"timing/{device}")
        print(f"  reference: {t_ref:8.2f}s   torch[{device}]: {t_new:6.3f}s   "
              f"speedup: {t_ref / t_new:8.1f}x   (outputs identical)")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
