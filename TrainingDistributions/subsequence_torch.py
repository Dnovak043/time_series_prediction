# -*- coding: utf-8 -*-
"""
Torch-based subsequence/class histogram estimation.

Vectorized replacement for the pure-Python
`estimate_subsequence_class_probabilities` in process_distributions.py
(ARCHITECTURE_AND_PERFORMANCE.md 7.1). Instead of enumerating the full
alphabet**length Cartesian product (16**6 ~ 16.8M tuples per call) and
probing a dict window-by-window in Python, it:

  1. builds all length-k windows at once with `tensor.unfold(0, k, 1)`
     (zero-copy view),
  2. reduces each window to a single big-endian mixed-radix integer key
     (integer order == lexicographic tuple order),
  3. joins the key with the terminal class label
     (`key * num_classes + class`) and counts everything with one
     `torch.bincount` per length.

Device-agnostic: cuda -> mps -> cpu fallback via `pick_device()`.

Numerical contract (verified by test_subsequence_torch.py): the second
return value (the distributions) is element-for-element identical to the
pure-Python reference — same entries, same lexicographic order, same
integer counts/totals, bit-identical probabilities. The first return
value differs deliberately: the reference returns the full Cartesian
product per length (the very object that made it slow); this version
returns only the subsequences it emitted rows for. The batch driver in
process_distributions.py consumes only the distributions (element [1]).
"""
from __future__ import annotations

import numpy as np
import torch


def pick_device(prefer: str | torch.device | None = None) -> torch.device:
    """cuda -> mps -> cpu, unless an explicit device is requested."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_int64_tensor(values, device: torch.device) -> torch.Tensor:
    """
    Convert a list / np.ndarray / pd.Series of integers to an int64 tensor.

    torch.from_numpy is the fast path; the .tolist() fallback covers builds
    where the torch<->numpy bridge is unavailable (e.g. torch 2.2.2 wheels
    running against numpy>=2 on Intel macOS).
    """
    arr = np.asarray(values).astype(np.int64, copy=False)
    try:
        t = torch.from_numpy(np.ascontiguousarray(arr))
    except (RuntimeError, TypeError):
        t = torch.tensor(arr.tolist(), dtype=torch.int64)
    return t.to(device)


def estimate_observed_subsequence_counts_torch(
    seq,
    max_subsequence_length: int,
    sample_size: float = 1.0,
    sample_after_length: int = 2,
    random_state=None,
    sort: str = "lexicographic",
    include_prob: bool = True,
    device: str | torch.device | None = None,
):
    """
    Drop-in replacement for read_databento_new.estimate_observed_subsequence_counts
    (the SEQ-distribution twin of the class estimator below): counts only the
    observed subsequences of each length via unfold + torch.unique instead of
    a Python tuple/dict pass per window.

    Exactness contract (verified by test_subsequence_torch.py): identical
    return structure — per length, rows [subseq, count, total_possible(, prob)]
    with the same integer counts, the same lexicographic order (ascending
    mixed-radix keys ARE lexicographic tuple order), the same subsampling
    decisions (same rng consumption on identically-ordered items), and
    bit-identical probs (same int/int division).

    `sort="none"` (first-encounter order) and non-integer/negative symbols
    fall back to the pure-Python original.
    """
    if max_subsequence_length < 1:
        raise ValueError("max_subsequence_length must be >= 1.")
    if not (0.0 <= sample_size <= 1.0):
        raise ValueError("sample_size must be in [0, 1].")
    if sort not in ("lexicographic", "count_desc", "none"):
        raise ValueError("sort must be 'lexicographic', 'count_desc', or 'none'.")

    def _fallback():
        from read_databento_new import estimate_observed_subsequence_counts
        return estimate_observed_subsequence_counts(
            seq, max_subsequence_length, sample_size=sample_size,
            sample_after_length=sample_after_length, random_state=random_state,
            sort=sort, include_prob=include_prob)

    if sort == "none":     # original preserves first-encounter dict order
        return _fallback()

    arr = np.asarray(seq)
    if arr.size and arr.dtype.kind not in "iu":   # non-integer symbols
        return _fallback()

    dev = pick_device(device)
    seq_t = _as_int64_tensor(arr, dev)
    n = seq_t.numel()
    if n > 0:
        if int(seq_t.min().item()) < 0:
            return _fallback()
        base = int(seq_t.max().item()) + 1
    else:
        base = 1
    if base ** max_subsequence_length >= 2 ** 62:
        return _fallback()

    rng = np.random.default_rng(random_state)
    observed_subsequences: list[list[tuple]] = []
    counts: list[list[list]] = []

    for length in range(1, max_subsequence_length + 1):
        total_possible = n - length + 1
        if total_possible <= 0:
            observed_subsequences.append([])
            counts.append([])
            continue

        windows = seq_t.unfold(0, length, 1)
        weights = base ** torch.arange(
            length - 1, -1, -1, dtype=torch.int64, device=dev)
        keys = (windows * weights).sum(dim=1)
        uniq, cnt = torch.unique(keys, return_counts=True)  # ascending == lex
        digit_mat = (uniq.unsqueeze(1) // weights) % base
        items = list(zip((tuple(r) for r in digit_mat.cpu().tolist()),
                         cnt.cpu().tolist()))

        if sort == "count_desc":
            items.sort(key=lambda kv: (-kv[1], kv[0]))
        # "lexicographic": already sorted by construction

        # ---- sample observed support only for longer lengths (identical
        # rng consumption to the original: same m, k, choice order)
        if length > sample_after_length and sample_size < 1.0:
            m = len(items)
            if sample_size == 0.0:
                items = []
            else:
                k = int(np.round(sample_size * m))
                k = min(max(k, 1), m)
                chosen = rng.choice(m, size=k, replace=False)
                chosen.sort()
                items = [items[idx] for idx in chosen]

        observed_subsequences.append([subseq for subseq, _ in items])
        if include_prob:
            counts.append([[subseq, count, total_possible,
                            count / total_possible]
                           for subseq, count in items])
        else:
            counts.append([[subseq, count, total_possible]
                           for subseq, count in items])

    return observed_subsequences, counts


def estimate_subsequence_class_probabilities_torch(
    seq,
    classes,
    num_classes: int,
    max_subsequence_length: int,
    non_zero_only: bool = True,
    device: str | torch.device | None = None,
):
    """
    Drop-in replacement for estimate_subsequence_class_probabilities
    (process_distributions.py): empirical class distribution at the
    terminal point of every observed subsequence of length 1..max.

    seq must contain non-negative integer symbols; classes must contain
    integer labels in [-num_classes, num_classes) — negative labels wrap
    to the tail slots exactly as the reference's list[-1] indexing does
    (label -1 -> slot num_classes-1).

    Returns (all_subsequences, distributions); distributions[k] rows are
    [subseq_tuple, class_counts, class_probs, total_occurrences] in
    lexicographic order, identical to the reference. all_subsequences
    holds only the emitted subsequences per length (see module docstring).
    """
    if len(seq) != len(classes):
        raise ValueError("seq and classes must have the same length.")
    if max_subsequence_length < 1:
        raise ValueError("max_subsequence_length must be >= 1.")
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1.")

    dev = pick_device(device)
    seq_t = _as_int64_tensor(seq, dev)
    cls_t = _as_int64_tensor(classes, dev)

    n = seq_t.numel()
    if n > 0:
        if int(seq_t.min().item()) < 0:
            raise ValueError("seq symbols must be non-negative integers.")
        cmin, cmax = int(cls_t.min().item()), int(cls_t.max().item())
        if cmin < -num_classes or cmax >= num_classes:
            raise ValueError(
                f"class labels must lie in [{-num_classes}, {num_classes}); "
                f"got range [{cmin}, {cmax}]."
            )
        # list[-1] indexing in the reference == modular wrap
        cls_t = cls_t % num_classes
        base = int(seq_t.max().item()) + 1
        observed_t = torch.unique(seq_t)  # sorted ascending
    else:
        base = 1
        observed_t = seq_t

    all_subsequences: list[list[tuple]] = []
    distributions: list[list[list]] = []

    for length in range(1, max_subsequence_length + 1):
        if base ** length * num_classes >= 2 ** 62:
            raise OverflowError(
                f"alphabet {base} at length {length} overflows int64 keys."
            )

        n_windows = n - length + 1

        if n_windows > 0:
            # [n_windows, length] zero-copy view of all sliding windows
            windows = seq_t.unfold(0, length, 1)
            # big-endian mixed-radix key: integer order == lex tuple order
            weights = base ** torch.arange(
                length - 1, -1, -1, dtype=torch.int64, device=dev
            )
            keys = (windows * weights).sum(dim=1)
            # joint (subsequence, terminal class) key, one bincount per length
            uniq_keys, inverse = torch.unique(keys, return_inverse=True)
            joint = inverse * num_classes + cls_t[length - 1:]
            counts = torch.bincount(
                joint, minlength=uniq_keys.numel() * num_classes
            ).reshape(-1, num_classes)
        else:
            weights = base ** torch.arange(
                length - 1, -1, -1, dtype=torch.int64, device=dev
            )
            uniq_keys = torch.empty(0, dtype=torch.int64, device=dev)
            counts = torch.empty(0, num_classes, dtype=torch.int64, device=dev)

        if not non_zero_only:
            # reference emits the full product over *observed* symbols,
            # zero rows included: generate all m**length keys vectorially
            # and look each up among the observed-window keys.
            m = observed_t.numel()
            total_tuples = m ** length
            idx = torch.arange(total_tuples, dtype=torch.int64, device=dev)
            m_weights = m ** torch.arange(
                length - 1, -1, -1, dtype=torch.int64, device=dev
            )
            digits = (idx.unsqueeze(1) // m_weights) % max(m, 1)
            symbols = observed_t[digits] if m > 0 else digits
            keys_all = (symbols * weights).sum(dim=1)

            full_counts = torch.zeros(
                total_tuples, num_classes, dtype=torch.int64, device=dev
            )
            if uniq_keys.numel() > 0:
                pos = torch.searchsorted(uniq_keys, keys_all)
                pos_c = pos.clamp(max=uniq_keys.numel() - 1)
                found = uniq_keys[pos_c] == keys_all
                full_counts[found] = counts[pos_c[found]]
            uniq_keys, counts = keys_all, full_counts

        # decode keys back to symbol tuples (big-endian digits)
        digit_mat = (uniq_keys.unsqueeze(1) // weights) % base
        tuples = [tuple(row) for row in digit_mat.cpu().tolist()]
        counts_list = counts.cpu().tolist()

        length_distributions = []
        kept_subseqs = []
        for subseq, class_counts in zip(tuples, counts_list):
            total_occurrences = sum(class_counts)
            if total_occurrences > 0:
                class_probs = [c / total_occurrences for c in class_counts]
            else:
                class_probs = [0.0] * num_classes
            if not non_zero_only or total_occurrences > 0:
                kept_subseqs.append(subseq)
                length_distributions.append(
                    [subseq, class_counts, class_probs, total_occurrences]
                )

        all_subsequences.append(kept_subseqs)
        if len(length_distributions) > 0:
            distributions.append(length_distributions)

    return all_subsequences, distributions
