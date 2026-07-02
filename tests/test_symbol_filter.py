# -*- coding: utf-8 -*-
"""
Verification for the new per-symbol instrument filter (real data, 1 day).

The raw .dbn.zst files carry BOTH symbols (20250401: ~5.2M NVDA + ~0.64M
INTC events interleaved); the legacy path featurized the mixed stream. This
test checks, on 20250401:

  1. DEFAULT UNCHANGED: instrument_filter=false produces a frame identical
     to calling the legacy generate_timeseries directly (no symbol arg) —
     i.e. existing behavior and the frozen-baseline contract are untouched.
  2. The NVDA-filtered frame differs from the unfiltered one (the filter
     actually does something).
  3. The INTC-filtered frame differs from the NVDA-filtered one and is much
     smaller (~0.64M vs ~5.2M events pre-resampling).

Run:  .env/bin/python tests/test_symbol_filter.py     (~3 featurize passes,
      several minutes; only READS data/; caches under outputs/filter_check/)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from parity_check import find_data_dir  # noqa: E402
from pipeline.config import RunConfig  # noqa: E402
from pipeline.features import DayFeatureCache  # noqa: E402

DATE = "20250401"


def build(symbol: str, instrument_filter: bool):
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.data_path = str(find_data_dir())
    cfg.data.instrument_filter = instrument_filter
    cfg.featurize.use_cache = False
    return DayFeatureCache(cfg.data, cfg.featurize).build(DATE)


def main():
    ok = True

    print("=== 1/3 featurize, filter OFF (legacy mixed stream) ===")
    mixed = build("NVDA", instrument_filter=False)

    from process_distributions import generate_timeseries  # noqa: E402
    import datetime
    legacy = generate_timeseries(
        DATE, datetime.time(9, 30), datetime.time(15, 30), 100, [1, 2, 3, 4],
        300, f"xnas-itch-{DATE}.mbp-10.dbn.zst", str(find_data_dir()))
    same = mixed.equals(legacy)
    print(f"  {'PASS' if same else 'FAIL'}  filter-off == legacy call "
          f"({len(mixed)} rows)")
    ok &= same

    print("=== 2/3 featurize, filter ON, NVDA ===")
    nvda = build("NVDA", instrument_filter=True)
    diff = not nvda.equals(mixed)
    print(f"  {'PASS' if diff else 'FAIL'}  NVDA-filtered != mixed "
          f"({len(nvda)} vs {len(mixed)} rows)")
    ok &= diff

    print("=== 3/3 featurize, filter ON, INTC ===")
    intc = build("INTC", instrument_filter=True)
    diff2 = not intc.equals(nvda)
    smaller = len(intc) < len(nvda) / 2
    print(f"  {'PASS' if diff2 else 'FAIL'}  INTC-filtered != NVDA-filtered")
    print(f"  {'PASS' if smaller else 'FAIL'}  INTC much smaller "
          f"({len(intc)} vs {len(nvda)} resampled rows)")
    ok &= diff2 and smaller

    if not ok:
        print("\nSYMBOL FILTER CHECK FAILED")
        sys.exit(1)
    print("\nSYMBOL FILTER CHECK PASSED "
          "(default unchanged; filter separates the instruments)")


if __name__ == "__main__":
    main()
