# -*- coding: utf-8 -*-
"""
Feature-file export for the colleague's February request.

Deliverable: ONE file per (day x symbol x aggregation x feature set).

    3 days      20250203, 20250204, 20250205
  x 2 symbols   AAPL (/feb/AAPL), NVDA (/feb/NVDA_INTC)
  x 2 aggr.     100 events, 1 second (physical time)
  x 1 set       core7
  ------------
   12 files     outputs/feb_features/FEATURES_{sym}_{date}_{aggr}_{set}.csv

This is a thin export layer: featurization is the pipeline's existing
DayFeatureCache (one decode+featurize per (symbol, day, aggregation)), so
the columns are produced by the same verified code path the distribution
runs use. This script only selects columns and writes them out — no new
math.

Feature-name mapping (colleague's names -> actual columns in
TrainingDistributions/process_distributions.py):

    microprice     -> micro_price      imbalance-weighted price
    vpin           -> vpin             (process_distributions.py:877-901)
    sigma_w        -> sigma_W          rolling RMS of log_mid_ret over W
    ovi_L_10       -> ofi_L10_norm_n   *** see note below ***
    TVI            -> tvi_n            (process_distributions.py:875)
    OBI            -> obi_L1           (process_distributions.py:980)
    log_midprice   -> log_mid          log mid-price

NOTE on ovi_L_10: there is no "OVI" in the codebase; this is read as the
deep (10-level) order-flow imbalance, which exists in FOUR variants:
    ofi_L10          event-level weighted deep OFI (raw)
    ofi_L10_n        rolling n-event sum of the above
    ofi_L10_norm     depth-normalized (divided by total bid+ask depth)
    ofi_L10_norm_n   rolling depth-normalized  <-- CHOSEN DEFAULT
`ofi_L10_norm_n` is the variant the colleague's own reference code uses
(ensemble_reference_2.py:1238) and the one the April spec trains on. If he
meant a different variant, change FEATURE_SETS below — it is a one-line
edit and a re-run.

Every knob is an explicit constant or config field below (project rule:
no invisible parameters); nothing is derived from a hidden default.

Run (compute box):
    python scripts/export_feb_features.py
    python scripts/export_feb_features.py --format parquet
    python scripts/export_feb_features.py --symbols AAPL
    python scripts/export_feb_features.py --asset-path AAPL=/feb/AAPL

Only READS the raw data. Outputs land in outputs/feb_features/.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pipeline  # noqa: E402,F401
from pipeline.config import RunConfig  # noqa: E402
from pipeline.features import DayFeatureCache  # noqa: E402

DATES = ["20250203", "20250204", "20250205"]

# symbol -> raw-data directory ON THE COMPUTE BOX (the colleague's paths).
# NVDA lives in an interleaved NVDA+INTC file, so instrument_filter must be
# on; AAPL's directory is single-symbol, where the filter is a no-op.
ASSET_PATHS = {
    "AAPL": "/feb/AAPL",
    "NVDA": "/feb/NVDA_INTC",
}
SYMBOLS = ["AAPL", "NVDA"]

# aggregation schemes: (tag, resampling mode, frequency, vol_window W)
# W is the trailing window for sigma_W. 0 would mean "auto = 3 x frequency"
# inside the pipeline; it is written out explicitly here so the value is
# visible rather than implied. events: 3 x 100 = 300 (the original driver's
# rule). seconds: 3 x 1 = 3 one-second bars, which is a very short RMS
# window -- raise SECONDS_VOL_WINDOW if the colleague wants a longer one.
EVENTS_VOL_WINDOW = 300
SECONDS_VOL_WINDOW = 3
AGGREGATIONS = [
    ("events100", "events", 100, EVENTS_VOL_WINDOW),
    ("sec1", "seconds", 1, SECONDS_VOL_WINDOW),
]

# named feature sets: set name -> ordered output columns
FEATURE_SETS = {
    "core7": ["log_mid", "micro_price", "vpin", "sigma_W",
              "ofi_L10_norm_n", "tvi_n", "obi_L1"],
}

# session window and forward horizons: the pipeline defaults, stated
# explicitly so the export is self-describing
SESSION_START = "09:30"
SESSION_END = "15:30"
FORWARD_INTERVALS = [1, 2, 3, 4]
FILE_PATTERN = "xnas-itch-{date}.mbp-10.dbn.zst"
INSTRUMENT_FILTER = True
OUTPUT_DIR = "outputs/feb_features"


def make_config(symbol: str, resampling: str, frequency: int,
                vol_window: int, asset_paths: dict) -> RunConfig:
    """One RunConfig per (symbol, aggregation) — the full parameter record."""
    cfg = RunConfig()
    cfg.data.symbol = symbol
    cfg.data.asset_paths = dict(asset_paths)
    cfg.data.dates = list(DATES)
    cfg.data.file_pattern = FILE_PATTERN
    cfg.data.instrument_filter = INSTRUMENT_FILTER
    cfg.data.session_start = SESSION_START
    cfg.data.session_end = SESSION_END
    cfg.featurize.resampling = resampling
    cfg.featurize.frequency = frequency
    cfg.featurize.vol_window = vol_window
    cfg.featurize.forward_intervals = list(FORWARD_INTERVALS)
    # own cache per (symbol, aggregation); the cache key already includes
    # resampling/frequency/vol_window, so this is belt-and-braces isolation
    cfg.featurize.cache_dir = f"{OUTPUT_DIR}/feature_cache"
    problems = cfg.validate()
    if problems:
        raise SystemExit(f"config invalid for {symbol}/{resampling}: {problems}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--dates", nargs="+", default=DATES)
    ap.add_argument("--sets", nargs="+", default=list(FEATURE_SETS),
                    help=f"feature sets to export (known: {list(FEATURE_SETS)})")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv",
                    help="output file format (default csv)")
    ap.add_argument("--asset-path", action="append", default=[],
                    metavar="SYMBOL=DIR",
                    help="override a symbol's raw-data directory "
                         "(repeatable), e.g. --asset-path AAPL=/feb/AAPL")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    asset_paths = dict(ASSET_PATHS)
    for override in args.asset_path:
        sym, _, rel = override.partition("=")
        if not sym or not rel:
            sys.exit(f"--asset-path expects SYMBOL=DIR, got {override!r}")
        asset_paths[sym] = rel

    unknown = [s for s in args.sets if s not in FEATURE_SETS]
    if unknown:
        sys.exit(f"unknown feature set(s): {unknown}; "
                 f"known: {list(FEATURE_SETS)}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_expected = (len(args.dates) * len(args.symbols)
                  * len(AGGREGATIONS) * len(args.sets))
    print(f"exporting {n_expected} files "
          f"({len(args.dates)} days x {len(args.symbols)} symbols x "
          f"{len(AGGREGATIONS)} aggregations x {len(args.sets)} sets) "
          f"-> {out_dir}")
    for name in args.sets:
        print(f"  set {name}: {FEATURE_SETS[name]}")
    print()

    written = []
    for symbol in args.symbols:
        if symbol not in asset_paths:
            sys.exit(f"no raw-data directory for {symbol!r}; pass "
                     f"--asset-path {symbol}=/path/to/dir")
        for tag, resampling, frequency, vol_window in AGGREGATIONS:
            cfg = make_config(symbol, resampling, frequency, vol_window,
                              asset_paths)
            cache = DayFeatureCache(cfg.data, cfg.featurize, repo_root=ROOT)
            print(f"=== {symbol} / {tag} "
                  f"({resampling} @ {frequency}, W={vol_window}, "
                  f"data={cfg.data.resolved_data_path()}) ===")
            for date in args.dates:
                df = cache.get(date)
                for name in args.sets:
                    columns = FEATURE_SETS[name]
                    missing = [c for c in columns if c not in df.columns]
                    if missing:
                        sys.exit(
                            f"{symbol} {date} {tag}: featurizer did not "
                            f"produce {missing}. Available: "
                            f"{sorted(df.columns)}")
                    out = df[columns]
                    fname = f"FEATURES_{symbol}_{date}_{tag}_{name}.{args.format}"
                    path = out_dir / fname
                    if args.format == "parquet":
                        out.to_parquet(path)
                    else:
                        out.to_csv(path, index=True)
                    written.append(path)
                    print(f"  {fname}  ({len(out)} rows x "
                          f"{len(columns)} features)")
            print()

    print(f"DONE: {len(written)}/{n_expected} files in {out_dir}")


if __name__ == "__main__":
    main()
