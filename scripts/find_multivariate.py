# -*- coding: utf-8 -*-
"""Report every multivariate (nested-list) entry in any config on disk.

A predictor/channel entry that is a LIST rather than a string is a
multivariate joint encoding. This scans the config files a run may have
written and says exactly which file, which section, and which entry -- so
"my config shows multivariate" can be pinned to a file instead of inferred.

Run:  .env/bin/python scripts/find_multivariate.py
      .env/bin/python scripts/find_multivariate.py path/to/dir_or_file.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# sections whose entries may legitimately be lists
LIST_FIELDS = [
    ("distributions", "predictors"),
    ("ensemble", "predictors"),
    ("ensemble_model", "channels"),
    ("training", "predictors"),
    ("training", "predictor"),
]
DEFAULT_GLOBS = ["configs/*.yaml", "outputs/**/configs/*.yaml",
                 "outputs/runs/*/config.yaml"]


def scan(path: Path) -> list[str]:
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except Exception as e:                                   # noqa: BLE001
        return [f"  ! could not parse: {e}"]
    hits = []
    for section, field in LIST_FIELDS:
        val = (doc.get(section) or {}).get(field)
        if val is None:
            continue
        entries = val if isinstance(val, list) else [val]
        # `training.predictor` is a single value, not a collection
        if (section, field) == ("training", "predictor"):
            entries = [val]
        for e in entries:
            if isinstance(e, list):
                hits.append(f"  MULTIVARIATE  {section}.{field}: {e}")
    return hits


def main(argv=None) -> None:
    args = (argv or sys.argv[1:])
    if args:
        targets = []
        for a in args:
            p = Path(a)
            targets += sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    else:
        targets = []
        for g in DEFAULT_GLOBS:
            targets += sorted(ROOT.glob(g))

    if not targets:
        print("no config files found")
        return

    total = 0
    for path in targets:
        hits = scan(path)
        rel = path.relative_to(ROOT) if ROOT in path.parents else path
        if hits:
            total += len(hits)
            print(f"\n{rel}")
            for h in hits:
                print(h)
        else:
            print(f"{rel}: all bivariate (no nested entries)")

    print(f"\n{total} multivariate entr{'y' if total == 1 else 'ies'} "
          f"across {len(targets)} config file(s)")
    if total:
        print("Note: configs/april_*.yaml belong to the April experiment "
              "(PR #11) and are SUPPOSED to contain one multivariate entry.\n"
              "The SQ_PRB experiment writes outputs/sqprb/configs/*.yaml and "
              "should show none.")


if __name__ == "__main__":
    main()
