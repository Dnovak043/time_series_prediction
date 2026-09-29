# -*- coding: utf-8 -*-
"""
Experiment: a folder per experiment under lab/runs/ (gitignored) holding the
spec you ran with, every table you logged, and every figure.

    exp = Experiment("nvda_pcmci", spec)      # spec: plain dict, saved as spec.yaml
    exp.table("pcmci_links", links)           # -> pcmci_links.csv
    exp.figure("graph", fig)                  # -> graph.png
    exp.note("tau_max=5 finds tvi_n -> d_log_mid at lag 1")

Experiment.list() / Experiment.open(name) bring old results back.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from . import REPO_ROOT, RUNS_DIR


def _rel(p: Path) -> Path:
    try:
        return p.relative_to(REPO_ROOT)
    except ValueError:
        return p


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


class Experiment:
    def __init__(self, name: str, spec: dict | None = None, root: Path = RUNS_DIR):
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec = dict(spec or {})
        meta = {"name": name, "created": stamp, "git": _git_rev(), "spec": self.spec}
        (self.dir / "spec.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(meta, default=str)),
                                                           sort_keys=False))
        print(f"experiment → {_rel(self.dir)}")

    def table(self, name: str, df: pd.DataFrame | pd.Series):
        p = self.dir / f"{name}.csv"
        df.to_csv(p)
        return p

    def figure(self, name: str, fig, dpi: int = 150):
        p = self.dir / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return p

    def note(self, text: str):
        with open(self.dir / "notes.md", "a") as f:
            f.write(f"- {_dt.datetime.now():%H:%M} {text}\n")

    @staticmethod
    def list(root: Path = RUNS_DIR) -> list[str]:
        return sorted(p.name for p in Path(root).glob("*_*") if p.is_dir())

    @staticmethod
    def open(name: str, root: Path = RUNS_DIR) -> dict:
        """{'spec': ..., '<table>': DataFrame, ...} for a past experiment (name or suffix)."""
        d = next(p for p in sorted(Path(root).iterdir(), reverse=True)
                 if p.name == name or p.name.endswith("_" + name))
        out = {"spec": yaml.safe_load((d / "spec.yaml").read_text()), "dir": d}
        for c in d.glob("*.csv"):
            out[c.stem] = pd.read_csv(c, index_col=0)
        return out
