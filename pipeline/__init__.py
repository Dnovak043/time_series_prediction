# -*- coding: utf-8 -*-
"""
Configurable pipeline layer over the TrainingDistributions research code.

Every stage of the original scripts (data selection, featurization,
z-encoding, distribution estimation, model training) is driven by the
dataclasses in pipeline.config, editable as YAML files, from the CLI
(``python -m pipeline``), or from the ipywidgets control panel
(pipeline_control.ipynb / pipeline.ui).

The underlying math is imported unchanged from TrainingDistributions/
and LearningKraus.py; this package only provides the seams
(PIPELINE_GUIDE.md documents the architecture).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The legacy modules import each other flat (e.g. `from read_databento_new
# import ...`), so both the repo root and TrainingDistributions must be
# importable regardless of where the interpreter was started.
for _p in (REPO_ROOT, REPO_ROOT / "TrainingDistributions"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__version__ = "0.1.0"
