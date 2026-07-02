# -*- coding: utf-8 -*-
"""
ipywidgets control panel for the pipeline (JupyterLab / VS Code notebooks).

Usage in a notebook (see pipeline_control.ipynb):

    from pipeline.ui import ControlPanel
    panel = ControlPanel("run.yaml")
    panel.show()

Widgets are generated from the dataclass schema in pipeline.config
(field types + metadata), so a field added to a config dataclass shows up
here without UI changes. Runs launch as detached subprocesses
(`python -m pipeline run/train`), so closing the notebook or dropping the
SSH connection does not kill them; the Run tab re-attaches via the
run's progress.json.
"""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

import ipywidgets as W
from IPython.display import display

from . import REPO_ROOT
from .config import STAGES, RunConfig, field_info
from .runner import RUNS_DIR, list_runs, read_progress, run_async, stop_run

_STYLE = {"description_width": "170px"}
_LAYOUT = W.Layout(width="440px")


# ---------------------------------------------------------------------------
# widget <-> field mapping
# ---------------------------------------------------------------------------
def _make_widget(info: dict) -> W.Widget:
    name, value = info["name"], info["value"]
    kw = dict(description=name, style=_STYLE, layout=_LAYOUT,
              tooltip=info["help"])
    if info["choices"]:
        return W.Dropdown(options=info["choices"], value=value, **kw)
    if isinstance(value, bool):
        return W.Checkbox(value=value, indent=False, **kw)
    if isinstance(value, int):
        return W.IntText(value=value, **kw)
    if isinstance(value, float):
        return W.FloatText(value=value, step=None, **kw)
    if isinstance(value, list):
        return W.Text(value=", ".join(str(v) for v in value), **kw)
    return W.Text(value="" if value is None else str(value), **kw)


def _read_widget(widget: W.Widget, template_value):
    v = widget.value
    if isinstance(template_value, list):
        items = [s.strip() for s in str(v).split(",") if s.strip()]
        if all(isinstance(t, int) for t in template_value) and template_value:
            return [int(s) for s in items]
        return items
    if isinstance(template_value, bool):
        return bool(v)
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(v)
    if isinstance(template_value, float):
        return float(v)
    return v


# ---------------------------------------------------------------------------
class ControlPanel:
    POLL_SECONDS = 2.0

    def __init__(self, config_path: str | Path = "run.yaml",
                 repo_root: Path | None = None):
        self.root = Path(repo_root or REPO_ROOT)
        self.config_path = self.root / config_path
        self.cfg = (RunConfig.load(self.config_path)
                    if self.config_path.exists() else RunConfig())
        self._defaults = RunConfig()          # template values for parsing
        self._widgets: dict[tuple[str, str], W.Widget] = {}
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self._build()

    # -- construction -----------------------------------------------------------
    def _stage_box(self, stage_name: str) -> W.Widget:
        stage_obj = getattr(self.cfg, stage_name)
        basic, advanced = [], []
        for info in field_info(stage_obj):
            w = _make_widget(info)
            self._widgets[(stage_name, info["name"])] = w
            row = W.HBox([w, W.HTML(
                f"<span style='color:#888;font-size:11px'>{info['help']}</span>",
                layout=W.Layout(max_width="380px"))])
            (advanced if info["advanced"] else basic).append(row)
        children = basic
        if advanced:
            acc = W.Accordion(children=[W.VBox(advanced)], selected_index=None)
            acc.set_title(0, "Advanced")
            children = basic + [acc]
        return W.VBox(children)

    def _build(self):
        # -- config bar --------------------------------------------------------
        self.w_path = W.Text(value=str(self.config_path),
                             description="config file", style=_STYLE,
                             layout=W.Layout(width="520px"))
        b_load = W.Button(description="Load", icon="folder-open")
        b_save = W.Button(description="Save", icon="save", button_style="primary")
        b_validate = W.Button(description="Validate", icon="check")
        self.w_status = W.HTML()
        b_load.on_click(lambda _: self.load())
        b_save.on_click(lambda _: self.save())
        b_validate.on_click(lambda _: self.validate())
        config_bar = W.HBox([self.w_path, b_load, b_save, b_validate,
                             self.w_status])

        # -- stage tabs ----------------------------------------------------------
        stage_names = list(STAGES)
        tabs = [self._stage_box(s) for s in stage_names]
        tabs.append(self._run_tab())
        tabs.append(self._results_tab())
        self.tabs = W.Tab(children=tabs)
        for i, name in enumerate(stage_names + ["run & monitor", "results"]):
            self.tabs.set_title(i, name)

        self.widget = W.VBox([config_bar, self.tabs])

    # -- config sync ---------------------------------------------------------------
    def collect(self) -> RunConfig:
        """Read every widget back into a RunConfig."""
        d = {}
        for stage_name in STAGES:
            stage_defaults = getattr(self._defaults, stage_name)
            sd = {}
            for info in field_info(getattr(self.cfg, stage_name)):
                w = self._widgets[(stage_name, info["name"])]
                template = getattr(stage_defaults, info["name"])
                sd[info["name"]] = _read_widget(w, template)
            d[stage_name] = sd
        return RunConfig.from_dict(d)

    def apply(self, cfg: RunConfig):
        """Push a RunConfig into the widgets."""
        self.cfg = cfg
        for stage_name in STAGES:
            for info in field_info(getattr(cfg, stage_name)):
                w = self._widgets[(stage_name, info["name"])]
                v = info["value"]
                if isinstance(v, list):
                    w.value = ", ".join(str(x) for x in v)
                elif isinstance(w, (W.IntText, W.FloatText, W.Checkbox, W.Dropdown)):
                    w.value = v
                else:
                    w.value = "" if v is None else str(v)

    def _flash(self, msg: str, ok: bool = True):
        color = "#2a2" if ok else "#c22"
        self.w_status.value = f"<b style='color:{color}'>&nbsp;{msg}</b>"

    def save(self):
        try:
            cfg = self.collect()
            path = Path(self.w_path.value)
            if not path.is_absolute():
                path = self.root / path
            cfg.save(path)
            self.config_path = path
            self._flash(f"saved {path.name}")
        except Exception as e:  # noqa: BLE001 - surfaced in the UI
            self._flash(f"save failed: {e}", ok=False)

    def load(self):
        try:
            path = Path(self.w_path.value)
            if not path.is_absolute():
                path = self.root / path
            self.apply(RunConfig.load(path))
            self.config_path = path
            self._flash(f"loaded {path.name}")
        except Exception as e:  # noqa: BLE001
            self._flash(f"load failed: {e}", ok=False)

    def validate(self):
        try:
            problems = self.collect().validate()
        except Exception as e:  # noqa: BLE001
            self._flash(f"invalid: {e}", ok=False)
            return
        if problems:
            self._flash("; ".join(problems), ok=False)
        else:
            self._flash("config OK")

    # -- run & monitor tab -------------------------------------------------------
    def _run_tab(self) -> W.Widget:
        b_run = W.Button(description="▶ Run distributions",
                         button_style="success", layout=W.Layout(width="180px"))
        b_train = W.Button(description="▶ Train model",
                           button_style="info", layout=W.Layout(width="180px"))
        b_train_all = W.Button(description="▶ Train all (multi-GPU)",
                               button_style="info",
                               tooltip="one training per predictor, parallel "
                                       "across visible GPUs (CPU fallback)",
                               layout=W.Layout(width="200px"))
        b_stop = W.Button(description="■ Stop", button_style="danger",
                          layout=W.Layout(width="100px"))
        self.w_runs = W.Dropdown(options=list_runs(self.root),
                                 description="run", style=_STYLE)
        b_refresh = W.Button(description="⟳", layout=W.Layout(width="40px"),
                             tooltip="refresh run list")
        b_attach = W.Button(description="Attach", tooltip="monitor selected run")

        self.w_progress = W.FloatProgress(value=0, min=0, max=100,
                                          layout=W.Layout(width="620px"))
        self.w_run_status = W.HTML()
        self.w_log = W.Textarea(layout=W.Layout(width="98%", height="260px"),
                                disabled=True)

        b_run.on_click(lambda _: self._launch("run"))
        b_train.on_click(lambda _: self._launch("train"))
        b_train_all.on_click(lambda _: self._launch("train-all"))
        b_stop.on_click(lambda _: self._stop_current())
        b_refresh.on_click(
            lambda _: setattr(self.w_runs, "options", list_runs(self.root)))
        b_attach.on_click(lambda _: self._monitor(self.w_runs.value))

        return W.VBox([
            W.HTML("<b>Launch</b> — saves the config, then starts a detached "
                   "<code>python -m pipeline</code> subprocess (survives "
                   "notebook restarts / SSH drops)."),
            W.HBox([b_run, b_train, b_train_all, b_stop]),
            W.HBox([self.w_runs, b_refresh, b_attach]),
            self.w_progress, self.w_run_status,
            W.HTML("<b>log tail</b>"), self.w_log,
        ])

    def _launch(self, command: str):
        self.save()
        try:
            run_id = run_async(self.config_path, command=command,
                               repo_root=self.root)
        except Exception as e:  # noqa: BLE001
            self._flash(f"launch failed: {e}", ok=False)
            return
        self.w_runs.options = list_runs(self.root)
        self.w_runs.value = run_id
        self._flash(f"launched {run_id}")
        self._monitor(run_id)

    def _stop_current(self):
        rid = self.w_runs.value
        if rid and stop_run(rid, self.root):
            self._flash(f"stop signal sent to {rid}")
        else:
            self._flash("nothing to stop", ok=False)

    def _monitor(self, run_id: str | None):
        if not run_id:
            return
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.POLL_SECONDS + 1)
        self._monitor_stop = threading.Event()
        stop = self._monitor_stop

        def loop():
            log_path = self.root / RUNS_DIR / run_id / "run.log"
            while not stop.is_set():
                st = read_progress(run_id, self.root)
                self.w_progress.value = float(st.get("pct", 0))
                color = {"completed": "#2a2", "failed": "#c22"}.get(
                    st.get("status"), "#06c")
                self.w_run_status.value = (
                    f"<b style='color:{color}'>{run_id} — {st.get('status')}"
                    f"</b> | {st.get('stage','')} | {st.get('message','')}"
                    + (f"<br><span style='color:#c22'>{st['error']}</span>"
                       if st.get("error") else ""))
                if log_path.exists():
                    text = log_path.read_text(errors="replace")
                    self.w_log.value = text[-8000:]
                if st.get("status") not in ("running", "unknown"):
                    break
                time.sleep(self.POLL_SECONDS)

        self._monitor_thread = threading.Thread(target=loop, daemon=True)
        self._monitor_thread.start()

    # -- results tab -----------------------------------------------------------
    def _results_tab(self) -> W.Widget:
        self.w_files = W.Dropdown(description="file", style=_STYLE,
                                  layout=W.Layout(width="620px"))
        b_refresh = W.Button(description="⟳ Refresh files")
        self.w_lengths = W.SelectMultiple(
            options=list(range(1, 13)), value=(1, 2),
            description="seq lengths", style=_STYLE, rows=6)
        self.w_topn = W.IntSlider(value=40, min=5, max=300, step=5,
                                  description="top N", style=_STYLE,
                                  layout=W.Layout(width="440px"))
        b_plot = W.Button(description="Plot", button_style="primary")
        self.w_plot_out = W.Output()

        b_refresh.on_click(lambda _: self._refresh_files())
        b_plot.on_click(lambda _: self._plot())
        self._refresh_files()

        return W.VBox([
            W.HTML("<b>Saved distributions</b> (SEQ_DISTR_* = sequence "
                   "probabilities; CLS_DISTR_* = class probabilities per "
                   "sequence)"),
            W.HBox([self.w_files, b_refresh]),
            W.HBox([self.w_lengths, self.w_topn, b_plot]),
            self.w_plot_out,
        ])

    def _refresh_files(self):
        out_dir = self.root / self.cfg.distributions.output_dir
        files = sorted(p.name for p in out_dir.glob("SEQ_DISTR_*"))
        files += sorted(p.name for p in out_dir.glob("CLS_DISTR_*"))
        self.w_files.options = files

    def _plot(self):
        import matplotlib.pyplot as plt

        name = self.w_files.value
        if not name:
            return
        path = self.root / self.cfg.distributions.output_dir / name
        lengths = set(self.w_lengths.value)
        top_n = self.w_topn.value

        with self.w_plot_out:
            self.w_plot_out.clear_output(wait=True)
            with open(path, "rb") as fh:
                payload = pickle.load(fh)

            fig, ax = plt.subplots(figsize=(14, 5))
            if name.startswith("SEQ_DISTR_"):
                distrs, _samples = payload
                rows = [(tuple(s), p) for s, p in distrs if len(s) in lengths]
                rows.sort(key=lambda r: -r[1])
                rows = rows[:top_n]
                ax.bar([str(list(s)) for s, _ in rows], [p for _, p in rows],
                       color="#3465a4")
                ax.set_ylabel("probability")
                ax.set_title(f"{name} — sequence probabilities "
                             f"(lengths {sorted(lengths)})")
            else:  # CLS_DISTR_*: rows [subseq, counts, probs, total]
                rows = [r for r in payload if len(r[0]) in lengths]
                rows.sort(key=lambda r: -r[-1])
                rows = rows[:top_n]
                labels = [str(list(r[0])) for r in rows]
                probs = [r[2] for r in rows]
                n_cls = len(probs[0]) if probs else 0
                x = range(len(rows))
                width = 0.8 / max(n_cls, 1)
                cls_names = {0: "flat(0)", 1: "up(1)", 2: "down(-1)"}
                for ci in range(n_cls):
                    ax.bar([xi + ci * width for xi in x],
                           [p[ci] for p in probs], width=width,
                           label=cls_names.get(ci, f"class {ci}"))
                ax.set_xticks([xi + width for xi in x])
                ax.set_xticklabels(labels)
                ax.legend()
                ax.set_ylabel("P(class | sequence)")
                ax.set_title(f"{name} — class probabilities "
                             f"(lengths {sorted(lengths)}, by frequency)")
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            fig.tight_layout()
            plt.show()

    # -- entry point -----------------------------------------------------------
    def show(self):
        display(self.widget)
        return self


def build_panel(config_path: str | Path = "run.yaml") -> ControlPanel:
    return ControlPanel(config_path).show()
