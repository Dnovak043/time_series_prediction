# -*- coding: utf-8 -*-
"""
Fast unit tests for the pipeline package (no market data needed).

Run:  .env/bin/python tests/test_pipeline_units.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import RunConfig  # noqa: E402
from pipeline.features import DayFeatureCache, _parse_hhmm  # noqa: E402


def test_yaml_roundtrip():
    cfg = RunConfig()
    cfg.encode.alpha = 0.123
    cfg.data.dates = ["20250415"]
    cfg.distributions.predictors = ["tvi_n"]
    cfg.training.predictor = "tvi_n"
    with tempfile.TemporaryDirectory() as d:
        path = cfg.save(Path(d) / "cfg.yaml")
        cfg2 = RunConfig.load(path)
    assert cfg.to_dict() == cfg2.to_dict()
    print("  PASS yaml roundtrip")


def test_unknown_key_rejected():
    try:
        RunConfig.from_dict({"encode": {"n_symbols": 4, "typo_field": 1}})
    except ValueError as e:
        assert "typo_field" in str(e)
        print("  PASS unknown key rejected")
        return
    raise AssertionError("unknown key was accepted")


def test_validate():
    cfg = RunConfig()
    cfg.data.dates = ["2025-04-01"]          # wrong format
    cfg.training.predictor = "nonexistent"
    problems = cfg.validate()
    assert any("yyyymmdd" in p for p in problems), problems
    assert any("nonexistent" in p for p in problems), problems
    assert RunConfig().validate() == []
    print("  PASS validate")


def test_cache_key():
    cfg = RunConfig()
    c1 = DayFeatureCache(cfg.data, cfg.featurize)
    k1 = c1.params_key()
    assert c1.params_key() == k1                      # stable
    cfg.featurize.frequency = 200
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() != k1
    cfg.featurize.frequency = 100
    cfg.featurize.use_cache = False                   # non-output param:
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() == k1
    print("  PASS cache key semantics")


def test_parse_hhmm():
    import datetime
    assert _parse_hhmm("09:30") == datetime.time(9, 30)
    assert _parse_hhmm(570) == datetime.time(9, 30)   # YAML base-60 footgun
    print("  PASS session time parsing")


def test_legacy_imports_side_effect_free():
    import matplotlib
    matplotlib.use("Agg")
    import pipeline  # noqa: F401  (sets sys.path)
    import process_distributions as pdst
    import LearningKraus as lk
    assert callable(pdst.run_legacy_driver) and callable(lk.main)
    print("  PASS legacy modules import without running")


def test_model_registry():
    from pipeline.models import REGISTRY
    assert "kraus" in REGISTRY
    print("  PASS model registry")


def test_multivariate_predictor():
    from pipeline.config import predictor_key

    trio = ["ofi_L10_norm_n", "micro_price", "vpin"]
    cfg = RunConfig()
    cfg.data.symbol = "NVDA"
    cfg.data.dates = ["20250401"]
    cfg.distributions.predictors = ["tvi_n", trio]
    cfg.training.predictor = list(trio)

    # keys, alphabet, naming
    assert predictor_key("tvi_n") == "tvi_n"
    assert predictor_key(trio) == "ofi_L10_norm_n+micro_price+vpin"
    assert cfg.alphabet_size_for("tvi_n") == 16
    assert cfg.alphabet_size_for(trio) == 4 ** 4
    # verbatim LearningKraus_multivariate naming: predicted-first-last
    assert cfg.seq_distr_name(trio) == (
        "SEQ_DISTR_NVDA_multivariate_log_mid-ofi_L10_norm_n-vpin_202504")
    assert cfg.seq_distr_name("tvi_n") == (
        "SEQ_DISTR_NVDA_bivariate_log_mid-tvi_n_202504")

    # validation: list training.predictor accepted iff listed; duplicate
    # variables in the joint encoding rejected
    assert cfg.validate() == [], cfg.validate()
    cfg.training.predictor = ["micro_price", "vpin"]
    assert any("training.predictor" in p for p in cfg.validate())
    cfg.training.predictor = list(trio)
    cfg.distributions.predictors = ["tvi_n", ["log_mid", "vpin"]]
    assert any("duplicate variables" in p.lower() for p in cfg.validate())

    # yaml roundtrip with nested predictors
    cfg.distributions.predictors = ["tvi_n", trio]
    with tempfile.TemporaryDirectory() as d:
        cfg2 = RunConfig.load(cfg.save(Path(d) / "cfg.yaml"))
    assert cfg2.distributions.predictors == ["tvi_n", trio]
    assert cfg2.training.predictor == trio
    print("  PASS multivariate predictor (keys/alphabet/naming/validation)")


def test_asset_paths():
    catalog = {"NVDA": "data/NVDA_INTC", "INTC": "data/NVDA_INTC",
               "AAPL": "data/AAPL", "IBM": "data/IBM"}

    # resolution: empty mapping and unlisted symbols fall back to data_path
    cfg = RunConfig()
    assert cfg.data.resolved_data_path() == cfg.data.data_path
    cfg.data.asset_paths = dict(catalog)
    cfg.data.symbol = "AAPL"
    assert cfg.data.resolved_data_path() == "data/AAPL"
    cfg.data.symbol = "INTC"
    assert cfg.data.resolved_data_path() == "data/NVDA_INTC"
    cfg.data.symbol = "TSLA"                        # not in the catalog
    assert cfg.data.resolved_data_path() == cfg.data.data_path
    assert cfg.validate() == [], cfg.validate()
    cfg.data.asset_paths = {"NVDA": 3}
    assert any("asset_paths" in p for p in cfg.validate())

    # cache keys: legacy configs (no resolution) keep their existing entries;
    # a resolved symbol keys on its directory, since same-named files in
    # different directories hold different assets
    legacy = RunConfig()
    legacy.data.symbol = "TSLA"
    legacy_key = DayFeatureCache(legacy.data, legacy.featurize).params_key()
    cfg = RunConfig()
    cfg.data.asset_paths = dict(catalog)
    cfg.data.symbol = "TSLA"                        # falls back -> same key
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() == legacy_key
    cfg.data.asset_paths["TSLA"] = "data/TSLA"      # resolves -> keyed on dir
    k_tsla = DayFeatureCache(cfg.data, cfg.featurize).params_key()
    assert k_tsla != legacy_key
    cfg.data.asset_paths["TSLA"] = "data/OTHER"     # different dir, new key
    assert DayFeatureCache(cfg.data, cfg.featurize).params_key() != k_tsla

    # yaml roundtrip with the mapping
    cfg = RunConfig()
    cfg.data.asset_paths = dict(catalog)
    with tempfile.TemporaryDirectory() as d:
        cfg2 = RunConfig.load(cfg.save(Path(d) / "cfg.yaml"))
    assert cfg2.data.asset_paths == catalog

    # the checked-in default config carries the catalog and validates
    cfg3 = RunConfig.load(Path(__file__).resolve().parent.parent
                          / "configs" / "default.yaml")
    assert cfg3.data.asset_paths == catalog
    assert cfg3.validate() == [], cfg3.validate()

    # control-panel round trip of a dict field
    from pipeline.ui import _make_widget, _read_widget
    from pipeline.config import field_info
    info = [i for i in field_info(cfg.data) if i["name"] == "asset_paths"][0]
    w = _make_widget(info)
    assert _read_widget(w, {}) == catalog
    w.value = ""
    assert _read_widget(w, {}) == {}
    print("  PASS asset paths (resolution/cache key/yaml/default.yaml/ui)")


def test_no_dead_knobs():
    """Every config field must actually be read by the code.

    Guards the project's no-invisible-parameters rule from its inverse: a
    knob that is exposed in YAML/UI/CLI but that nothing consumes, so
    changing it silently does nothing. Two of these have already shipped
    (training.num_workers, which LearningKraus.train() ignored until
    [vendoring fix 1], and the notebook's PREDICTED).
    """
    import glob
    import re
    from dataclasses import fields as dc_fields

    from pipeline.config import STAGES

    root = Path(__file__).resolve().parent.parent
    sources = []
    for pattern in ("pipeline/*.py", "scripts/*.py"):
        for path in glob.glob(str(root / pattern)):
            if path.endswith("config.py"):
                continue          # declaration site, not a use
            sources.append(Path(path).read_text())
    blob = "\n".join(sources)

    dead = []
    for stage, cls in STAGES.items():
        for f in dc_fields(cls):
            # attribute access (cfg.training.epochs) or getattr/string form
            # (getattr(d, "instrument_filter", False))
            if not (re.search(rf"\.{f.name}\b", blob)
                    or re.search(rf'["\']{f.name}["\']', blob)):
                dead.append(f"{stage}.{f.name}")
    assert not dead, f"config fields nothing reads: {dead}"
    n = sum(len(dc_fields(c)) for c in STAGES.values())
    print(f"  PASS no dead knobs ({n} config fields, all consumed)")


def test_training_knobs_reach_the_trainer():
    """The knobs added when training was consolidated must be wired, not
    just declared — models.py should read each of them."""
    root = Path(__file__).resolve().parent.parent
    models_src = (root / "pipeline" / "models.py").read_text()
    for field in ("seed", "num_workers", "eval_batch_size",
                  "plot_entries", "plot_dpi"):
        assert f"t.{field}" in models_src, f"models.py never reads t.{field}"

    # and LearningKraus.train() must actually honour num_workers
    # ([vendoring fix 1]) rather than hardcoding it
    lk_src = (root / "LearningKraus.py").read_text()
    train_body = lk_src[lk_src.index("def train("):]
    loader = train_body[train_body.index("DataLoader("):]
    # delimit on collate_fn rather than the first ')' — the [vendoring fix 1]
    # comment inside the call contains "main()'s"
    loader = loader[:loader.index("collate_fn")]
    # drop comment lines: the [vendoring fix 1] note quotes both the old
    # "num_workers=0" and "main()'s", which would confuse either check
    code = "\n".join(ln for ln in loader.splitlines()
                     if not ln.strip().startswith("#"))
    assert "num_workers=num_workers" in code, (
        "LearningKraus.train() DataLoader is not honouring num_workers")
    assert "num_workers=0" not in code, (
        "LearningKraus.train() DataLoader still hardcodes num_workers=0")
    print("  PASS training knobs reach the trainer (incl. vendoring fix 1)")


def test_training_jobs_handle_multivariate_predictors():
    """A list (multivariate) entry in training.predictors must survive job
    construction AND every display/format site in stage 3.

    Regression: the job listing formatted the raw predictor with a string
    spec (`f"{j['predictor']:16s}"`), which raises TypeError on a list —
    crashing the April run after distributions and ensembles had already
    completed. Jobs now carry a printable `label`.
    """
    import sys as _sys

    import matplotlib
    matplotlib.use("Agg")
    root = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "scripts"))
    from run_april import build_training_jobs

    trio = ["ofi_L10_norm_n", "micro_price", "vpin"]
    cfg = RunConfig()
    cfg.data.symbol = "NVDA"
    cfg.data.dates = ["20250401"]
    cfg.distributions.predictors = ["tvi_n", trio]
    cfg.training.predictor = "tvi_n"
    cfg.training.predictors = ["tvi_n", trio]      # one string, one list
    assert cfg.validate() == [], cfg.validate()

    with tempfile.TemporaryDirectory() as d:
        cfg_path = cfg.save(Path(d) / "april_nvda.yaml")
        jobs = build_training_jobs({"NVDA": cfg_path}, gpus=["0", "1"])

    assert len(jobs) == 2, jobs
    # the raw predictor is preserved for the trainer...
    assert jobs[0]["predictor"] == "tvi_n"
    assert jobs[1]["predictor"] == trio
    # ...and a printable label exists for every job
    assert jobs[0]["label"] == "tvi_n"
    assert jobs[1]["label"] == "ofi_L10_norm_n+micro_price+vpin"
    for j in jobs:
        assert isinstance(j["label"], str)
        # the exact format spec used by the stage-3 job listing, in both
        # run_april.py and april_run.ipynb — this is what used to raise
        f"    {j['symbol']:6s} x {j['label']:24s} -> {j['device']}"
        assert "+" not in j["run_id"] or not isinstance(j["predictor"], str)

    # round-robin still assigns distinct devices
    assert [j["device"] for j in jobs] == ["cuda:0", "cuda:1"]
    # run dirs stay unique and filesystem-safe per (symbol, predictor)
    assert len({j["run_id"] for j in jobs}) == 2
    print("  PASS multivariate predictors survive job build + listing")


def test_device_cuda_index_loads_in_control_panel():
    """The per-model config the April fan-out writes must open in the panel.

    Regression: `_train_one` sets training.device='cuda:3' and train_model
    persists it, but device declared choices=[auto,cuda,cpu,mps]; the panel
    built W.Dropdown(options=..., value='cuda:3') and raised TraitError, so
    exactly the configs a real run produces were unopenable. device is now
    free_form: dropdown of presets + a specification textbox.
    """
    import matplotlib
    matplotlib.use("Agg")
    from pipeline.ui import ChoiceOrCustom, ControlPanel, _make_widget, \
        _read_widget
    from pipeline.config import field_info

    info = [i for i in field_info(RunConfig().training)
            if i["name"] == "device"][0]
    assert info["free_form"], "training.device must stay free_form"
    assert info["choices"] == ["auto", "cuda", "cpu", "mps"]

    # presets remain a dropdown selection; the second chooser is greyed out
    w_preset = _make_widget(info)
    assert isinstance(w_preset, ChoiceOrCustom)
    assert w_preset.value == "auto" and w_preset._custom.disabled

    # an indexed device is representable, not a construction error
    w_custom = _make_widget(dict(info, value="cuda:3"))
    assert w_custom.value == "cuda:3"
    assert _read_widget(w_custom, "auto") == "cuda:3"      # collect() path
    w_custom.value = "cpu"                                  # apply() path
    assert w_custom.value == "cpu" and w_custom._custom.disabled
    w_custom.value = "cuda:7"
    assert w_custom.value == "cuda:7" and not w_custom._custom.disabled

    # end to end: save what the fan-out saves, reopen it in the panel
    cfg = RunConfig()
    cfg.data.dates = ["20250401"]
    cfg.training.device = "cuda:3"
    assert cfg.validate() == [], cfg.validate()
    with tempfile.TemporaryDirectory() as d:
        path = cfg.save(Path(d) / "config.yaml")
        panel = ControlPanel(path, repo_root=Path(d))
        assert panel.collect().training.device == "cuda:3"
    print("  PASS device 'cuda:N' round-trips through the control panel")


def test_device_chooser_offers_only_real_devices():
    """The device chooser must be a dropdown of devices that exist here, so
    an invalid device cannot be entered through the panel at all — and it
    must still represent a value from another machine, since configs are
    portable between the Mac and the compute box.
    """
    import ipywidgets as W
    import matplotlib
    matplotlib.use("Agg")
    from pipeline.config import field_info
    from pipeline.models import available_devices
    from pipeline.ui import _make_widget

    detected = available_devices()
    assert "cpu" in detected, detected
    for d in detected:
        assert d == "cpu" or d.startswith(("cuda", "mps")), d
    # indexed entries are well formed and pass validation
    for d in detected:
        cfg = RunConfig()
        cfg.training.device = d
        assert cfg.validate() == [], f"{d}: {cfg.validate()}"

    info = [i for i in field_info(RunConfig().training)
            if i["name"] == "device"][0]
    assert info["options_provider"] == "devices"

    # second chooser is a Dropdown (not free text) -> no invalid input path
    w = _make_widget(info)
    assert isinstance(w._custom, W.Dropdown), type(w._custom).__name__
    # it never re-offers what the preset dropdown already has
    assert not set(w._custom.options) & set(info["choices"])

    # a device absent from this machine still round-trips exactly
    foreign = "cuda:3"
    w2 = _make_widget(dict(info, value=foreign))
    assert foreign in w2._custom.options, w2._custom.options
    assert w2.value == foreign
    with tempfile.TemporaryDirectory() as d:
        cfg = RunConfig()
        cfg.data.dates = ["20250401"]
        cfg.training.device = foreign
        cfg2 = RunConfig.load(cfg.save(Path(d) / "c.yaml"))
        assert cfg2.training.device == foreign
    print("  PASS device chooser lists real devices, keeps foreign values")


def test_panel_save_reports_invalid_config():
    """Saving must not silently persist a config that fails validation."""
    import matplotlib
    matplotlib.use("Agg")
    from pipeline.ui import ControlPanel

    with tempfile.TemporaryDirectory() as d:
        cfg = RunConfig()
        cfg.data.dates = ["20250401"]
        path = cfg.save(Path(d) / "config.yaml")
        panel = ControlPanel(path, repo_root=Path(d))

        panel.save()
        assert "problem" not in panel.w_status.value, panel.w_status.value

        # force an invalid value the way a hand-edited YAML could
        panel._widgets[("data", "dates")].value = "2025-04-01"
        panel.save()
        status = panel.w_status.value
        assert "problem" in status and "yyyymmdd" in status, status
        # the edit is still written rather than silently dropped
        assert RunConfig.load(path).data.dates == ["2025-04-01"]
    print("  PASS panel save surfaces validation problems")


def test_choices_are_validated():
    """validate() rejects out-of-set values instead of deferring the failure
    to widget construction — for closed choices, and for free-form device."""
    for bad in ("gpu:1", "cuda:x", "CUDA", "cuda:", "cuda:1:2"):
        cfg = RunConfig()
        cfg.training.device = bad
        assert any("device" in p for p in cfg.validate()), f"accepted {bad!r}"
    for good in ("auto", "cuda", "cpu", "mps", "cuda:0", "cuda:7", "mps:0"):
        cfg = RunConfig()
        cfg.training.device = good
        assert cfg.validate() == [], f"rejected {good!r}: {cfg.validate()}"

    # closed (non-free-form) choices fields are now checked too
    cfg = RunConfig()
    cfg.featurize.resampling = "minutes"          # not in the choices
    problems = cfg.validate()
    assert any("featurize.resampling" in p for p in problems), problems
    cfg = RunConfig()
    cfg.training.optimizer = "adamax"
    assert any("training.optimizer" in p for p in cfg.validate())
    assert RunConfig().validate() == []
    print("  PASS choices validated in config, not at widget-construction")


def test_stage3_schedule_comes_from_the_config():
    """GPU selection and concurrency must be read from the config.

    Regression: the April fan-out called visible_gpu_ids() with the default
    'auto' and took concurrency from --train-parallel/TRAIN_PARALLEL, so
    training.gpus and training.max_parallel — fields that already mean
    exactly this for `train-all` — were silently ignored. Setting
    training.gpus: '0,2' to dodge busy cards had no effect (CLAUDE.md
    rule 4: every knob that affects a run must be a RunConfig field).
    """
    import sys as _sys

    import matplotlib
    matplotlib.use("Agg")
    root = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "scripts"))
    import run_april
    from run_april import build_training_jobs, make_config, training_schedule

    with tempfile.TemporaryDirectory() as d:
        original_root, run_april.ROOT = run_april.ROOT, Path(d)
        try:
            (Path(d) / "configs").mkdir()

            def generate(gpus, max_parallel):
                return {s: make_config(s, Path(d), ["20250401"], 0,
                                       epochs=5, gpus=gpus,
                                       max_parallel=max_parallel)
                        for s in ("NVDA", "INTC")}

            # an explicit GPU list reaches the scheduler
            paths = generate("0,2", 0)
            assert RunConfig.load(paths["NVDA"]).training.gpus == "0,2"
            gpus, n_par = training_schedule(paths)
            assert gpus == ["0", "2"], gpus
            jobs = build_training_jobs(paths, gpus)
            assert sorted({j["device"] for j in jobs}) == ["cuda:0", "cuda:2"]
            assert n_par == 2, n_par        # one per selected GPU

            # explicit concurrency wins over the GPU count
            gpus, n_par = training_schedule(generate("0,2", 1))
            assert n_par == 1, n_par

            # 'none' forces CPU scheduling regardless of hardware
            gpus, n_par = training_schedule(generate("none", 0))
            assert gpus == [] and n_par == 1
            jobs = build_training_jobs(generate("none", 0), gpus)
            assert {j["device"] for j in jobs} == {"cpu"}
        finally:
            run_april.ROOT = original_root
    print("  PASS stage-3 schedule is read from training.gpus/max_parallel")


def test_stage3_listing_format_sites_use_label():
    """Neither run surface may apply a format spec to the raw predictor."""
    import json as _json

    root = Path(__file__).resolve().parent.parent
    script = (root / "scripts" / "run_april.py").read_text()
    nb = _json.loads((root / "april_run.ipynb").read_text())
    notebook = "\n".join("".join(c["source"]) for c in nb["cells"]
                         if c["cell_type"] == "code")
    for name, src in (("run_april.py", script), ("april_run.ipynb", notebook)):
        assert "j['predictor']:" not in src, (
            f"{name} formats the raw predictor (breaks on list predictors)")
        assert 'j["predictor"]:' not in src, (
            f"{name} formats the raw predictor (breaks on list predictors)")
    print("  PASS stage-3 listings format the label, not the raw predictor")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for fn in [test_yaml_roundtrip, test_unknown_key_rejected, test_validate,
               test_cache_key, test_parse_hhmm,
               test_legacy_imports_side_effect_free, test_model_registry,
               test_multivariate_predictor, test_asset_paths,
               test_no_dead_knobs, test_training_knobs_reach_the_trainer,
               test_training_jobs_handle_multivariate_predictors,
               test_stage3_listing_format_sites_use_label,
               test_stage3_schedule_comes_from_the_config,
               test_device_cuda_index_loads_in_control_panel,
               test_device_chooser_offers_only_real_devices,
               test_panel_save_reports_invalid_config,
               test_choices_are_validated]:
        fn()
    print("ALL UNIT TESTS PASSED")
