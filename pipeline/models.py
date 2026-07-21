# -*- coding: utf-8 -*-
"""
Model registry: pluggable trainers over the SEQ_DISTR_* outputs.

A trainer is `fn(cfg: RunConfig, progress: RunProgress | None) -> dict`
registered under a name; the UI's model dropdown and the CLI's `train`
command are populated from REGISTRY, so adding a future model family is:

    @register_model("mymodel")
    def train_mymodel(cfg, progress=None): ...

plus (optionally) new TrainingConfig fields for its hyperparameters.
"""
from __future__ import annotations

import datetime
import pickle
import threading
import time
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig, predictor_key

REGISTRY: dict[str, callable] = {}

# pyplot is a process-global state machine, so the show-interception below
# is serialised. Separate processes each get their own lock and their own
# pyplot, so the GPU fan-out is unaffected either way.
_CHART_LOCK = threading.RLock()


def capture_plots_as_png(fig_base, dpi: int, draw) -> list[str]:
    """Run `draw()` and save every figure it would have shown, returning the
    PNG paths in draw order (`{fig_base}_1.png`, `_2.png`, ...).

    `LearningKraus.plotDistributions` renders in chunks and calls
    `plt.show()` once per chunk; intercepting `show` is the only way to
    capture those without editing vendored code.

    Three things make this safe for any scheduling:

    * **Concurrency** — the interception is held under a lock. Without it
      two in-process trainings (threads) would share one `plt.show`
      binding, interleave their figures into each other's filenames, and
      restore the original `show` out of order.
    * **The caller's figures** — only figures opened by `draw()` are
      closed. A blanket `plt.close("all")` would discard figures the
      calling notebook had open.
    * **The caller's backend** — deliberately not modified. Because the
      real `show` is never called no window can open, so forcing "Agg" (a
      process-global change that silently breaks a notebook's inline
      plotting for the rest of its session) is unnecessary.
    """
    import matplotlib.pyplot as plt

    paths: list[str] = []
    with _CHART_LOCK:
        pre_existing = set(plt.get_fignums())

        def _save_instead_of_show(*_a, **_k):
            paths.append(f"{fig_base}_{len(paths) + 1}.png")
            plt.savefig(paths[-1], dpi=dpi, bbox_inches="tight")
            plt.close()

        original_show = plt.show
        plt.show = _save_instead_of_show
        try:
            draw()
        finally:
            plt.show = original_show
            for num in set(plt.get_fignums()) - pre_existing:
                plt.close(num)
    return paths


def register_model(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def available_devices() -> list[str]:
    """Devices this machine can actually train on, most specific last:
    ['cpu', 'cuda', 'cuda:0', ...] or ['cpu', 'mps', 'mps:0'].

    Used by the control panel so the device chooser only offers what exists
    here. CUDA enumeration is delegated to pipeline.parallel.visible_gpus so
    this and `train-all` agree on the device list (and both respect an
    externally set CUDA_VISIBLE_DEVICES). Never raises: a missing or broken
    torch degrades to CPU-only rather than breaking the panel.
    """
    devices = ["cpu"]
    try:
        import torch

        from .parallel import visible_gpus
        if torch.cuda.is_available():
            devices.append("cuda")
            devices += [f"cuda:{i}" for i in visible_gpus("auto")]
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices += ["mps", "mps:0"]
    except Exception:      # torch missing/broken (or no backends attr)
        pass
    return devices


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    # deliberately not mps by default: complex-tensor ops (Kraus operators)
    # have incomplete MPS support; opt in via training.device="mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sequence_distribution(cfg: RunConfig, repo_root: Path | None = None):
    """Load [distrs, samples] and apply the length/probability filters."""
    t = cfg.training
    root = Path(repo_root or REPO_ROOT)
    path = Path(t.seq_distr_file) if t.seq_distr_file else (
        root / cfg.distributions.output_dir / cfg.seq_distr_name(t.predictor))
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(
            f"sequence distribution not found: {path}\n"
            "Run the distribution stage first, or set training.seq_distr_file.")

    with open(path, "rb") as fh:
        distrs, samples = pickle.load(fh)

    sequences, emp_probs = [], []
    for seq, prob in zip(samples, (row[1] for row in distrs)):
        if len(seq) <= t.max_seq_len and prob > t.min_seq_prob:
            sequences.append(seq)
            emp_probs.append(prob)
    return sequences, emp_probs, path


@register_model("kraus")
def train_kraus(cfg: RunConfig, progress=None, repo_root: Path | None = None) -> dict:
    """
    Train a KrausInstrument on one predictor's sequence distribution —
    the parameterized equivalent of LearningKraus.py's bivariate script body.
    """
    import LearningKraus as lk

    t = cfg.training
    root = Path(repo_root or REPO_ROOT)
    device = resolve_device(t.device)
    # n_symbols^2 for a string predictor (bivariate), n_symbols^(1+len)
    # for a list predictor (multivariate joint encoding)
    m = cfg.alphabet_size_for(t.predictor)
    sequences, emp_probs, seq_path = load_sequence_distribution(cfg, root)

    if progress:
        progress.update(stage="training", message=(
            f"kraus: {len(sequences)} sequences from {seq_path.name}, "
            f"device={device}"))

    # seeding: training.seed >= 0 makes the run reproducible; -1 is the
    # original main() behavior (unseeded, results vary run to run). This was
    # previously declared in the config but never applied here.
    if t.seed >= 0:
        import torch
        torch.manual_seed(t.seed)

    model0 = None
    if t.continue_from:
        model0, meta = lk.load_model_weights(
            t.continue_from, m, t.n_qubits, learn_rho0=t.learn_rho0, device="cpu")

    def on_epoch(ep, total, loss):
        if progress and (ep % max(1, total // 200) == 0 or ep == total):
            progress.update(stage="training", pct=100.0 * ep / total,
                            message=f"epoch {ep}/{total} loss {loss:.3e}")

    started = time.time()
    model = lk.train(
        sequences, emp_probs, t.max_seq_len,
        m, t.n_qubits,
        batch_size=t.batch_size,
        lr=t.lr,
        epochs=t.epochs,
        learn_rho0=t.learn_rho0,
        model=model0,
        num_workers=t.num_workers,
        device=device,
        optimizer_name=t.optimizer,
        loss_kind=t.loss_kind,
        length_mixture=t.length_mixture,
        on_epoch=on_epoch,
    )

    train_seconds = time.time() - started

    # evaluation + persistence (same artifacts as the original script)
    p_model = lk.predict_probs(model, sequences,
                               batch_size=t.eval_batch_size)
    total_loss = float(sum(pe * (pe - pm) ** 2
                           for pe, pm in zip(emp_probs, p_model)))

    model_dir = root / t.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(t.predictor, str):
        # original bivariate naming: MOD_<base>, WGHTS_MOD_<base>.pt
        base = seq_path.name.replace("SEQ_DISTR_", "")
        mod_path = model_dir / f"MOD_{base}_{t.n_qubits}q"
        wghts_path = model_dir / f"WGHTS_MOD_{base}_{t.n_qubits}q.pt"
        chart_tag = t.predictor
    else:
        # LearningKraus_multivariate driver naming, verbatim — including its
        # WGHTS_ prefix without MOD_; predictor part is the hand-written
        # abbreviation (training.predictor_abbrev) when given
        pred_tag = t.predictor_abbrev or (
            t.predictor[0] + "-" + t.predictor[-1])
        base = (cfg.data.symbol + "_multivariate_"
                + cfg.distributions.predicted + "-" + pred_tag
                + "_" + cfg.data.dates[0][:6])
        mod_path = model_dir / f"MOD_{base}_{t.n_qubits}q"
        wghts_path = model_dir / f"WGHTS_{base}_{t.n_qubits}q.pt"
        # charts carry the SAME predictor tag as the model files, so a
        # delivered bundle (2 result files + 4 PNGs) is internally
        # consistent; predictor_key's '+'-joined form would not match
        chart_tag = pred_tag

    # charts: LearningKraus.plotDistributions draws the first `plot_entries`
    # entries in chunks of 62 and calls plt.show() per chunk -> 4 figures for
    # the default 200. Interactively those are 4 windows; here plt.show is
    # intercepted so each becomes its own PNG. Ported from the April harness
    # so this is the single trainer that produces the full per-model bundle.
    # PERSIST FIRST. Charting is cosmetic; training is hours. Rendering
    # before the model reached disk meant any plotting error (too few
    # sequences to slice, a backend fault, a full disk) discarded the whole
    # run. Now a chart failure costs only the images.
    with open(mod_path, "wb") as fh:
        pickle.dump([model, sequences, emp_probs], fh)
    lk.save_model_weights(
        str(wghts_path), model,
        meta={"m": m, "n_qubits": t.n_qubits, "d": 2 ** t.n_qubits,
              "learn_rho0": t.learn_rho0, "seq_distr": str(seq_path),
              "loss": total_loss, "epochs": t.epochs,
              "trained": datetime.datetime.now().isoformat(timespec="seconds")})

    fig_paths: list[str] = []
    chart_error = None
    if t.plot_entries > 0:
        fig_base = model_dir / (
            f"{cfg.data.symbol}_{cfg.data.dates[0][:6]}_"
            f"{cfg.distributions.predicted}-{chart_tag}_"
            f"{t.n_qubits}q")
        n = t.plot_entries
        try:
            fig_paths = capture_plots_as_png(
                fig_base, t.plot_dpi,
                lambda: lk.plotDistributions(
                    emp_probs[:n], p_model[:n], sequences[:n],
                    f"{base} Cost={total_loss}",
                    "Target", "Model", c1="blue", c2="red"))
        except Exception as e:  # noqa: BLE001 - the model is already safe
            # reported, not raised: one model's missing charts must not kill
            # a sweep whose other trainings are still running
            chart_error = f"{type(e).__name__}: {e}"
            print(f"[warning] {mod_path.name}: charts failed ({chart_error}); "
                  f"model and weights were saved", flush=True)
            if progress:
                progress.update(stage="training",
                                message=f"charts failed: {chart_error}")

    result = {"model_file": str(mod_path), "weights_file": str(wghts_path),
              "loss": total_loss, "n_sequences": len(sequences),
              "device": device, "plots": fig_paths,
              "chart_error": chart_error,
              "symbol": cfg.data.symbol,
              "predictor": predictor_key(t.predictor),
              "train_seconds": round(train_seconds, 1)}
    if progress:
        progress.done(f"loss {total_loss:.3e} -> {wghts_path.name}")
    return result


def train_model(cfg: RunConfig, run_id: str | None = None,
                repo_root: Path | None = None) -> dict:
    """Entry point used by the CLI: dispatch to the registered trainer."""
    from .runner import RUNS_DIR, RunProgress, new_run_id

    name = cfg.training.model
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; registered: {sorted(REGISTRY)}")

    root = Path(repo_root or REPO_ROOT)
    run_id = run_id or new_run_id("train")
    progress = RunProgress(root / RUNS_DIR / run_id)
    cfg.save(progress.run_dir / "config.yaml")
    try:
        return REGISTRY[name](cfg, progress=progress, repo_root=root)
    except Exception as e:  # noqa: BLE001
        progress.fail(f"{type(e).__name__}: {e}")
        raise
