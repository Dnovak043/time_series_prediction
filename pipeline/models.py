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
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig, predictor_key

REGISTRY: dict[str, callable] = {}


def register_model(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


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

    import time as _time
    _t0 = _time.time()
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

    train_seconds = _time.time() - _t0

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

    # charts: LearningKraus.plotDistributions draws the first `plot_entries`
    # entries in chunks of 62 and calls plt.show() per chunk -> 4 figures for
    # the default 200. Interactively those are 4 windows; here plt.show is
    # intercepted so each becomes its own PNG. Ported from the April harness
    # so this is the single trainer that produces the full per-model bundle.
    fig_paths: list[str] = []
    if t.plot_entries > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_base = model_dir / (
            f"{cfg.data.symbol}_{cfg.data.dates[0][:6]}_"
            f"{cfg.distributions.predicted}-{predictor_key(t.predictor)}_"
            f"{t.n_qubits}q")
        n = t.plot_entries

        def _save_instead_of_show(*a, **k):
            fig_paths.append(f"{fig_base}_{len(fig_paths) + 1}.png")
            plt.savefig(fig_paths[-1], dpi=t.plot_dpi,
                        bbox_inches="tight")
            plt.close()

        orig_show = plt.show
        plt.show = _save_instead_of_show
        try:
            lk.plotDistributions(emp_probs[:n], p_model[:n], sequences[:n],
                                 f"{base} Cost={total_loss}",
                                 "Target", "Model", c1="blue", c2="red")
        finally:
            plt.show = orig_show
            plt.close("all")

    with open(mod_path, "wb") as fh:
        pickle.dump([model, sequences, emp_probs], fh)
    lk.save_model_weights(
        str(wghts_path), model,
        meta={"m": m, "n_qubits": t.n_qubits, "d": 2 ** t.n_qubits,
              "learn_rho0": t.learn_rho0, "seq_distr": str(seq_path),
              "loss": total_loss, "epochs": t.epochs,
              "trained": datetime.datetime.now().isoformat(timespec="seconds")})

    result = {"model_file": str(mod_path), "weights_file": str(wghts_path),
              "loss": total_loss, "n_sequences": len(sequences),
              "device": device, "plots": fig_paths,
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
