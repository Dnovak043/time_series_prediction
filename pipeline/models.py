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
from .config import RunConfig

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
    m = cfg.alphabet_size
    sequences, emp_probs, seq_path = load_sequence_distribution(cfg, root)

    if progress:
        progress.update(stage="training", message=(
            f"kraus: {len(sequences)} sequences from {seq_path.name}, "
            f"device={device}"))

    model0 = None
    if t.continue_from:
        model0, meta = lk.load_model_weights(
            t.continue_from, m, t.n_qubits, learn_rho0=t.learn_rho0, device="cpu")

    def on_epoch(ep, total, loss):
        if progress and (ep % max(1, total // 200) == 0 or ep == total):
            progress.update(stage="training", pct=100.0 * ep / total,
                            message=f"epoch {ep}/{total} loss {loss:.3e}")

    model = lk.train(
        sequences, emp_probs, t.max_seq_len,
        m, t.n_qubits,
        batch_size=t.batch_size,
        lr=t.lr,
        epochs=t.epochs,
        learn_rho0=t.learn_rho0,
        model=model0,
        device=device,
        optimizer_name=t.optimizer,
        loss_kind=t.loss_kind,
        length_mixture=t.length_mixture,
        on_epoch=on_epoch,
    )

    # evaluation + persistence (same artifacts as the original script)
    p_model = lk.predict_probs(model, sequences, batch_size=2048)
    total_loss = float(sum(pe * (pe - pm) ** 2
                           for pe, pm in zip(emp_probs, p_model)))

    base = seq_path.name.replace("SEQ_DISTR_", "")
    model_dir = root / t.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    mod_path = model_dir / f"MOD_{base}_{t.n_qubits}q"
    wghts_path = model_dir / f"WGHTS_MOD_{base}_{t.n_qubits}q.pt"

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
              "device": device}
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
