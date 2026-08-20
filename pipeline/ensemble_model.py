# -*- coding: utf-8 -*-
"""
Ensemble-model stage: the colleague's multi-encoder quantum ensemble
(LearningEnsemble.py). Frozen pre-trained Kraus encoders (the WGHTS_*
files written by the training stage) + a QuantumDecoder trained on the
ENS_TD_* tables (written by the ensemble stage) to predict the 3-class
distribution.

All math, model classes and the training loop are imported VERBATIM from
the vendored LearningEnsemble.py; this module replaces only the driver's
plumbing (paths, device, the per-class re-run) with config fields
(EnsembleModelConfig). The reproduced dataflow is his driver's active
path, in his order:

    ENS_TD_{sym}_{month}_SL_{L}_CL_{cls}_{predicted}_ALL  (per seq_len)
      -> extract_ensemble_arrays -> flatten_multilength_training_data
      -> per-channel aggregation (the alphabet m is derived from the data
         exactly the way his driver does it)
      -> load_model_weights for EVERY channel's WGHTS_* encoder
      -> MultiEncoderQuantumEnsemble over the FIRST n-1 encoders (his
         driver deliberately leaves the multivariate channel out of the
         trained ensemble)
      -> train_multi_encoder_ensemble (decoder-only, encoders frozen)
      -> save_ensemble_model -> agreement evaluation.

His ENS_MD_* file name carries no class tag (running his script for a
second class overwrites the first model), so each class trains into its
own directory: {ensemble_model.model_dir}/{cls}/ENS_MD_{sym}_{month}_ .
The two computations his driver stores but never uses
(same_length_joint_probs_from_original / _from_flat, marked "NOT USED
YET" in his file) are not repeated here — they have no effect on any
output.
"""
from __future__ import annotations

import datetime
import pickle
import time
from pathlib import Path

from . import REPO_ROOT
from .config import RunConfig
from .models import resolve_device


def _month(cfg: RunConfig) -> str:
    return cfg.data.dates[0][:6]


def ens_td_path(cfg: RunConfig, seq_len: int, class_name: str,
                repo_root: Path | None = None) -> Path:
    """His loader's file name, verbatim, in ensemble.output_dir."""
    name = (f"ENS_TD_{cfg.data.symbol}_{_month(cfg)}_SL_{seq_len}"
            f"_CL_{class_name}_{cfg.distributions.predicted}_ALL")
    return Path(repo_root or REPO_ROOT) / cfg.ensemble.output_dir / name


def encoder_weights_path(cfg: RunConfig, channel: int,
                         repo_root: Path | None = None) -> Path:
    """His model_file_name, verbatim, in training.model_dir."""
    em = cfg.ensemble_model
    # the SAME builder the trainer writes with (training.weights_scheme),
    # so the encoder this stage looks for is by construction the file the
    # training stage produced
    _mod, name = cfg.model_names(
        em.channel_names[channel], em.channel_qubits[channel],
        multivariate=not isinstance(em.channels[channel], str))
    return Path(repo_root or REPO_ROOT) / cfg.training.model_dir / name


def ensemble_model_dir(cfg: RunConfig, class_name: str,
                       repo_root: Path | None = None) -> Path:
    return Path(repo_root or REPO_ROOT) / cfg.ensemble_model.model_dir / class_name


def train_ensemble_model(cfg: RunConfig, class_name: str, progress=None,
                         repo_root: Path | None = None) -> dict:
    """Train ONE class's ensemble — the body of his driver for one clsName."""
    import LearningEnsemble as le    # vendored; import made safe by
                                     # [vendoring fix 1]

    em = cfg.ensemble_model
    root = Path(repo_root or REPO_ROOT)
    device = resolve_device(em.device)
    n_channels = len(em.channels)

    # ---- load the ENS_TD tables, one per sequence length ------------------
    training_data = {}
    for seq_len in em.seq_lens:
        path = ens_td_path(cfg, seq_len, class_name, root)
        if not path.exists():
            raise FileNotFoundError(
                f"ensemble table not found: {path}\n"
                "Run the ensemble stage first (python -m pipeline ensemble) "
                "— ensemble.class_names must include "
                f"{class_name!r} and ensemble.seq_lengths must include "
                f"{seq_len}.")
        with open(path, "rb") as fh:
            joint_data, component_data = pickle.load(fh)
        training_data[seq_len] = le.extract_ensemble_arrays(
            joint_data, component_data)

    flat = le.flatten_multilength_training_data(
        training_data_by_len=training_data,
        length_weighting=em.length_weighting,
        class_values=tuple(cfg.ensemble.class_values),
    )
    if progress:
        progress.update(stage="ensemble-model", message=(
            f"{class_name}: {len(flat['sequences'])} joint sequences from "
            f"{len(em.seq_lens)} ENS_TD tables, device={device}"))

    # ---- per-channel encoders --------------------------------------------
    # m (alphabet size) is derived from the data exactly as his driver does:
    # last lexicographically-ordered unique sequence at the longest length.
    # A wrong m fails loudly in load_state_dict, same as his run would.
    encoders = {}
    for channel in range(n_channels):
        channel_data = le.aggregate_flat_channel_data(flat, channel)
        dist = le.get_channel_fixed_length_distribution(
            channel_data, L=max(em.seq_lens))
        m = 1 + max(dist["X"][-1])
        w_path = encoder_weights_path(cfg, channel, root)
        if not w_path.exists():
            raise FileNotFoundError(
                f"pre-trained encoder not found: {w_path}\n"
                "Train the Kraus models first (stage 3 / `pipeline train`).")
        encoders[channel], _meta = le.load_model_weights(
            str(w_path), m, em.channel_qubits[channel],
            learn_rho0=True, device="cpu")

    # ---- ensemble over the FIRST n-1 encoders (his driver, verbatim:
    # the multivariate channel is loaded above but not ensembled) ----------
    ensemble = le.MultiEncoderQuantumEnsemble(
        encoders=[encoders[ch] for ch in range(n_channels - 1)],
        d_out=em.d_out,
        use_unitary=em.use_unitary,
        normalization_point=em.normalization_point,
    )
    sequences_list = flat["X_by_channel"][0:n_channels - 1]

    out_dir = ensemble_model_dir(cfg, class_name, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_name = f"ENS_MD_{cfg.data.symbol}_{_month(cfg)}_"   # his mName, verbatim
    meta = {
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": em.epochs,
        "freeze_decoder": em.freeze_decoder,
        "freeze_encoders": em.freeze_encoders,
        "n_encoders": n_channels,
        "predictors": list(em.channel_names)[:n_channels],
        "predicted": cfg.distributions.predicted,
        "#classes": em.d_out,
        "batch_size": em.batch_size,
        "symbol": cfg.data.symbol,
        "class_name": class_name,
    }

    started = time.time()
    trained, meta_out = le.train_multi_encoder_ensemble(
        ensemble_model=ensemble,
        sequences_list=sequences_list,
        target_distributions=flat["Y_dist"],
        global_weights=flat["weights"],
        d_out=em.d_out,
        batch_size=em.batch_size,
        lr=em.lr,
        epochs=em.epochs,
        freeze_encoders=em.freeze_encoders,
        freeze_decoder=em.freeze_decoder,
        device=device,
        prediction_loss=em.prediction_loss,
        lambda_enc=em.lambda_enc,
        lambda_pred=em.lambda_pred,
        checkpoint_every=None,
        checkpoint_path_prefix=str(out_dir / m_name),
        checkpoint_meta=meta,
    )
    train_seconds = time.time() - started

    meta_out["training_date"] = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    model_path = out_dir / m_name
    le.save_ensemble_model(str(model_path), trained, meta=meta_out)

    result = {"model_file": str(model_path), "class_name": class_name,
              "symbol": cfg.data.symbol, "device": device,
              "n_sequences": len(flat["sequences"]),
              "train_seconds": round(train_seconds, 1)}

    # ---- his evaluate_ensemble block -------------------------------------
    if em.evaluate:
        preds = le.get_ensemble_predictions_ordered(
            ensemble_model=trained,
            sequences_list=sequences_list,
            batch_size=em.eval_batch_size,
            device=device,
            class_values=tuple(cfg.ensemble.class_values),
        )
        eval_results = le.compute_ensemble_prediction_agreement(
            sequences=flat["sequences"],
            emp_target_dists=flat["Y_dist"],
            mod_target_dists=preds["pred_distributions"],
            class_values=flat["class_values"],
            class_names=["Down", "Neutral", "Up"],
            weights=flat["weights"],
            counts=flat["counts"],
            seq_lengths=flat["seq_lengths"],
        )
        print(f"\n[{cfg.data.symbol} {class_name}] agreement by class:")
        for name in ["Down", "Neutral", "Up"]:
            print(f"  {name:8s}: "
                  f"{eval_results['agreement_pct_by_class'][name]:6.2f}% "
                  f"({eval_results['agreements_by_class'][name]}"
                  f"/{eval_results['totals_by_class'][name]})")
        print(f"  unique-sequence agreement:      "
              f"{eval_results['agreement_pct']:.2f}%")
        print(f"  training-weighted agreement:    "
              f"{eval_results['weighted_agreement_pct']:.2f}%")
        print(f"  occurrence-weighted agreement:  "
              f"{eval_results['count_weighted_agreement_pct']:.2f}%")
        result["agreement_pct"] = eval_results["agreement_pct"]
        result["weighted_agreement_pct"] = eval_results[
            "weighted_agreement_pct"]
        result["count_weighted_agreement_pct"] = eval_results[
            "count_weighted_agreement_pct"]
        result["agreement_pct_by_class"] = dict(
            eval_results["agreement_pct_by_class"])

    if progress:
        progress.update(stage="ensemble-model",
                        message=f"{class_name} -> {model_path}")
    return result


def run_ensemble_models(cfg: RunConfig, run_id: str | None = None,
                        repo_root: Path | None = None) -> dict:
    """CLI entry point: one trained ensemble per ensemble_model.class_names
    entry, sequentially. (The April script fans the same per-class calls
    across GPUs instead.)"""
    from .runner import RUNS_DIR, RunProgress, new_run_id

    root = Path(repo_root or REPO_ROOT)
    run_id = run_id or new_run_id("ensemble-model")
    progress = RunProgress(root / RUNS_DIR / run_id)
    cfg.save(progress.run_dir / "config.yaml")
    results = {}
    try:
        classes = list(cfg.ensemble_model.class_names)
        for i, class_name in enumerate(classes):
            results[class_name] = train_ensemble_model(
                cfg, class_name, progress=progress, repo_root=root)
            progress.update(stage="ensemble-model",
                            pct=100.0 * (i + 1) / len(classes))
        progress.done(f"{len(results)} ensemble model(s) trained")
        return results
    except Exception as e:  # noqa: BLE001
        progress.fail(f"{type(e).__name__}: {e}")
        raise
