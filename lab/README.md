# lab/ — feature relationships & their effect on models

A sandbox for studying causal discovery, dependence metrics, and model ablation on
the pipeline's LOB features. It sits apart from the pipeline:

- **Read-only toward the pipeline.** It uses `RunConfig`, `DayFeatureCache` and
  the vendored `z_encoding`, and changes none of them. Nothing in `pipeline/`
  imports `lab/`, and `data/` is only ever read.
- **Results** go to `lab/runs/<stamp>_<name>/` (gitignored): `spec.yaml` (every
  knob + git rev), one CSV per table, one PNG per figure, `notes.md`.

## Start

Open `lab/playground.ipynb`. Edit `SPEC`, then run it top to bottom. Set
`source="synthetic"` first to see how each method does on a VAR with a known
graph.

```python
from lab import data, metrics, causal, effects, plots
panel = data.load("configs/april_nvda.yaml", dates=3)     # one row per block, `day` col
feats = [f for f in data.features_of(panel) if f != "log_mid_ret"]

metrics.target_table(panel, feats, "log_mid_return_fwd_1")  # IC, MI, best lag, Granger p
g     = causal.pc(panel, feats + ["log_mid_return_fwd_1"], alpha=1e-6,
                  sinks=["log_mid_return_fwd_1"])
links = causal.pcmci(panel, feats + ["log_mid_ret"], tau_max=5)
sets  = effects.feature_sets(feats, pc_graph=g, target="log_mid_return_fwd_1",
                             pcmci_links=links, pcmci_effect="log_mid_ret")
effects.ablate(panel, sets, "log_mid_return_fwd_1", reference="all")
```

## What's in it

| module | contents |
|---|---|
| `data` | `load` (config → panel), `from_parquet`, `stationarize` (levels → `d_*`), within-day `lagged`, `subsample` (whole days), `encode_symbols` (the pipeline's z-encoding) |
| `metrics` | correlation, kNN MI, VIF, lead-lag `xcorr`, pooled `granger` (pairwise / conditional), discrete `transfer_entropy` with surrogate p-values, `target_table` |
| `causal` | `pc` (PC-stable + Meek, sink knowledge), `pcmci` (PC1 + MCI, BH-FDR), `summary_graph`, optional `tigramite_pcmci` / `var_lingam` |
| `effects` | walk-forward-by-day `evaluate`, `ablate`, `drop_one`, `permutation_importance`, `feature_sets` (incl. size-matched random controls) |
| `plots` | `heatmap`, `graph`, `xcorr_plot`, `lag_heatmap`, `ablation_bars` |
| `synthetic` | `var_panel` — structural VAR with known links (Gaussian/Laplace, optionally nonlinear) |
| `bridge` | `pipeline_config` — a chosen feature set → pipeline YAML (outputs under `outputs/lab/`) so its effect on the Kraus models is measured by the pipeline itself |

## Caveats built into the methods

- Every lag, lead and diff is taken within a day, so no lag crosses an overnight gap.
- Price levels (`log_mid`, `micro_price`, `mid_price`) are replaced by their
  within-day differences by default. Tests on levels are mostly spurious.
- The in-house PC and PCMCI use partial correlation, which is linear-Gaussian.
  For nonlinear tests, use MI/TE or `tigramite_pcmci(test="cmiknn")` on a subsample.
- Lagged-only PCMCI can report `X[t-1] → Y` when the real link is `X[t] → Y` and X
  is autocorrelated. PCMCI+ (`tigramite_pcmci(plus=True)`) resolves this.
- With 10⁵–10⁶ rows almost every dependence is "significant". Use small alphas and
  read effect sizes (`val`, `delta_r2`, `TE_corrected`), not just presence.

Optional extras: `pip install -r lab/requirements-lab.txt` (tigramite, lingam).
