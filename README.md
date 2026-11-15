# STORM-PhysNet (journal version)

STORM-PhysNet forecasts the >2 MeV electron flux at geostationary orbit 1, 6, 12 and 24 hours ahead from a 72-hour
history of GOES electron flux, solar-wind and geomagnetic data, and tests how well GOES-15-trained models transfer
to other satellites (GOES-16, GOES-18 and India's GSAT-19 GRASP).

This repository holds the code, data, Kaggle scripts and results of the **journal version**
(prepared for submission to *Advances in Space Research*; results in `results/`). The manuscript itself is not
included; it will be linked here once published.
The conference version, *STORM-PhysNet: Multi-Horizon Forecasting of Relativistic Electron Hazards to Geostationary
Satellites with Transfer to GSAT-19* (AGERS 2026), is in the separate repository STORM-PhysNet-L1.

## Model

A delay–encode–gate–decode design:

1. a learned solar-wind propagation delay (0.5–1.5 h) on the driver channels;
2. a 2-layer, 4-head Transformer encoder (d_model 128) over the 72-hour input;
3. a gate conditioned on the southward IMF Bz;
4. forecast heads that predict a correction to persistence, with a predicted variance;
5. a loss with storm-weighted β-NLL and physics-informed penalty terms.

The journal study compares it with 28 other neural model families (ablations, alternative gates, a spectral head,
an LSTM encoder and its ablations, magnetopause features, the physics loss in physical units, Transformer controls,
a residual LSTM at three widths and standard baselines) and two classical baselines (ridge regression and
gradient-boosted trees on window summaries; `pipeline/classical_baselines.py`). Four networks are also retrained
with a second training protocol (learning rate 1e-4, one warm-up epoch, up to 60 epochs, selection on the mean
validation error over all lead times): 33 neural families and 990 training runs. Two networks and both classical
models are also retrained with 24 h and 168 h input windows, and STORM-PhysNet (protocol B) with a 24 h window;
the 24 h networks are then scored on every test, including the degraded-input probes and transfer (180 more
network runs, 1170 in all). See
`pipeline/README.md`.

## Evaluation protocol

- Leakage-controlled data pipeline: causal features, no window crosses a split boundary, scalers fitted on training
  rows only.
- Three test blocks: A (Sep 2018–Mar 2020, strictly chronological), B (2015–2016) and C (2012–2013), with a shared
  validation period; 10 seeds per model and block.
- Skill: prediction efficiency against climatology and persistence, storm-time scores, noise and driver-outage
  probes, and calibration of the predicted variance.
- Transfer: zero-shot, heads-only and full fine-tuning on chronological folds of GOES-16 (2018–2025),
  GOES-18 (2022–2026) and GSAT-19 GRASP (2017–2018).

## Repository contents

- `src/` — model, data, training and evaluation code
- `pipeline/` — block pipeline, model-family registry, transfer experiments, Kaggle script generator
- `datasets/` — GOES-15, GOES-16/18 (hourly >2 MeV), OMNI 2010–2026 and GRASP data
- `configs/` — training configuration
- `results/` — every number behind the paper's tables and figures (`numbers.json`, `extra.json`) and the run logs
- `tests/` — unit and integration tests (`python -m pytest -q tests`)
- `legacy/` — results, checkpoints, figures and notebook of the earlier single-split experiments (superseded; kept for reference)

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Full training runs on Kaggle with the scripts in `pipeline/kaggle/` (campaigns 1–8; campaign 5 trains the
physics terms at full weight from the first epoch, campaign 6 uses training protocol B, campaign 7 other input
windows, campaign 8 scores the 24 h-window networks on every test); see `pipeline/README.md`. Each Kaggle script
pins the commit it runs, and rerunning a family with the same seeds reproduces its test forecasts exactly (checked for
the 24 h residual Transformer, run in campaigns 7 and 8).

From the saved predictions (available on request, see `results/README.md`), the analysis runs in this order:

```bash
cd pipeline
python journal_analysis.py --blocks <72 h block dirs> --transfer <transfer dirs> --out <report>      # numbers.json
python journal_tables.py --numbers <report>/numbers.json --out <tables>
python gbm_importance.py --models <block-A tree dir> --out <report>/gbm_importance.json
python extra_analyses.py --blocks <72 h block dirs> --lookback <24 h and 168 h dirs> \
    --transfer72 <transfer dirs> --transfer24 <24 h transfer dirs> --numbers <report>/numbers.json \
    --out <report>/extra.json --tex <tables>/tab_extra.tex --tex24 <tables>/tab_window24.tex
python journal_figures.py --repo .. --numbers <report>/numbers.json --extra <report>/extra.json \
    --importance <report>/gbm_importance.json --blocks <72 h block dirs> --transfer <transfer dirs> --out <figures>
```

## Where each result of the paper comes from

| Paper item | Script | Key in `results/` |
|---|---|---|
| Main skill table, chronological block-A table | `journal_tables.py` | `numbers.json`: `main`, `persistence` |
| Paired tests (STORM-PhysNet, LSTM variant, protocol B, trees) | `journal_analysis.py` | `numbers.json`: `paired_*` |
| Storm-time RMSE and alert HSS | `journal_analysis.py` | `numbers.json`: `storm_rmse`, `alert_hss`, `n_storm_targets` |
| Degraded inputs (noise, outages, no drivers, late indices) | `journal_analysis.py` | `numbers.json`: `probes` |
| Calibration and seed ensembles | `journal_analysis.py` | `numbers.json`: `calibration`, `ensemble` |
| Transfer to GOES-16, GOES-18 and GRASP | `journal_analysis.py` | `numbers.json`: `transfer` |
| Skill by solar-cycle phase, daily means, local time, rises and falls | `extra_analyses.py` | `extra.json`: `by_block`, `daily`, `local_time`, `rises`, `falls` |
| Other input windows (24 h, 168 h) | `extra_analyses.py` | `extra.json`: `lookback`, `lookback_tests` |
| 24 h-window networks on every test | `extra_analyses.py` | `extra.json`: `window24`, `window24_transfer` |
| What the trees use | `gbm_importance.py` | `gbm_importance.json` |
| Training epochs, validation error, run time of every run | `train_block.py` | `runlogs/` |

All tests are paired over the ten seeds (42–51) after averaging each seed over the three test blocks, and each
family of comparisons is Holm-adjusted.

## License and data

Code is released under the **MIT License**. Follow the applicable terms for the GOES (NOAA NCEI), NASA OMNI and
GSAT-19 GRASP data products.
