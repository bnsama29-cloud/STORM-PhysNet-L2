# STORM-PhysNet (journal version)

STORM-PhysNet forecasts the >2 MeV electron flux at geostationary orbit 1, 6, 12 and 24 hours ahead from a 72-hour
history of GOES electron flux, solar-wind and geomagnetic data, and tests how well GOES-15-trained models transfer
to other satellites (GOES-16, GOES-18 and India's GSAT-19 GRASP).

This repository holds the code, data, Kaggle scripts and manuscript source of the **journal version**
(prepared for submission to *Advances in Space Research*; source in `paper/`, results in `paper/results/`).
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
validation error over all lead times): 33 neural families and 990 training runs in all. See `pipeline/README.md`.

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
- `paper/` — manuscript source (Elsevier CAS template), generated tables and figures, `results/` (numbers.json, run logs)
- `tests/` — unit and integration tests (`python -m pytest -q tests`)
- `legacy/` — results, checkpoints and notebook of the earlier single-split experiments (superseded; kept for reference)

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Full training runs on Kaggle with the scripts in `pipeline/kaggle/` (campaigns 1–7; campaign 5 trains the
physics terms at full weight from the first epoch, campaign 6 uses training protocol B, campaign 7 other input windows); see `pipeline/README.md`.
Tables and figures: `pipeline/journal_analysis.py`, `journal_tables.py`, `journal_figures.py`.

## License and data

Code is released under the **MIT License**. Follow the applicable terms for the GOES (NOAA NCEI), NASA OMNI and
GSAT-19 GRASP data products.
