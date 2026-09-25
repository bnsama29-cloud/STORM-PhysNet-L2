# STORM-PhysNet

## Build Next 2026 — Phase 1 Portfolio Project

**STORM-PhysNet is a pre-existing research and software project submitted by Samarth BN as a Phase 1 Portfolio Project for Build Next 2026.** The project was developed prior to the challenge and is presented here with its existing implementation, evaluation results, figures, and trained checkpoints.

This repository contains the implementation associated with the **IEEE Access manuscript**:

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer**

> **Publication status:** Manuscript under review.

## Project at a glance

STORM-PhysNet forecasts energetic relativistic electron fluxes at geostationary orbit over **1 h, 6 h, and 12 h** horizons using a **72-hour history** of GOES electron-flux observations together with solar-wind and geomagnetic measurements.

The model follows a delay–encode–gate–decode design:

1. Adaptive solar-wind propagation delay constrained to **0.5–1.5 h**
2. Feature projection to **128 dimensions**
3. **2-layer, 4-head Transformer** encoder
4. **Bz-conditioned** modulation gate
5. Residual prediction heads for **1 h / 6 h / 12 h** forecasts

The work also evaluates robustness, controlled ablations, multi-seed stability, and transfer from **GOES-15 to GSAT-19 GRASP** at Indian longitude.

## Key results — fifteen seeds

| System | PE_1h | PE_6h | PE_12h |
|---|---:|---:|---:|
| Transformer (default) | 0.978 | 0.895 | 0.845 |
| Transformer matched | 0.980 | 0.895 | 0.845 |
| **STORM-Bz** | **0.986** | 0.900 | 0.854 |
| **STORM-Bz bagged** | **0.987** | **0.910** | **0.870** |
| TF matched bagged | 0.984 | 0.908 | 0.861 |

**GRASP fine-tuning:** 6 h PE **0.740 → 0.841** and 12 h PE **0.567 → 0.762**.

The primary comparison is the architecture-matched Transformer. Controlled ablations show that the short-horizon gain is a property of the trained STORM package as a whole; the results do not attribute the gain to any single physics module alone.

## Repository contents

- `src/` — model, data, training, and evaluation implementation
- `configs/` — experiment configurations
- `notebooks/` — reproduction notebook
- `results/` — official aggregated evaluation tables and metric summaries
- `figures/` — paper figures and diagnostics
- `checkpoints/` — trained checkpoints for released configurations
- `datasets/` — dataset-related project files and references
- `requirements.txt` — runtime dependencies

## Evaluation protocol

The released evaluation uses purely chronological train/validation/test splits with no shuffling. The main GOES evaluation uses fifteen seeds (42–56). Test prediction efficiency is computed after training and is not used for model selection.

The project reports both climatology-relative prediction efficiency and persistence-relative performance, together with robustness and transfer analyses.

## GOES → GSAT-19 GRASP transfer

The transfer experiment adapts a GOES-pretrained model to GSAT-19 GRASP observations at Indian longitude. Fine-tuning improves:

- **6 h PE:** 0.740 → **0.841**
- **12 h PE:** 0.567 → **0.762**

Absolute GRASP PE remains below the GOES test scores under sensor and longitude shift; the result is presented as recovery after adaptation, not parity with GOES.

## Reproduce

```bash
pip install -r requirements.txt
```

Then run:

```
notebooks/STORM_PhysNet_Master.ipynb
```

Set `DEMO_MODE = False` only when a full retraining is intended. Headline evaluation tables are loaded from `results/*.csv`.

## Build Next 2026 disclosure

This repository is being submitted as an **existing Phase 1 Portfolio Project** for Build Next 2026. The implementation and reported experiments pre-date the challenge and are presented without representing them as newly created during the challenge.

## Citation

```bibtex
@article{samarth2026storm,
  title={STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer},
  author={Samarth BN and Samrudh S Malali and Dhyan M and Sanjana H V},
  journal={IEEE Access},
  year={2026},
  note={Manuscript under review}
}
```

## License and data

Code is released under the **MIT License**. Follow the applicable terms for the GOES, NASA OMNI, and GSAT-19 GRASP data products.
