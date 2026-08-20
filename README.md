# STORM-PhysNet

Official implementation of the **IEEE Access** paper:

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer**

Multi-horizon log-flux forecasts at **1 h / 6 h / 12 h** on hourly GOESâ€“OMNI, with transfer to GSAT-19 GRASP.

## What this repository includes

- Main systems: LSTM, default Transformer, architecture-matched Transformer, STORM-Bz
- Ablations: No-Delay, No-Physics, No-Gate
- Access-only extras: alternative gates (RDG / RDG-S / SDG), wider delay bounds, interpretability figures, bagged controls
- Official result CSVs in `results/` and checkpoints in `checkpoints/`, including FFN models.


## Key results (fifteen seeds)

| System | PE_1h | PE_6h | PE_12h |
|--------|-------|-------|--------|
| Transformer (default) | 0.978 | 0.895 | 0.845 |
| Transformer matched | 0.980 | 0.895 | 0.845 |
| STORM-Bz | **0.986** | 0.900 | 0.854 |
| STORM-Bz bagged | **0.987** | **0.910** | **0.870** |
| TF matched bagged | 0.984 | 0.908 | 0.861 |

GRASP fine-tune: 6 h PE 0.740 â†’ 0.841; 12 h 0.567 â†’ 0.762.

Primary comparison is the architecture-matched Transformer. Module ablations show the short-horizon gain is a training-package effect.

## Reproduce

```bash
pip install -r requirements.txt
# notebooks/STORM_PhysNet_Master.ipynb
# DEMO_MODE = False only for a full 15-seed retrain
```

Headline numbers load from `results/*.csv`. Test PE is not used for model selection.

## Citation

```bibtex
@article{samarth2026storm,
  title={STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer},
  author={Samarth BN},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

## License

MIT for code. Follow NOAA / NASA / ISSDC terms for GOES, OMNI, and GRASP.
