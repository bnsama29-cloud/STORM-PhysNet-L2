# STORM-PhysNet (IEEE Access)

Official code for the IEEE Access paper (extended version of the conference study):

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer**

Horizons: **1 h / 6 h / 12 h**. Transfer: GSAT-19 GRASP.

This repository includes the seven main systems **plus** Access-only probes: alternative gates (RDG / RDG-S / SDG), FFN-matched Transformer, residual-head control, noise robustness, and interpretability figures.

## Repository map

```
STORM-PhysNet-L2/
├── configs/          Official hyperparameters
├── datasets/         GOES, OMNI, GRASP
├── src/              Full package (including Access-only gate probes)
├── notebooks/        Reproduction notebook
├── checkpoints/      15-seed weights for main + extra systems
├── results/          Official CSVs behind Access tables
├── figures/          All Access figures
├── requirements.txt
└── LICENSE
```

### `configs/`
Same as the conference run (`config.yaml`: horizons `[1, 6, 12]`, 12× storm sampler). `config_transformer_baseline.yaml` is the default-width Transformer.

### `datasets/`
`goes/` · `omni/` · `grasp/` — real instrument files only. No synthetic training path.

### `src/`
Same layout as the conference repo. Access-only usage:
- `src/model/analogy_gates.py` — RDG (`cathode_anode`) and SDG (`radiotrophic`)
- `src/model/spectral_head.py` — RDG-S
- All other conference modules are unchanged

### `checkpoints/`
**Main / ablations:** `lstm/` `transformer/` `transformer_matched/` `storm_bz/` `storm_no_delay/` `storm_no_gate/` `storm_no_physics/`

**Access extras:** `storm_cathode/` (RDG) · `storm_cathode_spec/` (RDG-S) · `storm_radiotrophic/` (SDG) · `tf_ffn256/` · `tf_ffn256_res/`

### `results/`
| File | Paper use |
|------|-----------|
| `table_main_means.csv` | Table 3 seed means |
| `table_bagged.csv` | Bagged rows |
| `ablation_final_table.csv` | Ablation numbers |
| `alt_gates_summary.csv` | Table 5 (RDG / RDG-S / SDG) |
| `table_grasp_storm_bz.csv` | GRASP transfer |
| `table_parameter_counts.csv` | Table 4 |
| `tf_ffn256_summary.csv` | FFN-matched and residual-head PE |
| `noise_robustness.csv` | \(\sigma\)-noise probe |
| `all_seed_results_full.csv` | Per-seed rows |
| `ensemble_summary.json` | \(\alpha^*\) diagnostic |

### `figures/`
| File | Use |
|------|-----|
| `fig_system_architecture.png` | Architecture |
| `fig_horizon_pe.png` | Per-horizon PE |
| `fig_ablation_6h.png` | Ablations at 6 h |
| `fig_grasp_domain_gap.png` | GRASP zero-shot vs fine-tune |
| `fig_alt_gates_pe6h.png` | RDG / RDG-S / SDG |
| `fig_case_study_timeseries.png` | Storm case |
| `noise_robustness.png` | Input-noise curves |
| `fig_physics_tau_hist.png` | Delay histogram |
| `fig_physics_gate_activation.png` | Gate values |
| `fig_physics_gate_storm_quiet.png` | Storm vs quiet gate |
| `fig_feature_importance.png` | Permutation importance |
| `fig_dst_bins.png` | Dst-binned PE |
| `fig_seed_spread.png` | Seed scatter |
| `wider_delay_pe6h.png` | Delay-bound probe |

### `notebooks/`
`STORM_PhysNet_Master.ipynb` — loads official CSVs; set `DEMO_MODE = False` only for a full retrain.

## Key results (fifteen seeds)

Table I bagging is the mean of 15 independently trained checkpoints
(seeds 42–56), not the unused 5-member STORMPhysNetEnsemble.

| System | PE_1h | PE_6h | PE_12h |
|--------|-------|-------|--------|
| Transformer (default) | 0.978 | 0.895 | 0.845 |
| Transformer matched | 0.980 | 0.895 | 0.845 |
| STORM-Bz | **0.986** | 0.900 | 0.854 |
| STORM-Bz bagged | **0.987** | **0.910** | **0.870** |
| TF matched bagged | 0.984 | 0.908 | 0.861 |

GRASP fine-tune: 6 h 0.740 → 0.841; 12 h 0.567 → 0.762.

FFN-matched Transformer: PE_1h \(0.976\pm0.002\). Same encoder + persistence residual: \(0.987\pm0.001\). Alternative gates stay within seed noise of STORM-Bz.

## Reproduce

```bash
pip install -r requirements.txt
# notebooks/STORM_PhysNet_Master.ipynb
```

Official split: 63 394 / 7 924 / 15 850 hours.

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

MIT for code. Follow NOAA / NASA / ISSDC terms for data.
