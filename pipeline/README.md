# Pipeline: training, transfer and reporting (journal version)

This folder trains every model family, runs the cross-satellite transfer experiments, and builds the result tables
from the saved predictions. The same leakage-controlled pipeline produced the AGERS 2026 conference results
(repository STORM-PhysNet-L1); this repository extends it for the journal paper.

## Files

| File | Purpose |
|---|---|
| `storm_pipeline/data.py` | Causal data pipeline. Missing GOES hours are forward-filled from earlier hours and an observed-flux mask is an input channel; Kp is causal (last completed 3-h interval); rows are labelled per split, so no window crosses a boundary; scalers use training rows only. Defines the three test blocks (A chronological; B and C blocked) and the shared validation period. Horizons come from `STORM_HSTEPS` (default 1,6,12; the journal runs use 1,6,12,24). |
| `storm_pipeline/families.py` | Registry of all model families (see below), shared by training and transfer |
| `storm_pipeline/models_extra.py` | Control models: Transformer + residual head, flux-only Transformer, fixed two-point average |
| `storm_pipeline/transfer.py` | Shared pieces of the transfer experiments (building, freezing, fine-tuning) |
| `train_block.py` | Trains families × seeds for one block, saving test predictions, predicted log-variance, noise/driver-probe predictions, delay/gate values and (spectral variant) the spectral-head prediction. Resumable; `--stop_after_hours` ends a session cleanly. |
| `transfer_journal.py` | Block-A GOES-15 models → GOES-16, GOES-18 and GSAT-19 GRASP: zero-shot, heads-only, full fine-tuning and from-scratch, chronological folds |
| `goesr_download.py` | Builds `datasets/goesr/` (NOAA MPS-HI >2 MeV, telescope 4, hourly) and the 2021–2026 OMNI file |
| `make_journal_kaggle.py` | Writes the three Kaggle scripts of the journal campaign into `kaggle/` |
| `grasp_transfer.py`, `combine_report.py`, `combine_probes.py`, `camera_ready_stats.py`, `fixed_ma_stats.py` | Conference-version tools, kept for reproducibility |

Model architectures, the trainer and the loss come from `src/`. Settings are in `configs/config.yaml`: 40 epochs,
early-stopping patience 12, learning rate 3e-4, storm oversampling ×12.

## Model families

| Family | What it is |
|---|---|
| `storm_bz` | STORM-PhysNet: learned delay, Transformer encoder, Bz gate, persistence-residual heads, physics-informed loss |
| `storm_no_delay`, `storm_no_gate`, `storm_no_physics` | STORM-PhysNet without the delay stage / the Bz gate / the physics loss terms |
| `storm_fixed_ma` | STORM-PhysNet with the learned delay replaced by a fixed two-point average |
| `storm_cathode`, `storm_radiotrophic` | Alternative gates (cathode-anode and radiotrophic "dose" gates, `src/model/analogy_gates.py`) |
| `storm_cathode_spec` | Cathode-anode gate plus a kappa-spectrum head, trained through an extra loss term |
| `storm_lstm_backbone` | STORM-PhysNet with an LSTM encoder instead of the Transformer |
| `storm_magnetopause` | STORM-PhysNet plus Shue (1998) magnetopause stand-off features computed in physical units |
| `transformer_matched`, `transformer` | Transformer sized like STORM-PhysNet, and the default-size Transformer |
| `tf_ffn_res`, `tf_ffn256` | Transformer with an FFN head, with and without the persistence residual |
| `tf_res_fixed_ma`, `tf_flux_only` | Residual Transformer with the fixed two-point average; Transformer given only flux channels |
| `lstm`, `mlp`, `cnn` | Standard baselines |

The hybrid SSM/iTransformer backbone remains in `src/model` but is not part of the journal runs.

## Journal campaign on Kaggle

The nine conference families were run with horizons 1/6/12/24 h by the scripts of STORM-PhysNet-L1
(`journal24_[ABC].zip`, `journal_transfer.zip`). The remaining families run here:

1. `python make_journal_kaggle.py <commit>` writes `kaggle/journal_account{1,2,3}.py` (already generated in this repository).
2. Paste one script per Kaggle account into a new notebook. Settings: GPU T4 x2, Internet ON. Save & Run All.
3. Each session trains for at most 10 hours. If the log ends with `NOT FINISHED`, add the notebook's own output as an
   Input and run it again; finished runs are skipped.
4. Download `journal_l2_account{1,2,3}.zip` when the log says `ALL WORK DONE`.
