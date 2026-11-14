# Results behind the journal tables

- `numbers.json`: every number in the paper's tables and figures, written by `pipeline/journal_analysis.py`.
- `gbm_importance.json`: grouped split-gain importance of the block-A gradient-boosted trees (`pipeline/gbm_importance.py`).
- `runlogs/`: one JSON line per training run (block, family, seed, epochs trained, best validation MSE, seconds,
  parameters), named after the Kaggle campaign folder it came from. Early stopping uses patience 12, so the
  epoch whose weights were kept is `epochs - 12`.

The saved predictions of all runs (1.3 GB) are available from the author on request.
