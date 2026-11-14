# Results behind the journal tables

- `numbers.json`: every number in the paper's tables and figures, written by `pipeline/journal_analysis.py`.
- `extra.json`: skill per test block, on daily means, by local time, on rises and falls, and of other input windows (`pipeline/extra_analyses.py`).
- `gbm_importance.json`: grouped split-gain importance of the block-A gradient-boosted trees (`pipeline/gbm_importance.py`).
- `runlogs/`: one JSON line per training run (block, family, seed, epochs trained, best validation MSE, seconds,
  parameters), named after the Kaggle campaign folder it came from. Early stopping uses patience 12, so the
  epoch whose weights were kept is `epochs - 12`.

The saved predictions of all runs (1.4 GB) are available from the corresponding author on request.
