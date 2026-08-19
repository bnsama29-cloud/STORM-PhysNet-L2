# STORM-PhysNet Checkpoints

This directory contains the final trained weights for seeds 42-56 across all model configurations used in the paper.

**Loading Guidelines:**
- Always prefer the `*_best.pt` file when evaluating a model (e.g. `storm_physnet_bz_best.pt`).
- Ignore `*_last.pt` or intermediate saves if present.
- Note that the weights for models fine-tuned on the GRASP dataset are not included in this repository due to space constraints; however, the zero-shot main model weights are provided.
