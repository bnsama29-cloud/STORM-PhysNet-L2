# Build Next 2026 — Portfolio Project Notes

## Submission status

This project is submitted as a **Phase 1 existing Portfolio Project** for Build Next 2026.

**Participant:** Samarth BN

## What is being submitted

The submission points to the existing STORM-PhysNet-L2 repository and its pre-existing implementation, evaluation results, figures, and trained checkpoints.

No claim is made that this project was created during Build Next 2026.

## Project summary

STORM-PhysNet is a multi-horizon Transformer system for forecasting >2 MeV relativistic electron flux at geostationary orbit at 1 h, 6 h, and 12 h horizons. It combines a 72-hour GOES/OMNI history with a learnable L1–Earth propagation delay, a Bz-conditioned gate, and residual forecast heads. The evaluation includes fifteen seeds, matched Transformer controls, ablations, robustness checks, and GOES → GSAT-19 GRASP transfer.

## Headline evidence

| Evidence | Result |
|---|---:|
| STORM-Bz PE1h | 0.986 |
| STORM-Bz PE6h | 0.900 |
| STORM-Bz PE12h | 0.854 |
| STORM-Bz bagged PE6h | 0.910 |
| STORM-Bz bagged PE12h | 0.870 |
| GRASP fine-tuned PE6h | 0.841 |
| GRASP fine-tuned PE12h | 0.762 |

## Reviewer-safe wording

Use:

> The experiments support the performance of the complete STORM training package; the individual physics modules are not claimed to be independently responsible for the observed gain.

Avoid:

> The Bz gate alone causes the performance improvement.

Avoid describing the manuscript as published or accepted by IEEE Access. The correct status is **manuscript under review**.
