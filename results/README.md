# STORM-PhysNet Results

This directory strictly contains the final, aggregated evaluation tables and metric summaries that correspond directly to the results presented in the IEEE paper. 

### Main Evaluation Tables
- `table_main_means.csv` - Mean Performance across 15 seeds for all baselines vs STORM-PhysNet.
- `table_main_stats.csv` & `table_means_bootstrap_ci.csv` - Bootstrap confidence intervals and standard deviations for the main evaluation metrics.
- `all_seed_results_full.csv` - The complete, unaggregated 15-seed data block used to generate the summary tables.
- `table_parameter_counts.csv` - Model parameter sizes.

### Ensembles & Bagging
- `table_bagged.csv` - Ensemble Bagged prediction efficiency across models.
- `ensemble_summary.json` - Metrics for the val-selected $\alpha^*$ (mostly 0.4-0.6) and fixed $\alpha=0.3$ diagnostic.

### Ablations & Alternate Gates
- `ablation_final_table.csv` - Performance impact of removing the delay and physics-informed gates.
- `alt_gates_summary.csv` - Comparison of alternative physics gate formulations (e.g., Cathode, Radiotrophic).

### GRASP Domain Transfer
- `table_grasp_storm_bz.csv` - Formatted GRASP domain transfer table export.
- `grasp_summary.csv` - Summary of the zero-shot vs fine-tuning domain transfer performance.
