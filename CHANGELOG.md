# Changelog

## Algorithm 1.1.0

- The Cox model, `model_type="coxph"`: a penalised linear Cox model with `coxph_penalizer` and `coxph_l1_ratio`, for when failures are few.
- `get_global_analysis` and `get_local_analysis` explain a Cox model: coefficients and hazard ratios, each unit's risk drivers, feature groups and clusters of units, with plots.
- `examples/sagemaker_coxph_demo.ipynb` shows the Cox model on SageMaker, and `examples/sagemaker_coxph_fd003_demo.ipynb` uses it to tell two failure modes apart.
- The C-index functions are exported, and the metrics module ships with its tests.
- Fixes: each `deploy` gets its own model name; Harrell's C-index runs in n log n; stratified folds hold equal shares of units; `verbose` is no longer accepted.
