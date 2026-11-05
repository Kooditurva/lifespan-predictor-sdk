# Changelog

## Algorithm 1.2.0

- `fit` can choose `max_rul` for you: give it a `CategoricalRange` of caps and it picks the most accurate one.
- `decision_horizon` focuses that choice on the remaining life you plan maintenance around.
- A new score, `rmse_uncapped`: the error against the true remaining life.
- Fixes: after tuning on SageMaker, `model.config` shows the tuned values; training jobs are named after the model type; the same data trains the same model.
