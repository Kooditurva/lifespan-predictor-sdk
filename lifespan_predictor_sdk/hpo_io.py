"""
Hyperparameter search I/O: ranges, trial results, and stratified splitting.

The pieces a caller needs to describe a search and read its results —
serialising the ranges that go out, the trial records that come back, and the
unit-level stratified split that keeps a unit out of both halves.

Top-level imports are deliberately minimal (numpy and the standard library), so
importing this costs nothing at call sites that only need to describe a search.
"""

from dataclasses import dataclass

import numpy as np

# Keys that say where and how a job runs, not how the model trains. They are
# held back from the training job, and from anything shown to the person who
# runs it: a config they can read names only hyperparameters.
INTERNAL_CONFIG_KEYS = frozenset(
    {
        "backend",
        "role",
        "bucket",
        "image_uri",
        "algorithm_arn",
        "instance_type",
        "in_features",
    }
)


def without_internal_keys(config: dict) -> dict:
    """`config` with only its hyperparameters."""
    return {k: v for k, v in config.items() if k not in INTERNAL_CONFIG_KEYS}


# ---------------------------------------------------------------------------
# Range serialisation / deserialisation
# ---------------------------------------------------------------------------


def serialize_range(obj) -> dict:
    """Serialise a range object to a JSON-safe dict with a __type tag."""
    from lifespan_predictor_sdk.survival_model import CategoricalRange, ContinuousRange, IntegerRange

    if isinstance(obj, ContinuousRange):
        return {
            "__type": "ContinuousRange",
            "low": obj.low,
            "high": obj.high,
            "log": obj.log,
        }
    if isinstance(obj, IntegerRange):
        return {"__type": "IntegerRange", "low": obj.low, "high": obj.high}
    if isinstance(obj, CategoricalRange):
        return {"__type": "CategoricalRange", "choices": obj.choices}
    raise TypeError(f"Not a range object: {type(obj)}")


def deserialize_range(d: dict):
    """Deserialise a type-tagged dict back to a range object."""
    from lifespan_predictor_sdk.survival_model import CategoricalRange, ContinuousRange, IntegerRange

    t = d.get("__type")
    if t == "ContinuousRange":
        return ContinuousRange(low=d["low"], high=d["high"], log=d.get("log", False))
    if t == "IntegerRange":
        return IntegerRange(low=d["low"], high=d["high"])
    if t == "CategoricalRange":
        return CategoricalRange(choices=d["choices"])
    raise ValueError(f"Unknown range type tag: {t}")


def serialize_config(config: dict) -> dict:
    """Serialise a config dict (possibly containing range objects) to JSON-safe form."""
    from lifespan_predictor_sdk.survival_model import CategoricalRange, ContinuousRange, IntegerRange

    out = {}
    for k, v in config.items():
        if isinstance(v, (ContinuousRange, IntegerRange, CategoricalRange)):
            out[k] = serialize_range(v)
        else:
            out[k] = v
    return out


def deserialize_config(config: dict) -> dict:
    """Restore range objects from a JSON-safe config dict."""
    out = {}
    for k, v in config.items():
        if isinstance(v, dict) and "__type" in v:
            out[k] = deserialize_range(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# TrialResult / TuningTrials
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    number: int
    value: float  # mean score across folds (c-index or negated IBS)
    params: dict  # sampled scalar params for this trial
    config: dict  # full resolved config for this trial
    state: str  # "COMPLETE", "FAIL", etc.
    fold_scores_list: list = None  # per-fold scores
    metric: str = "c-index"  # scoring metric used: "c-index", "ibs", or "antolini"

    def fold_scores(self) -> list:
        """Return per-fold scores, or empty list if not available."""
        return self.fold_scores_list or []

    def get_best_fold_log(self):
        """Not supported — fold logs are not retained after HPO."""
        return None

    @classmethod
    def from_optuna(
        cls, trial, full_config: dict = None, metric: str = "c-index"
    ) -> "TrialResult":
        return cls(
            number=trial.number,
            value=trial.value if trial.value is not None else float("nan"),
            params=trial.params,
            config=full_config or trial.params,
            state=trial.state.name,
            fold_scores_list=trial.user_attrs.get("fold_scores"),
            metric=metric,
        )


class TuningTrials:
    """
    The trials of a tuning run, returned by `SurvivalModel.get_tuning_trials()`.

    Iterate over it for the trials in the order they ran; `len()` counts them.
    Each trial has:

    - `number`: its position in the run, from 0.
    - `params`: the values drawn from the config's ranges.
    - `config`: the whole config it trained with.
    - `value`: its mean score over the folds. Higher is better: for
      `hpo_metric="ibs"` it is the integrated Brier score negated.
    - `fold_scores()`: its score on each fold.
    - `metric`: the `hpo_metric` it was scored by.
    - `state`: `"COMPLETE"`, or `"FAIL"` if it could not be scored.

    Attributes:
        best_config: The config of the best trial, with every range replaced by
            its best value. This is the config `fit()` retrained with.
    """

    def __init__(self, trials: list):
        self._trials = trials
        self.best_config: dict | None = None

    def __bool__(self):
        return bool(self._trials)

    def __len__(self):
        return len(self._trials)

    def __iter__(self):
        return iter(self._trials)

    def get_best_trial(self):
        """
        The completed trial with the highest mean score.

        Returns:
            That trial, or None if no trial completed.
        """
        completed = [
            t
            for t in self._trials
            if t.state == "COMPLETE" and t.value != float("-inf")
        ]
        if not completed:
            return None
        return max(completed, key=lambda t: t.value)


# ---------------------------------------------------------------------------
# Asset-aware stratified splitting (used by HPO and by val_split= in fit())
# ---------------------------------------------------------------------------


def _assign_units_to_folds(unit_df, n_splits: int) -> list:
    """
    Distribute units across n_splits folds with stratification by event and duration.

    Strata (event_value × duration_quartile) are visited in turn, each sorted by
    mean duration, and their units are dealt round-robin in one continuous deal.
    Each fold receives a balanced mix of short/long durations and censored/failed
    units, and fold sizes differ by at most one.

    Args:
        unit_df: DataFrame with columns: unit, event, duration_mean.
        n_splits: Number of folds.

    Returns:
        List of n_splits lists, each containing unit IDs for one validation fold.
    """
    folds: list = [[] for _ in range(n_splits)]

    # Bin durations into quartiles (up to 4 bins)
    n_bins = min(4, max(1, len(unit_df) // n_splits))
    unit_df = unit_df.copy()
    unit_df["_dur_bin"] = (
        unit_df["duration_mean"]
        .rank(pct=True)
        .apply(lambda p: min(int(p * n_bins), n_bins - 1))
    )

    # One deal across all strata, not one per stratum: restarting at fold 0 in
    # each stratum made the first fold the largest and could leave the last
    # empty.
    dealt = 0
    for ev in sorted(unit_df["event"].unique()):
        for dur_bin in sorted(unit_df["_dur_bin"].unique()):
            stratum = unit_df[
                (unit_df["event"] == ev) & (unit_df["_dur_bin"] == dur_bin)
            ].sort_values("duration_mean")
            for unit in stratum["unit"].values:
                folds[dealt % n_splits].append(unit)
                dealt += 1

    return folds


def stratified_kfold(train_df, n_splits: int = 5, seed: int = 42):
    """
    Split training data into cross-validation folds, by unit.

    These are the folds `SurvivalModel.fit()` tunes on. Use them to search
    hyperparameters your own way, such as a grid search.

    - **By unit.** All of a unit's rows go to the same side, so no unit is ever
      in both halves of a fold, and each unit is in exactly one hold-out half.
    - **Balanced.** Units are spread so that each fold has a similar event rate
      and a similar spread of durations, which keeps fold scores comparable.
    - **Deterministic.** The same data and `seed` give the same folds.

    **Scoring a fold.** Score each model on its fold's hold-out half, which
    must play no part in training. Give the model its own validation data for
    early stopping, `val_split`, drawn from the training half. Do not pass the
    hold-out half as `val_df`: validation data chooses the epoch the model
    keeps and when its learning rate drops, so a score on it is optimistic.

    The model predicts once per unit, at the unit's last row. A hold-out half
    holds whole histories, so each unit's last row is at, or near, its end of
    life, and scoring there says little. Cut each hold-out unit at a random
    cycle first and label it with its remaining life at the cut, as a test set
    is built. Then pass it to `fit()` as `test_df` and `test_labels_df`: the
    model is scored on it after training, in the same training job, and never
    trains on it.

    ```python
    from itertools import product

    import numpy as np
    import pandas as pd

    from lifespan_predictor_sdk import SurvivalModel, stratified_kfold


    def as_test_set(fold, seed=0):
        # Cut each unit at a random cycle; label it with its remaining life there.
        rng = np.random.default_rng(seed)
        rows, labels = [], []
        for _, history in fold.groupby("unit"):
            cut = history.iloc[: rng.integers(1, len(history) + 1)]
            rows.append(cut.drop(columns=["duration", "event"]))
            labels.append(cut.iloc[-1][["unit", "duration", "event"]])
        labels = pd.DataFrame(labels).astype({"unit": int, "event": int})
        return pd.concat(rows), labels


    folds = [
        (fold_train, *as_test_set(hold_out))
        for fold_train, hold_out in stratified_kfold(train_df)
    ]
    grid = {"learning_rate": [1e-3, 1e-2], "num_nodes": [[16], [32, 32]]}

    best_params, best_score = None, float("-inf")
    for values in product(*grid.values()):
        params = dict(zip(grid, values))
        scores = []
        for fold_train, test_df, test_labels_df in folds:
            model = SurvivalModel({"model_type": "deepsurv", "epochs": 30, **params})
            model.fit(
                fold_train,
                val_split=0.2,
                test_df=test_df,
                test_labels_df=test_labels_df,
            )
            scores.append(model.get_evaluation_result().c_index)
        score = float(np.mean(scores))
        if score > best_score:
            best_params, best_score = params, score

    final = SurvivalModel({"model_type": "deepsurv", "epochs": 30, **best_params})
    final.fit(train_df, val_split=0.2)
    ```

    On SageMaker every `fit()` is a separate training job, so this grid of 4
    points over 5 folds runs 20 training jobs, plus the final one. Scoring
    needs no `deploy()`: each job scores its own hold-out.

    Args:
        train_df: Training data with `unit`, `event` and `duration` columns, as
            `SurvivalModel.fit()` takes it.
        n_splits: The number of folds.
        seed: Fixes how units are shuffled before they are assigned to folds.

    Yields:
        `n_splits` pairs `(fold_train_df, hold_out_df)` of DataFrames, each with
        `train_df`'s columns.

    Raises:
        ValueError: If `train_df` lacks `unit`, `event` or `duration`.
    """
    for col in ("unit", "event", "duration"):
        if col not in train_df.columns:
            raise ValueError(
                f"stratified_kfold requires column '{col}' in train_df. "
                f"Found columns: {list(train_df.columns)}"
            )

    unit_stats = (
        train_df.groupby("unit")
        .agg(event=("event", "last"), duration_mean=("duration", "mean"))
        .reset_index()
    )

    # Shuffle unit_stats with fixed seed so round-robin assignment is deterministic
    rng = np.random.default_rng(seed=seed)
    unit_stats = unit_stats.iloc[rng.permutation(len(unit_stats))].reset_index(
        drop=True
    )

    fold_assignments = _assign_units_to_folds(unit_stats, n_splits=n_splits)

    for val_units in fold_assignments:
        val_mask = train_df["unit"].isin(val_units)
        fold_train = train_df[~val_mask].reset_index(drop=True)
        fold_val = train_df[val_mask].reset_index(drop=True)
        yield fold_train, fold_val


def _stratified_val_split(train_df, val_fraction: float = 0.2, seed: int = 42):
    """
    Split train_df into (train, val) at unit level with event/duration stratification.

    Reuses _assign_units_to_folds so HPO cross-validation and final val_split
    use identical distribution-aware splitting logic (DRY).

    Args:
        train_df: Raw training DataFrame with 'unit', 'event', 'duration' columns.
        val_fraction: Fraction of units to place in validation set.
        seed: Random seed for determinism.

    Returns:
        (train_df_split, val_df_split) tuple.
    """
    for col in ("unit", "event", "duration"):
        if col not in train_df.columns:
            raise ValueError(
                f"_stratified_val_split requires column '{col}' in train_df. "
                f"Found columns: {list(train_df.columns)}"
            )

    unit_stats = (
        train_df.groupby("unit")
        .agg(event=("event", "last"), duration_mean=("duration", "mean"))
        .reset_index()
    )

    rng = np.random.default_rng(seed=seed)
    unit_stats = unit_stats.iloc[rng.permutation(len(unit_stats))].reset_index(
        drop=True
    )

    n_val = max(1, int(len(unit_stats) * val_fraction))
    # Use enough folds so one fold ≈ val_fraction of units
    n_splits = max(2, round(1.0 / val_fraction))
    fold_assignments = _assign_units_to_folds(unit_stats, n_splits=n_splits)

    # First fold is the val set; trim if it exceeds n_val
    val_units = fold_assignments[0][:n_val]
    val_mask = train_df["unit"].isin(val_units)
    return train_df[~val_mask].reset_index(drop=True), train_df[val_mask].reset_index(
        drop=True
    )
