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
    """Wrapper around a list of TrialResult objects with convenience methods."""

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
        """Return the TrialResult with the highest mean score (c-index or negated IBS)."""
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

    Within each stratum (event_value × duration_quartile) units are sorted by
    mean duration then assigned round-robin, so each fold receives a balanced
    mix of short/long durations and censored/failed units.

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

    for ev in sorted(unit_df["event"].unique()):
        for dur_bin in sorted(unit_df["_dur_bin"].unique()):
            stratum = unit_df[
                (unit_df["event"] == ev) & (unit_df["_dur_bin"] == dur_bin)
            ].sort_values("duration_mean")
            for i, unit in enumerate(stratum["unit"].values):
                folds[i % n_splits].append(unit)

    return folds


def _stratified_kfold(train_df, n_splits: int = 5, seed: int = 42):
    """
    Yield (fold_train_df, fold_val_df) pairs stratified by event and duration.

    Splitting is at unit level — no unit appears in both train and val.
    Stratification ensures each fold has a similar event rate and duration
    distribution, producing more reliable cross-validation scores.

    Requires columns: 'unit', 'event', 'duration'.
    """
    for col in ("unit", "event", "duration"):
        if col not in train_df.columns:
            raise ValueError(
                f"_stratified_kfold requires column '{col}' in train_df. "
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


# Alias for backwards compatibility
_asset_aware_kfold = _stratified_kfold
