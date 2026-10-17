"""
Survival analysis model facade.
Routes to the configured backend via the Strategy pattern.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lifespan_predictor_sdk.training_log import EvaluationResult, TrainingLog

# The name every survival-curve frame carries on its time index. One value in
# one place, because the frame is built by three different code paths and the
# only one that used to name it took the name from pycox by inheritance.
SURVIVAL_TIME_INDEX_NAME = "duration"


# Auto-detect available backends
def _make_backend(name=None):
    """Construct the backend that runs training and inference."""
    from lifespan_predictor_sdk.sagemaker_backend import SageMakerBackend

    return SageMakerBackend()


# ---------------------------------------------------------------------------
# Shared scoring functions (numpy only, no external deps)
# ---------------------------------------------------------------------------


def rmse_score(durations, predictions):
    """Calculate RMSE between true and predicted values."""
    return np.sqrt(np.mean((durations - predictions) ** 2))


def c_index_score(durations, events, predictions):
    """
    Calculate the concordance index (C-index) for survival analysis.

    NumPy-only implementation - no external dependencies required.

    Args:
        durations: Time to event or censoring (1D array)
        events: Event indicators (1=event, 0=censored) (1D array)
        predictions: Predicted scores (1D array, higher = longer survival)

    Returns:
        C-index score (float between 0 and 1)
    """
    n = len(durations)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue

            if durations[i] < durations[j]:
                shorter_idx, longer_idx = i, j
            elif durations[j] < durations[i]:
                shorter_idx, longer_idx = j, i
            else:
                if events[i] == 1 and events[j] == 1:
                    if predictions[i] == predictions[j]:
                        tied_risk += 1
                continue

            if events[shorter_idx] == 0:
                continue

            if predictions[shorter_idx] < predictions[longer_idx]:
                concordant += 1
            elif predictions[shorter_idx] > predictions[longer_idx]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def _censoring_survival(durations, events):
    """Kaplan-Meier estimate of the censoring distribution, ``G(t) = P(C > t)``.

    The reverse Kaplan-Meier: the "events" are the *censored* observations, so this
    estimates how long subjects remain under observation rather than how long they
    survive. Marginal, i.e. assuming censoring is independent of the covariates —
    the simpler of the two options Wolbers et al. allow, the other being a Cox model
    for ``G`` conditional on ``X``.

    Returns:
        ``(times, at, before)`` — the sorted distinct observed times, ``G(t)`` at
        each, and ``G(t⁻)`` just before each. Two arrays because the estimator needs
        both: ``G(t⁻) = P(C ≥ t)`` for a subject whose own event is at ``t``, and
        ``G(t) = P(C > t)`` for one who must still be under observation after ``t``.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(int)

    times = np.unique(durations)
    at_risk = (durations[:, None] >= times[None, :]).sum(axis=0)
    censored_here = (
        (durations[:, None] == times[None, :]) & (events == 0)[:, None]
    ).sum(axis=0)

    factors = 1.0 - np.divide(
        censored_here,
        np.maximum(at_risk, 1),
        out=np.zeros(times.shape, dtype=float),
        where=at_risk > 0,
    )
    at = np.cumprod(factors)
    before = np.concatenate(([1.0], at[:-1]))
    return times, at, before


def default_truncation_time(durations, events, min_censoring_survival=0.10):
    """Largest observed time at which the censoring estimate is still trustworthy.

    An IPCW weight goes as ``1/G²``, and ``G`` decays towards zero in the tail, so
    late cases can dominate the estimate. Measured on 400 subjects: at 25% censoring
    the largest pair weight is 8x and the top 1% of pairs carry 3% of the total
    weight; at 90% censoring those become 639x and **29%**. One percent of the
    comparisons deciding a third of the answer is not a stable estimate.

    Truncating where ``Ĝ`` is still appreciable bounds that. The default floor of
    0.10 caps a pair weight near 100x, which is a chosen constant — stated here
    rather than buried, and overridable by passing ``tau`` explicitly. It is a
    numerical-stability choice; the *domain* choice, "what follow-up horizon do I
    care about", is what an explicit ``tau`` expresses.

    Returns ``None`` when censoring is so heavy that ``Ĝ`` is already below the floor
    at the first observed time — truncating to nothing would be worse than not
    truncating, and the caller should distrust the result either way.
    """
    times, g_at, _ = _censoring_survival(durations, events)
    usable = g_at >= min_censoring_survival
    if not usable.any():
        return None
    return float(times[usable][-1])


def ipcw_c_index(durations, events, risk_scores, tau=None):
    """IPCW-corrected concordance for ordinary (single-risk) survival data.

    Harrell's C depends on the study's censoring distribution: run the same model
    against the same population with heavier drop-out and the number moves, though
    the model has not changed. Weighting each comparable pair by the inverse
    probability of being uncensored removes that dependence. Uno and others (2011)
    introduced this correction; Wolbers and others (2014) carry the same machinery
    into competing risks.

    This is that estimator's ordinary-survival case: with no competing events the
    competing-risks form has no ``B̃`` pairs, every comparable pair is
    ``I{T̃ᵢ < T̃ⱼ}``, and each carries weight ``1/(Ĝ(T̃ᵢ⁻)·Ĝ(T̃ᵢ))``. Implemented by
    delegation rather than as a second estimator, so there is one weighting code path
    to be right about.

    Args:
        durations: Observed time to event or censoring.
        events: ``1`` failed, ``0`` censored.
        risk_scores: Predicted risk — **higher means more likely to fail**, the
            opposite of :func:`c_index_score`'s survival-time convention.
        tau: Truncation time; ``None`` selects one via
            :func:`default_truncation_time`. See there for why truncation matters.

    Returns:
        Concordance in ``[0, 1]``, or ``nan`` when no pair is comparable.

    Note:
        Named for what it is rather than for a paper. The comparable-pair rules and
        the weighting were verified against Wolbers and others (2014), which is
        available; Uno's 2011 paper is not open-access and could not be checked, so
        this does not claim to reproduce that paper's exact published constant. The
        correction it applies is the same one.
    """
    events = np.asarray(events).astype(int)
    if tau is None:
        tau = default_truncation_time(durations, events)
    # Cause 1 with no other causes present: the competing-risks estimator's
    # degenerate case is exactly the single-risk IPCW concordance.
    return competing_risk_c_index(
        durations, np.where(events != 0, 1, 0), risk_scores, cause=1, ipcw=True, tau=tau
    )


def competing_risk_c_index(
    durations, events, risk_scores, cause=1, ipcw=True, tau=None
):
    """Competing-risks-aware concordance for one cause.

    Implements the non-truncated concordance of Wolbers, Blanche, Koller, Witteman &
    Gerds, "Concordance for prognostic models with competing risks", Biostatistics
    (2014), equation 2.1::

        C = P(M(Xi) > M(Xj) | Di = 1 and (Ti < Tj or Dj = 2))

    Answers the *predictive* question — does the model rank the units that actually
    experience cause ``cause``? — rather than the etiologic one that
    :func:`c_index_score` answers when competing events are relabelled as censored.

    A pair ``(i, j)`` is comparable when ``i`` has an observed ``cause`` event at
    ``T_i`` and either:

    1. ``T_j > T_i`` — ``j`` was still event-free, so it was genuinely at risk; or
    2. ``j`` suffered a **competing** event at **any** time, ``T_j < T_i`` included.

    Clause 2 is the whole difference. Censoring ``j`` at its competing event, as the
    etiologic treatment does, discards that pair whenever the competing event came
    first — yet ``j`` is exactly the unit we know will never fail from ``cause``, so
    ranking ``i`` above it is a real success. A *censored* ``j`` before ``T_i`` stays
    non-comparable: its fate is unknown, which is absence of information rather than
    information.

    Ordinary right-censoring is handled by inverse probability of censoring weights,
    the paper's equation 3.4. Without them the estimate inherits Harrell's C's
    dependence on the study's censoring distribution — *"a major problem of this
    estimator is that by ignoring non-evaluable pairs without any correction, bias is
    introduced"* — which is why the paper builds the weighted form and why it is the
    default here. The two pair types carry different weights::

        Ŵ_ij,1 = G(T̃_i⁻)·G(T̃_i)     j observed later: i's event seen, j still watched
        Ŵ_ij,2 = G(T̃_i⁻)·G(T̃_j⁻)    j's competing event: *both* events must be seen

    ``G`` is the Kaplan-Meier estimate of the censoring distribution, marginal — the
    simpler of the two the paper allows, the other being a Cox model conditional on
    covariates.

    So the metric divides the labour the way the literature does: Wolbers' rules say
    which pairs count when a competing event removes a unit from risk, and Uno's IPCW
    machinery keeps ordinary drop-out from biasing the result.

    Args:
        durations: Observed time to event or censoring.
        events: ``0`` censored, ``k >= 1`` failed from cause ``k``.
        risk_scores: Predicted risk of ``cause`` — **higher means more likely to
            fail**, the opposite of :func:`c_index_score`'s survival-time convention.
        cause: Which cause to score.
        ipcw: Weight pairs by inverse censoring probability. ``False`` gives the
            naive estimator, which is biased under censoring and is retained because
            it is what degenerates exactly to Harrell's C when no competing event
            occurs — a property worth testing, not a mode worth reporting.
        tau: Truncation time. Only cases with ``T̃ᵢ <= tau`` contribute, the paper's
            ``Ñ¹ᵢ(t)`` restriction (its equation 2.3). ``None`` uses every case.

            Not cosmetic under heavy censoring. Weights go as ``1/G²`` and ``G``
            falls towards zero in the tail, so late cases can dominate. Measured on
            400 subjects: at 25% censoring the largest pair weight is 8x and the top
            1% of pairs carry 3% of the total; at 90% censoring those become 639x and
            **29%**. Truncating where ``G`` is still appreciable is what keeps the
            estimate from being decided by a handful of late comparisons.

    Returns:
        Concordance in ``[0, 1]``, or ``nan`` when no pair is comparable.

    Note:
        ``nan`` rather than :func:`c_index_score`'s ``0.5``. A missing comparison is
        unknown, not average, and the per-risk reporting path already treats ``nan``
        as "undefined for this cause". Pairs with identical durations are skipped, as
        they are there: with no ordering between them there is nothing to be right or
        wrong about. Ties in the score count ½, the weighting the paper offers
        alongside its strict-inequality base definition.

        ``G`` is floored at 1e-8. Past the last observed censoring time it is not
        identified, and an unfloored zero would turn one pair into an infinite
        weight rather than an uncertain one.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(int)
    risk = np.asarray(risk_scores, dtype=float)

    # A unit that failed from any other cause can never fail from `cause`, so it
    # stays comparable no matter when that happened. Splitting the comparable set
    # this way reproduces the paper's disjoint Ã (later time) and B̃ (earlier or
    # equal time, competing event), which need different censoring weights.
    competing = (events != cause) & (events != 0)

    if ipcw:
        times, g_at, g_before = _censoring_survival(durations, events)
        index = np.searchsorted(times, durations)
        # A weight is 1/G·G; a zero would divide by zero. Beyond the last
        # observed censoring time G is not identified, so floor rather than blow up.
        floor = 1e-8
        own = np.maximum(g_before[index], floor)  # G(T̃⁻) for each subject
        after = np.maximum(g_at[index], floor)  # G(T̃)  for each subject

    cases = np.flatnonzero(events == cause)
    if tau is not None:
        cases = cases[durations[cases] <= tau]

    numerator = 0.0
    denominator = 0.0
    for i in cases:
        later = durations > durations[i]
        later[i] = False
        earlier_competing = (durations <= durations[i]) & competing
        earlier_competing[i] = False
        if not (later.any() or earlier_competing.any()):
            continue

        # I{M(X_i) > M(X_j)}, with ties at ½ — the weighting the paper offers
        # beside its strict-inequality definition.
        def _q(mask):
            others = risk[mask]
            return np.where(
                risk[i] > others, 1.0, np.where(risk[i] == others, 0.5, 0.0)
            )

        if ipcw:
            # Ŵ_ij,1 = G(T̃_i⁻)·G(T̃_i): i's event observed, and j still under
            # observation at that moment.
            w_later = np.full(int(later.sum()), 1.0 / (own[i] * after[i]))
            # Ŵ_ij,2 = G(T̃_i⁻)·G(T̃_j⁻): both events observed, since j's competing
            # event must itself have been seen for the pair to be usable.
            w_competing = 1.0 / (own[i] * own[earlier_competing])
        else:
            w_later = np.ones(int(later.sum()))
            w_competing = np.ones(int(earlier_competing.sum()))

        numerator += float(np.sum(w_later * _q(later)))
        numerator += float(np.sum(w_competing * _q(earlier_competing)))
        denominator += float(np.sum(w_later)) + float(np.sum(w_competing))

    if denominator == 0:
        return float("nan")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Dataclasses for HPO ranges
# ---------------------------------------------------------------------------


@dataclass
class ContinuousRange:
    """Continuous hyperparameter range for HPO."""

    low: float
    high: float
    log: bool = False


@dataclass
class IntegerRange:
    """Integer hyperparameter range for HPO."""

    low: int
    high: int


@dataclass
class CategoricalRange:
    """Categorical hyperparameter range for HPO."""

    choices: list


# ---------------------------------------------------------------------------
# SurvivalModel facade
# ---------------------------------------------------------------------------


class SurvivalModel:
    """
    A DeepSurv survival model: train it, then predict time to failure.

    A model is built from a config dict and trained with `fit()`. Preprocessing
    (scaling and the optional engineered features below) is part of the model, so
    `predict_mean()` and `predict()` take raw feature rows in the same shape as
    the training data.

    ```python
    model = SurvivalModel({
        "model_type": "deepsurv",
        "algorithm_arn": os.environ["SAGEMAKER_ALGORITHM_ARN"],
        "epochs": 30,
        "num_nodes": [24],
        "max_rul": 125,
    })
    model.fit(train_df, val_split=0.2)
    model.deploy()
    mean_times = model.predict_mean(test_df)
    ```

    ### Hyperparameters

    Every key is optional. A key marked *Range* may also be given as a
    `ContinuousRange`, `IntegerRange` or `CategoricalRange`, which makes `fit()`
    tune it (see `fit()`).

    | Key | Type | Default | Range | Meaning |
    |-----|------|---------|-------|---------|
    | `model_type` | str | `"deepsurv"` | no | The model. `"deepsurv"` is the one this release offers. |
    | `epochs` | int | `100` | yes | Maximum training epochs. With validation data, training stops early when the validation loss stops improving. |
    | `num_nodes` | list of int | `[32, 32]` | yes | Hidden layer sizes, one entry per layer. |
    | `dropout` | float | `0.1` | yes | Dropout rate after each hidden layer. |
    | `learning_rate` | float | `0.01` | yes | Initial learning rate. It is reduced when the validation loss plateaus. |
    | `batch_size` | int | `256` | yes | Rows per training batch. |
    | `max_rul` | float or None | `None` | yes | Caps the training durations at this value, and limits the C-index to units whose duration is at most this. Common for turbofan data, where early life carries no wear signal. |
    | `use_delta` | bool | `False` | yes | Adds, for each feature, its change from the unit's previous row. |
    | `use_per_unit_normalisation` | bool | `False` | yes | Adds, for each feature, its offset from the unit's first row. |
    | `rolling_means` | list of int or None | `None` | yes | Window sizes; adds, for each feature and window, the unit's rolling mean. |
    | `calibration` | str or None | `None` | yes | `"isotonic"` calibrates the predicted survival curves on the validation data, so `fit()` then needs `val_df` or `val_split`. |
    | `n_trials` | int | `2` | no | Tuning only: how many configurations to try. |
    | `hpo_metric` | str | `"c-index"` | no | Tuning only: the score trials are compared by: `"c-index"`, `"ibs"` (integrated Brier score) or `"antolini"` (time-dependent C-index). |
    | `verbose` | bool | `False` | no | Tuning only: log each trial in detail. |

    ### SageMaker settings

    Where training runs. They are not hyperparameters and cannot be tuned.

    | Key | Type | If left out | Meaning |
    |-----|------|-------------|---------|
    | `algorithm_arn` | str | `SAGEMAKER_ALGORITHM_ARN` is used; without it, `fit()` raises | The ARN of the algorithm you subscribed to. |
    | `instance_type` | str | `"ml.m5.large"` | The instance type of the training job. |
    | `bucket` | str | `SAGEMAKER_BUCKET` is used; without it, SageMaker's default bucket for your account and region | The S3 bucket for training data and model artifacts. |

    The job's IAM role is read from the `SAGEMAKER_ROLE_ARN` environment variable,
    which must be set.
    """

    def __init__(self, config):
        """
        Create an untrained model.

        Args:
            config: The model's configuration: hyperparameters and SageMaker
                settings, as listed in the tables above. Missing keys take their
                defaults.
        """
        # Minimal shared defaults
        defaults = {
            "model_type": "deepsurv",
            "epochs": 100,
            "max_rul": None,
        }

        self.config = {**defaults, **config}

        # Shared state
        self._is_fitted = False
        self._training_log = None
        self._tuning_trials = None
        self._evaluation_results = None
        self.in_features = self.config.get("in_features")

        # Stashed DataFrames from fit() for post-fit conversion methods
        self._fit_train_df = None
        self._fit_val_df = None
        self._fit_test_df = None
        self._fit_test_labels_df = None

        # Create backend. The backend reports its own name back into the config,
        # so `get_config()` still answers the question without this file having to
        # spell out which backends exist — that is _make_backend's business.
        self._backend = _make_backend(self.config.get("backend"))
        self.config["backend"] = self._backend.name

        # Let backend merge its defaults and initialize state
        self._backend.init_state(self.config)

    # ------------------------------------------------------------------
    # Property accessors — delegate to backend for test compatibility
    # ------------------------------------------------------------------





    @property
    def _calibrator(self):
        return getattr(self._backend, "_calibrator", None)

    @_calibrator.setter
    def _calibrator(self, value):
        self._backend._calibrator = value

    @property
    def _estimator(self):
        return getattr(self._backend, "_estimator", None)

    @property
    def _model_data(self):
        return getattr(self._backend, "_model_data", None)

    @_model_data.setter
    def _model_data(self, value):
        self._backend._model_data = value

    @property
    def _training_job_name(self):
        return getattr(self._backend, "_training_job_name", None)

    @_training_job_name.setter
    def _training_job_name(self, value):
        self._backend._training_job_name = value

    @property
    def _sagemaker_model_path(self):
        return getattr(self._backend, "_sagemaker_model_path", None)

    @_sagemaker_model_path.setter
    def _sagemaker_model_path(self, value):
        self._backend._sagemaker_model_path = value

    @property
    def _sagemaker_model_name(self):
        return getattr(self._backend, "_sagemaker_model_name", None)

    @_sagemaker_model_name.setter
    def _sagemaker_model_name(self, value):
        self._backend._sagemaker_model_name = value

    @property
    def _sagemaker_model(self):
        return getattr(self._backend, "_sagemaker_model", None)

    @_sagemaker_model.setter
    def _sagemaker_model(self, value):
        self._backend._sagemaker_model = value

    @property
    def _fleet_sub_models(self):
        return getattr(self._backend, "_fleet_sub_models", None)

    @_fleet_sub_models.setter
    def _fleet_sub_models(self, value):
        self._backend._fleet_sub_models = value

    @property
    def _fleet_n_risks(self):
        return getattr(self._backend, "_fleet_n_risks", None)

    @_fleet_n_risks.setter
    def _fleet_n_risks(self, value):
        self._backend._fleet_n_risks = value

    @property
    def _cr_kmeans(self):
        return getattr(self._backend, "_cr_kmeans", None)

    @_cr_kmeans.setter
    def _cr_kmeans(self, value):
        self._backend._cr_kmeans = value

    @property
    def _cr_resid_mean(self):
        return getattr(self._backend, "_cr_resid_mean", None)

    @_cr_resid_mean.setter
    def _cr_resid_mean(self, value):
        self._backend._cr_resid_mean = value

    @property
    def _cr_resid_std(self):
        return getattr(self._backend, "_cr_resid_std", None)

    @_cr_resid_std.setter
    def _cr_resid_std(self, value):
        self._backend._cr_resid_std = value

    @property
    def _cr_feature_scaler(self):
        return getattr(self._backend, "_cr_feature_scaler", None)

    @_cr_feature_scaler.setter
    def _cr_feature_scaler(self, value):
        self._backend._cr_feature_scaler = value

    @property
    def _sagemaker_session(self):
        return getattr(self._backend, "_sagemaker_session", None)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sdk_version():
        """This package's version, or ``None`` when it cannot be determined.

        Read from the installed distribution's metadata, which is the one place
        that has the answer at the time it is asked: the version is set at build
        time from the release tag, so no file in the source tree carries it.

        ``None`` rather than a placeholder. The container skips its compatibility
        check when no version reaches it, and a parseable stand-in misses that path
        and fails the comparison instead — telling the caller their version does
        not match, on the strength of a number nobody chose.
        """
        import importlib.metadata

        try:
            return importlib.metadata.version("lifespan-predictor")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df,
        val_df=None,
        val_split=None,
        test_df=None,
        test_labels_df=None,
    ):
        """
        Train the model, tuning its hyperparameters first if the config asks.

        **Training once.** When every config value is a plain value, `fit()`
        trains one model on `train_df` and returns its training log:

        ```python
        model = SurvivalModel({"model_type": "deepsurv", "epochs": 30})
        log = model.fit(train_df, val_split=0.2)
        print(log.best_epochs)
        ```

        To measure the model, give `fit()` a test set it does not train on, as
        `test_df` and `test_labels_df`, and read `get_evaluation_result()`.
        Validation data is no measure: it decides when training stops.

        **Tuning.** When any config value is a `ContinuousRange`, `IntegerRange`
        or `CategoricalRange`, `fit()` first searches for the best values:

        1. It splits `train_df` into 5 folds with `stratified_kfold`. The number
           of folds is fixed at 5.
        2. It tries `n_trials` configurations, each drawn from the ranges and
           scored by 5-fold cross-validation: trained on four folds,
           early-stopped on the fifth, scored on that fifth fold by
           `hpo_metric`, and averaged over the five.
        3. It retrains on all the training data with the best configuration,
           and returns that model's training log.

        The trials and the best configuration are then available from
        `get_tuning_trials()`. `val_df` and `val_split` do not change how trials
        are scored: each trial is scored on its own folds, and the validation
        data serves the final training only. With `val_split`, the held-out units
        are set aside first, so the folds are made from the rest. A config
        without ranges trains once, with no cross-validation at all.

        Each fold that scores a trial also decided when that trial stopped
        training, so trial scores lean optimistic. They rank configurations
        against each other; they do not measure the final model, which a test
        set does.

        ```python
        model = SurvivalModel({
            "model_type": "deepsurv",
            "learning_rate": ContinuousRange(1e-4, 1e-2, log=True),
            "num_nodes": CategoricalRange([[16], [32, 32]]),
            "n_trials": 20,
        })
        model.fit(train_df)
        print(model.get_tuning_trials().best_config)
        ```

        To search in a way of your own, such as a grid search, see
        `stratified_kfold`.

        **Data.** Each DataFrame has one row per unit and time step. `train_df`
        and `val_df` hold `unit` (the unit's ID), `duration` (the time from that
        row to the unit's failure or censoring), `event` (1 if the unit failed, 0
        if it was censored) and the feature columns, and may hold `cycles` (the
        time step). Every column other than `unit`, `cycles`, `duration`,
        `event` and `group` is a feature.

        Args:
            train_df: The training data.
            val_df: Validation data, with the same columns as `train_df`. It
                drives early stopping and the learning-rate reductions, so it is
                not a fair set to score the model on. Give either `val_df` or
                `val_split`, not both.
            val_split: The fraction of `train_df`'s units (0 to 1) to hold out
                as validation data instead, split by unit and balanced on event
                rate and duration.
            test_df: Test samples: rows of feature columns with `unit`, without
                labels. Needs `test_labels_df`.
            test_labels_df: One row per test unit, with `unit`, `duration` (the
                remaining life at the unit's last row in `test_df`) and `event`.
                With `test_df`, the model is evaluated after training, and
                `get_evaluation_result()` returns the scores.

        Returns:
            The `TrainingLog` of the model that was trained: of the single
            training, or of the final training after tuning.

        Raises:
            ValueError: If both `val_df` and `val_split` are given, or if
                `calibration` is `"isotonic"` and there is no validation data.
        """
        if val_df is not None and val_split is not None:
            raise ValueError(
                "val_df and val_split are mutually exclusive. "
                "Provide one or the other, not both."
            )

        if val_df is None and val_split is not None:
            from lifespan_predictor_sdk.hpo_io import _stratified_val_split

            train_df, val_df = _stratified_val_split(train_df, val_fraction=val_split)
            print(
                f"val_split={val_split}: {train_df['unit'].nunique()} train units, "
                f"{val_df['unit'].nunique()} val units "
                f"({len(train_df)} train rows, {len(val_df)} val rows)"
            )

        if self.config.get("calibration") == "isotonic" and val_df is None:
            raise ValueError(
                "calibration='isotonic' requires validation data. "
                "Provide val_df or val_split to fit()."
            )

        training_log = self._backend.fit(
            self,
            train_df,
            val_df=val_df,
            test_df=test_df,
            test_labels_df=test_labels_df,
        )

        self._training_log = training_log

        # Whatever a backend does once training has finished is the backend's own
        # business; the facade tells it that fitting is done and asks nothing about
        # what it is.
        self._backend._after_fit(
            self, val_df=val_df, test_df=test_df, test_labels_df=test_labels_df
        )

        return training_log

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_mean(self, X, **kwargs):
        """
        Predict each row's mean time to failure.

        Runs a batch transform job, or uses the real-time endpoint if `deploy()`
        created one.

        Args:
            X: Feature rows, a DataFrame with `unit` and the training data's
                feature columns, and no labels.
            **kwargs: Batch transform options: `model_name` (default: the model
                `deploy()` created), `instance_type` (default `"ml.m5.large"`),
                `instance_count` (default 1), `wait` (default True) and
                `max_payload` in MB (default 6).

        Returns:
            A numpy array of mean times to failure, one per row of `X`.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        return self._backend.predict_mean(self, X, **kwargs)

    def predict(self, X, time_points=None):
        """
        Predict each row's survival curve.

        Runs the same job as `predict_mean()`.

        Args:
            X: Feature rows, as for `predict_mean()`.
            time_points: Times to return the curves at. They select times from
                the model's own time grid; values are not interpolated. Without
                them, the whole grid is returned.

        Returns:
            A DataFrame of survival probabilities, indexed by time under the name
            `duration`, with one column per row of `X`.
        """
        curves = self._backend.predict(self, X, time_points=time_points)

        # Named here rather than in either backend because this is the one
        # function all of them pass through. Three paths build that frame —
        # pycox for DeepSurv, the Weibull helper for WTTE-RNN, and the batch
        # transform for SageMaker — and only the first named the index, having
        # inherited the name from pycox rather than chosen it. The other two
        # agreed with each other and disagreed with it, which is a difference a
        # caller sees when comparing or concatenating results.
        #
        # Set only when absent, so a backend with a reason to name its own
        # index keeps it.
        if curves is not None and curves.index.name is None:
            curves.index.name = SURVIVAL_TIME_INDEX_NAME
        return curves

    # ------------------------------------------------------------------
    # Evaluation (shared logic)
    # ------------------------------------------------------------------

    def evaluate(self, X, durations, events):
        """
        Score the model on labelled data.

        Typically `X` is a test set's last row per unit, and `durations` and
        `events` its labels. With `max_rul` set, the C-index counts only units
        whose duration is at most `max_rul`.

        Args:
            X: Feature rows, as for `predict_mean()`.
            durations: For each row of `X`, the time to failure or censoring.
            events: For each row of `X`, 1 if the unit failed and 0 if it was
                censored.

        Returns:
            An `EvaluationResult` with the C-index, RMSE, calibration and
            integrated Brier score.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before evaluation. Call fit() first."
            )

        predictions = self.predict_mean(X)

        max_rul = self.config.get("max_rul", None)
        print(f"Evaluating metrics with max_rul: {max_rul}")
        if max_rul is not None:
            mask_cindex = durations <= max_rul
        else:
            mask_cindex = np.ones(len(durations), dtype=bool)

        # C-index
        c_index = c_index_score(
            durations[mask_cindex], events[mask_cindex], predictions[mask_cindex]
        )
        # The censoring-corrected companion. `predictions` are survival
        # times, so they are negated into a risk for the IPCW estimator's
        # higher-means-worse convention.
        corrected_c_index = ipcw_c_index(
            durations[mask_cindex],
            events[mask_cindex],
            -np.asarray(predictions)[mask_cindex],
        )

        # RMSE (uncensored only)
        mask_rmse = (np.asarray(events) > 0) & mask_cindex
        rmse = rmse_score(durations[mask_rmse], predictions[mask_rmse])

        # Calibration metrics from survival curves
        cal_bias = None
        cal_mean_cdf = None
        cal_classification = None
        cal_sorted_cdf = None
        cal_empirical_cdf = None
        cal_sorted_cdf_events = None
        cal_empirical_cdf_events = None
        ibs_value = None
        ibs_class = None
        antolini_value = None

        try:
            import warnings as _warnings

            from lifespan_predictor_sdk.calibration import (
                calibration_bias,
                calibration_curve_data_from_survival,
                integrated_brier_score,
                interpret_calibration_bias,
                interpret_integrated_brier_score,
            )

            # Get survival curves from backend
            survival_curves, time_grid, surv_df = self._backend.get_survival_curves(
                self, X
            )

            if survival_curves is not None:
                sorted_cdf, empirical_cdf = calibration_curve_data_from_survival(
                    durations, survival_curves, time_grid
                )
                event_mask = events.astype(bool)
                if np.sum(event_mask) >= 2:
                    sorted_cdf_events, empirical_cdf_events = (
                        calibration_curve_data_from_survival(
                            durations[event_mask],
                            survival_curves[event_mask],
                            time_grid,
                        )
                    )
                else:
                    sorted_cdf_events, empirical_cdf_events = None, None
                bias = calibration_bias(sorted_cdf, empirical_cdf)
                mean_cdf = float(np.mean(sorted_cdf))
                bias_interp = interpret_calibration_bias(mean_cdf)

                cal_bias = float(bias)
                cal_mean_cdf = mean_cdf
                cal_classification = bias_interp["classification"]
                cal_sorted_cdf = sorted_cdf.tolist()
                cal_empirical_cdf = empirical_cdf.tolist()
                cal_sorted_cdf_events = (
                    sorted_cdf_events.tolist()
                    if sorted_cdf_events is not None
                    else None
                )
                cal_empirical_cdf_events = (
                    empirical_cdf_events.tolist()
                    if empirical_cdf_events is not None
                    else None
                )

                # IBS
                try:
                    pos_mask = durations > 0
                    if np.sum(pos_mask) > 0 and np.sum(events[pos_mask]) > 0:
                        dur_pos = durations[pos_mask]
                        ev_pos = events[pos_mask]
                        sc_pos = survival_curves[pos_mask]
                        t_min, t_max = float(np.min(dur_pos)), float(np.max(dur_pos))
                        tp_mask = (time_grid >= t_min) & (time_grid <= t_max)
                        if np.sum(tp_mask) >= 2:
                            with _warnings.catch_warnings(record=True) as caught:
                                _warnings.simplefilter("always")
                                ibs = integrated_brier_score(
                                    dur_pos,
                                    ev_pos,
                                    sc_pos[:, tp_mask],
                                    time_grid[tp_mask],
                                )
                            for w in caught:
                                print(f"  \u26a0\ufe0f  {w.message}")
                            if not np.isnan(ibs):
                                ibs_interp = interpret_integrated_brier_score(
                                    float(ibs)
                                )
                                ibs_value = float(ibs)
                                ibs_class = ibs_interp["classification"]
                except Exception as _e:
                    print(f"  \u26a0\ufe0f  Could not compute IBS: {_e}")

                # Antolini time-dependent C-index (requires pycox \u2014 optional)
                if surv_df is not None:
                    try:
                        from pycox.evaluation import EvalSurv

                        ev = EvalSurv(surv_df, durations, events, censor_surv="km")
                        score = ev.concordance_td("antolini")
                        if not np.isnan(score):
                            antolini_value = float(score)
                    except Exception as _e:
                        print(
                            f"  \u26a0\ufe0f  Could not compute Antolini C-index: {_e}"
                        )
        except Exception as e:
            print(f"  \u26a0\ufe0f  Could not compute calibration metrics: {e}")

        # Competing-risk metrics (backend-specific)
        (
            cr_per_risk_c,
            cr_per_risk_antolini,
            cr_per_risk_predictive,
        ) = self._backend.get_competing_risk_metrics(
            self, X, durations, events, mask_cindex=mask_cindex
        )

        return EvaluationResult(
            c_index=c_index,
            ipcw_c_index=corrected_c_index,
            rmse=rmse,
            predictions=predictions,
            mask_cindex=mask_cindex,
            mask_rmse=mask_rmse,
            y_test=durations,
            events=np.asarray(events),
            calibration_bias=cal_bias,
            calibration_mean_cdf=cal_mean_cdf,
            calibration_classification=cal_classification,
            calibration_sorted_cdf=cal_sorted_cdf,
            calibration_empirical_cdf=cal_empirical_cdf,
            calibration_sorted_cdf_events=cal_sorted_cdf_events,
            calibration_empirical_cdf_events=cal_empirical_cdf_events,
            ibs=ibs_value,
            ibs_classification=ibs_class,
            antolini_c_index=antolini_value,
            competing_risk_per_risk_cause_specific_c_index=cr_per_risk_c,
            competing_risk_per_risk_antolini=cr_per_risk_antolini,
            competing_risk_per_risk_predictive_c_index=cr_per_risk_predictive,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Delegated post-fit analysis methods
    # ------------------------------------------------------------------




    def get_mean_time_to_failure(self, X):
        """
        The same as `predict_mean()`.

        Args:
            X: Feature rows, as for `predict_mean()`.

        Returns:
            A numpy array of mean times to failure, one per row of `X`.
        """
        return self.predict_mean(X)

    def _predict_all_rows(self, df):
        """Return survival curves for every row in df (used by tuning)."""
        return self._backend._predict_all_rows(self, df)

    def _compute_val_antolini(self, val_df):
        """Compute Antolini's time-dependent C-index on val_df."""
        return self._backend._compute_val_antolini(self, val_df)

    def _compute_val_ibs(self, val_df, train_df):
        """Compute IBS on val_df."""
        return self._backend._compute_val_ibs(self, val_df, train_df)

    def _fit_calibrator_on_df(self, df):
        """Fit isotonic calibrator on df."""
        return self._backend._fit_calibrator_on_df(self, df)

    def _apply_calibrator(self, surv_df):
        """Apply isotonic calibrator to survival curves."""
        return self._backend._apply_calibrator(surv_df)



    # ------------------------------------------------------------------
    # Delegated SageMaker-only methods
    # ------------------------------------------------------------------

    def deploy(self, model_name=None, endpoint_instance_type=None):
        """
        Make the trained model available for prediction on SageMaker.

        By default this creates a SageMaker model that `predict_mean()` and
        `predict()` run batch transform jobs against. With
        `endpoint_instance_type`, it also hosts the model on a real-time endpoint,
        which predictions then use until `delete()`. An endpoint is billed for as
        long as it exists.

        Args:
            model_name: The name of the SageMaker model. Without it, a name is
                derived from the algorithm's.
            endpoint_instance_type: The instance type of a real-time endpoint,
                such as `"ml.m5.large"`. Without it, no endpoint is created.

        Returns:
            The name of the SageMaker model.
        """
        return self._backend.deploy(
            self, model_name=model_name, endpoint_instance_type=endpoint_instance_type
        )

    def delete(self):
        """Delete what `deploy()` created: the SageMaker model, and its endpoint."""
        self._backend.delete(self)

    def print_training_job_summary(self):
        """Print a summary of the model's SageMaker training job."""
        self._backend.print_training_job_summary(self)

    # ------------------------------------------------------------------
    # Shared accessors
    # ------------------------------------------------------------------

    def get_training_log(self):
        """
        The training log of the last `fit()`.

        Returns:
            The same `TrainingLog` `fit()` returned.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before accessing training log. Call fit() first."
            )
        if self._training_log is None:
            raise ValueError(
                "No training log available. This should not happen if model was fitted properly."
            )
        return self._training_log

    def get_tuning_trials(self):
        """
        The results of the tuning `fit()` ran.

        Returns:
            A `TuningTrials`, or None if the config had no ranges and `fit()`
            did not tune.
        """
        return getattr(self, "_tuning_trials", None)

    def get_config(self):
        """
        The model's config, with defaults filled in.

        Returns:
            The config dict.
        """
        return self.config

    def print_config(self):
        """Print the model configuration."""
        print("Model Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def get_evaluation_result(self, pre_calibration=False):
        """
        Get the evaluation result from training.

        Args:
            pre_calibration: If True, return the pre-calibration result.

        Returns:
            EvaluationResult if test data was provided during fit(), otherwise None.
        """
        if pre_calibration:
            if self._evaluation_results is None:
                raise ValueError(
                    "No pre-calibration result available: model has not been calibrated."
                )
            return self._evaluation_results["before"]
        if self._evaluation_results is not None:
            return self._evaluation_results.get("after")
        return None

