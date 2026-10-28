"""
Survival analysis model facade.
Routes to the configured backend via the Strategy pattern.
"""

import os
from dataclasses import dataclass

import numpy as np

from lifespan_predictor_sdk.metrics import capped_labels, score
from lifespan_predictor_sdk.training_log import EvaluationResult, TrainingLog

# The name every survival-curve frame carries on its time index. One value in
# one place, because the frame is built by three different code paths and the
# only one that used to name it took the name from pycox by inheritance.
SURVIVAL_TIME_INDEX_NAME = "duration"

# The model types with a global and a per-unit analysis: those whose log-hazard is
# linear in their features. Asking another for one is an error.
ANALYSED_MODEL_TYPES = ("coxph",)


# Auto-detect available backends
def _make_backend(name=None):
    """Construct the backend that runs training and inference."""
    from lifespan_predictor_sdk.sagemaker_backend import SageMakerBackend

    return SageMakerBackend()


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
    | `model_type` | str | `"deepsurv"` | no | The model: `"deepsurv"`, a neural network, or `"coxph"`, a penalised linear Cox model whose effects can be read directly, for when failures are few. A `"coxph"` model's `get_global_analysis()` and `get_local_analysis()` explain it. |
    | `coxph_penalizer` | float | `0.1` | yes | `"coxph"` only: the strength of the penalty on the coefficients. Larger values give smaller, more stable effects. |
    | `coxph_l1_ratio` | float | `0.0` | yes | `"coxph"` only: the share of the penalty that is lasso (L1), from `0` (ridge) to `1` (lasso). Lasso sets weak features' effects to exactly zero. |
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
        # Minimal shared defaults. The rest depend on the model type, so the
        # backend fills them in.
        defaults = {
            "model_type": "deepsurv",
            "max_rul": None,
        }

        self.config = {**defaults, **config}

        # Shared state
        self._is_fitted = False
        self._training_log = None
        self._tuning_trials = None
        self._evaluation_results = None
        self._global_analysis = None
        self._local_analysis = None
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
        # Scoring the test set predicted too; the caller has predicted nothing yet.
        self._local_analysis = None

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
        `events` its labels. Every metric scores every unit. With `max_rul` set,
        a unit whose duration is beyond it counts as censored at `max_rul`, and
        RMSE compares its prediction with `max_rul`; see `EvaluationResult` for
        how each metric is scored.

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

        predictions = np.asarray(self.predict_mean(X))
        max_rul = self.config.get("max_rul", None)

        # The backend supplies the predictions; `score` applies every metric to
        # them on one basis, so these figures are the ones its tests check.
        try:
            survival_curves, time_grid, _ = self._backend.get_survival_curves(self, X)
        except Exception as e:
            print(f"  \u26a0\ufe0f  Could not get survival curves: {e}")
            survival_curves, time_grid = None, None
        scores = score(
            durations, events, predictions, survival_curves, time_grid, max_rul
        )

        # Competing-risk metrics (backend-specific), on the same capped labels.
        durations_scored, events_scored, _ = capped_labels(durations, events, max_rul)
        (
            cr_per_risk_c,
            cr_per_risk_antolini,
            cr_per_risk_predictive,
        ) = self._backend.get_competing_risk_metrics(
            self, X, durations_scored, events_scored
        )

        return EvaluationResult(
            **scores,
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

    def get_global_analysis(self):
        """
        How the fitted model acts on any unit: its coefficients, each raw
        feature's effect, and the proportional-hazards test.

        Only a Cox model, `model_type="coxph"`, has one: its log-hazard is linear
        in its features, so each feature's effect is one number.

        Returns:
            A `GlobalAnalysis`, or None if the model was restored with `load()`:
            the analysis belongs to the fit, as the training log does.

        Raises:
            ValueError: If the model has not been fitted, or if its `model_type`
                is not `"coxph"`.
        """
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before accessing its global analysis. "
                "Call fit() first."
            )
        self._require_analysis("get_global_analysis")
        return self._global_analysis

    def get_local_analysis(self):
        """
        What drives each unit's risk in the most recent `predict()` or
        `predict_mean()`.

        Only a Cox model, `model_type="coxph"`, has one: its log-hazard is a sum
        over features, so each unit's risk splits exactly into their parts.

        Returns:
            A `LocalAnalysis`, or None before any prediction.

        Raises:
            ValueError: If the model's `model_type` is not `"coxph"`.
        """
        self._require_analysis("get_local_analysis")
        return self._local_analysis

    def _require_analysis(self, getter):
        """Refuse an analysis for a model type that has none."""
        model_type = self.config.get("model_type")
        if model_type not in ANALYSED_MODEL_TYPES:
            raise ValueError(
                f"{getter}() is available only for model_type='coxph'; this model "
                f"is {model_type!r}."
            )

