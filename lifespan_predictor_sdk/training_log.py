"""
Training log and fit quality analysis for survival models.
"""

from enum import Enum

import numpy as np
import pandas as pd


class EvaluationResult:
    """
    How well a model scored on labelled data.

    Returned by `SurvivalModel.evaluate()`, and by
    `SurvivalModel.get_evaluation_result()` when `fit()` was given test data.
    A metric that could not be computed is None.

    Attributes:
        c_index: Harrell's concordance index: the fraction of comparable pairs
            of units whose predicted times are in the right order. 0.5 is
            chance, 1.0 is perfect. It is the score tuning compares trials by.
        ipcw_c_index: The concordance index corrected for censoring, so it does
            not move with how many units were censored. With nothing censored it
            equals `c_index`.
        antolini_c_index: The time-dependent concordance of the whole survival
            curves.
        rmse: Root mean squared error of the predicted mean times, over the
            units that failed.
        ibs: Integrated Brier score of the survival curves; lower is better.
        ibs_classification: A plain-words reading of `ibs`.
        calibration_bias: How far the predicted probabilities are from the
            observed ones, signed.
        calibration_classification: A plain-words reading of
            `calibration_bias`.
        n_scored: How many units were scored.
        n_capped: How many had a true remaining life beyond `max_rul`.
        n_rmse: How many units RMSE was computed over: those that failed.
        n_calibration: How many units the calibration bias was computed over:
            those that failed within `max_rul`.
        max_rul: The model's `max_rul`, or None.

    ## How the metrics are scored

    Every metric scores every unit, on one basis. The model predicts no further
    than its `max_rul`, so a unit whose true remaining life is beyond it is
    scored as follows:

    - **The C-indices and the IBS** treat it as censored at `max_rul`: it is
      known to have lasted that long, but not when it failed.
    - **RMSE** compares its prediction with `max_rul`, the truth capped at the
      horizon, which is the usual convention on the NASA turbofan data. A unit
      censored in the data has no true time, so RMSE leaves it out.
    - **The calibration bias** compares each unit's predicted probability of
      having failed by its true time against an even spread, which holds only
      for units whose failure was observed. It therefore uses the units that
      failed within `max_rul`.

    `print_metrics()` and `report()` say how many units were scored and how
    many were capped, and the counts are on the result and in `get_metrics()`.

    A result can also be read like a dict: `result["c_index"]`,
    `result.get("ibs")`.
    """

    def __init__(
        self,
        c_index,
        rmse,
        predictions=None,
        mask_rmse=None,
        y_test=None,
        events=None,
        calibration_bias=None,
        calibration_mean_cdf=None,
        calibration_classification=None,
        calibration_sorted_cdf=None,
        calibration_empirical_cdf=None,
        calibration_sorted_cdf_events=None,
        calibration_empirical_cdf_events=None,
        ibs=None,
        ibs_classification=None,
        antolini_c_index=None,
        competing_risk_per_risk_cause_specific_c_index=None,
        competing_risk_per_risk_antolini=None,
        competing_risk_per_risk_predictive_c_index=None,
        ipcw_c_index=None,
        capped=None,
        max_rul=None,
        n_scored=None,
        n_capped=None,
        n_rmse=None,
        n_calibration=None,
    ):
        import numpy as np

        self.c_index = c_index
        self.rmse = rmse
        self.predictions = predictions
        self.mask_rmse = mask_rmse
        self.y_test = y_test
        self.events = events
        self.calibration_bias = calibration_bias
        self.calibration_mean_cdf = calibration_mean_cdf
        self.calibration_classification = calibration_classification
        self.calibration_sorted_cdf = calibration_sorted_cdf
        self.calibration_empirical_cdf = calibration_empirical_cdf
        self.calibration_sorted_cdf_events = calibration_sorted_cdf_events
        self.calibration_empirical_cdf_events = calibration_empirical_cdf_events
        self.ibs = ibs
        self.ibs_classification = ibs_classification
        self.antolini_c_index = antolini_c_index
        self.competing_risk_per_risk_cause_specific_c_index = (
            competing_risk_per_risk_cause_specific_c_index
        )
        self.competing_risk_per_risk_antolini = competing_risk_per_risk_antolini
        self.competing_risk_per_risk_predictive_c_index = (
            competing_risk_per_risk_predictive_c_index
        )
        self.ipcw_c_index = ipcw_c_index
        # Which units were scored, and how (see "How the metrics are scored").
        # The counts are passed directly when the per-unit arrays are not kept,
        # as for the result of a training job.
        self.capped = capped
        self.max_rul = max_rul
        if n_scored is None and y_test is not None:
            n_scored = len(y_test)
        if n_capped is None and capped is not None:
            n_capped = int(np.sum(capped))
        if n_rmse is None and mask_rmse is not None:
            n_rmse = int(np.sum(mask_rmse))
        self.n_scored = n_scored
        self.n_capped = n_capped
        self.n_rmse = n_rmse
        self.n_calibration = n_calibration

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        """
        A metric by name, like `dict.get`.

        Args:
            key: The attribute's name, such as `"c_index"`.
            default: What to return if there is no such attribute.

        Returns:
            The metric's value, or `default`.
        """
        return getattr(self, key, default)

    def get_metrics(self):
        """
        The scalar metrics, as a dict.

        Returns:
            A dict with `c_index` and `rmse`, each other metric that was
            computed, and the counts of units they were scored over.
        """
        metrics = {"c_index": self.c_index, "rmse": self.rmse}
        if self.ipcw_c_index is not None:
            metrics["ipcw_c_index"] = self.ipcw_c_index
        if self.antolini_c_index is not None:
            metrics["antolini_c_index"] = self.antolini_c_index
        if self.ibs is not None:
            metrics["ibs"] = self.ibs
        if self.calibration_bias is not None:
            metrics["calibration_bias"] = self.calibration_bias
        for name in ("n_scored", "n_capped", "n_rmse", "n_calibration", "max_rul"):
            if getattr(self, name) is not None:
                metrics[name] = getattr(self, name)
        if self.competing_risk_per_risk_cause_specific_c_index is not None:
            metrics["competing_risk_per_risk_cause_specific_c_index"] = list(
                self.competing_risk_per_risk_cause_specific_c_index
            )
        if self.competing_risk_per_risk_antolini is not None:
            metrics["competing_risk_per_risk_antolini"] = list(
                self.competing_risk_per_risk_antolini
            )
        if self.competing_risk_per_risk_predictive_c_index is not None:
            metrics["competing_risk_per_risk_predictive_c_index"] = list(
                self.competing_risk_per_risk_predictive_c_index
            )
        return metrics

    def _basis_lines(self):
        """The scoring basis in words: how many units, which were capped, and why."""
        import textwrap

        if self.n_scored is None:
            return []
        n = self.n_scored
        m = self.max_rul
        if m is not None and float(m).is_integer():
            m = int(m)
        capped = self.n_capped or 0
        if m is None:
            text = f"Scored on {n} units, against their observed times."
        elif not capped:
            text = (
                f"Scored on {n} units, against their observed times; none is beyond "
                f"max_rul={m}."
            )
        else:
            have = "has" if capped == 1 else "have"
            them = "it" if capped == 1 else "them"
            text = (
                f"Scored on {n} units. {capped} {have} a true remaining life beyond "
                f"max_rul={m}, the horizon the model predicts: RMSE counts {them} as "
                f"{m}, and the C-indices and IBS treat {them} as censored at {m}."
            )
        if self.n_rmse is not None and self.n_rmse < n:
            text += f" RMSE uses the {self.n_rmse} that failed."
        if (
            self.calibration_bias is not None
            and self.n_calibration is not None
            and self.n_calibration < n
        ):
            within = f" within max_rul={m}" if capped else ""
            text += (
                f" Calibration bias uses the {self.n_calibration} that failed{within}."
            )
        return textwrap.wrap(
            text, width=76, initial_indent="  ", subsequent_indent="  "
        )

    def print_metrics(self):
        """Print evaluation metrics in a formatted table.

        The table opens with the scoring basis: how many units were scored, and
        how those beyond `max_rul` were counted.
        """
        # A competing-risks model's top block is scored over *all* causes, so it says
        # so. Without the label it reads as though it described a single risk — which
        # is exactly how it was misread while these fields held risk 1's numbers.
        #
        # The predictive figure is the one every competing-risks model reports, so it
        # is what identifies one. The cause-specific figures are fleet-only diagnostics
        # and are absent for DeepHit, which has no sub-models to diagnose.
        is_competing_risk = (
            self.competing_risk_per_risk_predictive_c_index is not None
            or self.competing_risk_per_risk_cause_specific_c_index is not None
            or self.competing_risk_per_risk_antolini is not None
        )
        header = (
            "All-Cause Evaluation Metrics"
            if is_competing_risk
            else "Evaluation Metrics"
        )
        print(f"=== {header} ===")
        for line in self._basis_lines():
            print(line)
        # The IPCW-corrected figure leads when it is available. Harrell's C depends
        # on the study's censoring distribution, so the same model on the same
        # population scores differently under heavier drop-out; the corrected one
        # does not. Harrell's follows it: it is the figure tuning ranks trials by,
        # the one published results report, and the gap between the two shows how
        # much censoring moves the score.
        if self.ipcw_c_index is not None:
            print(
                f"  C-index (IPCW):       {self.ipcw_c_index:.4f}  (higher is better, "
                "0.5 = random; corrected for censoring)"
            )
            print(
                f"  C-index (Harrell):    {self.c_index:.4f}  (uncorrected; the tuning "
                "metric)"
            )
        else:
            print(
                f"  C-index (Harrell):    {self.c_index:.4f}  (higher is better, 0.5 = random, 1.0 = perfect)"
            )
        if self.antolini_c_index is not None:
            print(
                f"  C-index (Antolini):   {self.antolini_c_index:.4f}  (higher is better, time-dependent)"
            )
        print(f"  RMSE:                 {self.rmse:.4f}  (lower is better)")
        if self.ibs is not None:
            label = f"  ({self.ibs_classification})" if self.ibs_classification else ""
            print(
                f"  IBS:                  {self.ibs:.4f}  (lower is better, <0.1 = excellent){label}"
            )
        if self.calibration_bias is not None:
            label = (
                f"  ({self.calibration_classification})"
                if self.calibration_classification
                else ""
            )
            print(
                f"  Calibration bias:     {self.calibration_bias:.4f}  (closer to 0 is better){label}"
            )
        if (
            self.competing_risk_per_risk_cause_specific_c_index is not None
            or self.competing_risk_per_risk_antolini is not None
            or self.competing_risk_per_risk_predictive_c_index is not None
        ):

            def _fmt(c):
                return (
                    "n/a"
                    if c is None or (isinstance(c, float) and c != c)
                    else f"{c:.4f}"
                )

            print("=== Competing-Risk Metrics ===")
            # The predictive figure leads, and is the only per-risk number that means
            # the same thing for a DeepHit and for a CSH fleet — so it is the only one
            # the two may be compared on. It keeps a competing-event unit comparable,
            # because that unit is known never to fail from cause k.
            #
            # The figures below it are within-family diagnostics, printed under their
            # own heading so they are not read as competing answers to the same
            # question. A fleet's are its sub-models' own scores, in a counterfactual
            # world where cause k is the only cause; DeepHit's Antolini scores the
            # 1 - CIF_k curve in the real one. Useful for locating a weak sub-model,
            # not for ranking one model family against the other.
            if self.competing_risk_per_risk_predictive_c_index is not None:
                print("  Per-risk C-index (predictive, competing-risks-aware):")
                for k, c in enumerate(
                    self.competing_risk_per_risk_predictive_c_index, start=1
                ):
                    print(f"    Risk {k}: {_fmt(c)}")
            if (
                self.competing_risk_per_risk_cause_specific_c_index is not None
                or self.competing_risk_per_risk_antolini is not None
            ):
                print("  Diagnostics (within-family, not comparable across models):")
                if self.competing_risk_per_risk_cause_specific_c_index is not None:
                    print("    Per-sub-model C-index (cause-specific, Harrell):")
                    for k, c in enumerate(
                        self.competing_risk_per_risk_cause_specific_c_index, start=1
                    ):
                        print(f"      Risk {k}: {_fmt(c)}")
                if self.competing_risk_per_risk_antolini is not None:
                    print("    Per-risk C-index (time-dependent, Antolini):")
                    for k, c in enumerate(
                        self.competing_risk_per_risk_antolini, start=1
                    ):
                        print(f"      Risk {k}: {_fmt(c)}")

    def plot_predictions(self, figsize=(8, 6)):
        """Scatter plot of actual against predicted times, for the units that failed.

        The actual times are the ones RMSE scored: a unit whose true remaining
        life is beyond `max_rul` is plotted at `max_rul`, with its own marker.

        Args:
            figsize: The figure's size in inches, as `(width, height)`.

        Returns:
            A matplotlib `(fig, ax)` tuple, or None if the per-unit predictions
            were not kept, as for the result of a training job.
        """
        import matplotlib.pyplot as plt

        if self.predictions is None or self.mask_rmse is None or self.y_test is None:
            print("Prediction arrays not available (e.g. SageMaker backend).")
            return None

        import numpy as np

        y_test_masked = self.y_test[self.mask_rmse]
        y_pred_masked = self.predictions[self.mask_rmse]

        fig, ax = plt.subplots(figsize=figsize)

        # Color by competing-risk class when events are multi-risk
        events_masked = None
        if self.events is not None:
            events_masked = np.asarray(self.events)[self.mask_rmse].astype(int)

        if events_masked is not None and events_masked.max(initial=0) > 1:
            unique_risks = np.sort(np.unique(events_masked[events_masked > 0]))
            cmap = plt.get_cmap("tab10")
            for i, k in enumerate(unique_risks):
                mask = events_masked == k
                ax.scatter(
                    y_test_masked[mask],
                    y_pred_masked[mask],
                    color=cmap(i % 10),
                    label=f"Risk {int(k)}",
                    alpha=0.8,
                )
            ax.legend(title="Competing risk", loc="best")
        elif self.capped is not None and np.any(np.asarray(self.capped)):
            capped_masked = np.asarray(self.capped)[self.mask_rmse]
            ax.scatter(
                y_test_masked[~capped_masked],
                y_pred_masked[~capped_masked],
                label="Failed within max_rul",
            )
            ax.scatter(
                y_test_masked[capped_masked],
                y_pred_masked[capped_masked],
                marker="x",
                label=f"Beyond max_rul, counted at {self.max_rul:g}",
            )
            ax.legend(loc="best")
        else:
            ax.scatter(y_test_masked, y_pred_masked)

        ax.set_xlabel("Actual Values (y_test)")
        ax.set_ylabel("Predicted Values (y_pred)")
        ax.set_title("Actual vs Predicted")
        plt.show()
        return fig, ax

    def plot_calibration(self, figsize=(12, 5), show="uncensored"):
        """Plot the calibration curve, and the integrated Brier score.

        Args:
            figsize: The figure's size in inches, as `(width, height)`.
            show: `"uncensored"` draws the curve over the units that failed;
                `"all"` draws it over all units, with the uncensored curve
                beside it.

        Returns:
            A matplotlib `(fig, axes)` tuple, or None if there is no calibration
            data.
        """
        if self.calibration_sorted_cdf is None and self.calibration_bias is None:
            print("No calibration data available.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Model Calibration", fontsize=14)

        ax1 = axes[0]
        sorted_cdf = self.calibration_sorted_cdf
        empirical_cdf = self.calibration_empirical_cdf
        if sorted_cdf is not None and empirical_cdf is not None:
            sorted_cdf = np.atleast_1d(np.asarray(sorted_cdf, dtype=float))
            empirical_cdf = np.atleast_1d(np.asarray(empirical_cdf, dtype=float))
            sorted_cdf_events = self.calibration_sorted_cdf_events
            empirical_cdf_events = self.calibration_empirical_cdf_events
            if sorted_cdf_events is not None:
                sorted_cdf_events = np.atleast_1d(
                    np.asarray(sorted_cdf_events, dtype=float)
                )
            if empirical_cdf_events is not None:
                empirical_cdf_events = np.atleast_1d(
                    np.asarray(empirical_cdf_events, dtype=float)
                )
            ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            if show in ("all", "both"):
                ax1.plot(
                    sorted_cdf, empirical_cdf, "b-", lw=2, label="All observations"
                )
                ax1.fill_between(
                    sorted_cdf, sorted_cdf, empirical_cdf, alpha=0.1, color="blue"
                )
                if sorted_cdf_events is not None and empirical_cdf_events is not None:
                    ax1.plot(
                        sorted_cdf_events,
                        empirical_cdf_events,
                        "r-",
                        lw=2,
                        label="Uncensored only",
                    )
            elif show == "uncensored":
                cdf = sorted_cdf_events if sorted_cdf_events is not None else sorted_cdf
                ecdf = (
                    empirical_cdf_events
                    if empirical_cdf_events is not None
                    else empirical_cdf
                )
                label = (
                    "Uncensored only"
                    if sorted_cdf_events is not None
                    else "All observations"
                )
                ax1.plot(cdf, ecdf, "b-", lw=2, label=label)
                ax1.fill_between(cdf, cdf, ecdf, alpha=0.15, color="blue")
            else:
                ax1.plot(
                    sorted_cdf, empirical_cdf, "b-", lw=2, label="All observations"
                )
                ax1.fill_between(
                    sorted_cdf, sorted_cdf, empirical_cdf, alpha=0.15, color="blue"
                )
            ax1.set_xlabel("Predicted CDF")
            ax1.set_ylabel("Empirical CDF")
            ax1.set_title("Calibration Curve")
            ax1.legend()
            if (
                self.calibration_classification is not None
                or self.calibration_bias is not None
            ):
                parts = []
                if self.calibration_classification:
                    parts.append(self.calibration_classification)
                if self.calibration_bias is not None:
                    parts.append(f"bias={self.calibration_bias:.3f}")
                ax1.annotate(
                    "\n".join(parts), xy=(0.05, 0.85), xycoords="axes fraction"
                )
        else:
            ax1.text(
                0.5,
                0.5,
                "No calibration data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title("Calibration Curve")

        ax2 = axes[1]
        if self.ibs is not None:
            colors = {
                "excellent": "green",
                "good": "yellowgreen",
                "acceptable": "orange",
                "poor": "red",
            }
            color = colors.get(self.ibs_classification or "", "steelblue")
            ax2.barh(["IBS"], [self.ibs], color=color)
            ax2.axvline(
                0.25, color="gray", linestyle="--", label="Random baseline (0.25)"
            )
            ax2.set_xlim(0, max(0.4, self.ibs * 1.2))
            ax2.set_title("Integrated Brier Score")
            ax2.legend()
            ibs_label = f"{self.ibs:.3f}"
            if self.ibs_classification:
                ibs_label += f" — {self.ibs_classification}"
            ax2.annotate(ibs_label, xy=(self.ibs + 0.01, 0), va="center")
        else:
            ax2.text(
                0.5,
                0.5,
                "IBS not available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Integrated Brier Score")

        plt.tight_layout()
        return fig, axes

    def report(self):
        """Print the metrics and draw both plots."""
        self.print_metrics()
        self.plot_predictions()
        self.plot_calibration()


class GlobalAnalysis:
    """
    How a Cox model, `model_type="coxph"`, acts on any unit.

    Produced by `fit()`, and read with `SurvivalModel.get_global_analysis()`. Only
    the Cox model has one: its log-hazard is linear in its features, so each
    feature's effect is one number. For another model type, the getter raises
    `ValueError`.
    Model features are the columns the model was fitted on: the raw input
    features and any columns derived from them, such as rolling means. Every
    model feature is min-max scaled to [0, 1] on the training data.

    ### Reading the coefficients

    The hazard is a unit's instantaneous failure rate: how likely it is to fail
    in the next moment, given it has run until now. The model takes it as
    `baseline(t) × exp(coef₁·x₁ + coef₂·x₂ + …)`. The baseline is shared by every
    unit, and the features scale it.

    `coef` and `hazard_ratio` are the same effect on two scales. Because the
    features are scaled to [0, 1], each is the effect of moving that feature
    from its training minimum to its maximum, with the others held fixed:

    | | `coef` | `hazard_ratio` |
    |---|---|---|
    | Scale | log-hazard | hazard, as a multiplier |
    | No effect | `0` | `1` |
    | Raises risk | above `0` | above `1` |
    | Lowers risk | below `0` | between `0` and `1` |
    | Two features together | add | multiply |

    For example, a coefficient of `3.2` is a hazard ratio of about `25`: at its
    training maximum, the unit fails at about 25 times the rate it does at the
    minimum. A coefficient of `-2.6` is a ratio of about `0.07`, or 7% of the
    rate. Together they add to `0.6`, a ratio of about `1.8`, which is the two
    ratios multiplied: `24.5 × 0.074`.

    Do arithmetic with `coef`: effects add on its scale, which is why the parts
    of `LocalAnalysis.contributions` sum to the unit's log hazard ratio. Read
    `hazard_ratio` to say how large an effect is. Its interval is the
    coefficient's passed through `exp`, so it isn't symmetric around the ratio;
    `plot_hazard_ratios()` uses a log axis, on which it is.

    A hazard ratio is not a ratio of lifetimes. A ratio of `2` doubles the
    failure rate at every moment, and how much that shortens the remaining life
    depends on the baseline. With a weak penalty, near-copy features can split
    one effect into large pieces that cancel, so one coefficient can look huge
    on its own. Read `feature_effects`, which merges a feature's pieces, or refit
    with a stronger penalty.

    Attributes:
        coefficients: A DataFrame indexed by model feature, with columns `coef`
            (the effect on the log-hazard of moving from the feature's training
            minimum to its maximum), `hazard_ratio` (`exp(coef)`), its 95%
            interval `hr_lower_95` and `hr_upper_95`, and `p`. Standard errors
            treat all rows of one unit as one observation. With a penalty, the
            intervals are those of the penalised estimate, a guide rather than
            exact inference.
        feature_effects: A Series indexed by raw input feature: the change in
            log-hazard when that feature shifts by its training range and stays
            there. It sums the feature's own coefficient and those of the
            columns that follow its level (rolling means, per-unit normalised),
            each converted to the raw feature's units. Delta columns are left
            out, because a sustained shift leaves them at zero. Read this, not
            `coefficients`, when a feature enters the model more than once: a
            weak penalty can split its effect into large pieces that cancel.
        ph_test: A DataFrame indexed by model feature, with `test_statistic`
            and `p` of the proportional-hazards test. It treats rows as
            independent, so with many rows per unit its p-values are too small;
            read it as a ranking of which features break the assumption most.
        failed_units: The number of training units with an observed failure.
        failed_units_per_feature: `failed_units` divided by the number of
            model features.
    """

    def __init__(
        self,
        coefficients,
        feature_effects,
        ph_test,
        failed_units,
        failed_units_per_feature,
    ):
        self.coefficients = coefficients
        self.feature_effects = feature_effects
        self.ph_test = ph_test
        self.failed_units = failed_units
        self.failed_units_per_feature = failed_units_per_feature

    def to_dict(self):
        """The analysis as JSON-ready data.

        Returns:
            A dict of plain maps keyed by feature name, which `from_dict()` reads back.
        """
        return {
            "coefficients": self.coefficients.to_dict(orient="index"),
            "feature_effects": self.feature_effects.to_dict(),
            "ph_test": self.ph_test.to_dict(orient="index"),
            "failed_units": int(self.failed_units),
            "failed_units_per_feature": float(self.failed_units_per_feature),
        }

    @classmethod
    def from_dict(cls, data):
        """The analysis `to_dict()` wrote.

        Args:
            data: The dict `to_dict()` returned.

        Returns:
            A `GlobalAnalysis`.
        """
        return cls(
            coefficients=pd.DataFrame.from_dict(data["coefficients"], orient="index"),
            feature_effects=pd.Series(data["feature_effects"], dtype=float),
            ph_test=pd.DataFrame.from_dict(data["ph_test"], orient="index"),
            failed_units=data["failed_units"],
            failed_units_per_feature=data["failed_units_per_feature"],
        )

    def plot_hazard_ratios(self, figsize=None):
        """Each fitted feature's hazard ratio with its 95% interval, on a log axis.

        The largest effect is at the top. Features with no `p`, which the fit left
        out, are dropped.

        Args:
            figsize: The figure's size in inches, as `(width, height)`. By default
                the height grows with the number of features.

        Returns:
            A matplotlib `(fig, ax)` tuple.
        """
        import matplotlib.pyplot as plt

        fitted = self.coefficients.dropna(subset=["p"])
        fitted = fitted.reindex(fitted["coef"].abs().sort_values().index)

        fig, ax = plt.subplots(figsize=figsize or (7, 0.4 * len(fitted) + 1))
        ax.errorbar(
            fitted["hazard_ratio"],
            range(len(fitted)),
            xerr=[
                fitted["hazard_ratio"] - fitted["hr_lower_95"],
                fitted["hr_upper_95"] - fitted["hazard_ratio"],
            ],
            fmt="o",
            capsize=3,
        )
        ax.axvline(1.0, color="grey", linestyle="--")
        ax.set_xscale("log")
        ax.set_yticks(range(len(fitted)))
        ax.set_yticklabels(fitted.index)
        ax.set_xlabel("hazard ratio, training min to max (95% CI)")
        plt.show()
        return fig, ax


class LocalAnalysis:
    """
    What drives each predicted unit's risk, for a Cox model, `model_type="coxph"`.

    Produced by `predict()` and `predict_mean()`, and read with
    `SurvivalModel.get_local_analysis()`. It describes the most recent
    prediction; the next one replaces it. Only the Cox model has one: its
    log-hazard is a sum over features, so each unit's risk splits exactly into
    their parts. For another model type, the getter raises `ValueError`.

    Attributes:
        contributions: A DataFrame with one row per unit, indexed by unit ID
            under the name `unit` in the order of `predict()`'s columns, and one
            column per raw input feature. Each value is how much that feature
            raises (positive) or lowers (negative) the unit's log-hazard at its
            last row, compared with the average training row. Columns derived
            from a feature, such as its rolling means, count towards it.
        log_hazard_ratio: Each row's sum: the unit's log-hazard minus that of
            the average training row. The parts add up to it exactly.
    """

    def __init__(self, contributions):
        self.contributions = contributions

    @property
    def log_hazard_ratio(self):
        """Each unit's log-hazard minus that of the average training row.

        Returns:
            A `pandas.Series` indexed by unit: each row of `contributions`, summed.
        """
        return self.contributions.sum(axis=1).rename("log_hazard_ratio")

    def to_dict(self):
        """The analysis as JSON-ready data.

        Returns:
            A dict of `units`, `features` and `contributions`, one inner list per unit,
            which `from_dicts()` reads back.
        """
        return {
            "units": self.contributions.index.tolist(),
            "features": self.contributions.columns.tolist(),
            "contributions": self.contributions.to_numpy().tolist(),
        }

    @classmethod
    def from_dicts(cls, parts):
        """One analysis from the `to_dict()` of each part of a prediction.

        Args:
            parts: The dicts `to_dict()` returned, one per part, in unit order.

        Returns:
            A `LocalAnalysis` of every part's units.
        """
        return cls(
            pd.concat(
                pd.DataFrame(
                    part["contributions"],
                    index=pd.Index(part["units"], name="unit"),
                    columns=part["features"],
                )
                for part in parts
            )
        )

    def _active(self):
        """The contributions of features that move some unit's risk."""
        return self.contributions.loc[:, self.contributions.abs().max() > 0]

    def _risk_order(self):
        return self.log_hazard_ratio.sort_values(ascending=False).index

    def _feature_tree(self):
        """The correlation of contributions across units, and its clustering on 1 − |r|."""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Strongly correlated features carry the same information whether they push
        # risk the same way or opposite ways, so the distance uses |r|. A feature
        # whose contribution is the same for every unit has no correlation; it is
        # treated as unrelated to the others.
        correlation = self._active().corr().fillna(0.0)
        # On a copy: under copy-on-write, always on from pandas 3, `.values` is
        # read-only.
        values = correlation.to_numpy(copy=True)
        np.fill_diagonal(values, 1.0)
        correlation = pd.DataFrame(
            values, index=correlation.index, columns=correlation.columns
        )
        if len(correlation) < 2:
            return correlation, None
        distance = squareform(1 - correlation.abs().to_numpy(), checks=False)
        return correlation, linkage(distance, method="average")

    def feature_groups(self, min_correlation=0.7):
        """Which features carry the same information about risk.

        Features are clustered by average linkage on `1 − |r|`, where r is the
        correlation of their contributions across units, and the tree is cut at
        `1 − min_correlation`. Near-copy features split one effect between them,
        so summing a group (`group_contributions`) shows what they say together.

        Args:
            min_correlation: How closely, on average, a group's features must move
                together, as an absolute correlation.

        Returns:
            A `pandas.Series` indexed by feature, giving each feature's group label:
            the group's features joined by `" + "`, in column order. Features that
            contribute nothing to any unit are left out.
        """
        from scipy.cluster.hierarchy import fcluster

        correlation, tree = self._feature_tree()
        if tree is None:
            return pd.Series(correlation.index, index=correlation.index, name="group")
        labels = pd.Series(
            fcluster(tree, t=1 - min_correlation, criterion="distance"),
            index=correlation.index,
        )
        return labels.map(
            lambda label: " + ".join(labels.index[labels == label])
        ).rename("group")

    def group_contributions(self, groups):
        """Contributions summed per group of features.

        Args:
            groups: A `pandas.Series` mapping feature to group label, such as the
                result of `feature_groups()`. Features not in it must contribute
                nothing, or their part is lost.

        Returns:
            A DataFrame indexed like `contributions`, with one column per group
            label. Each row still adds up to `log_hazard_ratio`.
        """
        active = self._active()
        return active.T.groupby(groups.reindex(active.columns)).sum().T

    def top_drivers(self, groups=None):
        """The feature, or group of features, that moves each unit's risk most.

        Args:
            groups: Optional mapping of feature to group label, such as the result
                of `feature_groups()`. Given, the groups' summed contributions are
                compared instead of single features.

        Returns:
            A `pandas.Series` indexed by unit, naming the feature or group with the
            largest absolute contribution.
        """
        parts = self._active() if groups is None else self.group_contributions(groups)
        return parts.abs().idxmax(axis=1).rename("top_driver")

    def similarity(self):
        """How alike the units' explanations are.

        Returns:
            A unit × unit DataFrame of the cosine similarity of the units'
            contributions: 1 for the same pattern, −1 for the opposite one. A unit
            whose contributions are all zero has no similarity (NaN).
        """
        vectors = self._active().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        units = self.contributions.index
        return pd.DataFrame(unit_vectors @ unit_vectors.T, index=units, columns=units)

    def cluster_units(self, n_clusters=2, units=None):
        """Groups of units at risk for the same reason.

        Units are clustered by average linkage on `1 − similarity()`. The safest
        units' contributions are close to zero, so their direction says little;
        restricting the clustering to the riskiest units sharpens it.

        Args:
            n_clusters: How many clusters to form.
            units: Optional unit IDs to cluster, in the order to return them. By
                default, every unit.

        Returns:
            A `pandas.Series` indexed by unit, giving cluster numbers
            `1..n_clusters`.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        similarity = self.similarity()
        if units is not None:
            similarity = similarity.loc[units, units]
        distance = np.clip(1 - similarity.fillna(0.0).to_numpy(), 0, None)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        return pd.Series(
            fcluster(tree, n_clusters, criterion="maxclust"),
            index=similarity.index,
            name="cluster",
        )

    def plot_contributions(self, figsize=(9, 10)):
        """Heatmap of what drives each unit's risk.

        Units are ordered riskiest at the top, and features by mean absolute
        contribution. Red raises risk, blue lowers it.

        Args:
            figsize: The figure's size in inches, as `(width, height)`.

        Returns:
            A matplotlib `(fig, ax)` tuple.
        """
        import matplotlib.pyplot as plt

        active = self._active()
        features = active.abs().mean().sort_values(ascending=False).index
        grid = active.loc[self._risk_order(), features]
        limit = grid.abs().to_numpy().max()

        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=90)
        ax.set_yticks([])
        ax.set_ylabel("units, riskiest at the top")
        fig.colorbar(image, ax=ax, label="contribution to log-hazard")
        ax.set_title("What drives each unit's risk")
        plt.show()
        return fig, ax

    def plot_feature_correlation(self, figsize=(7, 6)):
        """Heatmap of how features' contributions correlate across units.

        Features are ordered by the clustering `feature_groups()` uses, so groups
        show as blocks along the diagonal.

        Args:
            figsize: The figure's size in inches, as `(width, height)`.

        Returns:
            A matplotlib `(fig, ax)` tuple.
        """
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import leaves_list

        correlation, tree = self._feature_tree()
        order = (
            correlation.index if tree is None else correlation.index[leaves_list(tree)]
        )

        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(correlation.loc[order, order], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        fig.colorbar(image, ax=ax, label="correlation of contributions across units")
        plt.show()
        return fig, ax

    def plot_similarity(self, figsize=(7, 6)):
        """Heatmap of `similarity()`, with units ordered riskiest first.

        Args:
            figsize: The figure's size in inches, as `(width, height)`.

        Returns:
            A matplotlib `(fig, ax)` tuple.
        """
        import matplotlib.pyplot as plt

        order = self._risk_order()
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(
            self.similarity().loc[order, order], cmap="RdBu_r", vmin=-1, vmax=1
        )
        ax.set_xlabel("units, riskiest first")
        ax.set_ylabel("units, riskiest first")
        fig.colorbar(image, ax=ax, label="cosine similarity of explanations")
        plt.show()
        return fig, ax


class FitQuality(Enum):
    """Quality of model fit based on training dynamics."""

    GOOD_FIT = "good_fit"
    OVER_FIT = "over_fit"
    UNDER_FIT = "under_fit"
    FULL_FIT = "full_fit"
    UNKNOWN_FIT = "unknown_fit"


class TrainingLog:
    """
    The history of one training run, returned by `SurvivalModel.fit()`.

    Attributes:
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch, or None without validation data.
        learning_rate: Learning rate per epoch.
        epochs: The configured maximum number of epochs.
        num_epochs: The number of epochs trained, fewer than `epochs` if
            training stopped early.
        best_epochs: The epoch with the lowest validation loss, whose weights
            the model keeps; None without validation data.
    """

    def __init__(self, train_loss, learning_rate, epochs, val_loss=None):
        self.train_loss = train_loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_loss = val_loss

        # Automatically determine num_epochs. This is the number of epochs trained.
        self.num_epochs = len(self.train_loss)
        if self.val_loss:
            self.best_epochs = int(self.val_loss.index(min(self.val_loss)) + 1)
        else:
            self.best_epochs = None
        self._val_c_index = None

    def to_pandas(self):
        """
        The losses as a DataFrame.

        Returns:
            A DataFrame with `epoch`, `train_loss` and, with validation data,
            `val_loss`.
        """
        data = {
            "epoch": list(range(1, len(self.train_loss) + 1)),
            "train_loss": self.train_loss,
        }
        if self.val_loss:
            data["val_loss"] = self.val_loss
        return pd.DataFrame(data)

    def plot(self, figsize=(12, 6), title="Training History"):
        """
        Plot training and validation losses with learning rate changes and best epoch markers.

        The plot automatically includes:
        - Training and validation loss curves
        - Learning rate reduction markers (vertical dashed lines)
        - Best epoch marker (star on validation loss)
        - Fit quality analysis in legend title (GOOD_FIT, OVER_FIT, etc.)

        Args:
            figsize: Tuple of (width, height) for the figure size
            title: Title for the plot

        Returns:
            matplotlib Figure and Axes objects

        Example:
            >>> log = model.fit(X_train, durations, events, X_val, durations_val, events_val)
            >>> fig, ax = log.plot()
            >>> plt.savefig('training_history.png')
            >>> # Legend will show fit quality automatically (e.g., "✓ Fit Quality: GOOD FIT")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. Install with: uv add matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        epochs = list(range(1, len(self.train_loss) + 1))

        # Plot training loss
        ax.plot(
            epochs,
            self.train_loss,
            label="Training Loss",
            color="#2E86AB",
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.8,
        )

        # Plot validation loss if available
        if self.val_loss:
            ax.plot(
                epochs,
                self.val_loss,
                label="Validation Loss",
                color="#A23B72",
                linewidth=2,
                marker="s",
                markersize=4,
                alpha=0.8,
            )

            # Mark best epoch with a larger dot
            if self.best_epochs and self.best_epochs <= len(self.val_loss):
                best_val_loss = self.val_loss[self.best_epochs - 1]
                ax.plot(
                    self.best_epochs,
                    best_val_loss,
                    marker="*",
                    markersize=20,
                    color="#F18F01",
                    label=f"Best Epoch ({self.best_epochs})",
                    zorder=5,
                )

        # Detect and mark learning rate reductions
        if self.learning_rate and len(self.learning_rate) > 1:
            lr_changes = []
            for i in range(1, len(self.learning_rate)):
                if self.learning_rate[i] < self.learning_rate[i - 1] * 0.95:
                    lr_changes.append(i + 1)  # +1 for 1-based epoch numbering

            # Draw vertical lines for LR changes
            for epoch_idx in lr_changes:
                ax.axvline(
                    x=epoch_idx,
                    color="#C73E1D",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

            # Add single legend entry for LR changes
            if lr_changes:
                ax.axvline(
                    x=-1,
                    color="#C73E1D",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="LR Reduction",
                )

        # Analyze fit quality for legend
        fit_quality = self.analyze_fit_quality()

        # Create legend title with fit quality
        legend_title = None
        if fit_quality is not None:
            quality_name = fit_quality.value.upper().replace("_", " ")
            # Add symbol for visual indication (using text symbols for better compatibility)
            quality_symbols = {
                "GOOD_FIT": "✓",
                "FULL_FIT": "★",
                "OVER_FIT": "⚠",
                "UNDER_FIT": "↓",
                "UNKNOWN_FIT": "?",
            }
            symbol = quality_symbols.get(fit_quality.value.upper(), "")
            legend_title = f"{symbol} Fit Quality: {quality_name}"

        # Styling
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add legend with fit quality title
        legend = ax.legend(loc="best", framealpha=0.9, fontsize=10)
        if legend_title:
            legend.set_title(legend_title, prop={"size": 10, "weight": "bold"})

        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set x-axis to show integer epochs
        ax.set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))

        plt.tight_layout()
        return fig, ax

    def analyze_fit_quality(self):
        """
        Judge from the losses whether the model fits its data well.

        Epochs after `best_epochs` are not considered, and a brief rise in
        validation loss just after a learning-rate reduction is tolerated.

        Returns:
            None without validation data; otherwise a `FitQuality`, whose
            `.value` is one of:

            - `"good_fit"`: both losses fell and the gap between them did not
              widen, and training ran to about its last epoch.
            - `"full_fit"`: like a good fit, but training stopped early and the
              last learning-rate reduction no longer helped.
            - `"over_fit"`: validation loss rose while training loss fell for
              more than three epochs in a row.
            - `"under_fit"`: both losses still falling, but by less than 1% per
              epoch over the last ten epochs.
            - `"unknown_fit"`: none of the above.
        """
        if self.val_loss is None or len(self.val_loss) == 0:
            return None
        # Detect learning rate reductions (95% threshold to account for floating point)
        lr_reductions = []
        for i in range(1, len(self.learning_rate)):
            if self.learning_rate[i] < self.learning_rate[i - 1] * 0.95:
                lr_reductions.append(i)

        # Truncate to best_epochs so post-best data doesn't skew fit analysis
        if self.best_epochs is not None:
            train_to_best = self.train_loss[: self.best_epochs]
            val_to_best = self.val_loss[: self.best_epochs]
        else:
            train_to_best = self.train_loss
            val_to_best = self.val_loss

        # Check if GOOD_FIT conditions hold
        is_good_fit = self._check_good_fit(train_to_best, val_to_best, lr_reductions)

        if is_good_fit:
            # Check if FULL_FIT: early stopping kicked in
            if self.epochs > self.num_epochs:
                return FitQuality.FULL_FIT
            else:
                return FitQuality.GOOD_FIT

        # If not GOOD_FIT, check for OVER_FIT (already truncated to best_epochs above)
        is_over_fit = self._check_over_fit(train_to_best, val_to_best, lr_reductions)
        if is_over_fit:
            return FitQuality.OVER_FIT

        # If not OVER_FIT, check for UNDER_FIT
        is_under_fit = self._check_under_fit(train_to_best, val_to_best)
        if is_under_fit:
            return FitQuality.UNDER_FIT

        # Neither condition holds
        return FitQuality.UNKNOWN_FIT

    def _check_good_fit(self, train_loss, val_loss, lr_reductions):
        """
        Check if training meets GOOD_FIT criteria:
        1. Both train_loss and val_loss are decreasing overall
        2. Gap between losses is not widening excessively (considering improvement rates)
        3. Temporary val_loss increases after LR reductions are tolerated (1-2 epochs)

        Returns:
            bool: True if GOOD_FIT conditions are met
        """
        if len(train_loss) < 5 or len(val_loss) < 5:
            # Not enough data to determine
            return False

        # 1. Check if both losses are decreasing overall
        # Compare first 5 epochs average with last 5 epochs average
        train_early = sum(train_loss[:5]) / 5
        train_late = sum(train_loss[-5:]) / 5
        val_early = sum(val_loss[:5]) / 5
        val_late = sum(val_loss[-5:]) / 5

        if train_late >= train_early or val_late >= val_early:
            # Losses not decreasing overall
            return False

        # Calculate improvement rates
        train_improvement = (train_early - train_late) / train_early
        val_improvement = (val_early - val_late) / val_early

        # 2. Check if gap is not widening excessively
        # Gap = val_loss - train_loss
        gap_early = val_early - train_early
        gap_late = val_late - train_late

        # If both losses are improving significantly (>10%), allow more gap widening
        # This handles the natural phenomenon where validation plateaus more than training
        if train_improvement > 0.1 and val_improvement > 0.1:
            # Both improving well, allow gap to widen up to 2.5x
            gap_threshold = 2.5
        else:
            # Slow improvement, be stricter about gap widening
            gap_threshold = 1.5

        if gap_late > gap_early * gap_threshold:
            # Gap widening too much even with lenient threshold
            return False

        # 3. Check for prolonged val_loss increases (ignoring LR reduction spikes)
        # Mark epochs that are within 1-2 epochs after LR reduction as "tolerance zones"
        tolerance_zones = set()
        for lr_idx in lr_reductions:
            tolerance_zones.add(lr_idx)  # Epoch of LR reduction
            if lr_idx + 1 < len(val_loss):
                tolerance_zones.add(lr_idx + 1)  # +1 epoch
            if lr_idx + 2 < len(val_loss):
                tolerance_zones.add(lr_idx + 2)  # +2 epochs

        # Count consecutive val_loss increases outside tolerance zones
        consecutive_increases = 0
        max_consecutive_increases = 0

        for i in range(1, len(val_loss)):
            if val_loss[i] > val_loss[i - 1]:
                # Val loss increased
                if i not in tolerance_zones:
                    consecutive_increases += 1
                    max_consecutive_increases = max(
                        max_consecutive_increases, consecutive_increases
                    )
            else:
                consecutive_increases = 0

        # If val_loss increased for >3 consecutive epochs (outside tolerance), not GOOD_FIT
        if max_consecutive_increases > 3:
            return False

        # 4. Check improvement rate is reasonable (not too slow)
        # GOOD_FIT should show meaningful progress, not just tiny decreases
        # Analyze last 10 epochs (or fewer if not available)
        if len(train_loss) >= 15:
            window_size = min(10, len(train_loss) // 2)
            train_recent = train_loss[-window_size:]
            val_recent = val_loss[-window_size:]

            train_start = train_recent[0]
            train_end = train_recent[-1]
            val_start = val_recent[0]
            val_end = val_recent[-1]

            if train_start > 0 and val_start > 0:
                train_improvement_per_epoch = (
                    (train_start - train_end) / train_start
                ) / window_size
                val_improvement_per_epoch = (
                    (val_start - val_end) / val_start
                ) / window_size

                # Context matters: when losses are high (>1.0), we expect faster improvement
                # When losses are low (<0.7), slower percentage improvement is acceptable
                avg_loss = (train_end + val_end) / 2

                if avg_loss > 1.0:
                    # High losses: expect at least 1% improvement per epoch
                    improvement_threshold = 0.01
                else:
                    # Low losses: accept slower improvement (0.3% per epoch)
                    improvement_threshold = 0.003

                # If BOTH improving less than threshold, it's too slow for GOOD_FIT
                if (
                    train_improvement_per_epoch < improvement_threshold
                    and val_improvement_per_epoch < improvement_threshold
                ):
                    return False

        # All conditions met
        return True

    def _check_over_fit(self, train_loss, val_loss, lr_reductions):
        """
        Check if training shows OVER_FIT characteristics:
        - Gap between train and val loss widening over time
        - Val_loss increasing while train_loss decreasing for >3 consecutive epochs
        - Tolerance for 1-2 epochs after LR reductions

        Returns:
            bool: True if OVER_FIT conditions are met
        """
        if len(train_loss) < 5 or len(val_loss) < 5:
            return False

        # Mark tolerance zones around LR reductions
        tolerance_zones = set()
        for lr_idx in lr_reductions:
            tolerance_zones.add(lr_idx)
            if lr_idx + 1 < len(val_loss):
                tolerance_zones.add(lr_idx + 1)
            if lr_idx + 2 < len(val_loss):
                tolerance_zones.add(lr_idx + 2)

        # Count consecutive epochs where val_loss increases AND train_loss decreases
        # (outside tolerance zones)
        consecutive_divergence = 0
        max_consecutive_divergence = 0

        for i in range(1, min(len(train_loss), len(val_loss))):
            val_increasing = val_loss[i] > val_loss[i - 1]
            train_decreasing = train_loss[i] < train_loss[i - 1]

            if val_increasing and train_decreasing:
                # Diverging behavior
                if i not in tolerance_zones:
                    consecutive_divergence += 1
                    max_consecutive_divergence = max(
                        max_consecutive_divergence, consecutive_divergence
                    )
            else:
                consecutive_divergence = 0

        # OVER_FIT if >3 consecutive epochs of divergence
        if max_consecutive_divergence > 3:
            return True

        # Also check if gap is widening significantly
        train_early = sum(train_loss[:5]) / 5
        train_late = sum(train_loss[-5:]) / 5
        val_early = sum(val_loss[:5]) / 5
        val_late = sum(val_loss[-5:]) / 5

        gap_early = val_early - train_early
        gap_late = val_late - train_late

        # OVER_FIT if gap more than tripled AND val_loss isn't improving much
        if gap_late > gap_early * 3.0 and val_late > val_early * 0.95:
            return True

        return False

    def _check_under_fit(self, train_loss, val_loss):
        """
        Check if training shows UNDER_FIT characteristics:
        - Both losses decreasing but average improvement per epoch <1% over last 10 epochs

        Returns:
            bool: True if UNDER_FIT conditions are met
        """
        if len(train_loss) < 15 or len(val_loss) < 15:
            # Need at least 15 epochs to analyze last 10 properly
            return False

        # Analyze last 10 epochs
        window_size = min(10, len(train_loss) // 2)  # Use at most half the data
        train_recent = train_loss[-window_size:]
        val_recent = val_loss[-window_size:]

        # Check if both are still decreasing
        train_start = train_recent[0]
        train_end = train_recent[-1]
        val_start = val_recent[0]
        val_end = val_recent[-1]

        if train_end >= train_start or val_end >= val_start:
            # Not decreasing, can't be UNDER_FIT
            return False

        # Calculate average improvement per epoch (as percentage)
        train_improvement_per_epoch = (
            (train_start - train_end) / train_start
        ) / window_size
        val_improvement_per_epoch = ((val_start - val_end) / val_start) / window_size

        # UNDER_FIT if both improving less than 1% per epoch
        if train_improvement_per_epoch < 0.01 and val_improvement_per_epoch < 0.01:
            return True

        return False
