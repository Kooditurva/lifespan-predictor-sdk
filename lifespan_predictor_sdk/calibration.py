"""
Model calibration utilities for survival analysis.

This module provides comprehensive tools for assessing survival model calibration quality,
including calibration curve generation, bias detection, and Integrated Brier Score computation.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Type aliases
ArrayLike = Union[np.ndarray, list, tuple]


# ============================================================================
# Input Validation Helpers
# ============================================================================


def _validate_positive_array(arr: np.ndarray, name: str) -> None:
    """Validate that array contains only positive values."""
    if np.any(arr <= 0):
        raise ValueError(f"{name} must contain only positive values")


def _validate_non_negative_array(arr: np.ndarray, name: str) -> None:
    """Validate that array contains only non-negative values."""
    if np.any(arr < 0):
        raise ValueError(f"{name} must contain only non-negative values")


def _validate_probability_array(arr: np.ndarray, name: str) -> None:
    """Validate that array contains values in [0, 1]."""
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} must contain values in range [0, 1]")


def _validate_no_nan_inf(arr: np.ndarray, name: str) -> None:
    """Validate that array contains no NaN or infinite values."""
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values")


def _validate_non_empty(arr: np.ndarray, name: str) -> None:
    """Validate that array is not empty."""
    if len(arr) == 0:
        raise ValueError(f"{name} must not be empty")


def _validate_matching_lengths(*arrays_and_names: Tuple[np.ndarray, str]) -> None:
    """Validate that all arrays have matching lengths."""
    lengths = [len(arr) for arr, _ in arrays_and_names]
    if len(set(lengths)) > 1:
        names_and_lengths = ", ".join(
            [f"{name} ({len(arr)})" for arr, name in arrays_and_names]
        )
        raise ValueError(
            f"Input arrays must have matching lengths: {names_and_lengths}"
        )


def _validate_monotonic_increasing(arr: np.ndarray, name: str) -> None:
    """Validate that array is monotonically increasing."""
    if len(arr) > 1 and np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be monotonically increasing")


def _validate_monotonic_non_increasing(
    arr: np.ndarray, name: str, axis: int = -1
) -> None:
    """Validate that array is monotonically non-increasing along specified axis."""
    if arr.ndim == 1:
        if len(arr) > 1 and np.any(np.diff(arr) > 0):
            raise ValueError(f"{name} must be monotonically non-increasing")
    else:
        if arr.shape[axis] > 1 and np.any(np.diff(arr, axis=axis) > 0):
            raise ValueError(
                f"{name} must be monotonically non-increasing along axis {axis}"
            )


def _validate_binary_array(arr: np.ndarray, name: str) -> None:
    """Validate that array contains only 0 and 1."""
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} must be binary (0 or 1)")


# ============================================================================
# Calibration Curve Generation
# ============================================================================


def calibration_curve_data(event_times, alpha, beta):
    """
    Generate calibration curve data from Weibull distribution parameters.

    Parameters
    ----------
    event_times : array-like of shape (n_samples,)
        Observed event times (must be positive)
    alpha : array-like of shape (n_samples,)
        Weibull scale parameters (must be positive)
    beta : array-like of shape (n_samples,)
        Weibull shape parameters (must be positive)

    Returns
    -------
    sorted_cdf : np.ndarray of shape (n_samples,)
        Predicted CDF values at true event times, sorted in ascending order
    empirical_cdf : np.ndarray of shape (n_samples,)
        Empirical CDF values, uniformly spaced from 1/n to 1

    Raises
    ------
    ValueError
        If event_times contains negative values
        If alpha or beta contain non-positive values
        If input arrays have mismatched lengths
        If input arrays are empty

    Notes
    -----
    For a perfectly calibrated model, sorted_cdf should approximately equal
    empirical_cdf. Deviations indicate calibration issues:
    - Points above diagonal: under-prediction (conservative)
    - Points below diagonal: over-prediction (dangerous in safety-critical systems)
    """
    # Convert to numpy arrays
    event_times = np.asarray(event_times, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    # Validate inputs
    _validate_non_empty(event_times, "event_times")
    _validate_non_empty(alpha, "alpha")
    _validate_non_empty(beta, "beta")
    _validate_matching_lengths(
        (event_times, "event_times"), (alpha, "alpha"), (beta, "beta")
    )
    _validate_positive_array(event_times, "event_times")
    _validate_positive_array(alpha, "alpha")
    _validate_positive_array(beta, "beta")

    # Compute CDF using Weibull formula
    cdf_true = 1 - np.exp(-((event_times / alpha) ** beta))

    # Sort CDF values
    sort_idx = np.argsort(cdf_true)
    sorted_cdf = cdf_true[sort_idx]

    # Compute empirical CDF
    empirical = np.arange(1, len(cdf_true) + 1) / len(cdf_true)

    return sorted_cdf, empirical


def calibration_curve_data_from_survival(
    event_times: ArrayLike,
    survival_curves: ArrayLike,
    time_grid: ArrayLike,
    interpolation: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate calibration curve data from survival curve predictions.

    Parameters
    ----------
    event_times : array-like of shape (n_samples,)
        Observed event times (must be positive)
    survival_curves : array-like of shape (n_samples, n_time_points) or (n_time_points,)
        Predicted survival probabilities at time_grid points
        If 1D, treated as single instance
        Must be monotonically non-increasing along time axis
        Values must be in [0, 1]
    time_grid : array-like of shape (n_time_points,)
        Time points at which survival probabilities are evaluated
        Must be monotonically increasing
    interpolation : {'linear', 'step'}, default='linear'
        Interpolation method for survival probabilities
        - 'linear': Linear interpolation between time points
        - 'step': Step function (right-continuous)

    Returns
    -------
    sorted_cdf : np.ndarray of shape (n_samples,)
        Predicted CDF values at true event times, sorted in ascending order
    empirical_cdf : np.ndarray of shape (n_samples,)
        Empirical CDF values, uniformly spaced from 1/n to 1

    Raises
    ------
    ValueError
        If event_times contains negative values
        If survival_curves not in [0, 1]
        If survival_curves not monotonically non-increasing
        If time_grid not monotonically increasing
        If time_grid length doesn't match survival_curves second dimension
        If survival_curves contains NaN or infinite values
        If input arrays are empty
        If interpolation method is not 'linear' or 'step'

    Notes
    -----
    For event times outside time_grid range, constant extrapolation is used:
    - Below minimum: uses survival probability at first time point
    - Above maximum: uses survival probability at last time point

    This function produces output in the same format as calibration_curve_data(),
    enabling downstream functions to work with both parametric and non-parametric
    predictions without modification.
    """
    # Convert to numpy arrays
    event_times = np.asarray(event_times, dtype=np.float64)
    survival_curves = np.asarray(survival_curves, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)

    # Validate interpolation method
    if interpolation not in ["linear", "step"]:
        raise ValueError("interpolation must be 'linear' or 'step'")

    # Validate inputs
    _validate_non_empty(event_times, "event_times")
    _validate_non_empty(time_grid, "time_grid")
    _validate_non_negative_array(event_times, "event_times")
    _validate_no_nan_inf(event_times, "event_times")
    _validate_no_nan_inf(time_grid, "time_grid")
    _validate_no_nan_inf(survival_curves, "survival_curves")
    _validate_monotonic_increasing(time_grid, "time_grid")

    # Handle 1D survival curves by reshaping to (1, n_time_points)
    if survival_curves.ndim == 1:
        survival_curves = survival_curves.reshape(1, -1)

    # Validate survival curves shape
    if survival_curves.ndim != 2:
        raise ValueError("survival_curves must be 1D or 2D array")

    n_samples = survival_curves.shape[0]
    n_time_points = survival_curves.shape[1]

    # Check dimension match
    if len(time_grid) != n_time_points:
        raise ValueError(
            f"time_grid length must match survival_curves second dimension: "
            f"time_grid ({len(time_grid)}) vs survival_curves ({n_time_points})"
        )

    # Check event_times length matches survival_curves first dimension
    if len(event_times) != n_samples:
        raise ValueError(
            f"event_times length must match survival_curves first dimension: "
            f"event_times ({len(event_times)}) vs survival_curves ({n_samples})"
        )

    # Validate survival probabilities
    _validate_probability_array(survival_curves, "survival_curves")
    _validate_monotonic_non_increasing(survival_curves, "survival_curves", axis=1)

    # Interpolate survival probabilities at event times
    survival_at_event = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        event_time = event_times[i]
        surv_curve = survival_curves[i]

        if interpolation == "linear":
            # Use numpy's interp for linear interpolation
            # np.interp handles extrapolation automatically (constant extrapolation)
            survival_at_event[i] = np.interp(event_time, time_grid, surv_curve)
        else:  # step function
            # Right-continuous step function: use value at previous time point
            idx = np.searchsorted(time_grid, event_time, side="right")
            if idx == 0:
                # Before first time point - use first survival probability
                survival_at_event[i] = surv_curve[0]
            elif idx >= len(time_grid):
                # After last time point - use last survival probability
                survival_at_event[i] = surv_curve[-1]
            else:
                # Use survival probability at the previous time point
                survival_at_event[i] = surv_curve[idx - 1]

    # Compute CDF = 1 - S(event_time)
    cdf_values = 1.0 - survival_at_event

    # Sort CDF values
    sorted_cdf = np.sort(cdf_values)

    # Compute empirical CDF
    empirical_cdf = np.arange(1, n_samples + 1, dtype=np.float64) / n_samples

    return sorted_cdf, empirical_cdf


# ============================================================================
# Bias Detection
# ============================================================================


def calibration_bias(predicted_cdf: ArrayLike, empirical_cdf: ArrayLike) -> float:
    """
    Compute calibration bias as mean CDF deviation.

    Parameters
    ----------
    predicted_cdf : array-like of shape (n_samples,)
        Predicted CDF values (typically from calibration_curve_data)
    empirical_cdf : array-like of shape (n_samples,)
        Empirical CDF values (typically from calibration_curve_data)

    Returns
    -------
    bias : float
        Mean calibration bias in range [-1, 1]
        - bias ≈ 0: well-calibrated
        - bias > 0: under-prediction (conservative)
        - bias < 0: over-prediction (dangerous)

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths
        If input arrays are empty

    Notes
    -----
    Bias is computed as mean(predicted_cdf - empirical_cdf).
    For perfectly calibrated models, predicted_cdf ≈ empirical_cdf, so bias ≈ 0.

    The mean CDF value under predicted distributions should be approximately 0.5
    for well-calibrated models. Use interpret_calibration_bias() for automated
    interpretation with domain-specific thresholds.
    """
    # Convert to numpy arrays
    predicted_cdf = np.asarray(predicted_cdf, dtype=np.float64)
    empirical_cdf = np.asarray(empirical_cdf, dtype=np.float64)

    # Validate inputs
    _validate_non_empty(predicted_cdf, "predicted_cdf")
    _validate_non_empty(empirical_cdf, "empirical_cdf")
    _validate_matching_lengths(
        (predicted_cdf, "predicted_cdf"), (empirical_cdf, "empirical_cdf")
    )

    # Compute bias as mean deviation
    bias = np.mean(predicted_cdf - empirical_cdf)

    return float(bias)


def interpret_calibration_bias(
    mean_cdf: float, threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Interpret calibration bias with domain-specific guidance.

    Parameters
    ----------
    mean_cdf : float
        Mean CDF value (typically from calibration_bias + 0.5)
        Must be in range [0, 1]
    threshold : float, default=0.05
        Threshold for well-calibrated classification
        Model is well-calibrated if |mean_cdf - 0.5| <= threshold

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'classification': str, one of {'well-calibrated', 'under-prediction', 'over-prediction'}
        - 'severity': str, one of {'none', 'mild', 'moderate', 'severe'}
        - 'mean_cdf': float, the input mean_cdf value
        - 'deviation': float, absolute deviation from 0.5
        - 'interpretation': str, human-readable interpretation
        - 'warning': str or None, domain-specific warning if applicable

    Raises
    ------
    ValueError
        If mean_cdf is outside range [0, 1]
        If threshold is negative or greater than 0.5

    Notes
    -----
    Classification thresholds:
    - Well-calibrated: |mean_cdf - 0.5| <= threshold
    - Under-prediction: mean_cdf > 0.5 + threshold (conservative, may waste resources)
    - Over-prediction: mean_cdf < 0.5 - threshold (dangerous in safety-critical systems)

    Severity levels based on deviation magnitude:
    - None: deviation <= threshold
    - Mild: threshold < deviation <= 0.10
    - Moderate: 0.10 < deviation <= 0.20
    - Severe: deviation > 0.20
    """
    # Validate inputs
    if not (0.0 <= mean_cdf <= 1.0):
        raise ValueError(f"mean_cdf must be in range [0, 1], got {mean_cdf}")

    if threshold < 0 or threshold > 0.5:
        raise ValueError(
            f"threshold must be positive and less than or equal to 0.5, got {threshold}"
        )

    # Compute deviation from perfect calibration
    deviation = abs(mean_cdf - 0.5)

    # Classify calibration
    if deviation <= threshold:
        classification = "well-calibrated"
    elif mean_cdf > 0.5 + threshold:
        classification = "under-prediction"
    else:  # mean_cdf < 0.5 - threshold
        classification = "over-prediction"

    # Assess severity based on deviation magnitude
    if deviation <= threshold:
        severity = "none"
    elif deviation <= 0.10:
        severity = "mild"
    elif deviation <= 0.20:
        severity = "moderate"
    else:
        severity = "severe"

    # Generate interpretation message
    if classification == "well-calibrated":
        interpretation = (
            f"Model is well-calibrated with mean CDF of {mean_cdf:.3f}. "
            f"Predicted survival distributions accurately reflect observed event times."
        )
        warning = None
    elif classification == "under-prediction":
        interpretation = (
            f"Model shows under-prediction (mean CDF = {mean_cdf:.3f}). "
            f"Predictions are conservative, potentially overestimating survival times. "
            f"This may lead to wasteful resource allocation but is safer than over-prediction."
        )
        warning = None
    else:  # over-prediction
        interpretation = (
            f"Model shows over-prediction (mean CDF = {mean_cdf:.3f}). "
            f"Predictions are optimistic, potentially underestimating survival times. "
            f"This pattern can be dangerous in safety-critical applications."
        )
        warning = (
            "SAFETY WARNING: Over-prediction detected. In safety-critical systems "
            "(e.g., predictive maintenance, medical applications), this may lead to "
            "insufficient preparation for failures or adverse events."
        )

    return {
        "classification": classification,
        "severity": severity,
        "mean_cdf": mean_cdf,
        "deviation": deviation,
        "interpretation": interpretation,
        "warning": warning,
    }


# ============================================================================
# Integrated Brier Score
# ============================================================================


def integrated_brier_score(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    predicted_surv_curves: ArrayLike,
    time_points: ArrayLike,
    censoring_times: ArrayLike = None,
    censoring_events: ArrayLike = None,
) -> float:
    """
    Compute Integrated Brier Score with IPCW for censored data.

    Parameters
    ----------
    event_times : array-like of shape (n_samples,)
        Observed event or censoring times (must be positive)
    event_observed : array-like of shape (n_samples,)
        Binary indicator of event occurrence (1 = event, 0 = censored)
    predicted_surv_curves : array-like of shape (n_samples, n_time_points)
        Predicted survival probabilities at time_points
        Must be monotonically non-increasing along time axis
        Values must be in [0, 1]
    time_points : array-like of shape (n_time_points,)
        Time points at which to evaluate Brier score
        Must be within range of observed event_times

    Returns
    -------
    ibs : float
        Integrated Brier Score in range [0, 1]
        Returns np.nan if all observations are censored

    Raises
    ------
    ValueError
        If event_times contains negative values
        If event_observed is not binary (0 or 1)
        If predicted_surv_curves not in [0, 1]
        If predicted_surv_curves not monotonically non-increasing
        If time_points outside range of event_times
        If input arrays have mismatched lengths
        If input arrays are empty

    Warnings
    --------
    - Issues warning if all observations are censored (returns np.nan)
    - Issues warning if censoring rate > 80% (IPCW may be unstable)

    Notes
    -----
    The Integrated Brier Score measures prediction accuracy across time:
    - IBS = 0: perfect predictions
    - IBS ≈ 0.25: random predictions
    - IBS = 1: worst possible predictions

    Uses Inverse Probability of Censoring Weighting (IPCW) to handle
    censored observations. Assumes censoring is non-informative.

    Integration uses trapezoidal rule normalized by time range.
    """
    from lifelines import KaplanMeierFitter

    # Convert to numpy arrays
    event_times = np.asarray(event_times, dtype=np.float64)
    event_observed = np.asarray(event_observed, dtype=np.int32)
    predicted_surv_curves = np.asarray(predicted_surv_curves, dtype=np.float64)
    time_points = np.asarray(time_points, dtype=np.float64)

    # Validate inputs
    _validate_non_empty(event_times, "event_times")
    _validate_non_empty(event_observed, "event_observed")
    _validate_non_empty(time_points, "time_points")
    _validate_positive_array(event_times, "event_times")
    _validate_no_nan_inf(event_times, "event_times")
    _validate_no_nan_inf(time_points, "time_points")
    _validate_no_nan_inf(predicted_surv_curves, "predicted_surv_curves")
    _validate_binary_array(event_observed, "event_observed")

    # Validate predicted_surv_curves shape
    if predicted_surv_curves.ndim != 2:
        raise ValueError("predicted_surv_curves must be 2D array")

    n_samples = predicted_surv_curves.shape[0]
    n_time_points = predicted_surv_curves.shape[1]

    # Check dimension matches
    if len(event_times) != n_samples:
        raise ValueError(
            f"event_times length must match predicted_surv_curves first dimension: "
            f"event_times ({len(event_times)}) vs predicted_surv_curves ({n_samples})"
        )

    if len(event_observed) != n_samples:
        raise ValueError(
            f"event_observed length must match predicted_surv_curves first dimension: "
            f"event_observed ({len(event_observed)}) vs predicted_surv_curves ({n_samples})"
        )

    if len(time_points) != n_time_points:
        raise ValueError(
            f"time_points length must match predicted_surv_curves second dimension: "
            f"time_points ({len(time_points)}) vs predicted_surv_curves ({n_time_points})"
        )

    # Validate survival probabilities
    _validate_probability_array(predicted_surv_curves, "predicted_surv_curves")
    _validate_monotonic_non_increasing(
        predicted_surv_curves, "predicted_surv_curves", axis=1
    )

    # Validate time_points are within range of event_times
    min_event_time = np.min(event_times)
    max_event_time = np.max(event_times)
    if np.any(time_points < min_event_time) or np.any(time_points > max_event_time):
        raise ValueError(
            f"time_points must be within range of event_times: "
            f"time_points [{np.min(time_points):.2f}, {np.max(time_points):.2f}] "
            f"vs event_times [{min_event_time:.2f}, {max_event_time:.2f}]"
        )

    # Handle edge case: all censored
    if np.sum(event_observed) == 0:
        warnings.warn(
            "All observations are censored. Cannot compute Integrated Brier Score. "
            "Returning np.nan.",
            UserWarning,
        )
        return np.nan

    # Warn if heavy censoring
    censoring_rate = 1.0 - np.mean(event_observed)
    if censoring_rate > 0.8:
        warnings.warn(
            f"Heavy censoring detected ({censoring_rate*100:.1f}%). "
            f"IPCW weights may be unstable.",
            UserWarning,
        )

    # Fit Kaplan-Meier estimator for censoring distribution
    # If training data provided, fit on that for better IPCW estimates
    kmf = KaplanMeierFitter()
    if censoring_times is not None and censoring_events is not None:
        kmf.fit(
            np.asarray(censoring_times, dtype=np.float64),
            event_observed=(1 - np.asarray(censoring_events, dtype=np.int32)),
        )
    else:
        kmf.fit(event_times, event_observed=(1 - event_observed))

    # Compute Brier score at each time point
    brier_scores = np.zeros(n_time_points, dtype=np.float64)

    for t_idx, t in enumerate(time_points):
        # Get predicted survival at time t for all instances
        pred_surv_t = predicted_surv_curves[:, t_idx]

        # Compute IPCW weights
        # For each instance, we need G(min(T_i, t)) where G is censoring survival
        weights = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            T_i = event_times[i]
            delta_i = event_observed[i]

            # Case 1: Event occurred before or at time t
            if delta_i == 1 and T_i <= t:
                # Weight = 1 / G(T_i)
                # Get censoring survival probability at T_i
                G_Ti = kmf.predict(T_i)
                if G_Ti > 0:
                    weights[i] = 1.0 / G_Ti
                else:
                    weights[i] = 0.0

            # Case 2: Still at risk at time t (T_i > t)
            elif T_i > t:
                # Weight = 1 / G(t)
                G_t = kmf.predict(t)
                if G_t > 0:
                    weights[i] = 1.0 / G_t
                else:
                    weights[i] = 0.0

            # Case 3: Censored before time t (delta_i == 0 and T_i <= t)
            # Weight = 0 (does not contribute)

        # Compute weighted Brier score at time t
        # BS(t) = sum_i weight_i * (I(T_i > t) - S(t|X_i))^2 / sum_i weight_i

        # True survival indicator at time t
        true_surv_t = (event_times > t).astype(np.float64)

        # Squared error
        squared_errors = (true_surv_t - pred_surv_t) ** 2

        # Weighted sum
        weighted_sum = np.sum(weights * squared_errors)
        weight_sum = np.sum(weights)

        if weight_sum > 0:
            brier_scores[t_idx] = weighted_sum / weight_sum
        else:
            brier_scores[t_idx] = 0.0

    # Integrate using trapezoidal rule
    if len(time_points) == 1:
        ibs = brier_scores[0]
    else:
        # Trapezoidal integration
        ibs = np.trapezoid(brier_scores, time_points)
        # Normalize by time range
        time_range = time_points[-1] - time_points[0]
        if time_range > 0:
            ibs = ibs / time_range

    return float(ibs)


def interpret_integrated_brier_score(
    ibs: float, baseline_ibs: Optional[float] = None
) -> Dict[str, Any]:
    """
    Interpret Integrated Brier Score with quality classification.

    Parameters
    ----------
    ibs : float
        Integrated Brier Score value
        Must be in range [0, 1]
    baseline_ibs : float, optional
        Baseline IBS for comparison (e.g., from Kaplan-Meier model)
        Must be in range [0, 1] if provided

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'ibs': float, the input IBS value
        - 'classification': str, one of {'excellent', 'good', 'acceptable', 'poor'}
        - 'quality_assessment': str, human-readable quality description
        - 'baseline_comparison': dict or None, comparison metrics if baseline provided
          - 'baseline_ibs': float
          - 'improvement': float, relative improvement (baseline - ibs) / baseline
          - 'better_than_baseline': bool
          - 'interpretation': str
        - 'warning': str or None, warning if not better than baseline

    Raises
    ------
    ValueError
        If ibs is outside range [0, 1]
        If baseline_ibs is outside range [0, 1]

    Notes
    -----
    Quality classification thresholds:
    - Excellent: IBS < 0.15 (very accurate predictions)
    - Good: 0.15 <= IBS < 0.25 (accurate predictions)
    - Acceptable: 0.25 <= IBS < 0.35 (moderate accuracy)
    - Poor: IBS >= 0.35 (low accuracy, consider model revision)

    For survival analysis:
    - IBS ≈ 0.25 indicates random predictions
    - IBS < 0.20 is generally considered good for clinical applications
    - Compare against Kaplan-Meier baseline for context
    """
    # Validate inputs
    if not (0.0 <= ibs <= 1.0):
        raise ValueError(f"ibs must be in range [0, 1], got {ibs}")

    if baseline_ibs is not None:
        if not (0.0 <= baseline_ibs <= 1.0):
            raise ValueError(
                f"baseline_ibs must be in range [0, 1], got {baseline_ibs}"
            )

    # Classify quality
    if ibs < 0.15:
        classification = "excellent"
        quality_assessment = (
            f"Excellent prediction quality (IBS = {ibs:.3f}). "
            f"The model provides very accurate survival predictions across time points."
        )
    elif ibs < 0.25:
        classification = "good"
        quality_assessment = (
            f"Good prediction quality (IBS = {ibs:.3f}). "
            f"The model provides accurate survival predictions suitable for most applications."
        )
    elif ibs < 0.35:
        classification = "acceptable"
        quality_assessment = (
            f"Acceptable prediction quality (IBS = {ibs:.3f}). "
            f"The model provides moderate accuracy. Consider refinement for critical applications."
        )
    else:
        classification = "poor"
        quality_assessment = (
            f"Poor prediction quality (IBS = {ibs:.3f}). "
            f"The model shows low accuracy. Model revision or alternative approaches recommended."
        )

    # Baseline comparison
    baseline_comparison = None
    warning = None

    if baseline_ibs is not None:
        improvement = (baseline_ibs - ibs) / baseline_ibs if baseline_ibs > 0 else 0.0
        better_than_baseline = ibs < baseline_ibs

        if better_than_baseline:
            comparison_interpretation = (
                f"Model outperforms baseline by {improvement*100:.1f}% "
                f"(baseline IBS = {baseline_ibs:.3f}, model IBS = {ibs:.3f}). "
                f"The predictive model provides meaningful improvement over non-parametric baseline."
            )
        else:
            comparison_interpretation = (
                f"Model does not outperform baseline "
                f"(baseline IBS = {baseline_ibs:.3f}, model IBS = {ibs:.3f}). "
                f"Consider using simpler baseline model or revising feature engineering."
            )
            warning = (
                "WARNING: Model performance is not better than baseline. "
                "The added complexity may not be justified."
            )

        baseline_comparison = {
            "baseline_ibs": baseline_ibs,
            "improvement": improvement,
            "better_than_baseline": better_than_baseline,
            "interpretation": comparison_interpretation,
        }

    return {
        "ibs": ibs,
        "classification": classification,
        "quality_assessment": quality_assessment,
        "baseline_comparison": baseline_comparison,
        "warning": warning,
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_calibration_curve(
    sorted_cdf: np.ndarray,
    empirical_cdf: np.ndarray,
    bias_classification: Optional[str] = None,
    bias_value: Optional[float] = None,
    sorted_cdf_events: Optional[np.ndarray] = None,
    empirical_cdf_events: Optional[np.ndarray] = None,
    show: str = "uncensored",
    ax=None,
):
    """Plot a calibration curve (predicted CDF vs empirical CDF).

    Parameters
    ----------
    sorted_cdf : np.ndarray
        Predicted CDF values, sorted ascending (all observations).
    empirical_cdf : np.ndarray
        Corresponding empirical CDF values (all observations).
    bias_classification : str, optional
        Calibration classification label (e.g. 'well-calibrated').
    bias_value : float, optional
        Calibration bias value to show in annotation.
    sorted_cdf_events : np.ndarray, optional
        Predicted CDF values for uncensored observations only.
    empirical_cdf_events : np.ndarray, optional
        Empirical CDF values for uncensored observations only.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if show in ("all", "both"):
        ax.plot(sorted_cdf, empirical_cdf, "b-", lw=2, label="All observations")
        ax.fill_between(sorted_cdf, sorted_cdf, empirical_cdf, alpha=0.1, color="blue")
        if sorted_cdf_events is not None and empirical_cdf_events is not None:
            ax.plot(
                sorted_cdf_events,
                empirical_cdf_events,
                "r-",
                lw=2,
                label="Uncensored only",
            )
    elif show == "uncensored":
        cdf = sorted_cdf_events if sorted_cdf_events is not None else sorted_cdf
        ecdf = (
            empirical_cdf_events if empirical_cdf_events is not None else empirical_cdf
        )
        label = (
            "Uncensored only" if sorted_cdf_events is not None else "All observations"
        )
        ax.plot(cdf, ecdf, "b-", lw=2, label=label)
        ax.fill_between(cdf, cdf, ecdf, alpha=0.15, color="blue")
    else:  # "all_only"
        ax.plot(sorted_cdf, empirical_cdf, "b-", lw=2, label="All observations")
        ax.fill_between(sorted_cdf, sorted_cdf, empirical_cdf, alpha=0.15, color="blue")
    ax.set_xlabel("Predicted CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Calibration Curve")
    ax.legend()

    if bias_classification is not None or bias_value is not None:
        parts = []
        if bias_classification:
            parts.append(bias_classification)
        if bias_value is not None:
            parts.append(f"bias={bias_value:.3f}")
        ax.annotate("\n".join(parts), xy=(0.05, 0.85), xycoords="axes fraction")

    return ax


def plot_ibs(
    ibs: float,
    ibs_classification: Optional[str] = None,
    ax=None,
):
    """Plot a horizontal bar showing the Integrated Brier Score.

    Parameters
    ----------
    ibs : float
        Integrated Brier Score value.
    ibs_classification : str, optional
        Classification label (e.g. 'excellent', 'good', 'acceptable', 'poor').
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    colors = {
        "excellent": "green",
        "good": "yellowgreen",
        "acceptable": "orange",
        "poor": "red",
    }
    color = colors.get(ibs_classification or "", "steelblue")

    ax.barh(["IBS"], [ibs], color=color)
    ax.axvline(0.25, color="gray", linestyle="--", label="Random baseline (0.25)")
    ax.set_xlim(0, max(0.4, ibs * 1.2))
    ax.set_title("Integrated Brier Score")
    ax.legend()

    label = f"{ibs:.3f}"
    if ibs_classification:
        label += f" — {ibs_classification}"
    ax.annotate(label, xy=(ibs + 0.01, 0), va="center")

    return ax


def plot_calibration(
    sorted_cdf: Optional[np.ndarray] = None,
    empirical_cdf: Optional[np.ndarray] = None,
    sorted_cdf_events: Optional[np.ndarray] = None,
    empirical_cdf_events: Optional[np.ndarray] = None,
    bias_classification: Optional[str] = None,
    bias_value: Optional[float] = None,
    ibs: Optional[float] = None,
    ibs_classification: Optional[str] = None,
    show: str = "uncensored",
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot calibration curve and IBS side by side.

    Convenience wrapper that combines plot_calibration_curve() and plot_ibs()
    into a single two-panel figure. Either panel is shown as 'not available'
    if its data is absent (e.g. SageMaker backend without curve artifact).

    Parameters
    ----------
    sorted_cdf : np.ndarray, optional
        Predicted CDF values for the calibration curve panel.
    empirical_cdf : np.ndarray, optional
        Empirical CDF values for the calibration curve panel.
    bias_classification : str, optional
    bias_value : float, optional
    ibs : float, optional
        IBS value for the IBS panel.
    ibs_classification : str, optional
    figsize : tuple, optional

    Returns
    -------
    fig, axes : matplotlib Figure and array of two Axes
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Model Calibration", fontsize=14)

    ax1 = axes[0]
    if sorted_cdf is not None and empirical_cdf is not None:
        plot_calibration_curve(
            sorted_cdf,
            empirical_cdf,
            bias_classification,
            bias_value,
            sorted_cdf_events=sorted_cdf_events,
            empirical_cdf_events=empirical_cdf_events,
            show=show,
            ax=ax1,
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
    if ibs is not None:
        plot_ibs(ibs, ibs_classification, ax=ax2)
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
