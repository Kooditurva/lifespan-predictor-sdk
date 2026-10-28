"""
Tests of every function in the metrics module.

They need nothing beyond what the metrics module needs, so they run in the SDK
repository too, against the same module: a sync change that altered a metric
fails there as well as here. Comparisons with pycox or lifelines, and tests that
train a model, live in other files.
"""

import numpy as np
import pandas as pd
import pytest

from lifespan_predictor_sdk import metrics

# ============================================================================
# Unit Tests for calibration_curve_data (existing function)
# ============================================================================


@pytest.mark.local
def test_calibration_curve_data_basic():
    """Test basic functionality of calibration_curve_data."""
    event_times = np.array([10.0, 20.0, 30.0])
    alpha = np.array([15.0, 25.0, 35.0])
    beta = np.array([2.0, 2.0, 2.0])

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data(event_times, alpha, beta)

    # Check output shapes
    assert len(sorted_cdf) == 3
    assert len(empirical_cdf) == 3

    # Check empirical CDF is correct
    np.testing.assert_array_almost_equal(empirical_cdf, [1 / 3, 2 / 3, 1.0])

    # Check sorted_cdf is sorted
    assert np.all(np.diff(sorted_cdf) >= 0)

    # Check values are in [0, 1]
    assert np.all((sorted_cdf >= 0) & (sorted_cdf <= 1))


@pytest.mark.local
def test_calibration_curve_data_validation_negative_event_times():
    """Test that negative event times raise ValueError."""
    event_times = np.array([-10.0, 20.0, 30.0])
    alpha = np.array([15.0, 25.0, 35.0])
    beta = np.array([2.0, 2.0, 2.0])

    with pytest.raises(
        ValueError, match="event_times must contain only positive values"
    ):
        metrics.calibration_curve_data(event_times, alpha, beta)


@pytest.mark.local
def test_calibration_curve_data_validation_non_positive_alpha():
    """Test that non-positive alpha raises ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    alpha = np.array([0.0, 25.0, 35.0])
    beta = np.array([2.0, 2.0, 2.0])

    with pytest.raises(ValueError, match="alpha must contain only positive values"):
        metrics.calibration_curve_data(event_times, alpha, beta)


@pytest.mark.local
def test_calibration_curve_data_validation_non_positive_beta():
    """Test that non-positive beta raises ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    alpha = np.array([15.0, 25.0, 35.0])
    beta = np.array([2.0, -1.0, 2.0])

    with pytest.raises(ValueError, match="beta must contain only positive values"):
        metrics.calibration_curve_data(event_times, alpha, beta)


@pytest.mark.local
def test_calibration_curve_data_validation_mismatched_lengths():
    """Test that mismatched array lengths raise ValueError."""
    event_times = np.array([10.0, 20.0])
    alpha = np.array([15.0, 25.0, 35.0])
    beta = np.array([2.0, 2.0, 2.0])

    with pytest.raises(ValueError, match="Input arrays must have matching lengths"):
        metrics.calibration_curve_data(event_times, alpha, beta)


@pytest.mark.local
def test_calibration_curve_data_validation_empty_arrays():
    """Test that empty arrays raise ValueError."""
    event_times = np.array([])
    alpha = np.array([])
    beta = np.array([])

    with pytest.raises(ValueError, match="event_times must not be empty"):
        metrics.calibration_curve_data(event_times, alpha, beta)


@pytest.mark.local
def test_calibration_curve_data_single_instance():
    """Test with single instance (n=1)."""
    event_times = np.array([10.0])
    alpha = np.array([15.0])
    beta = np.array([2.0])

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data(event_times, alpha, beta)

    assert len(sorted_cdf) == 1
    assert len(empirical_cdf) == 1
    assert empirical_cdf[0] == 1.0


@pytest.mark.local
def test_calibration_curve_data_array_like_inputs():
    """Test that function accepts lists and tuples."""
    event_times = [10.0, 20.0, 30.0]
    alpha = (15.0, 25.0, 35.0)
    beta = [2.0, 2.0, 2.0]

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data(event_times, alpha, beta)

    assert len(sorted_cdf) == 3
    assert len(empirical_cdf) == 3


# ============================================================================
# Unit Tests for calibration_bias (Task 3)
# ============================================================================


@pytest.mark.local
def test_calibration_bias_perfect_calibration():
    """Test bias for perfectly calibrated model (bias ≈ 0)."""
    # Perfect calibration: predicted_cdf = empirical_cdf
    predicted_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    empirical_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, float)
    np.testing.assert_almost_equal(bias, 0.0, decimal=10)


@pytest.mark.local
def test_calibration_bias_under_prediction():
    """Test bias for under-prediction (bias > 0)."""
    # Under-prediction: predicted_cdf > empirical_cdf (conservative)
    predicted_cdf = np.array([0.3, 0.5, 0.7, 0.9, 1.0])
    empirical_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, float)
    assert bias > 0
    expected_bias = np.mean([0.1, 0.1, 0.1, 0.1, 0.0])
    np.testing.assert_almost_equal(bias, expected_bias)


@pytest.mark.local
def test_calibration_bias_over_prediction():
    """Test bias for over-prediction (bias < 0)."""
    # Over-prediction: predicted_cdf < empirical_cdf (dangerous)
    predicted_cdf = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    empirical_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, float)
    assert bias < 0
    expected_bias = np.mean([-0.1, -0.1, -0.1, -0.1, -0.1])
    np.testing.assert_almost_equal(bias, expected_bias)


@pytest.mark.local
def test_calibration_bias_identical_predictions():
    """Test bias with identical predictions (edge case)."""
    # All predictions are the same
    predicted_cdf = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    empirical_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, float)
    # Bias should be 0.5 - mean(empirical_cdf) = 0.5 - 0.6 = -0.1
    expected_bias = 0.5 - 0.6
    np.testing.assert_almost_equal(bias, expected_bias)


@pytest.mark.local
def test_calibration_bias_single_instance():
    """Test bias with single instance (n=1)."""
    predicted_cdf = np.array([0.7])
    empirical_cdf = np.array([1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, float)
    np.testing.assert_almost_equal(bias, -0.3)


@pytest.mark.local
def test_calibration_bias_validation_mismatched_lengths():
    """Test that mismatched array lengths raise ValueError."""
    predicted_cdf = np.array([0.2, 0.4, 0.6])
    empirical_cdf = np.array([0.2, 0.4])

    with pytest.raises(ValueError, match="Input arrays must have matching lengths"):
        metrics.calibration_bias(predicted_cdf, empirical_cdf)


@pytest.mark.local
def test_calibration_bias_validation_empty_arrays():
    """Test that empty arrays raise ValueError."""
    predicted_cdf = np.array([])
    empirical_cdf = np.array([])

    with pytest.raises(ValueError, match="must not be empty"):
        metrics.calibration_bias(predicted_cdf, empirical_cdf)


@pytest.mark.local
def test_calibration_bias_return_type():
    """Test that bias is returned as float in range [-1, 1]."""
    predicted_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    empirical_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    bias = metrics.calibration_bias(predicted_cdf, empirical_cdf)

    assert isinstance(bias, (float, np.floating))
    assert -1.0 <= bias <= 1.0


# ============================================================================
# Unit Tests for interpret_calibration_bias (Task 4)
# ============================================================================


@pytest.mark.local
def test_interpret_calibration_bias_well_calibrated():
    """Test well-calibrated classification."""
    mean_cdf = 0.50  # Perfect calibration

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "well-calibrated"
    assert result["severity"] == "none"
    assert result["mean_cdf"] == 0.50
    assert result["deviation"] == 0.0
    assert "well-calibrated" in result["interpretation"].lower()
    assert result["warning"] is None


@pytest.mark.local
def test_interpret_calibration_bias_under_prediction_mild():
    """Test under-prediction with mild severity."""
    mean_cdf = 0.58  # Deviation = 0.08 (mild)

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "under-prediction"
    assert result["severity"] == "mild"
    assert result["mean_cdf"] == 0.58
    np.testing.assert_almost_equal(result["deviation"], 0.08)
    assert "under-prediction" in result["interpretation"].lower()
    assert (
        "conservative" in result["interpretation"].lower()
        or "wasteful" in result["interpretation"].lower()
    )


@pytest.mark.local
def test_interpret_calibration_bias_under_prediction_moderate():
    """Test under-prediction with moderate severity."""
    mean_cdf = 0.65  # Deviation = 0.15 (moderate)

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "under-prediction"
    assert result["severity"] == "moderate"
    np.testing.assert_almost_equal(result["deviation"], 0.15)


@pytest.mark.local
def test_interpret_calibration_bias_under_prediction_severe():
    """Test under-prediction with severe severity."""
    mean_cdf = 0.75  # Deviation = 0.25 (severe)

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "under-prediction"
    assert result["severity"] == "severe"
    assert result["deviation"] == 0.25


@pytest.mark.local
def test_interpret_calibration_bias_over_prediction_with_warning():
    """Test over-prediction classification with safety warning."""
    mean_cdf = 0.40  # Deviation = 0.10 (over-prediction)

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "over-prediction"
    assert result["severity"] == "mild"
    assert result["mean_cdf"] == 0.40
    np.testing.assert_almost_equal(result["deviation"], 0.10)
    assert "over-prediction" in result["interpretation"].lower()
    assert result["warning"] is not None
    assert (
        "safety" in result["warning"].lower()
        or "dangerous" in result["warning"].lower()
    )


@pytest.mark.local
def test_interpret_calibration_bias_over_prediction_severe():
    """Test over-prediction with severe severity."""
    mean_cdf = 0.25  # Deviation = 0.25 (severe)

    result = metrics.interpret_calibration_bias(mean_cdf)

    assert result["classification"] == "over-prediction"
    assert result["severity"] == "severe"
    assert result["warning"] is not None


@pytest.mark.local
def test_interpret_calibration_bias_custom_threshold():
    """Test with custom threshold value."""
    mean_cdf = 0.52
    threshold = 0.03

    result = metrics.interpret_calibration_bias(mean_cdf, threshold=threshold)

    # With threshold=0.03, deviation=0.02 should be well-calibrated
    assert result["classification"] == "well-calibrated"
    assert result["severity"] == "none"

    # But with default threshold=0.05, it would still be well-calibrated
    result_default = metrics.interpret_calibration_bias(mean_cdf)
    assert result_default["classification"] == "well-calibrated"


@pytest.mark.local
def test_interpret_calibration_bias_boundary_at_threshold():
    """Test boundary conditions at threshold edges."""
    threshold = 0.05

    # Just within threshold (deviation < 0.05, should be well-calibrated)
    result = metrics.interpret_calibration_bias(0.54, threshold=threshold)
    assert result["classification"] == "well-calibrated"

    # Just above threshold (should be under-prediction)
    result = metrics.interpret_calibration_bias(0.56, threshold=threshold)
    assert result["classification"] == "under-prediction"

    # Just below threshold (should be over-prediction)
    result = metrics.interpret_calibration_bias(0.44, threshold=threshold)
    assert result["classification"] == "over-prediction"


@pytest.mark.local
def test_interpret_calibration_bias_structure():
    """Test that result contains all required keys."""
    mean_cdf = 0.50

    result = metrics.interpret_calibration_bias(mean_cdf)

    required_keys = [
        "classification",
        "severity",
        "mean_cdf",
        "deviation",
        "interpretation",
        "warning",
    ]
    for key in required_keys:
        assert key in result


@pytest.mark.local
def test_interpret_calibration_bias_validation_invalid_mean_cdf():
    """Test that invalid mean_cdf raises ValueError."""
    with pytest.raises(ValueError, match="mean_cdf must be in range"):
        metrics.interpret_calibration_bias(1.5)

    with pytest.raises(ValueError, match="mean_cdf must be in range"):
        metrics.interpret_calibration_bias(-0.1)


@pytest.mark.local
def test_interpret_calibration_bias_validation_invalid_threshold():
    """Test that invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match="threshold must be"):
        metrics.interpret_calibration_bias(0.5, threshold=-0.01)

    with pytest.raises(ValueError, match="threshold must be"):
        metrics.interpret_calibration_bias(0.5, threshold=0.6)


# ============================================================================
# Unit Tests for calibration_curve_data_from_survival
# ============================================================================


@pytest.mark.local
def test_calibration_curve_data_from_survival_basic():
    """Test basic functionality with 2D survival curves."""
    event_times = np.array([10.0, 20.0, 30.0])
    time_grid = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    # Create simple decreasing survival curves
    survival_curves = np.array(
        [
            [1.0, 0.8, 0.6, 0.4, 0.2],
            [1.0, 0.9, 0.7, 0.5, 0.3],
            [1.0, 0.85, 0.65, 0.45, 0.25],
        ]
    )

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    assert len(sorted_cdf) == 3
    assert len(empirical_cdf) == 3
    np.testing.assert_array_almost_equal(empirical_cdf, [1 / 3, 2 / 3, 1.0])
    assert np.all(np.diff(sorted_cdf) >= 0)
    assert np.all((sorted_cdf >= 0) & (sorted_cdf <= 1))


@pytest.mark.local
def test_calibration_curve_data_from_survival_1d_input():
    """Test with 1D survival curve (single instance)."""
    event_times = np.array([15.0])
    time_grid = np.array([0.0, 10.0, 20.0, 30.0])
    survival_curves = np.array([1.0, 0.8, 0.6, 0.4])

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    assert len(sorted_cdf) == 1
    assert len(empirical_cdf) == 1
    assert empirical_cdf[0] == 1.0


@pytest.mark.local
def test_calibration_curve_data_from_survival_interpolation_exact():
    """Test interpolation at exact grid points."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0, 30.0])
    survival_curves = np.array([[1.0, 0.8, 0.6, 0.4], [1.0, 0.7, 0.5, 0.3]])

    sorted_cdf, empirical_cdf = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    # At exact grid points, should get exact survival values
    # For instance 0 at t=10: S(10)=0.8, so CDF=0.2
    # For instance 1 at t=20: S(20)=0.5, so CDF=0.5
    # After sorting: [0.2, 0.5]
    expected_sorted = np.sort([0.2, 0.5])
    np.testing.assert_array_almost_equal(sorted_cdf, expected_sorted)


@pytest.mark.local
def test_calibration_curve_data_from_survival_interpolation_at_grid_point():
    """Test that interpolation at exact grid points returns exact values."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0, 30.0])
    survival_curves = np.array([[1.0, 0.8, 0.6, 0.4], [1.0, 0.9, 0.7, 0.5]])

    sorted_cdf, _ = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    # At event_time=10, S=0.8, so CDF=0.2
    # At event_time=20, S=0.7, so CDF=0.3
    expected_cdfs = np.sort([0.2, 0.3])
    np.testing.assert_array_almost_equal(sorted_cdf, expected_cdfs)


@pytest.mark.local
def test_calibration_curve_data_from_survival_linear_interpolation():
    """Test linear interpolation between grid points."""
    event_times = np.array([15.0])
    time_grid = np.array([10.0, 20.0])
    survival_curves = np.array([[0.8, 0.6]])

    sorted_cdf, _ = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid, interpolation="linear"
    )

    # Linear interpolation: S(15) = 0.8 + (15-10)/(20-10) * (0.6-0.8) = 0.7
    # CDF = 1 - 0.7 = 0.3
    np.testing.assert_almost_equal(sorted_cdf[0], 0.3)


@pytest.mark.local
def test_calibration_curve_data_from_survival_step_interpolation():
    """Test step function interpolation."""
    event_times = np.array([15.0])
    time_grid = np.array([10.0, 20.0])
    survival_curves = np.array([[0.8, 0.6]])

    sorted_cdf, _ = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid, interpolation="step"
    )

    # Step function: S(15) = 0.8 (value at previous grid point)
    # CDF = 1 - 0.8 = 0.2
    np.testing.assert_almost_equal(sorted_cdf[0], 0.2)


@pytest.mark.local
def test_calibration_curve_data_from_survival_extrapolation_below():
    """Test extrapolation below time grid."""
    event_times = np.array([5.0])
    time_grid = np.array([10.0, 20.0, 30.0])
    survival_curves = np.array([[0.8, 0.6, 0.4]])

    sorted_cdf, _ = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    # Below grid: use first survival probability S=0.8
    # CDF = 1 - 0.8 = 0.2
    np.testing.assert_almost_equal(sorted_cdf[0], 0.2)


@pytest.mark.local
def test_calibration_curve_data_from_survival_extrapolation_above():
    """Test extrapolation above time grid."""
    event_times = np.array([40.0])
    time_grid = np.array([10.0, 20.0, 30.0])
    survival_curves = np.array([[0.8, 0.6, 0.4]])

    sorted_cdf, _ = metrics.calibration_curve_data_from_survival(
        event_times, survival_curves, time_grid
    )

    # Above grid: use last survival probability S=0.4
    # CDF = 1 - 0.4 = 0.6
    np.testing.assert_almost_equal(sorted_cdf[0], 0.6)


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_negative_event_times():
    """Test that negative event times raise ValueError."""
    event_times = np.array([-10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, 0.8, 0.6], [1.0, 0.9, 0.7]])

    with pytest.raises(
        ValueError, match="event_times must contain only non-negative values"
    ):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_invalid_probabilities():
    """Test that survival curves outside [0,1] raise ValueError."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, 0.8, 1.5], [1.0, 0.9, 0.7]])

    with pytest.raises(
        ValueError, match="survival_curves must contain values in range"
    ):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_non_monotonic():
    """Test that non-monotonic survival curves raise ValueError."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, 0.6, 0.8], [1.0, 0.9, 0.7]])  # Increases at end

    with pytest.raises(
        ValueError, match="survival_curves must be monotonically non-increasing"
    ):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_time_grid_not_increasing():
    """Test that non-increasing time grid raises ValueError."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 20.0, 10.0])  # Not increasing
    survival_curves = np.array([[1.0, 0.8, 0.6], [1.0, 0.9, 0.7]])

    with pytest.raises(ValueError, match="time_grid must be monotonically increasing"):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, 0.8], [1.0, 0.9]])  # Wrong second dimension

    with pytest.raises(
        ValueError, match="time_grid length must match survival_curves second dimension"
    ):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_nan_values():
    """Test that NaN values raise ValueError."""
    event_times = np.array([10.0, 20.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, np.nan, 0.6], [1.0, 0.9, 0.7]])

    with pytest.raises(
        ValueError, match="survival_curves must not contain NaN or infinite values"
    ):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid
        )


@pytest.mark.local
def test_calibration_curve_data_from_survival_validation_invalid_interpolation():
    """Test that invalid interpolation method raises ValueError."""
    event_times = np.array([10.0])
    time_grid = np.array([0.0, 10.0, 20.0])
    survival_curves = np.array([[1.0, 0.8, 0.6]])

    with pytest.raises(ValueError, match="interpolation must be 'linear' or 'step'"):
        metrics.calibration_curve_data_from_survival(
            event_times, survival_curves, time_grid, interpolation="invalid"
        )


# ============================================================================
# Unit Tests for integrated_brier_score (Task 6)
# ============================================================================


@pytest.mark.local
def test_integrated_brier_score_perfect_predictions():
    """Test IBS with perfect predictions (IBS ≈ 0)."""
    # Create perfect predictions where predicted survival matches true survival
    n_samples = 50
    n_time_points = 10

    # Generate event times
    np.random.seed(42)
    event_times = np.random.uniform(10, 100, n_samples)
    event_observed = np.ones(n_samples, dtype=int)  # All events observed

    # Time points for evaluation - ensure they're within event_times range
    min_event_time = np.min(event_times)
    max_event_time = np.max(event_times)
    time_points = np.linspace(min_event_time + 1, max_event_time - 1, n_time_points)

    # Create perfect survival curves: S(t) = I(T > t)
    predicted_surv_curves = np.zeros((n_samples, n_time_points))
    for i in range(n_samples):
        for j, t in enumerate(time_points):
            predicted_surv_curves[i, j] = 1.0 if event_times[i] > t else 0.0

    ibs = metrics.integrated_brier_score(
        event_times, event_observed, predicted_surv_curves, time_points
    )

    # Perfect predictions should have IBS ≈ 0
    assert isinstance(ibs, float)
    assert 0.0 <= ibs <= 1.0
    assert ibs < 0.1  # Should be very small


@pytest.mark.local
def test_integrated_brier_score_with_censoring():
    """Test IBS with censored data."""
    n_samples = 50
    n_time_points = 10

    np.random.seed(42)
    event_times = np.random.uniform(10, 100, n_samples)
    # 30% censoring
    event_observed = np.random.binomial(1, 0.7, n_samples)

    time_points = np.linspace(15, 95, n_time_points)

    # Create reasonable survival curves (exponential decay)
    predicted_surv_curves = np.zeros((n_samples, n_time_points))
    for i in range(n_samples):
        for j, t in enumerate(time_points):
            # Exponential survival: S(t) = exp(-lambda * t)
            predicted_surv_curves[i, j] = np.exp(-0.02 * t)

    ibs = metrics.integrated_brier_score(
        event_times, event_observed, predicted_surv_curves, time_points
    )

    assert isinstance(ibs, float)
    assert 0.0 <= ibs <= 1.0


@pytest.mark.local
def test_integrated_brier_score_all_censored():
    """Test IBS when all observations are censored (returns np.nan with warning)."""
    n_samples = 20
    n_time_points = 5

    event_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 4)
    event_observed = np.zeros(n_samples, dtype=int)  # All censored

    time_points = np.linspace(10, 50, n_time_points)
    predicted_surv_curves = np.random.uniform(0.3, 0.9, (n_samples, n_time_points))
    # Make monotonically non-increasing
    predicted_surv_curves = np.sort(predicted_surv_curves, axis=1)[:, ::-1]

    with pytest.warns(UserWarning, match="All observations are censored"):
        ibs = metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )

    assert np.isnan(ibs)


@pytest.mark.local
def test_integrated_brier_score_heavy_censoring_warning():
    """Test that heavy censoring (>80%) triggers warning."""
    n_samples = 50
    n_time_points = 10

    np.random.seed(42)
    event_times = np.random.uniform(10, 100, n_samples)
    # 85% censoring
    event_observed = np.random.binomial(1, 0.15, n_samples)
    # Ensure at least one event
    event_observed[0] = 1

    time_points = np.linspace(15, 95, n_time_points)
    predicted_surv_curves = np.random.uniform(0.3, 0.9, (n_samples, n_time_points))
    predicted_surv_curves = np.sort(predicted_surv_curves, axis=1)[:, ::-1]

    with pytest.warns(UserWarning, match="Heavy censoring detected"):
        ibs = metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )

    assert isinstance(ibs, float)
    assert 0.0 <= ibs <= 1.0


@pytest.mark.local
def test_integrated_brier_score_single_time_point():
    """Test IBS with single time point."""
    n_samples = 30

    np.random.seed(42)
    event_times = np.random.uniform(10, 100, n_samples)
    event_observed = np.ones(n_samples, dtype=int)

    time_points = np.array([50.0])
    predicted_surv_curves = np.random.uniform(0.3, 0.9, (n_samples, 1))

    ibs = metrics.integrated_brier_score(
        event_times, event_observed, predicted_surv_curves, time_points
    )

    assert isinstance(ibs, float)
    assert 0.0 <= ibs <= 1.0


@pytest.mark.local
def test_integrated_brier_score_validation_negative_times():
    """Test that negative event times raise ValueError."""
    event_times = np.array([-10.0, 20.0, 30.0])
    event_observed = np.array([1, 1, 1])
    time_points = np.array([10.0, 20.0])
    predicted_surv_curves = np.array([[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]])

    with pytest.raises(
        ValueError, match="event_times must contain only positive values"
    ):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


@pytest.mark.local
def test_integrated_brier_score_validation_non_binary_events():
    """Test that non-binary event_observed raises ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    event_observed = np.array([1, 2, 0])  # 2 is invalid
    time_points = np.array([10.0, 20.0])
    predicted_surv_curves = np.array([[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]])

    with pytest.raises(ValueError, match="event_observed must be binary"):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


@pytest.mark.local
def test_integrated_brier_score_validation_invalid_probabilities():
    """Test that survival probabilities outside [0,1] raise ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    event_observed = np.array([1, 1, 1])
    time_points = np.array([10.0, 20.0])
    predicted_surv_curves = np.array(
        [[0.8, 1.5], [0.7, 0.5], [0.9, 0.7]]
    )  # 1.5 invalid

    with pytest.raises(
        ValueError, match="predicted_surv_curves must contain values in range"
    ):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


@pytest.mark.local
def test_integrated_brier_score_validation_non_monotonic():
    """Test that non-monotonic survival curves raise ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    event_observed = np.array([1, 1, 1])
    time_points = np.array([10.0, 20.0])
    predicted_surv_curves = np.array(
        [[0.6, 0.8], [0.7, 0.5], [0.9, 0.7]]
    )  # First row increases

    with pytest.raises(
        ValueError, match="predicted_surv_curves must be monotonically non-increasing"
    ):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


@pytest.mark.local
def test_integrated_brier_score_validation_time_points_out_of_range():
    """Test that time_points outside event_times range raise ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    event_observed = np.array([1, 1, 1])
    time_points = np.array([5.0, 40.0])  # Outside [10, 30]
    predicted_surv_curves = np.array([[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]])

    with pytest.raises(
        ValueError, match="time_points must be within range of event_times"
    ):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


@pytest.mark.local
def test_integrated_brier_score_validation_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    event_times = np.array([10.0, 20.0, 30.0])
    event_observed = np.array([1, 1, 1])
    time_points = np.array([10.0, 20.0, 25.0])  # 3 time points
    predicted_surv_curves = np.array(
        [[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]]
    )  # Only 2 time points

    with pytest.raises(
        ValueError,
        match="time_points length must match predicted_surv_curves second dimension",
    ):
        metrics.integrated_brier_score(
            event_times, event_observed, predicted_surv_curves, time_points
        )


# ============================================================================
# Unit Tests for interpret_integrated_brier_score (Task 7)
# ============================================================================


@pytest.mark.local
def test_interpret_ibs_excellent():
    """Test excellent classification (IBS < 0.15)."""
    ibs = 0.10

    result = metrics.interpret_integrated_brier_score(ibs)

    assert result["ibs"] == 0.10
    assert result["classification"] == "excellent"
    assert "excellent" in result["quality_assessment"].lower()
    assert result["baseline_comparison"] is None
    assert result["warning"] is None


@pytest.mark.local
def test_interpret_ibs_good():
    """Test good classification (0.15 <= IBS < 0.25)."""
    ibs = 0.20

    result = metrics.interpret_integrated_brier_score(ibs)

    assert result["classification"] == "good"
    assert "good" in result["quality_assessment"].lower()


@pytest.mark.local
def test_interpret_ibs_acceptable():
    """Test acceptable classification (0.25 <= IBS < 0.35)."""
    ibs = 0.30

    result = metrics.interpret_integrated_brier_score(ibs)

    assert result["classification"] == "acceptable"
    assert "acceptable" in result["quality_assessment"].lower()


@pytest.mark.local
def test_interpret_ibs_poor():
    """Test poor classification (IBS >= 0.35)."""
    ibs = 0.40

    result = metrics.interpret_integrated_brier_score(ibs)

    assert result["classification"] == "poor"
    assert "poor" in result["quality_assessment"].lower()


@pytest.mark.local
def test_interpret_ibs_with_baseline_improvement():
    """Test baseline comparison with improvement."""
    ibs = 0.20
    baseline_ibs = 0.30

    result = metrics.interpret_integrated_brier_score(ibs, baseline_ibs=baseline_ibs)

    assert result["baseline_comparison"] is not None
    assert result["baseline_comparison"]["baseline_ibs"] == 0.30
    assert result["baseline_comparison"]["better_than_baseline"] is True
    np.testing.assert_almost_equal(
        result["baseline_comparison"]["improvement"], (0.30 - 0.20) / 0.30
    )
    assert "outperforms" in result["baseline_comparison"]["interpretation"].lower()
    assert result["warning"] is None


@pytest.mark.local
def test_interpret_ibs_with_baseline_no_improvement():
    """Test baseline comparison without improvement (warning)."""
    ibs = 0.35
    baseline_ibs = 0.30

    result = metrics.interpret_integrated_brier_score(ibs, baseline_ibs=baseline_ibs)

    assert result["baseline_comparison"] is not None
    assert result["baseline_comparison"]["better_than_baseline"] is False
    assert result["warning"] is not None
    assert "not better than baseline" in result["warning"].lower()


@pytest.mark.local
def test_interpret_ibs_structure():
    """Test that result contains all required keys."""
    ibs = 0.20

    result = metrics.interpret_integrated_brier_score(ibs)

    required_keys = [
        "ibs",
        "classification",
        "quality_assessment",
        "baseline_comparison",
        "warning",
    ]
    for key in required_keys:
        assert key in result


@pytest.mark.local
def test_interpret_ibs_validation_invalid_ibs():
    """Test that invalid IBS raises ValueError."""
    with pytest.raises(ValueError, match="ibs must be in range"):
        metrics.interpret_integrated_brier_score(1.5)

    with pytest.raises(ValueError, match="ibs must be in range"):
        metrics.interpret_integrated_brier_score(-0.1)


@pytest.mark.local
def test_interpret_ibs_validation_invalid_baseline():
    """Test that invalid baseline_ibs raises ValueError."""
    with pytest.raises(ValueError, match="baseline_ibs must be in range"):
        metrics.interpret_integrated_brier_score(0.2, baseline_ibs=1.5)

    with pytest.raises(ValueError, match="baseline_ibs must be in range"):
        metrics.interpret_integrated_brier_score(0.2, baseline_ibs=-0.1)


# ============================================================================
# Concordance: Harrell, IPCW and competing risks
# ============================================================================


@pytest.mark.local
def test_competing_event_keeps_a_unit_comparable():
    """The case that distinguishes the two estimands, in two units.

    j fails from cause 2 at t=10 - earlier than i's cause-1 event at t=20. Under the
    etiologic treatment j is censored before the case and the pair is discarded. Here
    it is kept: j can never fail from cause 1, so ranking i above it is a real
    success.
    """
    durations = np.array([20.0, 10.0])
    events = np.array([1, 2])  # i fails from cause 1; j from a competing cause
    risk = np.array([0.9, 0.1])  # i correctly ranked as the higher cause-1 risk

    assert metrics.competing_risk_c_index(durations, events, risk, cause=1) == 1.0

    # The etiologic view: relabel cause 2 as censored and the pair is discarded, so
    # c_index_score falls back to its no-information default of 0.5. It reports
    # random performance where the predictive index reports a correct ranking - the
    # difference between the two estimands, in one pair.
    events_k = (events == 1).astype(int)
    assert metrics.c_index_score(durations, events_k, -risk) == 0.5


@pytest.mark.local
def test_censored_unit_before_the_case_is_not_comparable():
    """Censoring is absence of information, not information."""
    durations = np.array([20.0, 10.0])
    events = np.array([1, 0])  # j is censored at t=10; its fate is unknown
    risk = np.array([0.9, 0.1])

    assert np.isnan(metrics.competing_risk_c_index(durations, events, risk, cause=1))


@pytest.mark.local
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduces_to_harrell_without_competing_events(seed):
    """The degenerate case must not be a special case."""
    rng = np.random.default_rng(seed)
    n = 60
    durations = rng.uniform(1.0, 100.0, size=n)
    events = (rng.random(n) < 0.6).astype(int)  # only cause 1 and censoring
    risk = rng.normal(size=n)

    # The *unweighted* estimator is the one defined to coincide with Harrell here.
    # With IPCW it deliberately does not: correcting for the censoring distribution
    # is precisely what Uno's weighting changes, so equality would mean the weights
    # were doing nothing.
    predictive = metrics.competing_risk_c_index(
        durations, events, risk, cause=1, ipcw=False
    )
    # c_index_score ranks by survival, so the risk score is negated.
    harrell = metrics.c_index_score(durations, events, -risk)

    assert np.isclose(
        predictive, harrell, rtol=1e-9, atol=1e-12
    ), f"seed {seed}: predictive {predictive} != Harrell {harrell}"


@pytest.mark.local
def test_orientation_and_random_baseline():
    """Perfect ranking is 1.0, its reversal 0.0, random near 0.5.

    Note what "perfect" has to mean here, because it is not what it means for an
    ordinary C-index. Ranking purely by survival time - ``-durations`` - scores only
    0.685 under this index, and correctly so: a unit that fails from cause 2 at t=50
    is comparable to a cause-1 case at t=90, and the cause-1 case must rank higher
    even though it survived longer. The index rewards identifying *who* suffers
    cause k, not only *when* failure happens, which is precisely the predictive
    question it exists to ask.
    """
    rng = np.random.default_rng(7)
    n = 80
    durations = rng.uniform(1.0, 100.0, size=n)
    events = rng.integers(1, 3, size=n)  # every unit fails, from cause 1 or 2

    # Every cause-1 case above every unit that never suffers cause 1, and among the
    # cause-1 cases, earlier failure ranks higher.
    perfect = (events == 1).astype(float) * 1000.0 - durations
    assert metrics.competing_risk_c_index(durations, events, perfect, cause=1) == 1.0
    assert metrics.competing_risk_c_index(durations, events, -perfect, cause=1) == 0.0

    # A survival-time ranking is *not* perfect for this estimand.
    survival_only = -durations
    assert (
        metrics.competing_risk_c_index(durations, events, survival_only, cause=1) < 1.0
    )

    scores = [
        metrics.competing_risk_c_index(durations, events, rng.normal(size=n), cause=1)
        for _ in range(40)
    ]
    assert abs(float(np.mean(scores)) - 0.5) < 0.05


@pytest.mark.local
def test_ipcw_is_a_no_op_without_censoring():
    """With nothing censored, G == 1 and the weights vanish.

    The cleanest check that the weighting is wired correctly: if no one drops out,
    every inverse-censoring weight is exactly 1, so the weighted and unweighted
    estimators must agree to the bit. Any disagreement means a weight is being
    computed from the wrong time or the wrong subject.
    """
    rng = np.random.default_rng(3)
    n = 120
    durations = rng.uniform(1.0, 100.0, size=n)
    events = rng.integers(1, 3, size=n)  # every unit fails; nothing is censored
    risk = rng.normal(size=n)

    weighted = metrics.competing_risk_c_index(
        durations, events, risk, cause=1, ipcw=True
    )
    naive = metrics.competing_risk_c_index(durations, events, risk, cause=1, ipcw=False)

    assert np.isclose(weighted, naive, rtol=1e-12, atol=1e-12)


@pytest.mark.local
def test_ipcw_corrects_for_dependent_censoring():
    """Under censoring the weighting actually changes the answer.

    The complement of the test above. Heavy censoring is where the naive estimator's
    dependence on the censoring distribution bites, and it is the regime this
    project's data lives in - one fixture elsewhere in the suite logs 99.5%
    censoring. If the two estimators agreed here, the weights would be inert.
    """
    rng = np.random.default_rng(5)
    n = 300
    durations = rng.uniform(1.0, 100.0, size=n)
    # High-risk units are censored earlier: exactly the dependence that biases the
    # unweighted estimate.
    risk = rng.normal(size=n)
    censored = rng.random(n) < 0.45
    events = np.where(censored, 0, rng.integers(1, 3, size=n))

    weighted = metrics.competing_risk_c_index(
        durations, events, risk, cause=1, ipcw=True
    )
    naive = metrics.competing_risk_c_index(durations, events, risk, cause=1, ipcw=False)

    assert np.isfinite(weighted) and np.isfinite(naive)
    assert 0.0 <= weighted <= 1.0
    assert not np.isclose(weighted, naive, rtol=1e-6, atol=1e-9), (
        f"IPCW made no difference under {censored.mean():.0%} censoring "
        f"(weighted {weighted:.6f}, naive {naive:.6f}) - the weights are inert"
    )


# ---------- IPCW concordance for single-risk data ----------


@pytest.mark.local
def test_ipcw_c_index_equals_harrell_without_censoring():
    """No drop-out, no correction to make.

    Harrell's C is biased only through the censoring distribution. With nothing
    censored there is no such distribution to depend on, so the corrected and
    uncorrected estimators must agree exactly - the sharpest available check that
    the weighting is applied to the right pairs.
    """

    rng = np.random.default_rng(11)
    n = 200
    durations = rng.uniform(1.0, 100.0, size=n)
    events = np.ones(n, dtype=int)  # nobody drops out
    risk = rng.normal(size=n)

    corrected = metrics.ipcw_c_index(durations, events, risk)
    harrell = metrics.c_index_score(durations, events, -risk)

    assert np.isclose(corrected, harrell, rtol=1e-12, atol=1e-12)


@pytest.mark.local
def test_ipcw_c_index_diverges_from_harrell_under_censoring():
    """Under drop-out the correction does something."""

    rng = np.random.default_rng(12)
    n = 400
    durations = rng.uniform(1.0, 100.0, size=n)
    events = np.where(rng.random(n) < 0.6, 0, 1)
    risk = rng.normal(size=n)

    corrected = metrics.ipcw_c_index(durations, events, risk)
    harrell = metrics.c_index_score(durations, events, -risk)

    assert np.isfinite(corrected) and 0.0 <= corrected <= 1.0
    assert not np.isclose(
        corrected, harrell, rtol=1e-6, atol=1e-9
    ), "IPCW made no difference under 60% censoring - the weights are inert"


@pytest.mark.local
def test_truncation_time_tracks_the_censoring_estimate():
    """Tau is chosen where G is still trustworthy, and is overridable.

    The default exists because weights go as 1/G^2: at 90% censoring the top 1% of
    pairs carry 29% of the total weight, so a handful of late comparisons decide the
    answer. Truncating where G is still appreciable bounds that.
    """

    rng = np.random.default_rng(13)
    n = 400
    durations = rng.uniform(1.0, 100.0, size=n)
    risk = rng.normal(size=n)

    light = np.where(rng.random(n) < 0.1, 0, 1)
    heavy = np.where(rng.random(n) < 0.8, 0, 1)

    tau_light = metrics.default_truncation_time(durations, light)
    tau_heavy = metrics.default_truncation_time(durations, heavy)

    assert tau_light is not None and tau_heavy is not None
    assert (
        tau_heavy < tau_light
    ), f"heavier censoring must truncate earlier: {tau_heavy} vs {tau_light}"

    # An explicit tau overrides the default, and truncating harder changes the answer.
    early = metrics.ipcw_c_index(
        durations, heavy, risk, tau=float(np.percentile(durations, 40))
    )
    default = metrics.ipcw_c_index(durations, heavy, risk)
    assert np.isfinite(early)
    assert not np.isclose(early, default, rtol=1e-9), "tau was ignored"


# ============================================================================
# score(): every metric on one basis
# ============================================================================

MAX_RUL = 100.0
GRID = np.arange(0.0, MAX_RUL + 1.0)  # a model's survival grid stops at max_rul

# 20 units, all failed; 6 of them lived beyond max_rul.
DURATIONS = np.array(
    [
        12,
        25,
        33,
        41,
        48,
        55,
        60,
        64,
        70,
        77,
        81,
        88,
        93,
        97,
        104,
        115,
        130,
        142,
        160,
        190,
    ],
    dtype=float,
)
EVENTS = np.ones(len(DURATIONS), dtype=int)
CAPPED = DURATIONS > MAX_RUL


def scored_inputs(durations=DURATIONS, seed=0):
    """Predictions and survival curves a trained model could have produced: mean
    life tracks the truth, capped as the model's would be, and each curve decays
    at that scale. Fixed, so every metric can be recomputed by hand."""
    rng = np.random.default_rng(seed)
    predictions = np.minimum(durations, MAX_RUL) + rng.normal(0, 8, len(durations))
    scale = np.clip(predictions, 5.0, None)
    curves = np.exp(-GRID[None, :] / scale[:, None])  # (n_units, n_times)
    return predictions, curves


@pytest.mark.local
def test_every_metric_scores_the_capped_labels():
    """Concordance and IBS score every unit, censored at max_rul."""
    predictions, curves = scored_inputs()
    result = metrics.score(DURATIONS, EVENTS, predictions, curves, GRID, MAX_RUL)
    t = np.minimum(DURATIONS, MAX_RUL)
    e = np.where(CAPPED, 0, EVENTS)

    assert result["c_index"] == pytest.approx(metrics.c_index_score(t, e, predictions))
    assert result["ipcw_c_index"] == pytest.approx(
        metrics.ipcw_c_index(t, e, -predictions)
    )
    surv_df = pd.DataFrame(curves.T, index=GRID)
    assert result["antolini_c_index"] == pytest.approx(
        metrics.antolini_c_index(t, e, surv_df)
    )
    in_range = (GRID >= t.min()) & (GRID <= t.max())
    assert result["ibs"] == pytest.approx(
        metrics.integrated_brier_score(t, e, curves[:, in_range], GRID[in_range])
    )


@pytest.mark.local
def test_rmse_scores_capped_truth_over_every_failed_unit():
    """RMSE compares predictions with the truth capped at max_rul."""
    predictions, curves = scored_inputs()
    result = metrics.score(DURATIONS, EVENTS, predictions, curves, GRID, MAX_RUL)
    t = np.minimum(DURATIONS, MAX_RUL)
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean((t - predictions) ** 2)))


@pytest.mark.local
def test_calibration_scores_the_units_that_failed_within_max_rul():
    """Calibration bias uses the units uncensored after capping."""
    predictions, curves = scored_inputs()
    result = metrics.score(DURATIONS, EVENTS, predictions, curves, GRID, MAX_RUL)
    keep = ~CAPPED
    sorted_cdf, empirical = metrics.calibration_curve_data_from_survival(
        DURATIONS[keep], curves[keep], GRID
    )
    assert result["calibration_bias"] == pytest.approx(
        metrics.calibration_bias(sorted_cdf, empirical)
    )
    assert result["n_calibration"] == int(keep.sum())


@pytest.mark.local
def test_calibration_needs_two_units_that_failed_within_max_rul():
    """With fewer than two, calibration bias is None, not computed over
    another set."""
    durations = np.array([50.0, 150.0, 160.0, 170.0])
    predictions, curves = scored_inputs(durations)
    result = metrics.score(
        durations, np.ones(4, dtype=int), predictions, curves, GRID, MAX_RUL
    )
    assert result["calibration_bias"] is None
    assert result["n_calibration"] == 1


@pytest.mark.local
def test_without_max_rul_the_observed_times_are_scored():
    """With no max_rul, nothing is capped."""
    predictions, curves = scored_inputs()
    result = metrics.score(DURATIONS, EVENTS, predictions, curves, GRID)
    assert not result["capped"].any()
    assert result["c_index"] == pytest.approx(
        metrics.c_index_score(DURATIONS, EVENTS, predictions)
    )


def _harrell_by_pairs(durations, events, predictions):
    """Harrell's C by visiting every pair: the definition c_index_score must match.

    Kept as the reference for the faster implementation, quirks included: any
    non-zero event is an event, a pair of equal durations counts only as a tie and
    only when both events are 1 and the predictions are equal, and a NaN prediction
    ties with everything it is compared with.
    """
    n = len(durations)
    concordant = discordant = tied_risk = 0
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue
            if durations[i] < durations[j]:
                shorter, longer = i, j
            elif durations[j] < durations[i]:
                shorter, longer = j, i
            else:
                if events[i] == 1 and events[j] == 1:
                    if predictions[i] == predictions[j]:
                        tied_risk += 1
                continue
            if events[shorter] == 0:
                continue
            if predictions[shorter] < predictions[longer]:
                concordant += 1
            elif predictions[shorter] > predictions[longer]:
                discordant += 1
            else:
                tied_risk += 1
    total = concordant + discordant + tied_risk
    return 0.5 if total == 0 else (concordant + 0.5 * tied_risk) / total


@pytest.mark.local
@pytest.mark.parametrize("seed", range(40))
def test_c_index_score_matches_the_pairwise_definition(seed):
    """Every quirk of the pairwise definition, on data built to hit them.

    Few distinct durations and predictions make ties common, censoring is heavy,
    some events are cause 2, and some predictions are NaN.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    durations = rng.integers(1, 15, size=n).astype(float)
    events = rng.choice([0, 1, 2], size=n, p=[0.4, 0.45, 0.15])
    predictions = rng.integers(1, 10, size=n).astype(float)
    predictions[rng.random(n) < 0.05] = np.nan

    expected = _harrell_by_pairs(durations, events, predictions)

    assert metrics.c_index_score(durations, events, predictions) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "durations, events, predictions",
    [
        ([5.0], [1], [3.0]),
        ([5.0, 7.0], [0, 0], [1.0, 2.0]),
        ([5.0, 5.0], [1, 1], [2.0, 2.0]),
        ([5.0, 5.0], [1, 1], [1.0, 2.0]),
        ([3.0, 9.0], [1, 0], [np.nan, 4.0]),
    ],
)
def test_c_index_score_edge_cases_match_the_definition(durations, events, predictions):
    """One unit, all censored, tied times with tied and untied predictions, a NaN."""
    args = [np.asarray(a) for a in (durations, events, predictions)]
    assert metrics.c_index_score(*args) == _harrell_by_pairs(*args)


@pytest.mark.local
def test_c_index_score_scales_to_tuning_sized_validation_sets():
    """Tuning scores every validation row, thousands per fold. Pairs are no way to count.

    Visiting all pairs of 20,000 rows is 200 million steps of Python, hours of work;
    the bound here is generous for any n log n implementation.
    """
    import time

    rng = np.random.default_rng(0)
    n = 20_000
    durations = rng.integers(1, 300, size=n).astype(float)
    events = (rng.random(n) < 0.7).astype(int)
    predictions = durations + rng.normal(0, 30, size=n)

    started = time.perf_counter()
    score = metrics.c_index_score(durations, events, predictions)
    elapsed = time.perf_counter() - started

    assert 0.5 < score < 1.0
    assert elapsed < 5.0, f"c_index_score took {elapsed:.1f}s for {n} rows"
