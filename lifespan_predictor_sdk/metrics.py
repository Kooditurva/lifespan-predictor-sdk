"""
Every metric a survival model is scored by, in one module.

Concordance (Harrell, IPCW, competing risks, Antolini), RMSE, calibration curves
and bias, and the Integrated Brier Score, with `score()`, which applies all of
them to one set of labels and predictions on one basis. It needs only NumPy and
pandas, and matplotlib for the plots.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Type aliases
ArrayLike = Union[np.ndarray, list, tuple]

# ---------------------------------------------------------------------------
# Concordance and error: NumPy only, no external dependencies
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
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events)
    predictions = np.asarray(predictions, dtype=float)
    n = len(durations)

    # A pair is comparable when the shorter-lived of the two had an event; it is
    # concordant when that one also has the smaller predicted time. Counted without
    # visiting pairs: units are taken longest-lived first, so when a unit is reached,
    # everything already counted outlived it. A Fenwick tree over prediction ranks
    # then says how many of those predicted longer, shorter, or the same. Units of
    # equal duration go in together, after all of them are queried, because a pair
    # of equal durations is never ordered. O(n log n) where visiting pairs was O(n²),
    # which made tuning, scoring thousands of rows per fold, take minutes per fold.
    #
    # A NaN prediction compares as neither smaller nor larger, so a pair with one is a
    # tie. NaNs are kept out of the tree and counted on their own.
    is_nan = np.isnan(predictions)
    ranks = np.zeros(n, dtype=np.int64)
    ranks[~is_nan] = np.unique(predictions[~is_nan], return_inverse=True)[1] + 1
    tree = [0] * (int(ranks.max(initial=0)) + 1)

    def counted_up_to(rank):
        total = 0
        while rank > 0:
            total += tree[rank]
            rank -= rank & -rank
        return total

    concordant = discordant = tied_risk = 0
    inserted = inserted_nan = 0
    order = np.argsort(-durations, kind="stable")
    sorted_durations = durations[order]
    start = 0
    while start < n:
        stop = start
        while stop < n and sorted_durations[stop] == sorted_durations[start]:
            stop += 1
        group = order[start:stop]
        for k in group:
            if events[k] == 0:
                continue
            if is_nan[k]:
                tied_risk += inserted + inserted_nan
                continue
            below = counted_up_to(ranks[k] - 1)
            at_or_below = counted_up_to(ranks[k])
            concordant += inserted - at_or_below
            discordant += below
            tied_risk += at_or_below - below + inserted_nan
        for k in group:
            if is_nan[k]:
                inserted_nan += 1
                continue
            rank = ranks[k]
            while rank < len(tree):
                tree[rank] += 1
                rank += rank & -rank
            inserted += 1
        start = stop

    # The one case the walk skips: two units with the same duration, both with an event
    # of cause 1 and the same prediction, count as a tie.
    both = (events == 1) & ~is_nan
    same = pd.DataFrame({"d": durations[both], "p": predictions[both]}).value_counts()
    tied_risk += int((same * (same - 1) // 2).sum())

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


def antolini_c_index(durations, events, survival_curves):
    """Antolini's time-dependent concordance of whole survival curves.

    Where Harrell's C ranks one number per unit, this ranks the curves themselves:
    a pair is concordant when the unit that failed first had the lower predicted
    survival at its own failure time. It therefore also scores models whose curves
    cross, which a single summary number cannot rank.

    Args:
        durations: Observed time to event or censoring, one per unit.
        events: ``1`` failed, ``0`` censored, one per unit.
        survival_curves: Predicted survival probabilities as a DataFrame indexed by
            time with one column per unit, in the order of ``durations``: the shape
            ``SurvivalModel.predict()`` returns.

    Returns:
        Concordance in ``[0, 1]``, or ``nan`` when no pair is comparable.
    """
    # NumPy only, like c_index_score, so the SDK computes it without pycox, and
    # every caller gets the same number. It is pycox's
    # concordance_td("antolini"): survival is read off the curve's grid at the
    # last time at or before each duration, a pair (i, j) is comparable when i
    # failed and j outlived it (or was censored at the same time), and it is
    # concordant only when i's survival is strictly lower.
    t = np.asarray(durations, dtype=float)
    failed = np.asarray(events) != 0
    grid = np.asarray(survival_curves.index, dtype=float)
    surv = np.asarray(survival_curves, dtype=float)  # (n_times, n_units)
    rows = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 1)

    concordant = 0
    comparable = 0
    for i in np.flatnonzero(failed):
        others = (t > t[i]) | ((t == t[i]) & ~failed)
        others[i] = False
        at_ti = surv[rows[i]]
        comparable += int(np.sum(others))
        concordant += int(np.sum(at_ti[i] < at_ti[others]))
    if comparable == 0:
        return float("nan")
    return concordant / comparable


# ---------------------------------------------------------------------------
# Calibration and the Integrated Brier Score
# ---------------------------------------------------------------------------


def kaplan_meier(durations: ArrayLike, observed: ArrayLike):
    """Kaplan-Meier estimate of S(t), returned as a step function of t.

    S is 1 before the first duration and keeps its last value after the last.
    """
    durations = np.asarray(durations, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    times = np.unique(durations)
    at_risk = len(durations) - np.searchsorted(np.sort(durations), times, side="left")
    events = np.bincount(
        np.searchsorted(times, durations), weights=observed, minlength=len(times)
    )
    survival = np.cumprod(1.0 - events / at_risk)

    def predict(t: float) -> float:
        index = np.searchsorted(times, t, side="right") - 1
        return 1.0 if index < 0 else float(survival[index])

    return predict


def mean_survival_times(surv_df) -> np.ndarray:
    """Mean survival time E[T] = ∫₀^∞ S(t) dt for each column of a curve frame.

    The frame is indexed by time, one column per sample. Its grid usually starts
    at the first event time, not at 0, so S(0) = 1 is added before integrating;
    otherwise every mean is short by about that first time.
    """
    times = np.concatenate([[0.0], surv_df.index.values.astype(float)])
    surv = np.vstack([np.ones((1, surv_df.shape[1])), surv_df.values.astype(float)])
    return np.trapezoid(surv, times, axis=0).astype(np.float32)


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
    if censoring_times is not None and censoring_events is not None:
        censoring_survival = kaplan_meier(
            np.asarray(censoring_times, dtype=np.float64),
            1 - np.asarray(censoring_events, dtype=np.int32),
        )
    else:
        censoring_survival = kaplan_meier(event_times, 1 - event_observed)

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
                G_Ti = censoring_survival(T_i)
                if G_Ti > 0:
                    weights[i] = 1.0 / G_Ti
                else:
                    weights[i] = 0.0

            # Case 2: Still at risk at time t (T_i > t)
            elif T_i > t:
                # Weight = 1 / G(t)
                G_t = censoring_survival(t)
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


# ---------------------------------------------------------------------------
# Every metric on one basis
# ---------------------------------------------------------------------------


def capped_labels(durations, events, max_rul=None):
    """The labels every metric is scored on.

    A model predicts no further than `max_rul`, so a unit whose true remaining
    life is beyond it counts as censored at `max_rul`: it is known to have lasted
    that long, not when it failed.

    Args:
        durations: For each unit, the time to failure or censoring.
        events: For each unit, nonzero if it failed and 0 if it was censored.
        max_rul: The model's prediction horizon, or None for no cap.

    Returns:
        A tuple `(durations, events, capped)`: the durations capped at `max_rul`,
        the events with the capped units censored, and a boolean mask of the
        units that were capped.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(int)
    if max_rul is None:
        return durations, events, np.zeros(len(durations), dtype=bool)
    capped = durations > max_rul
    return np.minimum(durations, max_rul), np.where(capped, 0, events), capped


def score(
    durations,
    events,
    predictions,
    survival_curves=None,
    time_grid=None,
    max_rul=None,
):
    """Score predictions against labels, every metric on one basis.

    Every metric scores every unit. With `max_rul` set, a unit whose duration is
    beyond it counts as censored at `max_rul` for the C-indices and the IBS, and
    RMSE compares its prediction with `max_rul`, the truth capped at the horizon.
    RMSE is over the units that failed. Calibration is over the units that failed
    within `max_rul`, since it compares each unit's predicted CDF at its true
    time against a uniform spread, which holds only where the failure was
    observed.

    Args:
        durations: For each unit, the time to failure or censoring.
        events: For each unit, nonzero if it failed and 0 if it was censored.
        predictions: For each unit, its predicted mean time to failure.
        survival_curves: Predicted survival probabilities, one row per unit and
            one column per time in `time_grid`. Without them, the calibration
            metrics, the IBS and Antolini's C-index are None.
        time_grid: The times the columns of `survival_curves` are at.
        max_rul: The model's prediction horizon, or None for no cap.

    Returns:
        A dict of the keyword arguments of `EvaluationResult` apart from the
        competing-risks figures: `c_index`, `ipcw_c_index`, `rmse`,
        `antolini_c_index`, `ibs` and `ibs_classification`, the `calibration_*`
        values and curves, the `predictions`, the scored durations as `y_test`,
        the observed `events`, the `capped` and `mask_rmse` masks,
        `n_calibration` and `max_rul`. A metric that could not be computed is
        None.
    """
    predictions = np.asarray(predictions)
    events = np.asarray(events).astype(int)
    durations_scored, events_scored, capped = capped_labels(durations, events, max_rul)
    # RMSE over every unit that failed, against the capped truth.
    mask_rmse = events > 0
    mask_calibration = events_scored > 0

    result = {
        "c_index": c_index_score(durations_scored, events_scored, predictions),
        # `predictions` are survival times, so they are negated into a risk for
        # the IPCW estimator's higher-means-worse convention.
        "ipcw_c_index": ipcw_c_index(durations_scored, events_scored, -predictions),
        "rmse": rmse_score(durations_scored[mask_rmse], predictions[mask_rmse]),
        "predictions": predictions,
        "mask_rmse": mask_rmse,
        "y_test": durations_scored,
        "events": events,
        "capped": capped,
        "max_rul": max_rul,
        "n_calibration": int(np.sum(mask_calibration)),
        "calibration_bias": None,
        "calibration_mean_cdf": None,
        "calibration_classification": None,
        "calibration_sorted_cdf": None,
        "calibration_empirical_cdf": None,
        "calibration_sorted_cdf_events": None,
        "calibration_empirical_cdf_events": None,
        "ibs": None,
        "ibs_classification": None,
        "antolini_c_index": None,
    }
    if survival_curves is None:
        return result

    survival_curves = np.asarray(survival_curves)
    time_grid = np.asarray(time_grid, dtype=float)

    try:
        result.update(
            _scored_calibration(
                durations_scored, mask_calibration, survival_curves, time_grid
            )
        )
    except Exception as e:
        print(f"  ⚠️  Could not compute calibration metrics: {e}")

    try:
        result.update(
            _scored_ibs(durations_scored, events_scored, survival_curves, time_grid)
        )
    except Exception as e:
        print(f"  ⚠️  Could not compute IBS: {e}")

    try:
        surv_df = pd.DataFrame(survival_curves.T, index=time_grid)
        antolini = antolini_c_index(durations_scored, events_scored, surv_df)
        if not np.isnan(antolini):
            result["antolini_c_index"] = float(antolini)
    except Exception as e:
        print(f"  ⚠️  Could not compute Antolini C-index: {e}")

    return result


def _scored_calibration(durations, mask_calibration, survival_curves, time_grid):
    """The calibration curve over every unit, and the bias over the units in
    `mask_calibration`. Calibration compares F(T) with a uniform spread, which
    holds only where the failure was observed, so the bias needs at least two."""
    sorted_cdf, empirical_cdf = calibration_curve_data_from_survival(
        durations, survival_curves, time_grid
    )
    scored = {
        "calibration_sorted_cdf": sorted_cdf.tolist(),
        "calibration_empirical_cdf": empirical_cdf.tolist(),
    }
    if np.sum(mask_calibration) < 2:
        return scored
    sorted_cdf, empirical_cdf = calibration_curve_data_from_survival(
        durations[mask_calibration], survival_curves[mask_calibration], time_grid
    )
    mean_cdf = float(np.mean(sorted_cdf))
    scored.update(
        calibration_bias=float(calibration_bias(sorted_cdf, empirical_cdf)),
        calibration_mean_cdf=mean_cdf,
        calibration_classification=interpret_calibration_bias(mean_cdf)[
            "classification"
        ],
        calibration_sorted_cdf_events=sorted_cdf.tolist(),
        calibration_empirical_cdf_events=empirical_cdf.tolist(),
    )
    return scored


def _scored_ibs(durations, events, survival_curves, time_grid):
    """The IBS over the units with a positive duration, on the part of the grid
    their durations span; empty when there is too little to integrate."""
    positive = durations > 0
    if not np.any(positive) or np.sum(events[positive]) == 0:
        return {}
    durations, events = durations[positive], events[positive]
    in_range = (time_grid >= np.min(durations)) & (time_grid <= np.max(durations))
    if np.sum(in_range) < 2:
        return {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ibs = integrated_brier_score(
            durations,
            events,
            survival_curves[positive][:, in_range],
            time_grid[in_range],
        )
    for w in caught:
        print(f"  ⚠️  {w.message}")
    if np.isnan(ibs):
        return {}
    ibs = float(ibs)
    return {
        "ibs": ibs,
        "ibs_classification": interpret_integrated_brier_score(ibs)["classification"],
    }
