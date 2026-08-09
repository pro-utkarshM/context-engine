"""Tests for the pure-stdlib stats module (bootstrap CI)."""

from __future__ import annotations

import math

import pytest

from context_engine.stats import (
    DistributionSummary,
    PairedDeltaSummary,
    summarize_distribution,
    summarize_paired_delta,
)


def test_summarize_distribution_reports_known_mean_and_std():
    values = [0.49, 0.50, 0.51, 0.50, 0.49, 0.51, 0.50, 0.50, 0.49, 0.50]
    dist = summarize_distribution(values, n_resamples=2000, seed=7)

    assert dist.n == 10
    assert math.isclose(dist.mean, 0.499, rel_tol=0, abs_tol=1e-12)
    assert dist.std > 0.0
    assert math.isclose(dist.median, 0.5, rel_tol=0, abs_tol=1e-12)
    assert dist.ci_low <= dist.mean <= dist.ci_high
    assert dist.ci_low >= 0.48
    assert dist.ci_high <= 0.52
    assert dist.ci_level == 0.95
    assert dist.n_resamples == 2000
    assert dist.seed == 7


def test_summarize_distribution_is_deterministic_for_same_seed():
    values = [0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.45, 0.55]
    a = summarize_distribution(values, seed=123)
    b = summarize_distribution(values, seed=123)
    assert a == b


def test_summarize_distribution_varies_with_seed():
    values = [0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.45, 0.55]
    a = summarize_distribution(values, seed=1)
    b = summarize_distribution(values, seed=99999)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_summarize_distribution_single_value_has_zero_spread():
    dist = summarize_distribution([0.7])
    assert dist.n == 1
    assert dist.std == 0.0
    assert dist.ci_low == 0.7
    assert dist.ci_high == 0.7


def test_summarize_distribution_identical_values_collapse_to_point():
    dist = summarize_distribution([0.5] * 20)
    assert dist.std == 0.0
    assert dist.ci_low == 0.5
    assert dist.ci_high == 0.5
    assert dist.median == 0.5


def test_summarize_distribution_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one value"):
        summarize_distribution([])


def test_summarize_distribution_rejects_non_finite_input():
    with pytest.raises(ValueError, match="finite numbers"):
        summarize_distribution([0.1, float("nan"), 0.3])
    with pytest.raises(ValueError, match="finite numbers"):
        summarize_distribution([0.1, float("inf"), 0.3])


def test_summarize_distribution_rejects_bad_resample_count():
    with pytest.raises(ValueError, match="n_resamples"):
        summarize_distribution([0.1, 0.2], n_resamples=0)


def test_summarize_paired_delta_computes_signed_difference():
    delta = summarize_paired_delta(
        [0.7, 0.8, 0.9, 0.6, 0.5],
        [0.5, 0.6, 0.7, 0.4, 0.3],
        n_resamples=2000,
        seed=11,
    )
    assert math.isclose(delta.mean_delta, 0.2, rel_tol=0, abs_tol=1e-12)
    assert delta.ci_low > 0.15
    assert delta.ci_high < 0.25
    assert delta.p_value_two_sided < 0.01


def test_summarize_paired_delta_zero_delta_has_p_value_one():
    delta = summarize_paired_delta([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert math.isclose(delta.mean_delta, 0.0, abs_tol=1e-12)
    assert delta.p_value_two_sided == 1.0


def test_summarize_paired_delta_flips_sign_when_left_right_swapped():
    """When the deltas are all distinct (no point collapse), swapping
    left/right should flip mean_delta and mirror the CI bounds across 0."""
    a = summarize_paired_delta(
        [0.71, 0.83, 0.92, 0.65, 0.51],
        [0.50, 0.62, 0.71, 0.41, 0.30],
        seed=5,
    )
    b = summarize_paired_delta(
        [0.50, 0.62, 0.71, 0.41, 0.30],
        [0.71, 0.83, 0.92, 0.65, 0.51],
        seed=5,
    )
    assert math.isclose(a.mean_delta, -b.mean_delta, abs_tol=1e-12)
    assert math.isclose(a.ci_low, -b.ci_high, abs_tol=1e-9)
    assert math.isclose(a.ci_high, -b.ci_low, abs_tol=1e-9)


def test_summarize_paired_delta_requires_equal_length():
    with pytest.raises(ValueError, match="equal length"):
        summarize_paired_delta([0.1, 0.2], [0.1])


def test_summarize_paired_delta_requires_at_least_one_pair():
    with pytest.raises(ValueError, match="at least one"):
        summarize_paired_delta([], [])


def test_summary_records_are_immutable():
    dist = summarize_distribution([0.1, 0.2, 0.3])
    delta = summarize_paired_delta([0.1, 0.2], [0.0, 0.1])
    with pytest.raises(Exception):
        dist.mean = 0.5  # type: ignore[misc]
    with pytest.raises(Exception):
        delta.mean_delta = 0.0  # type: ignore[misc]


def test_perfect_separation_yields_tiny_p_value_one_sided():
    """If every paired delta is positive, the one-sided tail probability
    should fall at the 1/n_resamples floor (no bootstrap sample flips
    sign). The two-sided p-value is 2 * min(p_lower, p_upper)."""
    delta = summarize_paired_delta(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        n_resamples=1000,
        seed=0,
    )
    # One-sided tail probability is the floor.
    assert math.isclose(delta.p_value_one_sided, 1.0 / 1000, abs_tol=1e-12)
    # Two-sided p-value: 2 * min(p_lower, p_upper). With most bootstrap
    # means positive, p_lower is small (only bootstrap samples that
    # picked the zero delta land at exactly 0). The floor of 1/n applies
    # via the <= 0 count.
    assert 0.0 <= delta.p_value_two_sided <= 2.0 / 1000


def test_perfect_separation_zero_deltas_yields_zero_two_sided_p():
    """If every paired delta is positive *and* the bootstrap distribution
    has no mass on the opposite side, the two-sided p-value is 0.
    """
    delta = summarize_paired_delta(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.0],  # all deltas strictly positive
        n_resamples=1000,
        seed=0,
    )
    assert math.isclose(delta.p_value_one_sided, 1.0 / 1000, abs_tol=1e-12)
    # All bootstrap means are strictly positive, so p_lower = 0 and
    # the two-sided p-value is 0.
    assert delta.p_value_two_sided == 0.0


def test_ci_level_changes_width():
    values = [0.4, 0.6, 0.5, 0.7, 0.3, 0.8, 0.45, 0.55, 0.5, 0.5]
    ci90 = summarize_distribution(values, ci_level=0.90, seed=42)
    ci99 = summarize_distribution(values, ci_level=0.99, seed=42)
    width_90 = ci90.ci_high - ci90.ci_low
    width_99 = ci99.ci_high - ci99.ci_low
    assert width_99 > width_90  # higher confidence -> wider interval
    assert ci90.ci_low >= ci99.ci_low
    assert ci90.ci_high <= ci99.ci_high


def test_summary_dataclass_export_shape():
    """Lock the dataclass field set so callers see a stable shape."""
    dist_fields = set(DistributionSummary.__dataclass_fields__.keys())
    assert dist_fields == {
        "n",
        "mean",
        "std",
        "median",
        "ci_low",
        "ci_high",
        "ci_level",
        "n_resamples",
        "seed",
    }
    delta_fields = set(PairedDeltaSummary.__dataclass_fields__.keys())
    assert delta_fields == {
        "n",
        "mean_delta",
        "std_delta",
        "ci_low",
        "ci_high",
        "ci_level",
        "p_value_one_sided",
        "p_value_two_sided",
        "n_resamples",
        "seed",
    }
