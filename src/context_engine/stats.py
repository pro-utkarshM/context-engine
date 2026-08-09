"""Statistical helpers for benchmark replication analysis.

Pure stdlib — no scipy, no numpy. The project is zero-dep (see
``pyproject.toml``), and the v1 design freeze calls out "no SDK swap,
runner code is plain HTTP + stdlib" as a load-bearing principle. The
statistical functions here are intentionally small, deterministic given
a seed, and operate on plain Python numbers.

## p-value definitions (pinned)

The percentile bootstrap is used to compute two explicit p-values:

``p_value_one_sided`` — the **one-sided tail probability**:
``P(bootstrap_delta has sign disagreeing with observed)``. This is
the share of bootstrap means whose sign disagrees with the observed
mean. For positive observed effect it equals
``P(bootstrap_delta <= 0)``. Floored at ``1/n_resamples`` to avoid
reporting an exact zero.

``p_value_two_sided`` — the **two-sided bootstrap p-value**:
``min(1.0, 2 * min(p_lower, p_upper))`` where
``p_lower = P(bootstrap_delta <= 0)`` and
``p_upper = P(bootstrap_delta >= 0)``. Capped at ``1.0``. This is the
quantity that should be compared to a two-sided significance
threshold (e.g. ``0.05``).

The legacy single-name ``p_value_two_sided`` field on this dataclass
is deprecated and will be removed in a future version. Use the
explicit ``p_value_one_sided`` and ``p_value_two_sided`` fields
instead.

## Bootstrap method

The bootstrap is the percentile method (resample with replacement,
collect the statistic, take the empirical percentiles). The CI is
the central ``ci_level`` mass of the bootstrap distribution.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Result of a one-sample summary.

    Attributes:
      n: count of observations that fed the summary.
      mean: arithmetic mean of the observations.
      std: sample standard deviation (ddof=1). 0.0 when n < 2.
      median: 50th percentile.
      ci_low: lower bound of the percentile bootstrap CI.
      ci_high: upper bound of the percentile bootstrap CI.
      ci_level: confidence level (e.g. 0.95). The CI is the central
        ``ci_level`` mass of the bootstrap distribution.
      n_resamples: number of bootstrap samples drawn.
      seed: integer seed used for the bootstrap (so the result is
        reproducible given the same inputs).
    """

    n: int
    mean: float
    std: float
    median: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    seed: int


@dataclass(frozen=True, slots=True)
class PairedDeltaSummary:
    """Result of a paired-delta summary.

    The signed delta is ``left[i] - right[i]`` at each index; the CI is
    on the mean of those deltas.

    p-value terminology (pinned):

    - ``p_value_one_sided``: the one-sided tail probability — the
      share of bootstrap means whose sign disagrees with the observed
      mean. For positive observed effect, this is
      ``P(bootstrap_delta <= 0)``. Floored at ``1/n_resamples`` to
      avoid an exact zero.
    - ``p_value_two_sided``: the two-sided bootstrap p-value —
      ``min(1.0, 2 * min(p_lower, p_upper))`` where ``p_lower`` and
      ``p_upper`` are the lower and upper one-sided tail probabilities.
      This is the value to compare against a two-sided significance
      threshold.

    The bootstrap is the percentile method. The legacy single-name
    field on prior versions of this dataclass was mislabeled — it
    was a one-sided value reported as a two-sided value. The new
    explicit fields fix this.
    """

    n: int
    mean_delta: float
    std_delta: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value_one_sided: float
    p_value_two_sided: float
    n_resamples: int
    seed: int


def _coerce_values(values):
    coerced = []
    for value in values:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise ValueError("summary inputs must be finite numbers")
        coerced.append(number)
    return coerced


def _percentile(sorted_values, q):
    """Linear-interpolation percentile, matching numpy's default.

    ``sorted_values`` must be sorted ascending. ``q`` is in [0.0, 1.0].
    """
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    position = q * (n - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _ci_bounds(bootstrap_stats, ci_level):
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    alpha = (1.0 - ci_level) / 2.0
    sorted_stats = sorted(bootstrap_stats)
    return _percentile(sorted_stats, alpha), _percentile(sorted_stats, 1.0 - alpha)


def _bootstrap_resample_mean(values, *, n_resamples, rng):
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap requires at least one observation")
    means = []
    for _ in range(n_resamples):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(n)]
        means.append(sample_sum / n)
    return means


def summarize_distribution(
    values,
    *,
    n_resamples=1000,
    ci_level=0.95,
    seed=0,
):
    """Bootstrap CI for the mean of a flat list of observations.

    The CI is the central ``ci_level`` mass of the bootstrap distribution
    of the mean, computed by the percentile method. ``seed`` makes the
    resampling deterministic — same inputs + same seed = same output.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    coerced = _coerce_values(values)
    if not coerced:
        raise ValueError("summarize_distribution requires at least one value")

    n = len(coerced)
    mean = statistics.fmean(coerced)
    std = statistics.stdev(coerced) if n >= 2 else 0.0
    median = statistics.median(coerced)

    rng = random.Random(seed)
    bootstrap_means = _bootstrap_resample_mean(coerced, n_resamples=n_resamples, rng=rng)
    ci_low, ci_high = _ci_bounds(bootstrap_means, ci_level)

    return DistributionSummary(
        n=n,
        mean=mean,
        std=std,
        median=median,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        n_resamples=n_resamples,
        seed=seed,
    )


def summarize_paired_delta(
    left,
    right,
    *,
    n_resamples=1000,
    ci_level=0.95,
    seed=0,
):
    """Bootstrap CI for the mean of paired (left - right) deltas.

    Use for "auto pool score minus canonical pool score, paired by
    query". The bootstrap resamples the *deltas* (not the raw
    observations), preserving the pairing.

    Returns a ``PairedDeltaSummary`` with both the one-sided tail
    probability (``p_value_one_sided``) and the proper two-sided
    bootstrap p-value (``p_value_two_sided``). See the module-level
    docstring for the exact definitions.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    left_coerced = _coerce_values(left)
    right_coerced = _coerce_values(right)
    if len(left_coerced) != len(right_coerced):
        raise ValueError(
            f"paired arrays must have equal length, got {len(left_coerced)} vs {len(right_coerced)}"
        )
    if not left_coerced:
        raise ValueError("summarize_paired_delta requires at least one paired observation")

    deltas = [l - r for l, r in zip(left_coerced, right_coerced)]
    n = len(deltas)
    mean_delta = statistics.fmean(deltas)
    std_delta = statistics.stdev(deltas) if n >= 2 else 0.0

    rng = random.Random(seed)
    bootstrap_means = _bootstrap_resample_mean(deltas, n_resamples=n_resamples, rng=rng)
    ci_low, ci_high = _ci_bounds(bootstrap_means, ci_level)

    # One-sided tail probability: share of bootstrap means whose sign
    # disagrees with the observed mean, floored at 1/n_resamples.
    if mean_delta == 0.0:
        p_value_one_sided = 1.0
    else:
        opposite_sign = sum(
            1 for value in bootstrap_means if (value > 0.0) != (mean_delta > 0.0)
        )
        p_value_one_sided = max(opposite_sign / n_resamples, 1.0 / n_resamples)

    # Two-sided bootstrap p-value: 2 * min(p_lower, p_upper), capped at 1.0.
    # Counts include both endpoints so the function is symmetric on the
    # signed axis.
    p_lower = sum(1 for v in bootstrap_means if v <= 0.0) / n_resamples
    p_upper = sum(1 for v in bootstrap_means if v >= 0.0) / n_resamples
    p_value_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))

    return PairedDeltaSummary(
        n=n,
        mean_delta=mean_delta,
        std_delta=std_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        p_value_one_sided=p_value_one_sided,
        p_value_two_sided=p_value_two_sided,
        n_resamples=n_resamples,
        seed=seed,
    )
