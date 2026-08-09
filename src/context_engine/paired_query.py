"""Multi-run paired-query aggregate statistics.

The independent experimental unit in this benchmark is the QUERY, not each
stochastic model repetition. With ``n_queries`` queries and ``n_reps``
model runs per (query, strategy), the aggregation follows:

  1. Average within-query reps -> one per-query mean per strategy.
  2. Subtract per-query -> ``n_queries`` paired deltas.
  3. Bootstrap the paired deltas -> 95% CI on the mean of those deltas.

This is the "right" unit for comparing strategies on the v1 corpus:
- CI is on the per-query mean, not on each individual rep.
- Repetitions reduce within-query model noise; they do not add
  independent evidence.

The legacy ``stats.summarize_paired_delta`` remains in the stats module
and is exposed for single-run paired comparisons (e.g. golden baseline
vs auto pool, where we do not have multiple runs). This module wraps it
with the per-query mean aggregation needed for multi-run analysis.

Naming: ``PairedQueryDeltaSummary`` is the multi-run analog of
``PairedDeltaSummary``. The two are deliberately distinct types so
callers cannot accidentally conflate single-run and multi-run
comparisons.

## p-value terminology

The bootstrap significance is reported as two explicit fields:

- ``p_value_one_sided``: ``P(bootstrap_delta has sign disagreeing with
  observed)``. For a positive observed effect this is
  ``P(bootstrap_delta <= 0)``. Floored at ``1/n_resamples``.
- ``p_value_two_sided``: ``min(1.0, 2 * min(p_lower, p_upper))`` where
  ``p_lower`` and ``p_upper`` are the lower and upper one-sided tail
  probabilities. This is the value to compare against a two-sided
  significance threshold.

## Rounding policy

The CI bounds and the mean delta are returned at full float precision
so that the raw quantile can be inspected (e.g. to determine whether
the CI lower bound is exactly zero vs slightly positive). A separate
``to_dict`` method exposes a rounded display form for human-readable
reports, alongside the raw values.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .stats import PairedDeltaSummary, summarize_paired_delta


@dataclass(frozen=True, slots=True)
class PairedQueryDeltaSummary:
    """Per-query paired-delta summary across multiple replications.

    Attributes:
      left_label: human-readable label for the left side.
      right_label: human-readable label for the right side.
      n_queries: count of independent queries with matched data on both sides.
      reps_per_query: number of stochastic reps averaged per query (>= 1).
      per_query_left: per-query mean of left-side scores (one float per query).
      per_query_right: per-query mean of right-side scores.
      per_query_deltas: per-query paired delta (left - right).
      per_query_raw_left: per-query list of all raw left-side scores.
      per_query_raw_right: per-query list of all raw right-side scores.
      mean_left: overall mean of left-side per-query means.
      mean_right: overall mean of right-side per-query means.
      delta_summary: ``PairedDeltaSummary`` produced by the percentile
        bootstrap on the per-query deltas.
      n_resamples: number of bootstrap samples drawn.
      seed: integer seed used for the bootstrap.
    """

    left_label: str
    right_label: str
    n_queries: int
    reps_per_query: int
    per_query_left: Mapping[str, float]
    per_query_right: Mapping[str, float]
    per_query_deltas: Mapping[str, float]
    mean_left: float
    mean_right: float
    delta_summary: PairedDeltaSummary
    n_resamples: int
    seed: int
    per_query_raw_left: Mapping[str, Sequence[float]] = field(default_factory=dict)
    per_query_raw_right: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation. CI bounds are kept at full float
        precision (no rounding) so the raw quantile is preserved.
        Use ``ci_low`` / ``ci_high`` directly to inspect the
        unrounded boundary."""

        return {
            "left_label": self.left_label,
            "right_label": self.right_label,
            "n_queries": self.n_queries,
            "reps_per_query": self.reps_per_query,
            "mean_left": round(self.mean_left, 6),
            "mean_right": round(self.mean_right, 6),
            "delta_summary": {
                "mean_delta": self.delta_summary.mean_delta,
                "std_delta": self.delta_summary.std_delta,
                "ci_low": self.delta_summary.ci_low,
                "ci_high": self.delta_summary.ci_high,
                "ci_level": self.delta_summary.ci_level,
                "p_value_one_sided": self.delta_summary.p_value_one_sided,
                "p_value_two_sided": self.delta_summary.p_value_two_sided,
                "n_resamples": self.delta_summary.n_resamples,
                "seed": self.delta_summary.seed,
            },
            "per_query": {
                qid: {
                    "left_mean": self.per_query_left[qid],
                    "right_mean": self.per_query_right[qid],
                    "delta": self.per_query_deltas[qid],
                }
                for qid in sorted(self.per_query_left)
            },
        }


def paired_query_summary(
    left_scores: Mapping[str, Sequence[float]],
    right_scores: Mapping[str, Sequence[float]],
    *,
    left_label: str,
    right_label: str,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> PairedQueryDeltaSummary:
    """Multi-run paired-query aggregate statistics.

    Both inputs are ``{query_id: [score, score, ...]}`` mappings. The
    function aligns by query_id, averages within each query, computes
    per-query deltas, and bootstraps the mean of those deltas.

    The result is deterministic given the same inputs and seed.

    Raises:
      ValueError: if either side is empty after the intersection, or if
        the two sides have no overlapping queries, or if no query has
        at least one rep on both sides.
    """
    if not left_scores or not right_scores:
        raise ValueError("paired_query_summary requires non-empty inputs on both sides")

    common_q = sorted(set(left_scores) & set(right_scores))
    if not common_q:
        raise ValueError(
            "paired_query_summary: no overlapping queries between left and right"
        )

    # Drop queries with no reps on either side.
    matched = [q for q in common_q if left_scores[q] and right_scores[q]]
    if not matched:
        raise ValueError("paired_query_summary: no queries with at least one rep on both sides")

    # Per-query means.
    left_means = {q: statistics.fmean(left_scores[q]) for q in matched}
    right_means = {q: statistics.fmean(right_scores[q]) for q in matched}

    # Per-query deltas.
    deltas = {q: left_means[q] - right_means[q] for q in matched}

    # Bootstrap CI on the mean of per-query deltas. The signed input
    # vector is (left[q] - right[q]) for each query, by index.
    left_vec = [left_means[q] for q in matched]
    right_vec = [right_means[q] for q in matched]
    delta_summary = summarize_paired_delta(
        left_vec,
        right_vec,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )

    # reps_per_query: surface the matched count. Heterogeneous matches
    # are rare in this benchmark; we report the max so the summary is
    # honest about the available coverage.
    rep_counts = {len(left_scores[q]) for q in matched} | {len(right_scores[q]) for q in matched}
    reps_per_query = max(rep_counts) if len(rep_counts) > 1 else next(iter(rep_counts))

    mean_left = statistics.fmean(left_means[q] for q in matched)
    mean_right = statistics.fmean(right_means[q] for q in matched)

    return PairedQueryDeltaSummary(
        left_label=left_label,
        right_label=right_label,
        n_queries=len(matched),
        reps_per_query=reps_per_query,
        per_query_left=left_means,
        per_query_right=right_means,
        per_query_deltas=deltas,
        mean_left=mean_left,
        mean_right=mean_right,
        delta_summary=delta_summary,
        n_resamples=n_resamples,
        seed=seed,
        per_query_raw_left=dict(left_scores),
        per_query_raw_right=dict(right_scores),
    )
