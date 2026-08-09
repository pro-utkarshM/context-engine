"""Aggregate multiple benchmark outcome runs into a CI-aware summary.

A "replication" is one complete invocation of the model-backed outcome
pipeline (50 outcome rows for the v1 corpus: 5 hand strategies × 10
queries). The replication summary is the per-strategy distribution of
the *run-level* mean across ``n_runs`` replications, with bootstrap
confidence intervals. This is the right level of aggregation for the
"is the +0.04 delta from r1 reliable?" question — the per-run mean is
one observation, and the run-to-run spread is what we are trying to
quantify.

Two comparison modes:

- **Within-run aggregation** (default): collapses per-query scores
  across a single run to a per-strategy mean. Used as the "single
  point" view of an existing artifact.

- **Across-run aggregation**: collapses per-run means across N runs to
  a per-strategy mean + bootstrap CI. Used for the replication summary.

- **Paired comparison**: given two pools (e.g. canonical vs auto) at
  the same model, produces per-strategy paired bootstrap CIs on the
  per-query delta. Used for "is the retrieval-aware delta stable?".

The module is pure stdlib and operates on outcome rows already loaded
in memory. The on-disk JSONL contract for the replication summary is
defined in ``docs/data-contract.md`` (replication-summary v1).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from .artifacts import ContextSet, Outcome
from .stats import (
    DistributionSummary,
    PairedDeltaSummary,
    summarize_distribution,
    summarize_paired_delta,
)


@dataclass(frozen=True, slots=True)
class ReplicationStrategySummary:
    """Per-strategy summary across N replications.

    ``run_means`` is the per-run mean of overall score (one number per
    replication). ``within_run_query_means`` is the per-query mean of
    overall score for the most recent run (informational; useful for
    "what does a single run look like?"). The across-run CI is the
    primary deliverable.
    """

    strategy: str
    n_runs: int
    n_queries_per_run: int
    run_means: list[float]
    run_mean_summary: DistributionSummary
    within_run_query_means: list[float]
    within_run_query_summary: DistributionSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_runs": self.n_runs,
            "n_queries_per_run": self.n_queries_per_run,
            "run_means": [round(value, 6) for value in self.run_means],
            "run_mean_summary": _summary_to_dict(self.run_mean_summary),
            "within_run_query_means": [
                round(value, 6) for value in self.within_run_query_means
            ],
            "within_run_query_summary": _summary_to_dict(self.within_run_query_summary),
        }


@dataclass(frozen=True, slots=True)
class ReplicationPairedDelta:
    """Per-strategy paired-delta summary across two pool sources."""

    strategy: str
    n_queries: int
    left_pool_source: str
    right_pool_source: str
    delta_summary: PairedDeltaSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_queries": self.n_queries,
            "left_pool_source": self.left_pool_source,
            "right_pool_source": self.right_pool_source,
            "delta_summary": _paired_summary_to_dict(self.delta_summary),
        }


@dataclass(frozen=True, slots=True)
class ReplicationSummary:
    """Top-level replication summary artifact.

    ``strategies`` is one entry per strategy present in the run set,
    sorted by strategy name. ``paired`` is empty unless the caller
    passed a paired comparison; the shape is identical to what
    ``docs/data-contract.md`` calls the "comparison_summary" section.
    """

    experiment_name: str
    runner: str
    model_name: str
    artifact_version: str
    n_runs: int
    n_resamples: int
    ci_level: float
    seed: int
    strategies: list[ReplicationStrategySummary]
    paired: list[ReplicationPairedDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "runner": self.runner,
            "model_name": self.model_name,
            "artifact_version": self.artifact_version,
            "n_runs": self.n_runs,
            "n_resamples": self.n_resamples,
            "ci_level": self.ci_level,
            "seed": self.seed,
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "paired": [pair.to_dict() for pair in self.paired],
        }

    def with_paired(self, paired: list["ReplicationPairedDelta"]) -> "ReplicationSummary":
        """Return a new summary with the paired-delta table attached.

        The base object is frozen; this is the public way to add a
        paired comparison after the run aggregation has been built.
        """
        return ReplicationSummary(
            experiment_name=self.experiment_name,
            runner=self.runner,
            model_name=self.model_name,
            artifact_version=self.artifact_version,
            n_runs=self.n_runs,
            n_resamples=self.n_resamples,
            ci_level=self.ci_level,
            seed=self.seed,
            strategies=self.strategies,
            paired=paired,
        )


def _summary_to_dict(summary: DistributionSummary) -> dict[str, Any]:
    return {
        "n": summary.n,
        "mean": round(summary.mean, 6),
        "std": round(summary.std, 6),
        "median": round(summary.median, 6),
        "ci_low": round(summary.ci_low, 6),
        "ci_high": round(summary.ci_high, 6),
        "ci_level": round(summary.ci_level, 6),
        "n_resamples": summary.n_resamples,
        "seed": summary.seed,
    }


def _paired_summary_to_dict(summary: PairedDeltaSummary) -> dict[str, Any]:
    return {
        "n": summary.n,
        "mean_delta": round(summary.mean_delta, 6),
        "std_delta": round(summary.std_delta, 6),
        "ci_low": round(summary.ci_low, 6),
        "ci_high": round(summary.ci_high, 6),
        "ci_level": round(summary.ci_level, 6),
        "p_value_one_sided": round(summary.p_value_one_sided, 6),
        "p_value_two_sided": round(summary.p_value_two_sided, 6),
        "n_resamples": summary.n_resamples,
        "seed": summary.seed,
    }


def _per_run_overall(context_sets: list[ContextSet], outcomes: list[Outcome]) -> dict[str, float]:
    """Per-strategy mean overall score across all outcomes in a single run."""
    by_strategy: dict[str, list[float]] = {}
    sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    for outcome in outcomes:
        context_set = sets_by_id.get(outcome.set_id)
        if context_set is None:
            continue
        by_strategy.setdefault(context_set.strategy, []).append(outcome.scores.overall)
    return {
        strategy: statistics.fmean(scores) for strategy, scores in by_strategy.items() if scores
    }


def _per_query_overall(
    context_sets: list[ContextSet], outcomes: list[Outcome]
) -> dict[str, dict[str, float]]:
    """Per-(strategy, query_id) overall score in a single run."""
    sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    table: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        context_set = sets_by_id.get(outcome.set_id)
        if context_set is None:
            continue
        table.setdefault(context_set.strategy, {})[context_set.query_id] = outcome.scores.overall
    return table


def _first_run_query_count(
    runs: list[tuple[list[ContextSet], list[Outcome]]],
) -> int:
    """Sanity-check query count by looking at the first run's largest strategy."""
    if not runs:
        return 0
    context_sets, outcomes = runs[0]
    table = _per_query_overall(context_sets, outcomes)
    if not table:
        return 0
    return max(len(query_scores) for query_scores in table.values())


def summarize_replications(
    runs: Sequence[tuple[list[ContextSet], list[Outcome]]],
    *,
    experiment_name: str,
    runner: str,
    model_name: str,
    artifact_version: str,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> ReplicationSummary:
    """Aggregate ``runs`` (each run is (context_sets, outcomes)) into a summary.

    The first run's per-query counts populate ``n_queries_per_run`` for
    every strategy; later runs are expected to match. Mismatched query
    counts are not validated here — the caller is responsible for
    passing matching runs.
    """
    if not runs:
        raise ValueError("summarize_replications requires at least one run")

    n_runs = len(runs)
    n_queries_per_run = _first_run_query_count(runs)

    # Per-strategy run means, across runs.
    per_strategy_run_means: dict[str, list[float]] = {}
    for context_sets, outcomes in runs:
        for strategy, mean in _per_run_overall(context_sets, outcomes).items():
            per_strategy_run_means.setdefault(strategy, []).append(mean)

    # Within-run per-query means (using the first run as the reference view).
    first_context_sets, first_outcomes = runs[0]
    per_strategy_query_means = _per_query_overall(first_context_sets, first_outcomes)

    strategies: list[ReplicationStrategySummary] = []
    for strategy in sorted(per_strategy_run_means):
        run_means = per_strategy_run_means[strategy]
        run_mean_summary = summarize_distribution(
            run_means,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        query_means = sorted(per_strategy_query_means.get(strategy, {}).values())
        within_run_summary = summarize_distribution(
            query_means,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        strategies.append(
            ReplicationStrategySummary(
                strategy=strategy,
                n_runs=n_runs,
                n_queries_per_run=n_queries_per_run,
                run_means=run_means,
                run_mean_summary=run_mean_summary,
                within_run_query_means=query_means,
                within_run_query_summary=within_run_summary,
            )
        )

    return ReplicationSummary(
        experiment_name=experiment_name,
        runner=runner,
        model_name=model_name,
        artifact_version=artifact_version,
        n_runs=n_runs,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        strategies=strategies,
    )


def paired_summary(
    left_runs: Sequence[tuple[list[ContextSet], list[Outcome]]],
    right_runs: Sequence[tuple[list[ContextSet], list[Outcome]]],
    *,
    left_pool_source: str,
    right_pool_source: str,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> list[ReplicationPairedDelta]:
    """Per-query paired delta between two pool sources at the same model.

    Both inputs must contain the same set of (strategy, query_id) pairs;
    the aggregation selects the *first* run from each side as the paired
    comparison. For multi-run paired analysis, the caller should pass
    matched-run pairs as a list of (left, right) pairs and average the
    per-run deltas — that case is out of scope for the v1 implementation.
    """
    if not left_runs or not right_runs:
        raise ValueError("paired_summary requires at least one run on each side")

    left_table = _per_query_overall(*left_runs[0])
    right_table = _per_query_overall(*right_runs[0])

    out: list[ReplicationPairedDelta] = []
    for strategy in sorted(set(left_table) & set(right_table)):
        left_by_query = left_table[strategy]
        right_by_query = right_table[strategy]
        common_queries = sorted(set(left_by_query) & set(right_by_query))
        if not common_queries:
            continue
        left_scores = [left_by_query[query_id] for query_id in common_queries]
        right_scores = [right_by_query[query_id] for query_id in common_queries]
        delta = summarize_paired_delta(
            left_scores,
            right_scores,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
        )
        out.append(
            ReplicationPairedDelta(
                strategy=strategy,
                n_queries=len(common_queries),
                left_pool_source=left_pool_source,
                right_pool_source=right_pool_source,
                delta_summary=delta,
            )
        )

    return out
