"""Benchmark result aggregation and reporting utilities."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .artifacts import ContextSet, MarginalImpact, Outcome, Query


@dataclass(frozen=True, slots=True)
class StrategySummary:
    strategy: str
    run_count: int
    avg_correctness: float
    avg_support: float
    avg_overall: float
    avg_prompt_tokens: float


@dataclass(frozen=True, slots=True)
class QueryBestResult:
    query_id: str
    best_strategy: str
    best_overall: float


@dataclass(frozen=True, slots=True)
class PerSetRow:
    set_id: str
    query_id: str
    strategy: str
    token_count: int
    distractor_types: list[str]
    correctness: float
    support: float
    overall: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    evaluator_version: str


def summarize_by_strategy(context_sets: list[ContextSet], outcomes: list[Outcome]) -> list[StrategySummary]:
    context_sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    grouped: dict[str, list[tuple[ContextSet, Outcome]]] = defaultdict(list)

    for outcome in outcomes:
        context_set = context_sets_by_id[outcome.set_id]
        grouped[context_set.strategy].append((context_set, outcome))

    summaries: list[StrategySummary] = []
    for strategy, records in sorted(grouped.items()):
        run_count = len(records)
        summaries.append(
            StrategySummary(
                strategy=strategy,
                run_count=run_count,
                avg_correctness=sum(record[1].scores.correctness for record in records) / run_count,
                avg_support=sum(record[1].scores.support for record in records) / run_count,
                avg_overall=sum(record[1].scores.overall for record in records) / run_count,
                avg_prompt_tokens=sum(record[1].prompt_tokens for record in records) / run_count,
            )
        )
    return summaries


def best_strategy_per_query(context_sets: list[ContextSet], outcomes: list[Outcome]) -> list[QueryBestResult]:
    context_sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    best: dict[str, tuple[str, float]] = {}

    for outcome in outcomes:
        context_set = context_sets_by_id[outcome.set_id]
        current = best.get(outcome.query_id)
        if current is None or outcome.scores.overall > current[1]:
            best[outcome.query_id] = (context_set.strategy, outcome.scores.overall)

    return [
        QueryBestResult(query_id=query_id, best_strategy=strategy, best_overall=overall)
        for query_id, (strategy, overall) in sorted(best.items())
    ]


def per_set_rows(context_sets: list[ContextSet], outcomes: list[Outcome]) -> list[PerSetRow]:
    context_sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    rows: list[PerSetRow] = []
    for outcome in outcomes:
        context_set = context_sets_by_id[outcome.set_id]
        rows.append(
            PerSetRow(
                set_id=context_set.set_id,
                query_id=context_set.query_id,
                strategy=context_set.strategy,
                token_count=context_set.token_count,
                distractor_types=list(context_set.metadata.distractor_types),
                correctness=outcome.scores.correctness,
                support=outcome.scores.support,
                overall=outcome.scores.overall,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                latency_ms=outcome.latency_ms,
                evaluator_version=outcome.evaluator_version,
            )
        )
    rows.sort(key=lambda row: row.set_id)
    return rows


def render_text_report(context_sets: list[ContextSet], outcomes: list[Outcome]) -> str:
    strategy_summaries = summarize_by_strategy(context_sets, outcomes)
    best_results = best_strategy_per_query(context_sets, outcomes)

    lines = ["Strategy Summary"]
    for summary in strategy_summaries:
        lines.append(
            f"- {summary.strategy}: runs={summary.run_count}, "
            f"correctness={summary.avg_correctness:.3f}, "
            f"support={summary.avg_support:.3f}, "
            f"overall={summary.avg_overall:.3f}, "
            f"prompt_tokens={summary.avg_prompt_tokens:.1f}"
        )

    lines.append("")
    lines.append("Best Strategy Per Query")
    for result in best_results:
        lines.append(
            f"- {result.query_id}: {result.best_strategy} "
            f"(overall={result.best_overall:.3f})"
        )

    return "\n".join(lines)


def render_json_report(
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    *,
    marginal_impacts: list[MarginalImpact] | None = None,
    queries_by_id: dict[str, Query] | None = None,
    replication_summary: Any | None = None,
) -> str:
    """Stable, machine-readable strategy + per-set + impact summary.

    The output is sorted by strategy name and by set_id so the JSON is
    byte-stable across runs against the same artifacts.
    """
    strategy_summaries = summarize_by_strategy(context_sets, outcomes)
    best_results = best_strategy_per_query(context_sets, outcomes)
    rows = per_set_rows(context_sets, outcomes)

    payload: dict[str, Any] = {
        "strategy_summary": [
            {
                "strategy": summary.strategy,
                "run_count": summary.run_count,
                "avg_correctness": round(summary.avg_correctness, 6),
                "avg_support": round(summary.avg_support, 6),
                "avg_overall": round(summary.avg_overall, 6),
                "avg_prompt_tokens": round(summary.avg_prompt_tokens, 6),
            }
            for summary in strategy_summaries
        ],
        "best_strategy_per_query": [
            {
                "query_id": result.query_id,
                "best_strategy": result.best_strategy,
                "best_overall": round(result.best_overall, 6),
            }
            for result in best_results
        ],
        "per_set": [
            {
                "set_id": row.set_id,
                "query_id": row.query_id,
                "strategy": row.strategy,
                "token_count": row.token_count,
                "distractor_types": row.distractor_types,
                "correctness": round(row.correctness, 6),
                "support": round(row.support, 6),
                "overall": round(row.overall, 6),
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "latency_ms": row.latency_ms,
                "evaluator_version": row.evaluator_version,
            }
            for row in rows
        ],
    }

    if marginal_impacts is not None:
        payload["marginal_impact_summary"] = _summarize_marginal_impacts(
            marginal_impacts, queries_by_id=queries_by_id
        )

    if replication_summary is not None:
        payload["replication_summary"] = _replication_summary_payload(replication_summary)

    return json.dumps(payload, indent=2, sort_keys=True)


def render_csv_per_query(context_sets: list[ContextSet], outcomes: list[Outcome]) -> str:
    """One row per set_id with a stable column order.

    Use this for PR-attached tables or spreadsheet pivots. The
    ``distractor_types`` column is a pipe-joined list to keep the CSV
    parser-friendly.
    """
    rows = per_set_rows(context_sets, outcomes)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "set_id",
            "query_id",
            "strategy",
            "token_count",
            "distractor_types",
            "correctness",
            "support",
            "overall",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "evaluator_version",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.set_id,
                row.query_id,
                row.strategy,
                row.token_count,
                "|".join(row.distractor_types),
                f"{row.correctness:.6f}",
                f"{row.support:.6f}",
                f"{row.overall:.6f}",
                row.prompt_tokens,
                row.completion_tokens,
                row.latency_ms,
                row.evaluator_version,
            ]
        )
    return buffer.getvalue()


def render_markdown_report(
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    *,
    marginal_impacts: list[MarginalImpact] | None = None,
    queries_by_id: dict[str, Any] | None = None,
    replication_summary: Any | None = None,
) -> str:
    """Human-readable markdown report.

    Sections:
      - Strategy Summary
      - Best Strategy Per Query
      - Distractor-Heavy vs Concise-Context Wins
      - Predicted vs Measured Marginal Impact (only when marginal_impacts is provided)
      - Replication Confidence Intervals (only when replication_summary is provided)
      - Pool Comparison (only when replication_summary carries paired deltas)
    """
    strategy_summaries = summarize_by_strategy(context_sets, outcomes)
    best_results = best_strategy_per_query(context_sets, outcomes)

    lines: list[str] = ["# Benchmark Report", "", "## Strategy Summary", ""]
    lines.append("| strategy | runs | correctness | support | overall | prompt_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for summary in strategy_summaries:
        lines.append(
            f"| {summary.strategy} | {summary.run_count} | "
            f"{summary.avg_correctness:.3f} | {summary.avg_support:.3f} | "
            f"{summary.avg_overall:.3f} | {summary.avg_prompt_tokens:.1f} |"
        )

    lines.extend(["", "## Best Strategy Per Query", ""])
    lines.append("| query_id | best_strategy | best_overall |")
    lines.append("|---|---|---:|")
    for result in best_results:
        lines.append(
            f"| {result.query_id} | {result.best_strategy} | {result.best_overall:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Distractor-Heavy vs Concise-Context Wins",
            "",
            "Queries where the distractor-heavy strategy (`gold_plus_distractors`) "
            "out-performs the concise strategy (`gold_only`), and vice versa. "
            "These cases reveal whether the model is being misled by added context "
            "or whether additional gold support is needed.",
            "",
            "| query_id | gold_only | gold_plus_distractors | winner |",
            "|---|---:|---:|---|",
        ]
    )
    wins = _distractor_wins(context_sets, outcomes)
    if not wins:
        lines.append("| _no data_ | _ | _ | _ |")
    for row in wins:
        lines.append(
            f"| {row['query_id']} | {row['gold_only']:.3f} | "
            f"{row['gold_plus_distractors']:.3f} | {row['winner']} |"
        )

    if marginal_impacts is not None:
        lines.extend(
            [
                "",
                "## Predicted vs Measured Marginal Impact",
                "",
                "Predicted utility uses a heuristic slot (`is_gold`) so this view "
                "is meaningful even before the learned selector lands. A future "
                "release will replace the heuristic with the selector's score; "
                "the view's shape stays the same.",
                "",
                "| predicted_role | chunk_count | mean_delta | sum_delta | positive_share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        impact_summary = _summarize_marginal_impacts(marginal_impacts, queries_by_id=queries_by_id)
        for row in impact_summary["by_predicted_role"]:
            lines.append(
                f"| {row['predicted_role']} | {row['chunk_count']} | "
                f"{row['mean_delta']:.4f} | {row['sum_delta']:.4f} | "
                f"{row['positive_share']:.3f} |"
            )
        lines.append("")
        lines.append(
            f"_Total impact rows: {impact_summary['row_count']}; "
            f"queries covered: {impact_summary['query_count']}._"
        )

    if replication_summary is not None:
        lines.extend(
            [
                "",
                "## Replication Confidence Intervals",
                "",
                "Per-strategy mean of overall score across replications. "
                "The CI is the central 95% mass of the bootstrap distribution "
                "of the mean across runs. `within_run_query_std` is the std "
                "of per-query overall scores within the first run - it shows "
                "the spread attributable to query difficulty, not model variance. "
                "Reliability flags use a 0.05 half-width threshold: CI half-width "
                "< 0.05 -> `reliable`, otherwise `provisional`.",
                "",
                "| strategy | n_runs | mean | ci_low | ci_high | within_run_query_std | reliability |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in _replication_table_rows(replication_summary):
            flag = _reliability_flag(row["ci_low"], row["ci_high"])
            lines.append(
                f"| {row['strategy']} | {row['n_runs']} | "
                f"{row['mean']:.3f} | {row['ci_low']:.3f} | {row['ci_high']:.3f} | "
                f"{row['within_std']:.3f} | {flag} |"
            )

        if replication_summary.paired:
            lines.extend(
                [
                    "",
                    "## Pool Comparison",
                    "",
                    "Paired bootstrap CI on the per-query difference "
                    f"({replication_summary.paired[0].left_pool_source} - "
                    f"{replication_summary.paired[0].right_pool_source}). "
                    "The CI excludes 0 -> the difference is reliable at the 95% "
                    "level; otherwise it is provisional.",
                    "",
                    "| strategy | n_queries | delta | ci_low | ci_high | p |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for pair in replication_summary.paired:
                ds = pair.delta_summary
                lines.append(
                    f"| {pair.strategy} | {pair.n_queries} | "
                    f"{ds.mean_delta:+.3f} | {ds.ci_low:+.3f} | {ds.ci_high:+.3f} | "
                    f"{ds.p_value_two_sided:.3f} |"
                )

    return "\n".join(lines) + "\n"


def _distractor_wins(
    context_sets: list[ContextSet], outcomes: list[Outcome]
) -> list[dict[str, Any]]:
    """For each query, compare gold_only vs gold_plus_distractors overall scores."""
    by_query_strategy: dict[str, dict[str, float]] = defaultdict(dict)
    context_sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    for outcome in outcomes:
        context_set = context_sets_by_id[outcome.set_id]
        by_query_strategy[context_set.query_id][context_set.strategy] = outcome.scores.overall

    wins: list[dict[str, Any]] = []
    for query_id in sorted(by_query_strategy):
        scores = by_query_strategy[query_id]
        gold_only = scores.get("gold_only")
        gold_plus = scores.get("gold_plus_distractors")
        if gold_only is None or gold_plus is None:
            continue
        if gold_plus > gold_only:
            winner = "distractor_heavy_wins"
        elif gold_only > gold_plus:
            winner = "concise_wins"
        else:
            winner = "tie"
        wins.append(
            {
                "query_id": query_id,
                "gold_only": gold_only,
                "gold_plus_distractors": gold_plus,
                "winner": winner,
            }
        )
    return wins


def _summarize_marginal_impacts(
    impacts: list[MarginalImpact],
    *,
    queries_by_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate marginal impacts by predicted-role (heuristic = is_gold).

    When the learned selector lands, swap the ``_predict_role`` predicate
    for the selector's score and the view stays valid.
    """
    by_role: dict[str, list[float]] = defaultdict(list)
    queries_seen: set[str] = set()
    for impact in impacts:
        queries_seen.add(impact.query_id)
        role = _predict_role(impact, queries_by_id)
        by_role[role].append(impact.delta)

    rows: list[dict[str, Any]] = []
    for role in sorted(by_role):
        deltas = by_role[role]
        rows.append(
            {
                "predicted_role": role,
                "chunk_count": len(deltas),
                "mean_delta": round(sum(deltas) / len(deltas), 6),
                "sum_delta": round(sum(deltas), 6),
                "positive_share": round(
                    sum(1 for delta in deltas if delta > 0) / len(deltas), 6
                ),
            }
        )

    return {
        "row_count": len(impacts),
        "query_count": len(queries_seen),
        "by_predicted_role": rows,
    }


def _predict_role(impact: MarginalImpact, queries_by_id: dict[str, Any] | None) -> str:
    """Heuristic predictor. Replaced by learned selector in Phase E (#3)."""
    if queries_by_id is None:
        return "unknown"
    query = queries_by_id.get(impact.query_id)
    if query is None:
        return "unknown"
    gold_ids = set(query.gold_support_ids or [])
    if impact.chunk_id in gold_ids:
        return "gold"
    return "distractor"


def _replication_summary_payload(replication_summary) -> dict[str, Any]:
    """Convert a ReplicationSummary to a stable JSON payload.

    The shape is locked by ``docs/data-contract.md`` (replication-summary
    v1). Adding nested fields is *additive* (allowed), removing or
    renaming fields is a breaking change.
    """
    return replication_summary.to_dict()


def _replication_table_rows(replication_summary) -> list[dict[str, Any]]:
    """Markdown-friendly rows for the per-strategy CI table."""
    rows: list[dict[str, Any]] = []
    for strategy in replication_summary.strategies:
        rms = strategy.run_mean_summary
        rows.append(
            {
                "strategy": strategy.strategy,
                "n_runs": strategy.n_runs,
                "mean": rms.mean,
                "ci_low": rms.ci_low,
                "ci_high": rms.ci_high,
                "std": rms.std,
                "within_mean": strategy.within_run_query_summary.mean,
                "within_std": strategy.within_run_query_summary.std,
            }
        )
    return rows


def _reliability_flag(ci_low: float, ci_high: float, threshold: float = 0.05) -> str:
    """Mark a CI as 'narrow' (reliable) or 'wide' (provisional).

    The threshold is the half-width of the CI. CIs narrower than 2*threshold
    exclude enough effect to be reportable; wider CIs are flagged so the
    reader knows the number is provisional.
    """
    half_width = (ci_high - ci_low) / 2.0
    return "reliable" if half_width < threshold else "provisional"


def render_json_report(
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    *,
    marginal_impacts: list[MarginalImpact] | None = None,
    queries_by_id: dict[str, Query] | None = None,
    replication_summary: Any | None = None,
) -> str:
    """Stable, machine-readable strategy + per-set + impact + replication summary.

    The output is sorted by strategy name and by set_id so the JSON is
    byte-stable across runs against the same artifacts. When a
    ``replication_summary`` is provided, the JSON also includes a
    ``replication_summary`` section with per-strategy bootstrap CIs.
    """
    strategy_summaries = summarize_by_strategy(context_sets, outcomes)
    best_results = best_strategy_per_query(context_sets, outcomes)
    rows = per_set_rows(context_sets, outcomes)

    payload: dict[str, Any] = {
        "strategy_summary": [
            {
                "strategy": summary.strategy,
                "run_count": summary.run_count,
                "avg_correctness": round(summary.avg_correctness, 6),
                "avg_support": round(summary.avg_support, 6),
                "avg_overall": round(summary.avg_overall, 6),
                "avg_prompt_tokens": round(summary.avg_prompt_tokens, 6),
            }
            for summary in strategy_summaries
        ],
        "best_strategy_per_query": [
            {
                "query_id": result.query_id,
                "best_strategy": result.best_strategy,
                "best_overall": round(result.best_overall, 6),
            }
            for result in best_results
        ],
        "per_set": [
            {
                "set_id": row.set_id,
                "query_id": row.query_id,
                "strategy": row.strategy,
                "token_count": row.token_count,
                "distractor_types": row.distractor_types,
                "correctness": round(row.correctness, 6),
                "support": round(row.support, 6),
                "overall": round(row.overall, 6),
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "latency_ms": row.latency_ms,
                "evaluator_version": row.evaluator_version,
            }
            for row in rows
        ],
    }

    if marginal_impacts is not None:
        payload["marginal_impact_summary"] = _summarize_marginal_impacts(
            marginal_impacts, queries_by_id=queries_by_id
        )

    if replication_summary is not None:
        payload["replication_summary"] = _replication_summary_payload(replication_summary)

    return json.dumps(payload, indent=2, sort_keys=True)


def render_markdown_report(
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    *,
    marginal_impacts: list[MarginalImpact] | None = None,
    queries_by_id: dict[str, Any] | None = None,
    replication_summary: Any | None = None,
) -> str:
    """Human-readable markdown report.

    Sections:
      - Strategy Summary
      - Best Strategy Per Query
      - Distractor-Heavy vs Concise-Context Wins
      - Predicted vs Measured Marginal Impact (only when marginal_impacts is provided)
      - Replication Confidence Intervals (only when replication_summary is provided)
      - Pool Comparison (only when replication_summary carries paired deltas)
    """
    strategy_summaries = summarize_by_strategy(context_sets, outcomes)
    best_results = best_strategy_per_query(context_sets, outcomes)

    lines: list[str] = ["# Benchmark Report", "", "## Strategy Summary", ""]
    lines.append("| strategy | runs | correctness | support | overall | prompt_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for summary in strategy_summaries:
        lines.append(
            f"| {summary.strategy} | {summary.run_count} | "
            f"{summary.avg_correctness:.3f} | {summary.avg_support:.3f} | "
            f"{summary.avg_overall:.3f} | {summary.avg_prompt_tokens:.1f} |"
        )

    lines.extend(["", "## Best Strategy Per Query", ""])
    lines.append("| query_id | best_strategy | best_overall |")
    lines.append("|---|---|---:|")
    for result in best_results:
        lines.append(
            f"| {result.query_id} | {result.best_strategy} | {result.best_overall:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Distractor-Heavy vs Concise-Context Wins",
            "",
            "Queries where the distractor-heavy strategy (`gold_plus_distractors`) "
            "out-performs the concise strategy (`gold_only`), and vice versa. "
            "These cases reveal whether the model is being misled by added context "
            "or whether additional gold support is needed.",
            "",
            "| query_id | gold_only | gold_plus_distractors | winner |",
            "|---|---:|---:|---|",
        ]
    )
    wins = _distractor_wins(context_sets, outcomes)
    if not wins:
        lines.append("| _no data_ | _ | _ | _ |")
    for row in wins:
        lines.append(
            f"| {row['query_id']} | {row['gold_only']:.3f} | "
            f"{row['gold_plus_distractors']:.3f} | {row['winner']} |"
        )

    if marginal_impacts is not None:
        lines.extend(
            [
                "",
                "## Predicted vs Measured Marginal Impact",
                "",
                "Predicted utility uses a heuristic slot (`is_gold`) so this view "
                "is meaningful even before the learned selector lands. A future "
                "release will replace the heuristic with the selector's score; "
                "the view's shape stays the same.",
                "",
                "| predicted_role | chunk_count | mean_delta | sum_delta | positive_share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        impact_summary = _summarize_marginal_impacts(marginal_impacts, queries_by_id=queries_by_id)
        for row in impact_summary["by_predicted_role"]:
            lines.append(
                f"| {row['predicted_role']} | {row['chunk_count']} | "
                f"{row['mean_delta']:.4f} | {row['sum_delta']:.4f} | "
                f"{row['positive_share']:.3f} |"
            )
        lines.append("")
        lines.append(
            f"_Total impact rows: {impact_summary['row_count']}; "
            f"queries covered: {impact_summary['query_count']}._"
        )

    if replication_summary is not None:
        lines.extend(
            [
                "",
                "## Replication Confidence Intervals",
                "",
                "Per-strategy mean of overall score across replications. "
                "The CI is the central 95% mass of the bootstrap distribution "
                "of the mean across runs. `within_run_query_std` is the std "
                "of per-query overall scores within the first run — it shows "
                "the spread attributable to query difficulty, not model variance. "
                "Reliability flags use a 0.05 half-width threshold: CI half-width "
                "< 0.05 -> `reliable`, otherwise `provisional`.",
                "",
                "| strategy | n_runs | mean | ci_low | ci_high | within_run_query_std | reliability |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in _replication_table_rows(replication_summary):
            flag = _reliability_flag(row["ci_low"], row["ci_high"])
            lines.append(
                f"| {row['strategy']} | {row['n_runs']} | "
                f"{row['mean']:.3f} | {row['ci_low']:.3f} | {row['ci_high']:.3f} | "
                f"{row['within_std']:.3f} | {flag} |"
            )

        if replication_summary.paired:
            lines.extend(
                [
                    "",
                    "## Pool Comparison",
                    "",
                    "Paired bootstrap CI on the per-query difference "
                    f"({replication_summary.paired[0].left_pool_source} - "
                    f"{replication_summary.paired[0].right_pool_source}). "
                    "The CI excludes 0 -> the difference is reliable at the 95% "
                    "level; otherwise it is provisional.",
                    "",
                    "| strategy | n_queries | delta | ci_low | ci_high | p |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for pair in replication_summary.paired:
                ds = pair.delta_summary
                lines.append(
                    f"| {pair.strategy} | {pair.n_queries} | "
                    f"{ds.mean_delta:+.3f} | {ds.ci_low:+.3f} | {ds.ci_high:+.3f} | "
                    f"{ds.p_value_two_sided:.3f} |"
                )

    return "\n".join(lines) + "\n"
