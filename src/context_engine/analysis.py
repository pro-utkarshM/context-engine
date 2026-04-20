"""Benchmark result aggregation utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .artifacts import ContextSet, Outcome


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
