"""Model-backed outcome generation path."""

from __future__ import annotations

from .artifacts import ContextSet, CorpusChunk, Outcome, Query
from .authoring import make_outcome
from .evaluation import ScoringWeights, score_correctness, score_efficiency, score_support
from .prompting import assemble_prompt
from .runner import ModelRunner


def evaluate_with_runner(
    *,
    query: Query,
    context_set: ContextSet,
    chunks_by_id: dict[str, CorpusChunk],
    runner: ModelRunner,
    model_name: str,
    weights: ScoringWeights = ScoringWeights(),
    evaluator_version: str = "eval_v1_model_runner",
    max_token_budget: int = 1500,
) -> Outcome:
    payload = assemble_prompt(query=query, context_set=context_set, chunks_by_id=chunks_by_id)
    response = runner.run(payload, model_name=model_name)

    correctness = score_correctness(query, response.answer)
    support = score_support(query, context_set)
    efficiency = score_efficiency(context_set, max_token_budget)
    overall = (
        weights.correctness * correctness
        + weights.support * support
        + weights.efficiency * efficiency
    )

    return make_outcome(
        set_id=context_set.set_id,
        query_id=query.query_id,
        answer=response.answer,
        correctness=correctness,
        support=support,
        overall=min(max(overall, 0.0), 1.0),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        evaluator_version=evaluator_version,
    )
