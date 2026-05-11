"""Deterministic context-set generation over fixed candidate pools."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import CandidatePool, ContextSet, CorpusChunk, Query
from .authoring import make_context_set


@dataclass(frozen=True, slots=True)
class GenerationStrategy:
    name: str
    ordering_type: str


DEFAULT_STRATEGIES: tuple[GenerationStrategy, ...] = (
    GenerationStrategy(name="gold_only", ordering_type="best_first"),
    GenerationStrategy(name="topk_pool_order", ordering_type="pool_order"),
    GenerationStrategy(name="gold_plus_distractors", ordering_type="best_first"),
    GenerationStrategy(name="shuffled_order", ordering_type="shuffled"),
    GenerationStrategy(name="minimal_support", ordering_type="best_first"),
)


def _token_count(chunk_ids: list[str], chunks_by_id: dict[str, CorpusChunk]) -> int:
    return sum(chunks_by_id[chunk_id].token_count for chunk_id in chunk_ids)


def _gold_ids(query: Query) -> list[str]:
    return list(query.gold_support_ids or [])


def _pool_distractor_ids(candidate_pool: CandidatePool, query: Query) -> list[str]:
    gold_ids = set(_gold_ids(query))
    distractor_ids = [chunk_id for chunk_id in candidate_pool.candidate_ids if chunk_id not in gold_ids]
    if not candidate_pool.candidate_metadata:
        return distractor_ids

    typed = [
        chunk_id
        for chunk_id in distractor_ids
        if candidate_pool.candidate_metadata.get(chunk_id, {}).get("role") == "distractor"
    ]
    untyped = [chunk_id for chunk_id in distractor_ids if chunk_id not in typed]
    return typed + untyped


def _selected_distractor_types(candidate_pool: CandidatePool, selected_ids: list[str], gold_ids: set[str]) -> list[str]:
    if not candidate_pool.candidate_metadata:
        return ["unknown" for chunk_id in selected_ids if chunk_id not in gold_ids]

    distractor_types: list[str] = []
    for chunk_id in selected_ids:
        if chunk_id in gold_ids:
            continue
        metadata = candidate_pool.candidate_metadata.get(chunk_id, {})
        distractor_types.append(str(metadata.get("distractor_type", "unknown")))
    return distractor_types


def _strategy_selected_ids(strategy_name: str, candidate_pool: CandidatePool, query: Query) -> list[str]:
    gold_ids = _gold_ids(query)
    distractor_ids = _pool_distractor_ids(candidate_pool, query)

    if strategy_name == "gold_only":
        return gold_ids
    if strategy_name == "topk_pool_order":
        return list(candidate_pool.candidate_ids)
    if strategy_name == "gold_plus_distractors":
        return gold_ids + distractor_ids[:2]
    if strategy_name == "shuffled_order":
        return list(reversed(candidate_pool.candidate_ids))
    if strategy_name == "minimal_support":
        return gold_ids[:1]
    raise ValueError(f"Unknown context-set strategy: {strategy_name}")


def generate_context_set(
    *,
    query: Query,
    candidate_pool: CandidatePool,
    chunks_by_id: dict[str, CorpusChunk],
    strategy: GenerationStrategy,
) -> ContextSet:
    selected_ids = _strategy_selected_ids(strategy.name, candidate_pool, query)
    gold_ids = set(_gold_ids(query))
    missing_gold_count = len(gold_ids.difference(selected_ids))
    distractor_types = _selected_distractor_types(candidate_pool, selected_ids, gold_ids)

    return make_context_set(
        set_id=f"{query.query_id}_{strategy.name}",
        query_id=query.query_id,
        candidate_pool_id=candidate_pool.candidate_pool_id,
        strategy=strategy.name,
        selected_ids=selected_ids,
        ordering_type=strategy.ordering_type,
        token_count=_token_count(selected_ids, chunks_by_id),
        contains_all_gold=missing_gold_count == 0,
        missing_gold_count=missing_gold_count,
        distractor_types=distractor_types,
    )


def generate_context_sets(
    *,
    queries: list[Query],
    candidate_pools: list[CandidatePool],
    chunks_by_id: dict[str, CorpusChunk],
    strategies: tuple[GenerationStrategy, ...] = DEFAULT_STRATEGIES,
) -> list[ContextSet]:
    pools_by_query_id = {pool.query_id: pool for pool in candidate_pools}
    generated: list[ContextSet] = []
    for query in queries:
        candidate_pool = pools_by_query_id[query.query_id]
        for strategy in strategies:
            generated.append(
                generate_context_set(
                    query=query,
                    candidate_pool=candidate_pool,
                    chunks_by_id=chunks_by_id,
                    strategy=strategy,
                )
            )
    return generated
