"""Helpers for authoring benchmark artifacts from structured inputs."""

from __future__ import annotations

from .artifacts import (
    CandidatePool,
    ChunkMetadata,
    ContextSet,
    ContextSetMetadata,
    CorpusChunk,
    MarginalImpact,
    Outcome,
    Query,
    QueryMetadata,
    RetrievalComposition,
    ScoreBundle,
)


def make_corpus_chunk(
    *,
    chunk_id: str,
    doc_version: str,
    doc_path: str,
    section_path: list[str],
    text: str,
    token_count: int,
    chunk_index: int,
    topic: str,
    subtopic: str | None = None,
    source_type: str = "doc",
    prev_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        doc_version=doc_version,
        doc_path=doc_path,
        section_path=section_path,
        source_type=source_type,
        text=text,
        token_count=token_count,
        chunk_index=chunk_index,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
        metadata=ChunkMetadata(topic=topic, subtopic=subtopic),
    )


def make_query(
    *,
    query_id: str,
    query: str,
    difficulty: str,
    gold_answer: str,
    topic: str,
    question_family: str,
    task_type: str = "doc_qa",
    gold_support_ids: list[str] | None = None,
    requires_multi_hop: bool = False,
) -> Query:
    return Query(
        query_id=query_id,
        query=query,
        task_type=task_type,
        difficulty=difficulty,
        gold_answer=gold_answer,
        gold_support_ids=gold_support_ids,
        metadata=QueryMetadata(
            topic=topic,
            requires_multi_hop=requires_multi_hop,
            question_family=question_family,
        ),
    )


def make_candidate_pool(
    *,
    query_id: str,
    candidate_pool_id: str,
    candidate_ids: list[str],
    gold_count: int,
    plausible_count: int,
    distractor_count: int,
    neutral_count: int,
    gold_in_pool: bool,
) -> CandidatePool:
    return CandidatePool(
        query_id=query_id,
        candidate_pool_id=candidate_pool_id,
        candidate_ids=candidate_ids,
        composition=RetrievalComposition(
            gold_count=gold_count,
            plausible_count=plausible_count,
            distractor_count=distractor_count,
            neutral_count=neutral_count,
        ),
        gold_in_pool=gold_in_pool,
    )


def make_context_set(
    *,
    set_id: str,
    query_id: str,
    candidate_pool_id: str,
    strategy: str,
    selected_ids: list[str],
    ordering_type: str,
    token_count: int,
    contains_all_gold: bool,
    missing_gold_count: int,
    distractor_types: list[str],
) -> ContextSet:
    return ContextSet(
        set_id=set_id,
        query_id=query_id,
        candidate_pool_id=candidate_pool_id,
        strategy=strategy,
        selected_ids=selected_ids,
        ordering_type=ordering_type,
        token_count=token_count,
        metadata=ContextSetMetadata(
            contains_all_gold=contains_all_gold,
            missing_gold_count=missing_gold_count,
            distractor_types=distractor_types,
        ),
    )


def make_outcome(
    *,
    set_id: str,
    query_id: str,
    answer: str,
    correctness: float,
    support: float,
    overall: float,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    evaluator_version: str,
) -> Outcome:
    return Outcome(
        set_id=set_id,
        query_id=query_id,
        answer=answer,
        scores=ScoreBundle(correctness=correctness, support=support, overall=overall),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        evaluator_version=evaluator_version,
    )


def make_marginal_impact(
    *,
    query_id: str,
    base_set_id: str,
    chunk_id: str,
    operation: str,
    base_score: float,
    new_score: float,
) -> MarginalImpact:
    delta = round(new_score - base_score, 10)
    return MarginalImpact.from_dict(
        {
            "query_id": query_id,
            "base_set_id": base_set_id,
            "chunk_id": chunk_id,
            "operation": operation,
            "base_score": base_score,
            "new_score": new_score,
            "delta": delta,
        }
    )
