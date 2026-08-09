"""Learned utility-based selector baseline.

Implements the first learned selector for Phase E (#3). The selector
estimates per-chunk utility from existing benchmark outcomes and greedily
packs the highest-value chunks under a token budget.

Design choices:

* Utility is the empirical mean ``overall`` score across context sets
  that include the chunk. Positive = including the chunk helped on
  average; negative = it hurt. No learned model — this is a baseline.
* Packing is greedy: sort candidates by utility desc, take them while
  budget remains. Ties break by candidate-pool position for stability.
* The selector respects the existing context-set contract: pool-bound,
  budget-respecting, deterministic for a fixed inputs.
* The ``score_axis`` parameter lets callers re-derive utility from a
  different score (e.g. ``correctness``-only) without touching code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .artifacts import (
    CandidatePool,
    ContextSet,
    CorpusChunk,
    Outcome,
    Query,
)
from .authoring import make_context_set


ScoreAxis = str  # one of correctness, support, efficiency, overall


@dataclass(frozen=True, slots=True)
class SelectedChunk:
    chunk_id: str
    rank: int
    selector_score: float


@dataclass(frozen=True, slots=True)
class ChunkUtility:
    chunk_id: str
    utility: float
    sample_size: int  # number of context sets that included this chunk


def estimate_chunk_utility(
    context_sets: Iterable[ContextSet],
    outcomes: Iterable[Outcome],
    *,
    score_axis: ScoreAxis = "overall",
) -> dict[str, ChunkUtility]:
    """Estimate empirical utility for each chunk that appears in any context set.

    Utility for chunk c = mean(outcome.scores.<axis> over context sets whose
    selected_ids contain c). Chunks never selected get no entry.
    """
    context_sets_by_id = {context_set.set_id: context_set for context_set in context_sets}
    grouped: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        context_set = context_sets_by_id.get(outcome.set_id)
        if context_set is None:
            continue
        score = getattr(outcome.scores, score_axis)
        for chunk_id in context_set.selected_ids:
            grouped[chunk_id].append(score)

    utilities: dict[str, ChunkUtility] = {}
    for chunk_id, scores in grouped.items():
        utilities[chunk_id] = ChunkUtility(
            chunk_id=chunk_id,
            utility=sum(scores) / len(scores),
            sample_size=len(scores),
        )
    return utilities


def select_with_budget(
    *,
    candidate_pool: CandidatePool,
    utilities: dict[str, ChunkUtility],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
) -> list[SelectedChunk]:
    """Greedy budget-respecting packer.

    Sort candidates by utility desc, break ties by candidate-pool position.
    Take a chunk iff (running_token_count + chunk.token_count) <= budget.

    Returns an ordered list of ``SelectedChunk`` records (rank 1-based).
    Empty pool or zero budget yields an empty list.
    """
    if token_budget <= 0:
        return []

    # Pre-seed all candidates with utility 0.0 so chunks never observed in
    # the training data still compete on tiebreak (pool position).
    indexed = list(enumerate(candidate_pool.candidate_ids))
    scored: list[tuple[int, str, float]] = []
    for position, chunk_id in indexed:
        if chunk_id not in chunks_by_id:
            continue
        utility = utilities[chunk_id].utility if chunk_id in utilities else 0.0
        # Sort key: (-utility, position). Negative utility so higher is first;
        # position so ties resolve by pool order.
        scored.append((position, chunk_id, utility))
    scored.sort(key=lambda item: (-item[2], item[0]))

    selected: list[SelectedChunk] = []
    used = 0
    for rank, (_, chunk_id, utility) in enumerate(scored, start=1):
        chunk = chunks_by_id[chunk_id]
        if used + chunk.token_count > token_budget:
            continue
        selected.append(SelectedChunk(chunk_id=chunk_id, rank=rank, selector_score=utility))
        used += chunk.token_count

    return selected


def build_learned_context_set(
    *,
    query: Query,
    candidate_pool: CandidatePool,
    utilities: dict[str, ChunkUtility],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
) -> ContextSet:
    """Build a single ``ContextSet`` whose strategy is ``learned``.

    The set uses ``ordering_type = best_first`` (utility desc). Distractor
    metadata is propagated from the candidate pool when available so the
    analysis view in #7 stays accurate.
    """
    selected = select_with_budget(
        candidate_pool=candidate_pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=token_budget,
    )
    selected_ids = [record.chunk_id for record in selected]
    token_count = sum(chunks_by_id[chunk_id].token_count for chunk_id in selected_ids)

    gold_ids = set(query.gold_support_ids or [])
    missing_gold_count = len(gold_ids.difference(selected_ids))

    if candidate_pool.candidate_metadata:
        distractor_types = [
            str(candidate_pool.candidate_metadata.get(chunk_id, {}).get("distractor_type", "unknown"))
            for chunk_id in selected_ids
            if chunk_id not in gold_ids
        ]
    else:
        distractor_types = ["unknown" for chunk_id in selected_ids if chunk_id not in gold_ids]

    return make_context_set(
        set_id=f"{query.query_id}_learned",
        query_id=query.query_id,
        candidate_pool_id=candidate_pool.candidate_pool_id,
        strategy="learned",
        selected_ids=selected_ids,
        ordering_type="best_first",
        token_count=token_count,
        contains_all_gold=missing_gold_count == 0,
        missing_gold_count=missing_gold_count,
        distractor_types=distractor_types,
    )


def build_learned_context_sets(
    *,
    queries: list[Query],
    candidate_pools: list[CandidatePool],
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
    score_axis: ScoreAxis = "overall",
) -> list[ContextSet]:
    """Build a learned context set for every query.

    The selector is global — it learns from *all* queries' outcomes, then
    applies the same utility ranking to each query's candidate pool. This
    is the simplest possible shared-rank baseline; per-query adaptation is
    out of scope for v1.
    """
    utilities = estimate_chunk_utility(
        context_sets=context_sets, outcomes=outcomes, score_axis=score_axis
    )
    pools_by_query_id = {pool.query_id: pool for pool in candidate_pools}
    return [
        build_learned_context_set(
            query=query,
            candidate_pool=pools_by_query_id[query.query_id],
            utilities=utilities,
            chunks_by_id=chunks_by_id,
            token_budget=token_budget,
        )
        for query in queries
        if query.query_id in pools_by_query_id
    ]