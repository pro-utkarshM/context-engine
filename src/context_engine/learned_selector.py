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

# --- M6 estimator upgrade additions (Phase J) ---

def estimate_chunk_utility_from_marginal_impact(
    marginal_impacts: Iterable[object],
) -> dict[str, ChunkUtility]:
    """Estimate per-chunk utility from marginal-impact deltas.

    Sign convention (from the marginal_impact contract):
      delta = new_score - base_score
      delta < 0 → removing the chunk HURT the score (chunk is useful)
      delta > 0 → removing the chunk HELPED the score (chunk is harmful)

    To make this signal compatible with the outcome-based utility
    estimator (which uses raw overall scores in [0, 1], higher = better),
    we INVERT the sign: utility = -mean(delta). A chunk with negative
    mean delta (removing hurt) gets positive utility; a chunk with
    positive mean delta (removing helped) gets negative utility.

    This is the M6 estimator. Chunks with no marginal-impact rows get
    no entry; callers should fall back to the outcome-based estimator
    for those.
    """
    by_chunk: dict[str, list[float]] = defaultdict(list)
    for mi in marginal_impacts:
        by_chunk[mi.chunk_id].append(mi.delta)

    utilities: dict[str, ChunkUtility] = {}
    for chunk_id, deltas in by_chunk.items():
        utilities[chunk_id] = ChunkUtility(
            chunk_id=chunk_id,
            utility=-sum(deltas) / len(deltas),  # invert sign
            sample_size=len(deltas),
        )
    return utilities


def estimate_combined_utility(
    *,
    context_sets: Iterable[ContextSet],
    outcomes: Iterable[Outcome],
    marginal_impacts: Iterable[object] | None = None,
    score_axis: ScoreAxis = "overall",
) -> dict[str, ChunkUtility]:
    """Estimate per-chunk utility using marginal impact + outcome mean.

    Priority order (highest first):
      1. Marginal-impact signal (most direct: "did including this chunk
         help or hurt the score on a counterfactual probe?").
      2. Outcome-mean signal (fallback for chunks never probed via
         marginal impact).

    The marginal-impact signal is more informative when available because
    it isolates the chunk's contribution from the context-set's other
    chunks. The outcome-mean signal conflates the chunk's contribution
    with everything else in the context set.
    """
    utilities: dict[str, ChunkUtility] = {}

    if marginal_impacts is not None:
        mi_utilities = estimate_chunk_utility_from_marginal_impact(marginal_impacts)
        utilities.update(mi_utilities)

    outcome_utilities = estimate_chunk_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        score_axis=score_axis,
    )
    for chunk_id, util in outcome_utilities.items():
        if chunk_id not in utilities:
            utilities[chunk_id] = util

    return utilities


def select_with_negative_tiebreak(
    *,
    candidate_pool: CandidatePool,
    utilities: dict[str, ChunkUtility],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
) -> list[SelectedChunk]:
    """Greedy budget-respecting packer with negative-utility deprioritization.

    Sort key: (-utility, is_negative, pool_position) where is_negative is
    False (=0) for non-negative utility and True (=1) for negative utility.
    Effect: positive-utility chunks pack first; among same-utility chunks,
    non-negative comes before negative; ties break by pool position.

    This is the M6 packing strategy. The M5 packing (select_with_budget)
    only used pool-position tiebreak, which meant a chunk with utility
    -0.05 (mildly harmful) would tie with a chunk with utility +0.0 (no
    signal). The new tiebreak deprioritizes harmful chunks even when
    their utility is close to zero.
    """
    if token_budget <= 0:
        return []

    indexed = list(enumerate(candidate_pool.candidate_ids))
    scored: list[tuple[int, str, float, bool]] = []
    for position, chunk_id in indexed:
        if chunk_id not in chunks_by_id:
            continue
        utility_obj = utilities.get(chunk_id)
        utility = utility_obj.utility if utility_obj is not None else 0.0
        is_negative = utility < 0.0
        scored.append((position, chunk_id, utility, is_negative))

    # Sort: higher utility first; non-negative before negative; pool position last.
    scored.sort(key=lambda item: (-item[2], item[3], item[0]))

    selected: list[SelectedChunk] = []
    used = 0
    for rank, (_, chunk_id, utility, _) in enumerate(scored, start=1):
        chunk = chunks_by_id[chunk_id]
        if used + chunk.token_count > token_budget:
            continue
        selected.append(
            SelectedChunk(chunk_id=chunk_id, rank=rank, selector_score=utility)
        )
        used += chunk.token_count

    return selected


def build_learned_context_sets_v2(
    *,
    queries: list[Query],
    candidate_pools: list[CandidatePool],
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    marginal_impacts: list[object],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
    use_negative_tiebreak: bool = True,
    score_axis: ScoreAxis = "overall",
) -> list[ContextSet]:
    """Build learned context sets using the M6 estimator + packing.

    Differences vs the v1 estimator (M5):
      - Marginal-impact signal is used as the primary utility source.
      - Outcome-mean signal is the fallback for chunks without marginal
        impact data.
      - Packing deprioritizes chunks with negative utility (M5 used
        only pool-position tiebreak).

    Set ``use_negative_tiebreak=False`` to reproduce M5 packing behavior
    with M6 utility estimation; useful for ablation analysis.
    """
    utilities = estimate_combined_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        score_axis=score_axis,
    )

    select_fn = select_with_negative_tiebreak if use_negative_tiebreak else select_with_budget
    pools_by_query_id = {pool.query_id: pool for pool in candidate_pools}

    return [
        _build_v2_context_set(
            query=query,
            candidate_pool=pools_by_query_id[query.query_id],
            utilities=utilities,
            chunks_by_id=chunks_by_id,
            token_budget=token_budget,
            select_fn=select_fn,
        )
        for query in queries
        if query.query_id in pools_by_query_id
    ]


def _build_v2_context_set(
    *,
    query: Query,
    candidate_pool: CandidatePool,
    utilities: dict[str, ChunkUtility],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
    select_fn,
) -> ContextSet:
    """Build a single M6 learned context set."""
    selected = select_fn(
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


def estimate_chunk_utility_per_query_marginal_impact(
    marginal_impacts: Iterable[object],
) -> dict[tuple[str, str], ChunkUtility]:
    """Per-(query, chunk) utility from marginal impact, with global fallback.

    Returns a dict keyed by (query_id, chunk_id). For pairs without data,
    the caller falls back to the global marginal-impact signal.
    """
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for mi in marginal_impacts:
        by_pair[(mi.query_id, mi.chunk_id)].append(mi.delta)

    utilities: dict[tuple[str, str], ChunkUtility] = {}
    for (query_id, chunk_id), deltas in by_pair.items():
        utilities[(query_id, chunk_id)] = ChunkUtility(
            chunk_id=chunk_id,
            utility=-sum(deltas) / len(deltas),  # invert sign
            sample_size=len(deltas),
        )
    return utilities


def estimate_combined_utility_per_query(
    *,
    context_sets: Iterable[ContextSet],
    outcomes: Iterable[Outcome],
    marginal_impacts: Iterable[object] | None = None,
    score_axis: ScoreAxis = "overall",
) -> dict[str, dict[str, ChunkUtility]]:
    """Per-query utility table.

    Returns ``{query_id: {chunk_id: ChunkUtility}}``. For each query,
    the marginal-impact signal is used when available (per-query deltas
    are more informative than the global mean for the chunks that have
    them); the global marginal-impact signal is the fallback for chunks
    not in the per-query data; outcome-mean is the final fallback.

    This is the per-query adaptation that the M5 follow-ups called out.
    """
    # 1. Per-query marginal impact.
    per_query_mi: dict[str, dict[str, ChunkUtility]] = {}
    if marginal_impacts is not None:
        for (query_id, chunk_id), util in estimate_chunk_utility_per_query_marginal_impact(
            marginal_impacts
        ).items():
            per_query_mi.setdefault(query_id, {})[chunk_id] = util

    # 2. Global marginal impact (fallback).
    global_mi: dict[str, ChunkUtility] = {}
    if marginal_impacts is not None:
        global_mi = estimate_chunk_utility_from_marginal_impact(marginal_impacts)

    # 3. Outcome mean (final fallback).
    outcome_util = estimate_chunk_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        score_axis=score_axis,
    )

    # Assemble per-query tables.
    queries_seen = set(per_query_mi.keys()) | set(
        cs.query_id for cs in context_sets
    )
    result: dict[str, dict[str, ChunkUtility]] = {}
    for query_id in queries_seen:
        table: dict[str, ChunkUtility] = {}
        # Per-query marginal impact first.
        if query_id in per_query_mi:
            table.update(per_query_mi[query_id])
        # Global marginal impact as fallback.
        for chunk_id, util in global_mi.items():
            if chunk_id not in table:
                table[chunk_id] = util
        # Outcome mean as final fallback.
        for chunk_id, util in outcome_util.items():
            if chunk_id not in table:
                table[chunk_id] = util
        result[query_id] = table

    return result


def build_learned_context_sets_v3(
    *,
    queries: list[Query],
    candidate_pools: list[CandidatePool],
    context_sets: list[ContextSet],
    outcomes: list[Outcome],
    marginal_impacts: list[object],
    chunks_by_id: dict[str, CorpusChunk],
    token_budget: int,
    use_negative_tiebreak: bool = True,
    score_axis: ScoreAxis = "overall",
) -> list[ContextSet]:
    """Build learned context sets with per-query marginal-impact utility.

    This is the v3 estimator: per-query marginal impact first, global
    marginal impact second, outcome mean third. Chunks not in the
    per-query data inherit the global marginal impact estimate for that
    chunk, which is more informative than the per-query mean (which has
    only 1-2 observations per chunk on the v1 corpus).

    Differences from v2:
      - Per-query signal where available
      - Per-query selection (each query gets its own utility table)

    This is the M5-follow-up #1 (per-query adaptation) implementation.
    """
    per_query_utilities = estimate_combined_utility_per_query(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        score_axis=score_axis,
    )

    select_fn = select_with_negative_tiebreak if use_negative_tiebreak else select_with_budget
    pools_by_query_id = {pool.query_id: pool for pool in candidate_pools}

    return [
        _build_v2_context_set(
            query=query,
            candidate_pool=pools_by_query_id[query.query_id],
            utilities=per_query_utilities.get(query.query_id, {}),
            chunks_by_id=chunks_by_id,
            token_budget=token_budget,
            select_fn=select_fn,
        )
        for query in queries
        if query.query_id in pools_by_query_id
    ]
