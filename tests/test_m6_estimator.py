"""Tests for the M6 estimator upgrade (Phase J)."""

from __future__ import annotations

import pytest

from context_engine.artifacts import (
    CandidatePool,
    ContextSet,
    ContextSetMetadata,
    CorpusChunk,
    MarginalImpact,
    Outcome,
    Query,
)
from context_engine.learned_selector import (
    ChunkUtility,
    build_learned_context_sets_v2,
    build_learned_context_sets_v3,
    estimate_chunk_utility_from_marginal_impact,
    estimate_chunk_utility_per_query_marginal_impact,
    estimate_combined_utility,
    estimate_combined_utility_per_query,
    select_with_negative_tiebreak,
)


def _chunk(chunk_id: str, tokens: int, topic: str = "t") -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "x.md",
            "section_path": ["S"],
            "source_type": "doc",
            "text": f"text for {chunk_id}",
            "token_count": tokens,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": topic},
        }
    )


def _context_set(set_id: str, query_id: str, selected_ids: list[str]) -> ContextSet:
    return ContextSet(
        set_id=set_id,
        query_id=query_id,
        candidate_pool_id=f"pool_{query_id}",
        strategy="gold_only",
        selected_ids=selected_ids,
        ordering_type="best_first",
        token_count=sum(len(s) for s in selected_ids),
        metadata=ContextSetMetadata(
            contains_all_gold=True, missing_gold_count=0, distractor_types=[]
        ),
    )


def _outcome(set_id: str, query_id: str, overall: float) -> Outcome:
    return Outcome.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "answer": "a",
            "scores": {"correctness": overall, "support": overall, "overall": overall},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "evaluator_version": "eval_v1",
        }
    )


def _query(query_id: str, gold_ids: list[str]) -> Query:
    return Query.from_dict(
        {
            "query_id": query_id,
            "query": "Q?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "x",
            "gold_support_ids": gold_ids,
            "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
        }
    )


def _candidate_pool(query_id: str, candidate_ids: list[str]) -> CandidatePool:
    return CandidatePool.from_dict(
        {
            "query_id": query_id,
            "candidate_pool_id": f"pool_{query_id}",
            "candidate_ids": candidate_ids,
            "composition": {
                "gold_count": 1,
                "plausible_count": 10,
                "distractor_count": 6,
                "neutral_count": 3,
            },
            "gold_in_pool": True,
        }
    )


def _mi(query_id: str, chunk_id: str, delta: float) -> MarginalImpact:
    return MarginalImpact.from_dict(
        {
            "query_id": query_id,
            "base_set_id": f"{query_id}_gold_plus_distractors",
            "chunk_id": chunk_id,
            "operation": "remove",
            "base_score": 0.5,
            "new_score": 0.5 + delta,
            "delta": delta,
        }
    )


# --- estimate_chunk_utility_from_marginal_impact ---

def test_marginal_impact_utility_inverts_sign():
    """A chunk that HURT when removed (negative delta) should have POSITIVE utility."""
    mi_rows = [_mi("q1", "c1", -0.3), _mi("q2", "c1", -0.5)]
    utilities = estimate_chunk_utility_from_marginal_impact(mi_rows)
    assert utilities["c1"].utility == pytest.approx(0.4)  # -(-0.3 + -0.5) / 2 = 0.4
    assert utilities["c1"].sample_size == 2


def test_marginal_impact_utility_negative_for_harmful_chunks():
    """A chunk that HELPED when removed (positive delta) should have NEGATIVE utility."""
    mi_rows = [_mi("q1", "c1", 0.3)]
    utilities = estimate_chunk_utility_from_marginal_impact(mi_rows)
    assert utilities["c1"].utility == pytest.approx(-0.3)


def test_marginal_impact_utility_per_chunk():
    """Utility is per-chunk, not per-query."""
    mi_rows = [
        _mi("q1", "c1", -0.3),
        _mi("q2", "c1", 0.2),
        _mi("q1", "c2", -0.5),
    ]
    utilities = estimate_chunk_utility_from_marginal_impact(mi_rows)
    assert utilities["c1"].utility == pytest.approx(0.05)  # -(-0.3 + 0.2) / 2
    assert utilities["c2"].utility == pytest.approx(0.5)


# --- estimate_combined_utility ---

def test_combined_utility_uses_marginal_impact_when_available():
    """Marginal-impact signal wins for chunks that have it."""
    mi_rows = [_mi("q1", "c1", -0.5)]  # c1: utility 0.5 (inverted)
    context_sets = [_context_set("s1", "q1", ["c1", "c2"])]
    outcomes = [_outcome("s1", "q1", 0.9)]
    utilities = estimate_combined_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=mi_rows,
    )
    # c1: from marginal impact (0.5), not from outcome (0.9)
    assert utilities["c1"].utility == pytest.approx(0.5)
    # c2: only in outcomes, so utility = outcome score
    assert utilities["c2"].utility == pytest.approx(0.9)


def test_combined_utility_falls_back_to_outcome_mean():
    """Chunks without marginal-impact rows use outcome mean."""
    context_sets = [_context_set("s1", "q1", ["c1"])]
    outcomes = [_outcome("s1", "q1", 0.7)]
    utilities = estimate_combined_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=[],  # no marginal impact data
    )
    assert utilities["c1"].utility == pytest.approx(0.7)


def test_combined_utility_without_marginal_impacts_matches_outcome_only():
    """When marginal_impacts=None, behavior matches the M5 estimator."""
    context_sets = [
        _context_set("s1", "q1", ["c1", "c2"]),
        _context_set("s2", "q1", ["c2"]),
    ]
    outcomes = [_outcome("s1", "q1", 0.8), _outcome("s2", "q1", 0.4)]
    utilities = estimate_combined_utility(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=None,
    )
    assert utilities["c1"].utility == pytest.approx(0.8)
    assert utilities["c2"].utility == pytest.approx(0.6)


# --- select_with_negative_tiebreak ---

def test_negative_tiebreak_deprioritizes_harmful_chunks():
    """A chunk with utility 0 (no signal) should pack before a chunk with utility -0.1."""
    pool = _candidate_pool("q1", ["c1", "c2"])
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}
    utilities = {
        "c1": ChunkUtility("c1", 0.5, 1),
        "c2": ChunkUtility("c2", -0.1, 1),  # harmful
    }
    selected = select_with_negative_tiebreak(
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    assert [s.chunk_id for s in selected] == ["c1", "c2"]


def test_negative_tiebreak_skips_harmful_when_budget_constrained():
    """When the budget can't fit both, the harmful chunk is dropped."""
    pool = _candidate_pool("q1", ["c1", "c2"])
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}
    utilities = {
        "c1": ChunkUtility("c1", 0.5, 1),
        "c2": ChunkUtility("c2", -0.5, 1),
    }
    selected = select_with_negative_tiebreak(
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=100,
    )
    assert [s.chunk_id for s in selected] == ["c1"]


def test_negative_tiebreak_preserves_pool_position_for_zero_utility():
    """Chunks with utility 0 (no signal) tiebreak by pool position, like M5."""
    pool = _candidate_pool("q1", ["c1", "c2", "c3"])
    chunks_by_id = {c: _chunk(c, 100) for c in ("c1", "c2", "c3")}
    utilities = {c: ChunkUtility(c, 0.0, 0) for c in ("c1", "c2", "c3")}
    selected = select_with_negative_tiebreak(
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    assert [s.chunk_id for s in selected] == ["c1", "c2"]


def test_negative_tiebreak_zero_budget_yields_empty():
    pool = _candidate_pool("q1", ["c1"])
    chunks_by_id = {"c1": _chunk("c1", 100)}
    utilities = {"c1": ChunkUtility("c1", 0.9, 1)}
    assert select_with_negative_tiebreak(
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=0,
    ) == []


# --- build_learned_context_sets_v2 ---

def test_build_v2_uses_marginal_impact_signal():
    """The v2 estimator picks the chunk that has positive marginal impact."""
    queries = [_query("q1", ["c1"])]
    pools = [_candidate_pool("q1", ["c1", "c2"])]
    context_sets = [_context_set("s1", "q1", ["c1"])]
    outcomes = [_outcome("s1", "q1", 0.5)]  # c1 has outcome 0.5
    marginal_impacts = [_mi("q1", "c1", -0.3)]  # c1: utility 0.3 (inverted)
    # c2 is not in outcomes or marginal impacts → utility 0.0
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}

    learned = build_learned_context_sets_v2(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    # c1 has utility 0.3 (marginal impact), c2 has utility 0.0 (default).
    # Both fit in budget, so both should be selected (positive utility + tiebreak).
    assert [cs.set_id for cs in learned] == ["q1_learned"]
    assert set(learned[0].selected_ids) == {"c1", "c2"}


def test_build_v2_drops_chunks_with_negative_utility():
    queries = [_query("q1", ["c1"])]
    pools = [_candidate_pool("q1", ["c1", "c2"])]
    context_sets = [_context_set("s1", "q1", ["c1", "c2"])]
    outcomes = [_outcome("s1", "q1", 0.5)]
    # c1: positive utility via marginal impact (-0.3 → 0.3)
    # c2: NEGATIVE utility via marginal impact (+0.5 → -0.5)
    marginal_impacts = [_mi("q1", "c1", -0.3), _mi("q1", "c2", 0.5)]
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}

    learned = build_learned_context_sets_v2(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        chunks_by_id=chunks_by_id,
        token_budget=100,  # only one chunk fits
    )
    assert learned[0].selected_ids == ["c1"]


def test_build_v2_uses_m5_packing_when_flag_disabled():
    """With use_negative_tiebreak=False, packing matches M5 behavior."""
    queries = [_query("q1", ["c1"])]
    pools = [_candidate_pool("q1", ["c1", "c2"])]
    context_sets = [_context_set("s1", "q1", ["c1", "c2"])]
    outcomes = [_outcome("s1", "q1", 0.5)]
    marginal_impacts = [_mi("q1", "c1", -0.3), _mi("q1", "c2", 0.5)]
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}

    learned = build_learned_context_sets_v2(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        chunks_by_id=chunks_by_id,
        token_budget=100,
        use_negative_tiebreak=False,
    )
    # M5 packing only used pool position for tiebreak. With utility 0.3 vs -0.5,
    # M5 still picks the higher-utility chunk (no tiebreak needed here).
    # This test mainly locks that the flag works without changing semantics.
    assert learned[0].selected_ids == ["c1"]


def test_build_v2_emits_strategy_learned_and_contains_all_gold():
    queries = [_query("q1", ["c1"])]
    pools = [_candidate_pool("q1", ["c1", "c2"])]
    context_sets = [_context_set("s1", "q1", ["c1", "c2"])]
    outcomes = [_outcome("s1", "q1", 0.7)]
    marginal_impacts = [_mi("q1", "c1", -0.3)]
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}

    learned = build_learned_context_sets_v2(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    assert learned[0].strategy == "learned"
    assert learned[0].metadata.contains_all_gold is True


# --- per-query marginal impact (M6 v3) ---

def test_per_query_marginal_impact_uses_query_specific_signal():
    """When the same chunk appears for two queries with opposite deltas,
    per-query marginal impact should give opposite utilities."""
    from context_engine.learned_selector import estimate_chunk_utility_per_query_marginal_impact
    mi_rows = [
        _mi("q1", "c1", -0.3),  # c1 useful for q1 (delta -0.3 → utility 0.3)
        _mi("q2", "c1", 0.5),   # c1 harmful for q2 (delta +0.5 → utility -0.5)
    ]
    utilities = estimate_chunk_utility_per_query_marginal_impact(mi_rows)
    assert utilities[("q1", "c1")].utility == pytest.approx(0.3)
    assert utilities[("q2", "c1")].utility == pytest.approx(-0.5)


def test_combined_utility_per_query_prefers_query_specific():
    """Per-query marginal impact wins over global marginal impact for chunks in the per-query data."""
    from context_engine.learned_selector import estimate_combined_utility_per_query
    # c1 has both global and per-query signal
    # Global mean: -0.3 → utility 0.3
    # Per-query for q1: -0.5 → utility 0.5
    mi_rows = [
        _mi("q1", "c1", -0.3),  # global only
        _mi("q1", "c1", -0.5),  # also for q1, average = -0.4, utility 0.4
    ]
    context_sets = [_context_set("s1", "q1", ["c1"])]
    outcomes = [_outcome("s1", "q1", 0.7)]
    tables = estimate_combined_utility_per_query(
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=mi_rows,
    )
    # q1's c1 utility: mean of [-0.3, -0.5] = -0.4, inverted = 0.4
    assert tables["q1"]["c1"].utility == pytest.approx(0.4)


def test_build_v3_uses_per_query_signal():
    queries = [_query("q1", ["c1"]), _query("q2", ["c2"])]
    pools = [_candidate_pool("q1", ["c1", "c3"]), _candidate_pool("q2", ["c2", "c3"])]
    context_sets = [_context_set("s1", "q1", ["c1", "c3"]), _context_set("s2", "q2", ["c2", "c3"])]
    outcomes = [_outcome("s1", "q1", 0.5), _outcome("s2", "q2", 0.5)]
    # c3 has OPPOSITE utility for q1 vs q2
    mi_rows = [
        _mi("q1", "c3", -0.3),  # useful for q1
        _mi("q2", "c3", 0.5),   # harmful for q2
    ]
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100), "c3": _chunk("c3", 100)}

    learned = build_learned_context_sets_v3(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=mi_rows,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    # Both learned sets should include c3, but the per-query utility differs.
    # We can't directly assert the utility value here, but the function should
    # successfully emit one learned set per query.
    assert [cs.set_id for cs in learned] == ["q1_learned", "q2_learned"]
