"""Baseline answer generation and scoring for benchmark bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .artifacts import ContextSet, CorpusChunk, Outcome, Query
from .authoring import make_outcome


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    correctness: float = 0.6
    support: float = 0.3
    efficiency: float = 0.1


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
TOKEN_PATTERN = re.compile(r"[a-z0-9_./-]+")


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1}


def _score_sentence(query: Query, sentence: str, chunk_id: str, gold_ids: set[str], rank: int) -> tuple[float, int]:
    query_tokens = _tokenize(query.query)
    sentence_tokens = _tokenize(sentence)
    overlap = len(query_tokens.intersection(sentence_tokens))
    gold_bonus = 2.0 if chunk_id in gold_ids else 0.0
    # Prefer earlier selected chunks on ties while still allowing later better matches to win.
    rank_penalty = rank * 0.01
    return (overlap + gold_bonus - rank_penalty, overlap)


def generate_baseline_answer(
    query: Query,
    context_set: ContextSet,
    chunks_by_id: dict[str, CorpusChunk],
) -> str:
    if not context_set.selected_ids:
        return ""

    gold_ids = set(query.gold_support_ids or [])
    best_sentence = ""
    best_score = float("-inf")
    best_overlap = -1

    for rank, chunk_id in enumerate(context_set.selected_ids):
        chunk = chunks_by_id[chunk_id]
        for sentence in _split_sentences(chunk.text):
            score, overlap = _score_sentence(query, sentence, chunk_id, gold_ids, rank)
            if score > best_score or (score == best_score and overlap > best_overlap):
                best_sentence = sentence
                best_score = score
                best_overlap = overlap

    return best_sentence


def score_correctness(query: Query, answer: str) -> float:
    normalized_answer = answer.strip().lower()
    normalized_gold = query.gold_answer.strip().lower()
    if not normalized_answer:
        return 0.0
    if normalized_answer == normalized_gold:
        return 1.0
    if normalized_answer in normalized_gold or normalized_gold in normalized_answer:
        return 0.7
    return 0.0


def score_support(query: Query, context_set: ContextSet) -> float:
    gold_ids = set(query.gold_support_ids or [])
    if not gold_ids:
        return 0.0
    selected_ids = set(context_set.selected_ids)
    overlap = len(gold_ids.intersection(selected_ids))
    return overlap / len(gold_ids)


def score_efficiency(context_set: ContextSet, max_token_budget: int) -> float:
    if max_token_budget <= 0:
        return 0.0
    ratio = min(context_set.token_count / max_token_budget, 1.0)
    return 1.0 - ratio


def evaluate_context_set(
    *,
    query: Query,
    context_set: ContextSet,
    chunks_by_id: dict[str, CorpusChunk],
    weights: ScoringWeights = ScoringWeights(),
    evaluator_version: str = "eval_v1_rule_based",
    max_token_budget: int = 1500,
) -> Outcome:
    answer = generate_baseline_answer(query, context_set, chunks_by_id)
    correctness = score_correctness(query, answer)
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
        answer=answer,
        correctness=correctness,
        support=support,
        overall=min(max(overall, 0.0), 1.0),
        prompt_tokens=context_set.token_count,
        completion_tokens=max(len(answer.split()), 1) if answer else 0,
        latency_ms=0,
        evaluator_version=evaluator_version,
    )
