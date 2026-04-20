"""Typed benchmark artifacts defined by docs/data-contract.md."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ArtifactValidationError(ValueError):
    """Raised when a benchmark artifact violates the frozen data contract."""


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ArtifactValidationError(f"{field_name} must be an integer")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactValidationError(f"{field_name} must be a boolean")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field_name} must be numeric")
    return float(value)


def _require_list_of_str(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ArtifactValidationError(f"{field_name} must be a list of non-empty strings")
    return value


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{field_name} must be an object")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _score_in_unit_interval(value: Any, field_name: str) -> float:
    score = _require_number(value, field_name)
    if score < 0.0 or score > 1.0:
        raise ArtifactValidationError(f"{field_name} must be in [0.0, 1.0]")
    return score


@dataclass(slots=True)
class ChunkMetadata:
    topic: str
    subtopic: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkMetadata":
        data = _require_object(payload, "metadata")
        topic = _require_str(data.get("topic"), "metadata.topic")
        subtopic = _optional_str(data.get("subtopic"), "metadata.subtopic")
        return cls(topic=topic, subtopic=subtopic)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorpusChunk:
    chunk_id: str
    doc_version: str
    doc_path: str
    section_path: list[str]
    source_type: str
    text: str
    token_count: int
    chunk_index: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    metadata: ChunkMetadata

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusChunk":
        data = _require_object(payload, "corpus_chunk")
        return cls(
            chunk_id=_require_str(data.get("chunk_id"), "chunk_id"),
            doc_version=_require_str(data.get("doc_version"), "doc_version"),
            doc_path=_require_str(data.get("doc_path"), "doc_path"),
            section_path=_require_list_of_str(data.get("section_path"), "section_path"),
            source_type=_require_str(data.get("source_type"), "source_type"),
            text=_require_str(data.get("text"), "text"),
            token_count=_require_int(data.get("token_count"), "token_count"),
            chunk_index=_require_int(data.get("chunk_index"), "chunk_index"),
            prev_chunk_id=_optional_str(data.get("prev_chunk_id"), "prev_chunk_id"),
            next_chunk_id=_optional_str(data.get("next_chunk_id"), "next_chunk_id"),
            metadata=ChunkMetadata.from_dict(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(slots=True)
class QueryMetadata:
    topic: str
    requires_multi_hop: bool
    question_family: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryMetadata":
        data = _require_object(payload, "metadata")
        return cls(
            topic=_require_str(data.get("topic"), "metadata.topic"),
            requires_multi_hop=_require_bool(data.get("requires_multi_hop"), "metadata.requires_multi_hop"),
            question_family=_require_str(data.get("question_family"), "metadata.question_family"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Query:
    query_id: str
    query: str
    task_type: str
    difficulty: str
    gold_answer: str
    gold_support_ids: list[str] | None
    metadata: QueryMetadata

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Query":
        data = _require_object(payload, "query")
        gold_support_ids = data.get("gold_support_ids")
        if gold_support_ids is not None:
            gold_support_ids = _require_list_of_str(gold_support_ids, "gold_support_ids")
        return cls(
            query_id=_require_str(data.get("query_id"), "query_id"),
            query=_require_str(data.get("query"), "query"),
            task_type=_require_str(data.get("task_type"), "task_type"),
            difficulty=_require_str(data.get("difficulty"), "difficulty"),
            gold_answer=_require_str(data.get("gold_answer"), "gold_answer"),
            gold_support_ids=gold_support_ids,
            metadata=QueryMetadata.from_dict(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(slots=True)
class RetrievalComposition:
    gold_count: int
    plausible_count: int
    distractor_count: int
    neutral_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalComposition":
        data = _require_object(payload, "composition")
        return cls(
            gold_count=_require_int(data.get("gold_count"), "composition.gold_count"),
            plausible_count=_require_int(data.get("plausible_count"), "composition.plausible_count"),
            distractor_count=_require_int(data.get("distractor_count"), "composition.distractor_count"),
            neutral_count=_require_int(data.get("neutral_count"), "composition.neutral_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidatePool:
    query_id: str
    candidate_pool_id: str
    candidate_ids: list[str]
    composition: RetrievalComposition
    gold_in_pool: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidatePool":
        data = _require_object(payload, "candidate_pool")
        return cls(
            query_id=_require_str(data.get("query_id"), "query_id"),
            candidate_pool_id=_require_str(data.get("candidate_pool_id"), "candidate_pool_id"),
            candidate_ids=_require_list_of_str(data.get("candidate_ids"), "candidate_ids"),
            composition=RetrievalComposition.from_dict(data.get("composition")),
            gold_in_pool=_require_bool(data.get("gold_in_pool"), "gold_in_pool"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["composition"] = self.composition.to_dict()
        return payload


@dataclass(slots=True)
class ContextSetMetadata:
    contains_all_gold: bool
    missing_gold_count: int
    distractor_types: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextSetMetadata":
        data = _require_object(payload, "metadata")
        return cls(
            contains_all_gold=_require_bool(data.get("contains_all_gold"), "metadata.contains_all_gold"),
            missing_gold_count=_require_int(data.get("missing_gold_count"), "metadata.missing_gold_count"),
            distractor_types=_require_list_of_str(data.get("distractor_types"), "metadata.distractor_types"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextSet:
    set_id: str
    query_id: str
    candidate_pool_id: str
    strategy: str
    selected_ids: list[str]
    ordering_type: str
    token_count: int
    metadata: ContextSetMetadata

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextSet":
        data = _require_object(payload, "context_set")
        return cls(
            set_id=_require_str(data.get("set_id"), "set_id"),
            query_id=_require_str(data.get("query_id"), "query_id"),
            candidate_pool_id=_require_str(data.get("candidate_pool_id"), "candidate_pool_id"),
            strategy=_require_str(data.get("strategy"), "strategy"),
            selected_ids=_require_list_of_str(data.get("selected_ids"), "selected_ids"),
            ordering_type=_require_str(data.get("ordering_type"), "ordering_type"),
            token_count=_require_int(data.get("token_count"), "token_count"),
            metadata=ContextSetMetadata.from_dict(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(slots=True)
class ScoreBundle:
    correctness: float
    support: float
    overall: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreBundle":
        data = _require_object(payload, "scores")
        return cls(
            correctness=_score_in_unit_interval(data.get("correctness"), "scores.correctness"),
            support=_score_in_unit_interval(data.get("support"), "scores.support"),
            overall=_score_in_unit_interval(data.get("overall"), "scores.overall"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Outcome:
    set_id: str
    query_id: str
    answer: str
    scores: ScoreBundle
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    evaluator_version: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Outcome":
        data = _require_object(payload, "outcome")
        return cls(
            set_id=_require_str(data.get("set_id"), "set_id"),
            query_id=_require_str(data.get("query_id"), "query_id"),
            answer=_require_str(data.get("answer"), "answer"),
            scores=ScoreBundle.from_dict(data.get("scores")),
            prompt_tokens=_require_int(data.get("prompt_tokens"), "prompt_tokens"),
            completion_tokens=_require_int(data.get("completion_tokens"), "completion_tokens"),
            latency_ms=_require_int(data.get("latency_ms"), "latency_ms"),
            evaluator_version=_require_str(data.get("evaluator_version"), "evaluator_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = self.scores.to_dict()
        return payload


@dataclass(slots=True)
class MarginalImpact:
    query_id: str
    base_set_id: str
    chunk_id: str
    operation: str
    base_score: float
    new_score: float
    delta: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarginalImpact":
        data = _require_object(payload, "marginal_impact")
        operation = _require_str(data.get("operation"), "operation")
        if operation not in {"add", "remove"}:
            raise ArtifactValidationError("operation must be one of 'add' or 'remove'")

        base_score = _require_number(data.get("base_score"), "base_score")
        new_score = _require_number(data.get("new_score"), "new_score")
        delta = _require_number(data.get("delta"), "delta")
        expected_delta = round(new_score - base_score, 10)
        if abs(delta - expected_delta) > 1e-9:
            raise ArtifactValidationError("delta must equal new_score - base_score")

        return cls(
            query_id=_require_str(data.get("query_id"), "query_id"),
            base_set_id=_require_str(data.get("base_set_id"), "base_set_id"),
            chunk_id=_require_str(data.get("chunk_id"), "chunk_id"),
            operation=operation,
            base_score=base_score,
            new_score=new_score,
            delta=delta,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
