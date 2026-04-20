"""Artifact file validation against the frozen benchmark contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .artifacts import CandidatePool, ContextSet, CorpusChunk, MarginalImpact, Outcome, Query
from .io import load_jsonl


ValidatorFn = Callable[[dict], object]


ARTIFACT_VALIDATORS: dict[str, ValidatorFn] = {
    "corpus_chunks": CorpusChunk.from_dict,
    "queries": Query.from_dict,
    "candidate_pools": CandidatePool.from_dict,
    "context_sets": ContextSet.from_dict,
    "outcomes": Outcome.from_dict,
    "marginal_impact": MarginalImpact.from_dict,
}


@dataclass(slots=True)
class ValidationSummary:
    artifact_name: str
    path: Path
    row_count: int


def infer_artifact_name(path: str | Path) -> str:
    stem = Path(path).name
    for artifact_name in ARTIFACT_VALIDATORS:
        prefix = f"{artifact_name}_"
        if stem.startswith(prefix) and stem.endswith(".jsonl"):
            return artifact_name
    raise ValueError(f"Could not infer artifact type from filename: {stem}")


def validate_jsonl_file(path: str | Path, artifact_name: str | None = None) -> ValidationSummary:
    resolved_path = Path(path)
    artifact_key = artifact_name or infer_artifact_name(resolved_path)
    validator = ARTIFACT_VALIDATORS[artifact_key]
    rows = load_jsonl(resolved_path)
    for row in rows:
        validator(row)
    return ValidationSummary(artifact_name=artifact_key, path=resolved_path, row_count=len(rows))
