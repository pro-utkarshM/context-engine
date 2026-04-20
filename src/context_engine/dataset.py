"""Dataset loaders for versioned benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    CandidatePool,
    ContextSet,
    CorpusChunk,
    MarginalImpact,
    Outcome,
    Query,
)
from .io import load_jsonl


@dataclass(slots=True)
class BenchmarkDataset:
    corpus_chunks: list[CorpusChunk]
    queries: list[Query]
    candidate_pools: list[CandidatePool]
    context_sets: list[ContextSet]
    outcomes: list[Outcome]
    marginal_impact: list[MarginalImpact]

    @classmethod
    def from_directory(cls, directory: str | Path, version: str = "v1") -> "BenchmarkDataset":
        base = Path(directory)
        return cls(
            corpus_chunks=[CorpusChunk.from_dict(row) for row in load_jsonl(base / f"corpus_chunks_{version}.jsonl")],
            queries=[Query.from_dict(row) for row in load_jsonl(base / f"queries_{version}.jsonl")],
            candidate_pools=[
                CandidatePool.from_dict(row) for row in load_jsonl(base / f"candidate_pools_{version}.jsonl")
            ],
            context_sets=[ContextSet.from_dict(row) for row in load_jsonl(base / f"context_sets_{version}.jsonl")],
            outcomes=[Outcome.from_dict(row) for row in load_jsonl(base / f"outcomes_{version}.jsonl")],
            marginal_impact=[
                MarginalImpact.from_dict(row) for row in load_jsonl(base / f"marginal_impact_{version}.jsonl")
            ],
        )

    def chunk_by_id(self) -> dict[str, CorpusChunk]:
        return {chunk.chunk_id: chunk for chunk in self.corpus_chunks}

    def query_by_id(self) -> dict[str, Query]:
        return {query.query_id: query for query in self.queries}

    def candidate_pool_by_id(self) -> dict[str, CandidatePool]:
        return {pool.candidate_pool_id: pool for pool in self.candidate_pools}

    def context_set_by_id(self) -> dict[str, ContextSet]:
        return {context_set.set_id: context_set for context_set in self.context_sets}
