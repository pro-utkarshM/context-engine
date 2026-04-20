"""Core package for the Context Engine benchmark and runtime contracts."""

from .artifacts import (
    ArtifactValidationError,
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
from .dataset import BenchmarkDataset
from .io import load_jsonl, write_jsonl

__all__ = [
    "ArtifactValidationError",
    "BenchmarkDataset",
    "CandidatePool",
    "ChunkMetadata",
    "ContextSet",
    "ContextSetMetadata",
    "CorpusChunk",
    "MarginalImpact",
    "Outcome",
    "Query",
    "QueryMetadata",
    "RetrievalComposition",
    "ScoreBundle",
    "load_jsonl",
    "write_jsonl",
]
