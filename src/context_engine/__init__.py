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
from .authoring import (
    make_candidate_pool,
    make_context_set,
    make_corpus_chunk,
    make_marginal_impact,
    make_outcome,
    make_query,
)
from .context_sets import DEFAULT_STRATEGIES, GenerationStrategy, generate_context_set, generate_context_sets
from .dataset import BenchmarkDataset
from .evaluation import ScoringWeights, evaluate_context_set, generate_baseline_answer
from .io import load_jsonl, write_jsonl
from .validation import ValidationSummary, validate_jsonl_file

__all__ = [
    "ArtifactValidationError",
    "BenchmarkDataset",
    "CandidatePool",
    "ChunkMetadata",
    "ContextSet",
    "ContextSetMetadata",
    "CorpusChunk",
    "DEFAULT_STRATEGIES",
    "GenerationStrategy",
    "MarginalImpact",
    "Outcome",
    "Query",
    "QueryMetadata",
    "RetrievalComposition",
    "ScoreBundle",
    "load_jsonl",
    "make_candidate_pool",
    "make_context_set",
    "make_corpus_chunk",
    "make_marginal_impact",
    "make_outcome",
    "make_query",
    "generate_context_set",
    "generate_context_sets",
    "generate_baseline_answer",
    "evaluate_context_set",
    "ValidationSummary",
    "validate_jsonl_file",
    "write_jsonl",
    "ScoringWeights",
]
