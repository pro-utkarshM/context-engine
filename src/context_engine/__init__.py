"""Core package for the Context Engine benchmark and runtime contracts."""

from .analysis import QueryBestResult, StrategySummary, best_strategy_per_query, render_text_report, summarize_by_strategy
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
from .model_outcomes import evaluate_with_runner
from .prompting import PromptPayload, assemble_prompt
from .runner import ModelResponse, ModelRunner, OpenAIResponsesRunner, StubModelRunner
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
    "ModelResponse",
    "ModelRunner",
    "OpenAIResponsesRunner",
    "Outcome",
    "PromptPayload",
    "Query",
    "QueryBestResult",
    "QueryMetadata",
    "RetrievalComposition",
    "ScoreBundle",
    "StrategySummary",
    "StubModelRunner",
    "assemble_prompt",
    "best_strategy_per_query",
    "evaluate_with_runner",
    "load_jsonl",
    "make_candidate_pool",
    "make_context_set",
    "make_corpus_chunk",
    "make_marginal_impact",
    "make_outcome",
    "make_query",
    "render_text_report",
    "generate_context_set",
    "generate_context_sets",
    "generate_baseline_answer",
    "evaluate_context_set",
    "summarize_by_strategy",
    "ValidationSummary",
    "validate_jsonl_file",
    "write_jsonl",
    "ScoringWeights",
]
