"""Core package for the Context Engine benchmark and runtime contracts."""

from .analysis import (
    PerSetRow,
    QueryBestResult,
    StrategySummary,
    best_strategy_per_query,
    per_set_rows,
    render_csv_per_query,
    render_json_report,
    render_markdown_report,
    render_text_report,
    summarize_by_strategy,
)
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
from .context_sets import (
    DEFAULT_STRATEGIES,
    GenerationStrategy,
    generate_context_set,
    generate_context_sets,
)
from .dataset import BenchmarkDataset
from .env import load_dotenv
from .evaluation import (
    ScoringWeights,
    evaluate_context_set,
    generate_baseline_answer,
)
from .io import load_jsonl, write_jsonl
from .learned_selector import (
    ChunkUtility,
    SelectedChunk,
    build_learned_context_set,
    build_learned_context_sets,
    estimate_chunk_utility,
    select_with_budget,
)
from .marginal_impact import (
    MarginalImpactError,
    Operation,
    ScoreKey,
    compute_marginal_impact,
    evaluate_marginal_impact,
)
from .model_outcomes import evaluate_with_runner
from .replications import (
    ReplicationPairedDelta,
    ReplicationStrategySummary,
    ReplicationSummary,
    paired_summary,
    summarize_replications,
)
from .retrieval import BM25ExactMatchRetriever, BM25Retriever, RetrievalResult, Retriever, tokenize
from .stats import (
    DistributionSummary,
    PairedDeltaSummary,
    summarize_distribution,
    summarize_paired_delta,
)
from .prompting import PromptPayload, assemble_prompt
from .runner import (
    MINIMAX_DEFAULT_MODEL,
    MiniMaxResponsesRunner,
    ModelResponse,
    ModelRunner,
    OpenAIResponsesRunner,
    StubModelRunner,
)
from .validation import ValidationSummary, validate_jsonl_file

__all__ = [
    "ArtifactValidationError",
    "BenchmarkDataset",
    "BM25ExactMatchRetriever",
    "BM25Retriever",
    "CandidatePool",
    "ChunkMetadata",
    "ChunkUtility",
    "ContextSet",
    "ContextSetMetadata",
    "CorpusChunk",
    "DEFAULT_STRATEGIES",
    "GenerationStrategy",
    "MINIMAX_DEFAULT_MODEL",
    "MarginalImpact",
    "MarginalImpactError",
    "MiniMaxResponsesRunner",
    "ModelResponse",
    "ModelRunner",
    "OpenAIResponsesRunner",
    "Operation",
    "Outcome",
    "PerSetRow",
    "PromptPayload",
    "Query",
    "QueryBestResult",
    "QueryMetadata",
    "RetrievalComposition",
    "RetrievalResult",
    "Retriever",
    "ScoreBundle",
    "ScoreKey",
    "SelectedChunk",
    "StrategySummary",
    "StubModelRunner",
    "assemble_prompt",
    "best_strategy_per_query",
    "build_learned_context_set",
    "build_learned_context_sets",
    "compute_marginal_impact",
    "evaluate_marginal_impact",
    "evaluate_with_runner",
    "evaluate_context_set",
    "estimate_chunk_utility",
    "generate_baseline_answer",
    "generate_context_set",
    "generate_context_sets",
    "load_dotenv",
    "load_jsonl",
    "make_candidate_pool",
    "make_context_set",
    "make_corpus_chunk",
    "make_marginal_impact",
    "make_outcome",
    "make_query",
    "per_set_rows",
    "render_csv_per_query",
    "render_json_report",
    "render_markdown_report",
    "render_text_report",
    "select_with_budget",
    "summarize_by_strategy",
    "tokenize",
    "validate_jsonl_file",
    "write_jsonl",
    "ScoringWeights",
    "ValidationSummary",
]