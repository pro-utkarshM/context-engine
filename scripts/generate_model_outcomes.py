from __future__ import annotations

import argparse
import os
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.io import load_jsonl, write_jsonl
from context_engine.model_outcomes import evaluate_with_runner
from context_engine.runner import OpenAIResponsesRunner, StubModelRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_model_outcomes")
    parser.add_argument("--model", default=None, help="Model name to send to the runner.")
    parser.add_argument(
        "--runner",
        choices=("stub", "openai"),
        default="stub",
        help="Runner backend to use.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename. Defaults to outcomes_model_<runner>_v1.jsonl",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-5")
    base = Path("data/processed")
    corpus_chunks = [CorpusChunk.from_dict(row) for row in load_jsonl(base / "corpus_chunks_v1.jsonl")]
    queries = [Query.from_dict(row) for row in load_jsonl(base / "queries_v1.jsonl")]
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(base / "context_sets_v1.jsonl")]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    queries_by_id = {query.query_id: query for query in queries}
    runner = StubModelRunner() if args.runner == "stub" else OpenAIResponsesRunner()

    outcomes = [
        evaluate_with_runner(
            query=queries_by_id[context_set.query_id],
            context_set=context_set,
            chunks_by_id=chunks_by_id,
            runner=runner,
            model_name=model_name,
        ).to_dict()
        for context_set in context_sets
    ]

    target_name = args.output or f"outcomes_model_{args.runner}_v1.jsonl"
    target = base / target_name
    write_jsonl(target, outcomes)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
