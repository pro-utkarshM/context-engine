from __future__ import annotations

import argparse

from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.evaluation import evaluate_context_set
from context_engine.io import load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_outcomes")
    add_config_args(parser)
    parser.add_argument(
        "--context-sets",
        default=None,
        help="Optional explicit input context-sets filename (under dataset_dir).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. Defaults to <dataset_dir>/outcomes_<version>.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)

    corpus_chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "corpus_chunks"))
    ]
    queries = [
        Query.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "queries"))
    ]

    context_sets_path = (
        config.dataset_dir / args.context_sets
        if args.context_sets
        else resolved_artifact_path(config, "context_sets")
    )
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(context_sets_path)]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    queries_by_id = {query.query_id: query for query in queries}

    outcomes = [
        evaluate_context_set(
            query=queries_by_id[context_set.query_id],
            context_set=context_set,
            chunks_by_id=chunks_by_id,
            weights=config.scoring_weights,
            max_token_budget=config.token_budget,
        ).to_dict()
        for context_set in context_sets
    ]

    target = args.output if args.output else resolved_artifact_path(config, "outcomes")
    write_jsonl(target, outcomes)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
