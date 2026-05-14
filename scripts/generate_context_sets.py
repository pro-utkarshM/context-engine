from __future__ import annotations

import argparse

from context_engine.artifacts import CandidatePool, CorpusChunk, Query
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.context_sets import generate_context_sets
from context_engine.io import load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_context_sets")
    add_config_args(parser)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. Defaults to <dataset_dir>/context_sets_<version>.jsonl.",
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
    candidate_pools = [
        CandidatePool.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "candidate_pools"))
    ]

    context_sets = generate_context_sets(
        queries=queries,
        candidate_pools=candidate_pools,
        chunks_by_id={chunk.chunk_id: chunk for chunk in corpus_chunks},
    )

    target = args.output if args.output else resolved_artifact_path(config, "context_sets")
    write_jsonl(target, [context_set.to_dict() for context_set in context_sets])
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
