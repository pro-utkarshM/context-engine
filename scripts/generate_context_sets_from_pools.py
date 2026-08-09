"""Run context-set generation against a custom candidate-pools path.

Tiny utility used for the r1 retrieval→evaluation live validation. Reads
queries + chunks from the canonical ``data/processed``, reads candidate
pools from a user-specified path (so we can drive the chain from
auto-built retrieval pools), and writes context sets to a user-specified
output path. The canonical ``generate_context_sets.py`` always reads
the canonical candidate-pools artifact; this entrypoint exists so
experiments can drive the chain end-to-end without overwriting it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from context_engine.artifacts import CandidatePool, CorpusChunk, Query
from context_engine.config import (
    add_config_args,
    config_from_args,
    resolved_artifact_path,
)
from context_engine.context_sets import generate_context_sets
from context_engine.io import load_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_context_sets_from_pools")
    add_config_args(parser)
    parser.add_argument(
        "--candidate-pools",
        required=True,
        help="Path to a candidate_pools JSONL artifact (overrides the canonical default).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <dataset_dir>/context_sets_<version>.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)

    chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "corpus_chunks"))
    ]
    queries = [
        Query.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "queries"))
    ]
    pools = [
        CandidatePool.from_dict(row) for row in load_jsonl(Path(args.candidate_pools))
    ]

    context_sets = generate_context_sets(
        queries=queries,
        candidate_pools=pools,
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
    )

    target = (
        Path(args.output)
        if args.output
        else resolved_artifact_path(config, "context_sets")
    )
    write_jsonl(target, [context_set.to_dict() for context_set in context_sets])
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())