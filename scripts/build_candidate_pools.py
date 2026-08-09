"""Build candidate pools from queries + corpus via the retrieval component.

Closes the docs/component-interface-spec.md gap: the v1 benchmark had no
Retriever implementation, so candidate pools were pre-baked into the
artifact. This script retrieves top-K candidates per query, classifies
them by role (gold / plausible / distractor), injects the gold chunk if
the retriever missed it, and writes a ``candidate_pools_v1.jsonl`` that
satisfies the data contract.

Pool composition mirrors the existing v1 artifact: 1 gold, 2 plausible,
2 distractor, 0 neutral (5 candidates total). The classification rules
are intentionally simple — gold comes from ``gold_support_ids``,
``pg15_*`` chunks are labelled stale, everything else is topical_wrong.
A future iteration can plug in the distractor decision tree from
``data/annotation/distractor_guidelines.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from context_engine.artifacts import (
    CandidatePool,
    CorpusChunk,
    Query,
)
from context_engine.authoring import make_candidate_pool
from context_engine.config import (
    add_config_args,
    config_from_args,
    resolved_artifact_path,
)
from context_engine.io import load_jsonl, write_jsonl
from context_engine.retrieval import BM25ExactMatchRetriever, BM25Retriever


DEFAULT_POOL_SIZE = 5
DEFAULT_PLAUSIBLE_COUNT = 2
DEFAULT_DISTRACTOR_COUNT = 2
DEFAULT_NEUTRAL_COUNT = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_candidate_pools")
    add_config_args(parser)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=f"Max candidates per pool (default: {DEFAULT_POOL_SIZE}).",
    )
    parser.add_argument(
        "--plausible-count",
        type=int,
        default=DEFAULT_PLAUSIBLE_COUNT,
        help="Number of plausible (non-gold, non-distractor) candidates per pool.",
    )
    parser.add_argument(
        "--distractor-count",
        type=int,
        default=DEFAULT_DISTRACTOR_COUNT,
        help="Number of distractor candidates per pool.",
    )
    parser.add_argument(
        "--neutral-count",
        type=int,
        default=DEFAULT_NEUTRAL_COUNT,
        help="Number of neutral candidates per pool.",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "bm25_exact"),
        default="bm25",
        help="Retriever to use. Default: bm25. "
        "bm25_exact is a hybrid BM25 + exact-phrase rerank.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Default: candidate_pools_<version>.jsonl under dataset_dir.",
    )
    return parser


def _classify(chunk_id: str, gold_id: str) -> str:
    """Classify a candidate by its role in the pool.

    Simple deterministic rules: gold (the query's gold support id),
    stale (a pg15 chunk when target is pg16), topical_wrong otherwise.
    Plausible / distractor / neutral counts are applied by the caller
    based on BM25 rank; this just labels the chunk for metadata.
    """
    if chunk_id == gold_id:
        return "gold"
    if chunk_id.startswith("pg15_"):
        return "stale"
    return "topical_wrong"


def build_pool_for_query(
    *,
    query: Query,
    artifact_version: str,
    retriever: BM25Retriever | BM25ExactMatchRetriever,
    chunks_by_id: dict[str, CorpusChunk],
    pool_size: int,
    plausible_count: int,
    distractor_count: int,
    neutral_count: int,
) -> CandidatePool:
    results = retriever.retrieve(query.query, pool_size=pool_size)
    retrieved_ids = [r.chunk_id for r in results]

    gold_ids = list(query.gold_support_ids or [])
    if not gold_ids:
        raise ValueError(
            f"query {query.query_id} has no gold_support_ids; cannot guarantee gold_in_pool"
        )

    gold_id = gold_ids[0]
    if gold_id not in retrieved_ids:
        retrieved_ids.insert(0, gold_id)

    candidate_ids = retrieved_ids[:pool_size]

    # Defend against the truncate step dropping gold.
    if gold_id not in candidate_ids:
        candidate_ids[-1] = gold_id

    candidate_metadata: dict[str, dict[str, str]] = {
        cid: {
            "role": _classify(cid, gold_id),
            "distractor_type": _classify(cid, gold_id),
        }
        for cid in candidate_ids
    }

    return make_candidate_pool(
        query_id=query.query_id,
        candidate_pool_id=f"pool_{query.query_id}_{artifact_version}",
        candidate_ids=candidate_ids,
        gold_count=1,
        plausible_count=plausible_count,
        distractor_count=distractor_count,
        neutral_count=neutral_count,
        gold_in_pool=gold_id in candidate_ids,
    )


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
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}

    if args.retriever == "bm25":
        retriever = BM25Retriever()
    elif args.retriever == "bm25_exact":
        retriever = BM25ExactMatchRetriever()
    else:
        raise SystemExit(f"unknown retriever: {args.retriever}")
    retriever.index(corpus_chunks)

    pools = [
        build_pool_for_query(
            query=query,
            artifact_version=config.artifact_version,
            retriever=retriever,
            chunks_by_id=chunks_by_id,
            pool_size=args.pool_size,
            plausible_count=args.plausible_count,
            distractor_count=args.distractor_count,
            neutral_count=args.neutral_count,
        )
        for query in queries
    ]

    target = (
        Path(args.output)
        if args.output
        else resolved_artifact_path(config, "candidate_pools")
    )
    write_jsonl(target, [pool.to_dict() for pool in pools])
    print(
        f"wrote {len(pools)} candidate pools "
        f"(pool_size={args.pool_size}, retriever={args.retriever}) -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())