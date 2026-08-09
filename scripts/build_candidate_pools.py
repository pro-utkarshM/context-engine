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
from context_engine.retrieval import (
    BM25ExactMatchRetriever,
    BM25Retriever,
    RandomRetriever,
    Retriever,
)


def _load_retriever_from_module(spec: str) -> Retriever:
    """Load a retriever class from a ``<module>:<ClassName>`` spec.

    Resolves the module via ``importlib.import_module`` and pulls the
    named attribute. The class is instantiated with no arguments; if
    the contributor's class needs configuration, they should provide
    a no-arg constructor that picks up env vars / config files / etc.

    Validation:
      - spec must contain a single ``:`` separator
      - the resolved attribute must be a class with the Retriever
        Protocol shape (name + index + retrieve)
      - instantiation must succeed with no arguments
    """
    import importlib

    if ":" not in spec:
        raise SystemExit(
            f"--retriever-module expected '<module>:<ClassName>'; got {spec!r}"
        )
    module_name, _, class_name = spec.partition(":")
    module_name = module_name.strip()
    class_name = class_name.strip()
    if not module_name or not class_name:
        raise SystemExit(
            f"--retriever-module expected '<module>:<ClassName>'; got {spec!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"--retriever-module: cannot import {module_name!r}: {exc}")
    cls = getattr(module, class_name, None)
    if cls is None:
        raise SystemExit(
            f"--retriever-module: {module_name!r} has no attribute {class_name!r}"
        )
    for required in ("name", "index", "retrieve"):
        if not hasattr(cls, required):
            raise SystemExit(
                f"--retriever-module: {class_name!r} missing Protocol attribute {required!r}"
            )
    try:
        instance = cls()
    except TypeError as exc:
        raise SystemExit(
            f"--retriever-module: {class_name!r}() constructor raised: {exc}. "
            "The class must be instantiable with no arguments."
        )
    return instance


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
        choices=("bm25", "bm25_exact", "random"),
        default=None,
        help="Retriever to use. Default: bm25. "
        "bm25_exact is a hybrid BM25 + exact-phrase rerank. "
        "random is a uniform-random no-signal baseline (used to validate "
        "the extension mechanism in #13). Mutually exclusive with --retriever-module.",
    )
    parser.add_argument(
        "--retriever-module",
        default=None,
        help="Custom retriever via '<module>:<ClassName>'. Lets contributors "
        "register a custom retriever at runtime without modifying "
        "scripts/build_candidate_pools.py. The class must implement the "
        "Retriever Protocol (have index() and retrieve() methods, and a "
        "name attribute). Mutually exclusive with --retriever.",
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
    """Build the candidate pool for a single query.

    Algorithm:
      1. Retrieve top-K candidates from the retriever.
      2. Ensure ALL gold chunks are in the pool (multi-hop safe).
         - Insert each missing gold chunk at the front, in gold_id order.
         - If the list exceeds pool_size after gold insertion, truncate
           from the END (drop distractors, not gold).
      3. If gold_in_pool still fails (e.g., pool_size < number of gold
         chunks), raise a clear error.
    """
    results = retriever.retrieve(query.query, pool_size=pool_size)
    retrieved_ids = [r.chunk_id for r in results]

    gold_ids = list(query.gold_support_ids or [])
    if not gold_ids:
        raise ValueError(
            f"query {query.query_id} has no gold_support_ids; cannot guarantee gold_in_pool"
        )

    # Insert all gold chunks at the front, in order, deduping.
    pool = list(retrieved_ids)
    for gold_id in gold_ids:
        if gold_id not in pool:
            pool.insert(0, gold_id)

    # Truncate to pool_size. If truncating would drop a gold chunk,
    # raise an error rather than silently violate the contract.
    if len(gold_ids) > pool_size:
        raise ValueError(
            f"query {query.query_id} has {len(gold_ids)} gold chunks, "
            f"but pool_size={pool_size}; cannot fit all gold chunks"
        )
    candidate_ids = pool[:pool_size]
    for gold_id in gold_ids:
        if gold_id not in candidate_ids:
            raise ValueError(
                f"query {query.query_id}: gold chunk {gold_id} dropped "
                f"from pool after truncation; this should be impossible. "
                f"pool={candidate_ids}"
            )

    # Use the first gold id as the "primary" gold for classification.
    primary_gold_id = gold_ids[0]
    candidate_metadata: dict[str, dict[str, str]] = {
        cid: {
            "role": _classify(cid, primary_gold_id),
            "distractor_type": _classify(cid, primary_gold_id),
        }
        for cid in candidate_ids
    }

    return make_candidate_pool(
        query_id=query.query_id,
        candidate_pool_id=f"pool_{query.query_id}_{artifact_version}",
        candidate_ids=candidate_ids,
        gold_count=len(gold_ids),
        plausible_count=plausible_count,
        distractor_count=distractor_count,
        neutral_count=neutral_count,
        gold_in_pool=all(g in candidate_ids for g in gold_ids),
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

    if args.retriever_module and args.retriever is not None:
        # Both flags explicitly set => ambiguous dispatch.
        raise SystemExit(
            "--retriever-module is mutually exclusive with --retriever"
        )
    if args.retriever_module:
        retriever = _load_retriever_from_module(args.retriever_module)
    elif args.retriever is None or args.retriever == "bm25":
        retriever = BM25Retriever()
    elif args.retriever == "bm25_exact":
        retriever = BM25ExactMatchRetriever()
    elif args.retriever == "random":
        retriever = RandomRetriever()
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
    retriever_label = args.retriever_module or retriever.name
    print(
        f"wrote {len(pools)} candidate pools "
        f"(pool_size={args.pool_size}, retriever={retriever_label}) -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())