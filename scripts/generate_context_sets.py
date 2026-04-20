from __future__ import annotations

from pathlib import Path

from context_engine.context_sets import generate_context_sets
from context_engine.io import write_jsonl
from context_engine.io import load_jsonl
from context_engine.artifacts import CandidatePool, CorpusChunk, Query


def main() -> int:
    base = Path("data/processed")
    corpus_chunks = [CorpusChunk.from_dict(row) for row in load_jsonl(base / "corpus_chunks_v1.jsonl")]
    queries = [Query.from_dict(row) for row in load_jsonl(base / "queries_v1.jsonl")]
    candidate_pools = [CandidatePool.from_dict(row) for row in load_jsonl(base / "candidate_pools_v1.jsonl")]
    context_sets = generate_context_sets(
        queries=queries,
        candidate_pools=candidate_pools,
        chunks_by_id={chunk.chunk_id: chunk for chunk in corpus_chunks},
    )
    target = base / "context_sets_v1.jsonl"
    write_jsonl(target, [context_set.to_dict() for context_set in context_sets])
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
