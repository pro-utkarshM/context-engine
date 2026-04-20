from __future__ import annotations

from pathlib import Path

from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.evaluation import evaluate_context_set
from context_engine.io import load_jsonl, write_jsonl


def main() -> int:
    base = Path("data/processed")
    corpus_chunks = [CorpusChunk.from_dict(row) for row in load_jsonl(base / "corpus_chunks_v1.jsonl")]
    queries = [Query.from_dict(row) for row in load_jsonl(base / "queries_v1.jsonl")]
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(base / "context_sets_v1.jsonl")]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    queries_by_id = {query.query_id: query for query in queries}

    outcomes = [
        evaluate_context_set(
            query=queries_by_id[context_set.query_id],
            context_set=context_set,
            chunks_by_id=chunks_by_id,
        ).to_dict()
        for context_set in context_sets
    ]

    target = base / "outcomes_v1.jsonl"
    write_jsonl(target, outcomes)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
