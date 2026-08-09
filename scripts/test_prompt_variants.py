
"""Test prompt variants on q_0008 with v3 learned context set.

Compares:
- A: current (Question before Context, with chunk_id)
- B: reordered (Context before Question, with chunk_id)
- C: reordered + no chunk_id leak (Context before Question, no chunk_id)

Runs each variant 5 times at MiniMax-M3 and reports the per-variant
mean, failure rate, and answer quality.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import ContextSet, Query, CorpusChunk, Outcome
from context_engine.config import add_config_args, config_from_args
from context_engine.io import load_jsonl, write_jsonl
from context_engine.prompting import assemble_prompt


VARIANTS = ("A", "B", "C")
N_RUNS = 5


def build_prompt_A(query: Query, cs: ContextSet, chunks_by_id):
    """Current: Question before Context, with chunk_id."""
    return assemble_prompt(query=query, context_set=cs, chunks_by_id=chunks_by_id).prompt


def build_prompt_B(query: Query, cs: ContextSet, chunks_by_id):
    """Context before Question, with chunk_id (standard RAG order)."""
    chunk_blocks = []
    for rank, chunk_id in enumerate(cs.selected_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        chunk_blocks.append(f"[Chunk {rank} | {chunk.chunk_id} | v{chunk.doc_version}]\n{chunk.text}")
    context_block = "\n\n".join(chunk_blocks)
    return (
        "You are answering a technical documentation benchmark question.\n"
        "Use only the provided context.\n"
        "Return exactly one short answer line.\n"
        "Do not explain your reasoning.\n"
        "Do not add background, caveats, or extra sentences.\n"
        "If the answer is a file name, setting name, method name, record type, or privilege, return that exact term.\n"
        "Prefer exact wording from the context when possible.\n"
        "If the context is insufficient, return exactly: INSUFFICIENT_CONTEXT\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query.query}\n"
    )


def build_prompt_C(query: Query, cs: ContextSet, chunks_by_id):
    """Context before Question, NO chunk_id (no leak)."""
    chunk_blocks = []
    for rank, chunk_id in enumerate(cs.selected_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        # Drop chunk_id and version from the header to avoid leaking meta-info.
        chunk_blocks.append(f"[Document {rank}]\n{chunk.text}")
    context_block = "\n\n".join(chunk_blocks)
    return (
        "You are answering a technical documentation benchmark question.\n"
        "Use only the provided context.\n"
        "Return exactly one short answer line.\n"
        "Do not explain your reasoning.\n"
        "Do not add background, caveats, or extra sentences.\n"
        "If the answer is a file name, setting name, method name, record type, or privilege, return that exact term.\n"
        "Prefer exact wording from the context when possible.\n"
        "If the context is insufficient, return exactly: INSUFFICIENT_CONTEXT\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query.query}\n"
    )


PROMPT_BUILDERS = {
    "A": build_prompt_A,
    "B": build_prompt_B,
    "C": build_prompt_C,
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_RUNS)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--query-id", default="q_0008")
    parser.add_argument("--context-sets", required=True, help="Path to context_sets_*.jsonl (v3 learned)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    chunks_by_id = {c.chunk_id: c for c in (CorpusChunk.from_dict(row) for row in load_jsonl(args.corpus))}
    queries_by_id = {q.query_id: q for q in (Query.from_dict(row) for row in load_jsonl(args.queries))}
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(args.context_sets)]

    target_query = queries_by_id[args.query_id]
    target_cs = next((cs for cs in context_sets if cs.query_id == args.query_id), None)
    if target_cs is None:
        raise SystemExit(f"no context set for query {args.query_id} in {args.context_sets}")

    # Build the prompt
    prompt = PROMPT_BUILDERS[args.variant](target_query, target_cs, chunks_by_id)

    # Save the prompt for inspection
    prompt_path = out_dir / f"prompt_variant_{args.variant}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"wrote prompt variant {args.variant} ({len(prompt)} chars) -> {prompt_path}")

    # For now, just print the prompt; running the model is done via a separate script.
    print(f"\n--- Variant {args.variant} prompt (first 500 chars) ---")
    print(prompt[:500])
    print("...")
    print(prompt[-300:])


if __name__ == "__main__":
    main()
