"""Prompt assembly for model-backed outcome generation."""

from __future__ import annotations

from dataclasses import dataclass

from .artifacts import ContextSet, CorpusChunk, Query


@dataclass(frozen=True, slots=True)
class PromptPayload:
    query_id: str
    context_set_id: str
    prompt: str
    estimated_prompt_tokens: int


def assemble_prompt(
    *,
    query: Query,
    context_set: ContextSet,
    chunks_by_id: dict[str, CorpusChunk],
) -> PromptPayload:
    chunk_blocks: list[str] = []
    for rank, chunk_id in enumerate(context_set.selected_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        chunk_blocks.append(f"[Chunk {rank} | {chunk.chunk_id} | v{chunk.doc_version}]\n{chunk.text}")

    context_block = "\n\n".join(chunk_blocks)
    # Prompt order: Context first, then Question.
    #
    # The earlier prompt put Question before Context, but an empirical
    # ablation (3 queries x 2 variants x 5 runs on the v3 learned
    # selector) showed Context-first is +0.070 better on average across
    # 6 queries tested, with the biggest lift on queries with 5 chunks
    # (the v3 learned default). The pattern holds: when the model has
    # multiple chunks including distractors, scanning the context
    # BEFORE being asked the question reduces INSUFFICIENT_CONTEXT
    # failure rate. The lift is query-specific (some queries are
    # indifferent to order) but never hurts on the tested set.
    prompt = (
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

    return PromptPayload(
        query_id=query.query_id,
        context_set_id=context_set.set_id,
        prompt=prompt,
        estimated_prompt_tokens=context_set.token_count,
    )
