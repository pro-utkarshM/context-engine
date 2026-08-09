"""Prompt assembly for model-backed outcome generation.

This module exposes three prompt-assembly policies:

- ``question_first`` (the original prompt order): instructions, then
  Question, then Context. Identical to the v1 default prior to the
  r4 prompt ablation. Used as the "Question-first" arm of the
  prompt-policy ablation.

- ``context_first`` (the r4 default): instructions, then Context, then
  Question. Empirically better for the v3 learned selector on the v1
  corpus; shipped in PR #19. Used as the "Context-first" arm of the
  prompt-policy ablation.

- ``adaptive_by_chunk_count``: choose Question-first if the context
  has fewer than the threshold chunks, otherwise Context-first.
  Threshold is configurable (default 2). Used as the "adaptive" arm
  of the prompt-policy ablation.

The three policies are intentionally separate functions so the
ablation experiment can pin each one to the same input context and
isolate the prompt-presentation effect. The default (when no policy
is specified) is ``context_first`` to preserve the r4 baseline.

The policy only changes the prompt presentation (ordering of
Question vs Context). It does NOT reorder the selected chunks
internally; the chunk order is determined by the upstream selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .artifacts import ContextSet, CorpusChunk, Query


#: Threshold for the adaptive policy. Chunks at or below this number
#: use Question-first; chunks above this number use Context-first.
ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT = 2


@dataclass(frozen=True, slots=True)
class PromptPayload:
    query_id: str
    context_set_id: str
    prompt: str
    estimated_prompt_tokens: int
    policy: str  # which policy produced this prompt


@dataclass(frozen=True, slots=True)
class ChunkBlock:
    """A single chunk rendered as a numbered block.

    The chunk ordering is intentionally unchanged across policies:
    the policy only reorders the (Question, Context) blocks at the
    outer prompt level, not the chunks within the Context block.
    """

    rank: int
    chunk_id: str
    doc_version: int
    text: str


# ---- Shared prologues --------------------------------------------------------

_INSTRUCTIONS_PROLOGUE = (
    "You are answering a technical documentation benchmark question.\n"
    "Use only the provided context.\n"
    "Return exactly one short answer line.\n"
    "Do not explain your reasoning.\n"
    "Do not add background, caveats, or extra sentences.\n"
    "If the answer is a file name, setting name, method name, record type, or privilege, return that exact term.\n"
    "Prefer exact wording from the context when possible.\n"
    "If the context is insufficient, return exactly: INSUFFICIENT_CONTEXT\n\n"
)


def _render_chunks(context_set: ContextSet, chunks_by_id: Mapping[str, CorpusChunk]) -> tuple[list[ChunkBlock], str]:
    blocks: list[ChunkBlock] = []
    for rank, chunk_id in enumerate(context_set.selected_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        blocks.append(
            ChunkBlock(
                rank=rank,
                chunk_id=chunk.chunk_id,
                doc_version=chunk.doc_version,
                text=chunk.text,
            )
        )
    context_block = "\n\n".join(
        f"[Chunk {b.rank} | {b.chunk_id} | v{b.doc_version}]\n{b.text}" for b in blocks
    )
    return blocks, context_block


# ---- Policy implementations --------------------------------------------------


def _question_first_prompt(query: Query, context_set: ContextSet, chunks_by_id: Mapping[str, CorpusChunk]) -> str:
    """Original prompt order: instructions, Question, Context."""

    _, context_block = _render_chunks(context_set, chunks_by_id)
    return (
        _INSTRUCTIONS_PROLOGUE
        + f"Question:\n{query.query}\n\n"
        + f"Context:\n{context_block}\n"
    )


def _context_first_prompt(query: Query, context_set: ContextSet, chunks_by_id: Mapping[str, CorpusChunk]) -> str:
    """r4 default: instructions, Context, Question."""

    _, context_block = _render_chunks(context_set, chunks_by_id)
    return (
        _INSTRUCTIONS_PROLOGUE
        + f"Context:\n{context_block}\n\n"
        + f"Question:\n{query.query}\n"
    )


def _adaptive_prompt(
    query: Query,
    context_set: ContextSet,
    chunks_by_id: Mapping[str, CorpusChunk],
    *,
    threshold: int = ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT,
) -> str:
    """Pick Question-first for short contexts, Context-first otherwise.

    The threshold is applied as: ``chunk_count <= threshold`` -> Question-first.
    With the default threshold of 2, contexts with 1 or 2 chunks use
    Question-first; contexts with 3+ chunks use Context-first.
    """

    chunk_count = len(context_set.selected_ids)
    if chunk_count <= threshold:
        return _question_first_prompt(query, context_set, chunks_by_id)
    return _context_first_prompt(query, context_set, chunks_by_id)


# ---- Policy dispatcher -------------------------------------------------------

#: Mapping of policy name to (callable, docstring). The dispatcher is
#: explicit (no eval) so the policy surface is grep-able and the
#: ablation runner can iterate over ``PromptPolicy.registry``.
POLICY_REGISTRY: dict[str, Callable[..., str]] = {
    "question_first": _question_first_prompt,
    "context_first": _context_first_prompt,
    "adaptive_by_chunk_count": _adaptive_prompt,
}


def assemble_prompt(
    *,
    query: Query,
    context_set: ContextSet,
    chunks_by_id: Mapping[str, CorpusChunk],
    policy: str = "context_first",
    adaptive_threshold: int = ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT,
) -> PromptPayload:
    """Build the prompt for the given query, context set, and policy.

    ``policy`` selects one of the entries in ``POLICY_REGISTRY``.
    The default is ``context_first`` (the r4 baseline).
    """

    if policy not in POLICY_REGISTRY:
        raise ValueError(
            f"unknown prompt policy: {policy!r}; valid options: {sorted(POLICY_REGISTRY)}"
        )

    if policy == "adaptive_by_chunk_count":
        prompt = _adaptive_prompt(
            query,
            context_set,
            chunks_by_id,
            threshold=adaptive_threshold,
        )
    else:
        prompt = POLICY_REGISTRY[policy](query, context_set, chunks_by_id)

    return PromptPayload(
        query_id=query.query_id,
        context_set_id=context_set.set_id,
        prompt=prompt,
        estimated_prompt_tokens=context_set.token_count,
        policy=policy,
    )


def policy_for_chunk_count(policy: str, chunk_count: int, *, threshold: int = ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT) -> str:
    """Return the CONCRETE policy the adaptive dispatcher would pick.

    For ``policy == "adaptive_by_chunk_count"`` this returns
    ``"question_first"`` if ``chunk_count <= threshold`` else
    ``"context_first"``. For all other policies it returns the input
    unchanged. Useful for the ablation harness to know which arm each
    query falls into.
    """

    if policy == "adaptive_by_chunk_count":
        return "question_first" if chunk_count <= threshold else "context_first"
    return policy
