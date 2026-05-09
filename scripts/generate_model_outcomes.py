from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.io import load_jsonl, write_jsonl
from context_engine.model_outcomes import evaluate_with_runner
from context_engine.runner import OpenAIResponsesRunner, StubModelRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_model_outcomes")
    parser.add_argument("--model", default=None, help="Model name to send to the runner.")
    parser.add_argument(
        "--runner",
        choices=("stub", "openai"),
        default="stub",
        help="Runner backend to use.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename. Defaults to outcomes_model_<runner>_v1.jsonl",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior and regenerate all outcomes from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pending context sets after filtering/resume.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Restrict execution to one or more query IDs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--set-id",
        action="append",
        default=None,
        help="Restrict execution to one or more exact context set IDs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Skip pending sets until this set_id is reached, then continue from there.",
    )
    return parser


def _load_existing_outcomes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _filter_context_sets(
    context_sets: list[ContextSet],
    *,
    query_ids: set[str] | None,
    set_ids: set[str] | None,
    start_at: str | None,
    limit: int | None,
) -> list[ContextSet]:
    filtered = context_sets
    if query_ids:
        filtered = [context_set for context_set in filtered if context_set.query_id in query_ids]
    if set_ids:
        filtered = [context_set for context_set in filtered if context_set.set_id in set_ids]
    if start_at:
        seen = False
        resumed: list[ContextSet] = []
        for context_set in filtered:
            if context_set.set_id == start_at:
                seen = True
            if seen:
                resumed.append(context_set)
        filtered = resumed
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-5")
    base = Path("data/processed")
    corpus_chunks = [CorpusChunk.from_dict(row) for row in load_jsonl(base / "corpus_chunks_v1.jsonl")]
    queries = [Query.from_dict(row) for row in load_jsonl(base / "queries_v1.jsonl")]
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(base / "context_sets_v1.jsonl")]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    queries_by_id = {query.query_id: query for query in queries}
    runner = StubModelRunner() if args.runner == "stub" else OpenAIResponsesRunner()

    target_name = args.output or f"outcomes_model_{args.runner}_v1.jsonl"
    target = base / target_name
    if args.no_resume and target.exists():
        print(
            f"warning: --no-resume will restart from scratch and overwrite progress in {target}",
            file=sys.stderr,
        )
    existing_rows = [] if args.no_resume else _load_existing_outcomes(target)
    completed_set_ids = {row["set_id"] for row in existing_rows}
    outcomes = list(existing_rows)

    pending_context_sets = [
        context_set for context_set in context_sets if context_set.set_id not in completed_set_ids
    ]
    pending_context_sets = _filter_context_sets(
        pending_context_sets,
        query_ids=set(args.query_id) if args.query_id else None,
        set_ids=set(args.set_id) if args.set_id else None,
        start_at=args.start_at,
        limit=args.limit,
    )

    print(
        f"pending {len(pending_context_sets)} context sets "
        f"(existing={len(existing_rows)}, runner={args.runner}, model={model_name})",
        flush=True,
    )

    for index, context_set in enumerate(pending_context_sets, start=1):
        outcome = evaluate_with_runner(
            query=queries_by_id[context_set.query_id],
            context_set=context_set,
            chunks_by_id=chunks_by_id,
            runner=runner,
            model_name=model_name,
        ).to_dict()
        outcomes.append(outcome)
        write_jsonl(target, outcomes)
        print(
            f"[{index}/{len(pending_context_sets)}] wrote {context_set.set_id} -> {target}",
            flush=True,
        )

    write_jsonl(target, outcomes)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
