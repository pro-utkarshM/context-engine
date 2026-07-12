"""Generate marginal-impact rows for chunk-level utility validation.

Drives the ``MarginalImpact`` generator across selected base context sets.
For each base set, it evaluates the add/remove of one chunk at a time and
records the signed score delta.

The script is config-driven (shares ``--config``/``--dataset-dir``/``--
artifact-version`` with the other generators), defaults to the stub runner,
and writes incrementally so a long run can be resumed by re-invoking.

Default base-set selection: every context set where
``metadata.contains_all_gold == True`` and ``strategy != minimal_support``
(distractor-heavy bases give the cleanest add/remove signal). The gold-only
base is kept as the canonical reference.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import ContextSet, CorpusChunk, Outcome, Query
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.io import load_jsonl, write_jsonl
from context_engine.marginal_impact import (
    MarginalImpactError,
    Operation,
    ScoreKey,
    compute_marginal_impact,
)
from context_engine.model_outcomes import evaluate_with_runner
from context_engine.runner import (
    MINIMAX_DEFAULT_MODEL,
    MiniMaxResponsesRunner,
    ModelRunner,
    OpenAIResponsesRunner,
    StubModelRunner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_marginal_impact")
    add_config_args(parser)
    parser.add_argument("--model", default=None, help="Model name to send to the runner. Overrides config.model_name.")
    parser.add_argument(
        "--runner",
        choices=("stub", "openai", "minimax"),
        default="stub",
        help="Runner backend to use.",
    )
    parser.add_argument(
        "--operations",
        default="remove",
        help="Comma-separated subset of {add, remove}. Default: remove (add requires external chunk ids).",
    )
    parser.add_argument(
        "--score-key",
        choices=("correctness", "support", "efficiency", "overall"),
        default="overall",
        help="Which score axis to compute deltas over.",
    )
    parser.add_argument(
        "--strategies",
        default="gold_plus_distractors",
        help="Comma-separated base-set strategy names to evaluate. Default: gold_plus_distractors (gives 3-chunk sets with remove signal).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. Defaults to marginal_impact_<runner>_<version>.jsonl under dataset_dir.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior and regenerate from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pending (base_set_id, chunk_id, operation) triples after filtering.",
    )
    parser.add_argument("--query-id", action="append", default=None, help="Restrict to one or more query IDs.")
    parser.add_argument("--set-id", action="append", default=None, help="Restrict to one or more exact context set IDs.")
    parser.add_argument("--start-at", default=None, help="Skip pending triples until this (set_id|chunk_id|operation) is reached.")
    return parser


def _make_scorer(
    *,
    query: Query,
    chunks_by_id: dict[str, CorpusChunk],
    runner: ModelRunner,
    model_name: str,
    weights,
    evaluator_version: str,
    max_token_budget: int,
):
    def scorer(context_set: ContextSet) -> Outcome:
        return evaluate_with_runner(
            query=query,
            context_set=context_set,
            chunks_by_id=chunks_by_id,
            runner=runner,
            model_name=model_name,
            weights=weights,
            evaluator_version=evaluator_version,
            max_token_budget=max_token_budget,
        )

    return scorer


def _select_base_context_sets(
    context_sets: list[ContextSet],
    *,
    strategies: set[str],
    query_ids: set[str] | None,
    set_ids: set[str] | None,
) -> list[ContextSet]:
    selected = [
        context_set
        for context_set in context_sets
        if context_set.strategy in strategies and context_set.metadata.contains_all_gold
    ]
    if query_ids:
        selected = [cs for cs in selected if cs.query_id in query_ids]
    if set_ids:
        selected = [cs for cs in selected if cs.set_id in set_ids]
    return selected


def _enumerate_triples(
    base_context_sets: list[ContextSet],
    operations: tuple[Operation, ...],
) -> list[tuple[str, str, Operation]]:
    triples: list[tuple[str, str, Operation]] = []
    for base_set in base_context_sets:
        for operation in operations:
            if operation == "remove":
                for chunk_id in base_set.selected_ids:
                    triples.append((base_set.set_id, chunk_id, operation))
            else:  # add — requires a chunk that is NOT in the base set
                # Without a richer candidate-pool handle here, add is a no-op
                # at this layer. The script still records the intent so users
                # can supply extra chunk ids via a future extension.
                continue
    return triples


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    config = config_from_args(args)

    if args.runner == "minimax":
        env_model = os.environ.get("MINIMAX_MODEL")
        model_name = args.model or env_model or MINIMAX_DEFAULT_MODEL
    else:
        model_name = args.model or os.environ.get("OPENAI_MODEL") or config.model_name

    operations: tuple[Operation, ...] = tuple(
        operation.strip()
        for operation in args.operations.split(",")
        if operation.strip() in {"add", "remove"}
    )
    if not operations:
        print("error: --operations must include at least one of add, remove", file=sys.stderr)
        return 2

    strategies: set[str] = {s.strip() for s in args.strategies.split(",") if s.strip()}
    if not strategies:
        print("error: --strategies must list at least one base strategy", file=sys.stderr)
        return 2

    score_key: ScoreKey = args.score_key  # type: ignore[assignment]

    corpus_chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "corpus_chunks"))
    ]
    queries = [
        Query.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "queries"))
    ]
    context_sets = [
        ContextSet.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "context_sets"))
    ]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    queries_by_id = {query.query_id: query for query in queries}
    base_context_sets = _select_base_context_sets(
        context_sets,
        strategies=strategies,
        query_ids=set(args.query_id) if args.query_id else None,
        set_ids=set(args.set_id) if args.set_id else None,
    )

    if args.runner == "stub":
        runner = StubModelRunner()
    elif args.runner == "openai":
        runner = OpenAIResponsesRunner()
    else:
        runner = MiniMaxResponsesRunner()

    triples = _enumerate_triples(base_context_sets, operations)
    if args.start_at:
        try:
            start_set_id, start_chunk_id, start_op = args.start_at.split("|", 2)
        except ValueError:
            print(
                "error: --start-at must be in the form '<set_id>|<chunk_id>|<operation>'",
                file=sys.stderr,
            )
            return 2
        seen = False
        resumed = []
        for triple in triples:
            if triple == (start_set_id, start_chunk_id, start_op):
                seen = True
            if seen:
                resumed.append(triple)
        triples = resumed

    if args.limit is not None:
        triples = triples[: args.limit]

    if args.output:
        target = Path(args.output)
    else:
        target = resolved_artifact_path(config, f"marginal_impact_{args.runner}")
    if args.no_resume and target.exists():
        print(
            f"warning: --no-resume will restart from scratch and overwrite progress in {target}",
            file=sys.stderr,
        )

    existing_rows = [] if args.no_resume or not target.exists() else load_jsonl(target)
    completed = {(row["base_set_id"], row["chunk_id"], row["operation"]) for row in existing_rows}
    rows = list(existing_rows)
    pending = [triple for triple in triples if triple not in completed]

    print(
        f"pending {len(pending)} marginal-impact triples "
        f"(existing={len(existing_rows)}, base_sets={len(base_context_sets)}, "
        f"runner={args.runner}, model={model_name}, score_key={score_key})",
        flush=True,
    )

    base_set_by_id = {base_set.set_id: base_set for base_set in base_context_sets}
    for index, (set_id, chunk_id, operation) in enumerate(pending, start=1):
        base_set = base_set_by_id[set_id]
        query = queries_by_id[base_set.query_id]
        scorer = _make_scorer(
            query=query,
            chunks_by_id=chunks_by_id,
            runner=runner,
            model_name=model_name,
            weights=config.scoring_weights,
            evaluator_version=config.evaluator_version,
            max_token_budget=config.token_budget,
        )
        try:
            base_outcome = scorer(base_set)
            variant_selected = (
                list(base_set.selected_ids) + [chunk_id]
                if operation == "add"
                else [cid for cid in base_set.selected_ids if cid != chunk_id]
            )
            if not variant_selected:
                continue
            variant = ContextSet(
                set_id=base_set.set_id,
                query_id=base_set.query_id,
                candidate_pool_id=base_set.candidate_pool_id,
                strategy=base_set.strategy,
                selected_ids=variant_selected,
                ordering_type=base_set.ordering_type,
                token_count=base_set.token_count,
                metadata=base_set.metadata,
            )
            new_outcome = scorer(variant)
        except MarginalImpactError as exc:
            print(f"skip {set_id}|{chunk_id}|{operation}: {exc}", file=sys.stderr)
            continue

        row = compute_marginal_impact(
            base_set=base_set,
            chunk_id=chunk_id,
            operation=operation,
            base_score=getattr(base_outcome.scores, score_key),
            new_score=getattr(new_outcome.scores, score_key),
        ).to_dict()
        rows.append(row)
        write_jsonl(target, rows)
        print(f"[{index}/{len(pending)}] wrote {set_id}|{chunk_id}|{operation} -> {target}", flush=True)

    write_jsonl(target, rows)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())