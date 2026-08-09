"""Generate learned context sets from existing benchmark outcomes.

Reads ``context_sets_v1.jsonl`` and any outcome artifact (default:
``outcomes_model_<runner>_v1.jsonl``), estimates per-chunk utility, and
emits ``context_sets_learned_v1.jsonl`` with one set per query. The
emitted artifact satisfies the same ``context_sets`` contract as the
hand-authored strategies, so the existing outcome pipeline picks it up
without code changes.

Selector algorithm is the first learned baseline (Phase E, #3): utility
is the empirical mean ``overall`` across context sets that include the
chunk; packing is greedy by utility desc with candidate-pool position
as the tiebreak; total selected tokens are bounded by the configured
budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from context_engine.artifacts import (
    CandidatePool,
    ContextSet,
    CorpusChunk,
    Outcome,
    Query,
)
from context_engine.config import (
    add_config_args,
    config_from_args,
    resolved_artifact_path,
)
from context_engine.io import load_jsonl, write_jsonl
from context_engine.learned_selector import build_learned_context_sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_learned_context_sets")
    add_config_args(parser)
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Outcomes artifact filename under dataset_dir. Default: outcomes_<version>.jsonl.",
    )
    parser.add_argument(
        "--score-axis",
        choices=("correctness", "support", "efficiency", "overall"),
        default="overall",
        help="Score axis used to estimate per-chunk utility. Default: overall.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Default: context_sets_learned_<version>.jsonl under dataset_dir.",
    )
    return parser


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
    candidate_pools = [
        CandidatePool.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "candidate_pools"))
    ]
    context_sets = [
        ContextSet.from_dict(row)
        for row in load_jsonl(resolved_artifact_path(config, "context_sets"))
    ]
    outcomes_path = (
        config.dataset_dir / args.outcomes
        if args.outcomes
        else resolved_artifact_path(config, "outcomes")
    )
    outcomes = [Outcome.from_dict(row) for row in load_jsonl(outcomes_path)]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}

    learned = build_learned_context_sets(
        queries=queries,
        candidate_pools=candidate_pools,
        context_sets=context_sets,
        outcomes=outcomes,
        chunks_by_id=chunks_by_id,
        token_budget=config.token_budget,
        score_axis=args.score_axis,
    )

    target = (
        Path(args.output)
        if args.output
        else config.dataset_dir / f"context_sets_learned_{config.artifact_version}.jsonl"
    )
    write_jsonl(target, [cs.to_dict() for cs in learned])
    print(
        f"wrote {len(learned)} learned context sets "
        f"(token_budget={config.token_budget}, score_axis={args.score_axis}) -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())