"""Build M6 learned context sets and run the model on them.

Closes the M6 estimator-upgrade cycle: ships the v2 estimator
(marginal-impact signal + negative-utility tiebreak) and produces
the empirical evidence needed to validate the thesis that a learned
selector can outperform heuristic baselines.

Pipeline:
1. Load canonical pools, queries, corpus, context sets (M7 replications).
2. Load marginal_impact_v1.jsonl as the v2 estimator's primary signal.
3. Build M6 learned context sets via build_learned_context_sets_v2.
4. Run the model on the new context sets (5 replications).
5. Aggregate the outcomes into a replications summary.

Run with:
    .venv/bin/python scripts/generate_learned_context_sets_v2.py \
        --n 5 --config configs/experiment_v1_baseline.json \
        --out-dir data/processed/learned_v2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import (
    CandidatePool,
    ContextSet,
    CorpusChunk,
    MarginalImpact,
    Outcome,
    Query,
)
from context_engine.config import add_config_args, config_from_args
from context_engine.io import load_jsonl, write_jsonl
from context_engine.learned_selector import (
    build_learned_context_sets_v2,
    build_learned_context_sets_v3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_learned_context_sets_v2")
    add_config_args(parser)
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of replications to run. Default: 5.",
    )
    parser.add_argument(
        "--runner",
        choices=("stub", "openai", "minimax"),
        default="minimax",
        help="Runner backend. Default: minimax.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to send to the runner. Overrides config / env defaults.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for per-run outcome files and the summary. "
        "Defaults to dataset_dir.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete existing per-run outcome files before regenerating.",
    )
    parser.add_argument(
        "--marginal-impact",
        default="marginal_impact_minimax_v1.jsonl",
        help="Marginal-impact artifact filename (under dataset_dir). "
        "Defaults to marginal_impact_minimax_v1.jsonl.",
    )
    parser.add_argument(
        "--estimator",
        choices=("v2", "v3"),
        default="v3",
        help="Estimator version. v2 uses global marginal impact; v3 (default) "
        "uses per-query marginal impact with global fallback. v3 is the "
        "M5-follow-up #1 (per-query adaptation) implementation.",
    )
    parser.add_argument(
        "--no-negative-tiebreak",
        action="store_true",
        help="Use M5 pool-position tiebreak instead of the v2 negative-utility tiebreak.",
    )
    parser.add_argument(
        "--score-axis",
        default="overall",
        choices=("correctness", "support", "efficiency", "overall"),
        help="Outcome score axis for the outcome-mean fallback. Default: overall.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    config = config_from_args(args)
    out_dir = Path(args.out_dir) if args.out_dir else config.dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2

    # 1. Load inputs.
    corpus_chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(config.dataset_dir / f"corpus_chunks_{config.artifact_version}.jsonl")
    ]
    queries = [
        Query.from_dict(row)
        for row in load_jsonl(config.dataset_dir / f"queries_{config.artifact_version}.jsonl")
    ]
    candidate_pools = [
        CandidatePool.from_dict(row)
        for row in load_jsonl(config.dataset_dir / f"candidate_pools_{config.artifact_version}.jsonl")
    ]
    # Context sets and outcomes come from the canonical r2 set; we use a single
    # representative run to fit the estimator (the v2 estimator only needs the
    # marginal_impact signal at the chunk level; per-run outcome mean is a
    # tiebreaker-level signal that we can populate from any one run).
    context_sets = [
        ContextSet.from_dict(row)
        for row in load_jsonl(config.dataset_dir / f"context_sets_{config.artifact_version}.jsonl")
    ]
    outcomes = [
        Outcome.from_dict(row)
        for row in load_jsonl(
            config.dataset_dir / f"outcomes_model_minimax_{config.artifact_version}.jsonl"
        )
    ]
    marginal_impacts = [
        MarginalImpact.from_dict(row)
        for row in load_jsonl(config.dataset_dir / args.marginal_impact)
    ]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus_chunks}

    # 2. Build v2 learned context sets.
    if args.estimator == "v3":
        build_fn = build_learned_context_sets_v3
        learned_label = "v3"
    else:
        build_fn = build_learned_context_sets_v2
        learned_label = "v2"

    learned_sets = build_fn(
        queries=queries,
        candidate_pools=candidate_pools,
        context_sets=context_sets,
        outcomes=outcomes,
        marginal_impacts=marginal_impacts,
        chunks_by_id=chunks_by_id,
        token_budget=config.token_budget,
        use_negative_tiebreak=not args.no_negative_tiebreak,
        score_axis=args.score_axis,
    )

    # 3. Mirror the source corpus/queries/pools into out_dir so the subprocess
    # can read them under the same --dataset-dir it writes outcomes to. The
    # learned_v2 context sets are the only new artifact.
    for artifact in ("corpus_chunks", "queries", "candidate_pools"):
        src = config.dataset_dir / f"{artifact}_{config.artifact_version}.jsonl"
        dst = out_dir / src.name
        if not dst.exists():
            write_jsonl(dst, list(load_jsonl(src)))

    learned_path = out_dir / f"context_sets_learned_{learned_label}_{config.artifact_version}.jsonl"
    write_jsonl(learned_path, [cs.to_dict() for cs in learned_sets])
    print(f"wrote {len(learned_sets)} v2 learned context sets -> {learned_path}")

    # 4. Run the model on the v2 context sets, N replications.
    for run_index in range(args.n):
        output_path = out_dir / f"outcomes_model_{args.runner}_learned_{learned_label}_{config.artifact_version}_run{run_index:03d}.jsonl"
        if output_path.exists() and not args.no_resume:
            print(f"run {run_index:03d}: existing, skipping -> {output_path}", flush=True)
            continue
        if output_path.exists() and args.no_resume:
            output_path.unlink()
        print(f"run {run_index:03d}: starting -> {output_path}", flush=True)
        cmd: list[str] = [
            sys.executable,
            "scripts/generate_model_outcomes.py",
            "--runner", args.runner,
            "--dataset-dir", str(out_dir),
            "--artifact-version", config.artifact_version,
            "--context-sets", learned_path.name,
            "--output", str(output_path),
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.model:
            cmd.extend(["--model", args.model])
        result = subprocess.run(cmd, env={"PYTHONPATH": "src", **os.environ})
        if result.returncode != 0:
            raise RuntimeError(f"generate_model_outcomes failed for {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
