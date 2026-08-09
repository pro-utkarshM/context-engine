"""Run N replications of a specific prompt policy for a subset of strategies.

Generates per-run outcome files under the dataset_dir, named:
  outcomes_model_<runner>_<version>_<strategy>_run<i:03d>.jsonl

Then aggregates via scripts/run_replications.py if --summarize is passed.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from context_engine.env import load_dotenv


def run_n_reps(
    *,
    dataset_dir: str,
    artifact_version: str,
    runner: str,
    policy: str,
    set_ids: list[str],
    n: int,
    output_basename: str,
    adaptive_threshold: int = 2,
    context_sets_file: str | None = None,
) -> None:
    load_dotenv()
    for i in range(n):
        out_path = Path(dataset_dir) / f"{output_basename}_run{i:03d}.jsonl"
        cmd = [
            sys.executable,
            "scripts/generate_model_outcomes.py",
            "--runner", runner,
            "--dataset-dir", dataset_dir,
            "--artifact-version", artifact_version,
            "--policy", policy,
            "--adaptive-threshold", str(adaptive_threshold),
            "--no-resume",
            "--output", str(out_path),
        ]
        for sid in set_ids:
            cmd.extend(["--set-id", sid])
        if context_sets_file:
            cmd.extend(["--context-sets", context_sets_file])
        print(f"  [{i+1}/{n}] running policy={policy} -> {out_path}")
        result = subprocess.run(cmd, env={"PYTHONPATH": "src", **os.environ}, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[-500:]}")
            sys.exit(1)
        else:
            # Just print the last line of the output
            lines = result.stdout.strip().split("\n")
            for line in lines[-3:]:
                print(f"    {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--artifact-version", default="v1")
    parser.add_argument("--runner", default="minimax")
    parser.add_argument("--policy", required=True, choices=("question_first", "context_first", "adaptive_by_chunk_count"))
    parser.add_argument("--adaptive-threshold", type=int, default=2)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--strategy", required=True, help="e.g. gold_only, gold_plus_distractors, topk_pool_order, learned_v3")
    parser.add_argument("--context-sets-file", default=None, help="Override context-sets file (defaults to context_sets_v1.jsonl)")
    parser.add_argument("--output-basename", default=None, help="Defaults to outcomes_model_minimax_v1_<strategy>")
    args = parser.parse_args()

    # Build the set_ids for this strategy
    # For canonical strategies: q_0001_<strategy>, q_0002_<strategy>, ...
    # For learned_v3: q_0001_learned, q_0002_learned, ...
    queries = [f"q_{i:04d}" for i in range(1, 11)]
    if args.strategy == "learned_v3":
        set_ids = [f"{q}_learned" for q in queries]
        if not args.context_sets_file:
            args.context_sets_file = "context_sets_learned_v1.jsonl"
    else:
        set_ids = [f"{q}_{args.strategy}" for q in queries]

    output_basename = args.output_basename or f"outcomes_model_{args.runner}_{args.artifact_version}_{args.strategy}"

    print(f"Running {args.n} reps for policy={args.policy}, strategy={args.strategy}")
    print(f"  Set IDs: {set_ids}")
    print(f"  Output basename: {output_basename}")
    print()

    run_n_reps(
        dataset_dir=args.dataset_dir,
        artifact_version=args.artifact_version,
        runner=args.runner,
        policy=args.policy,
        set_ids=set_ids,
        n=args.n,
        output_basename=output_basename,
        adaptive_threshold=args.adaptive_threshold,
        context_sets_file=args.context_sets_file,
    )


if __name__ == "__main__":
    main()
