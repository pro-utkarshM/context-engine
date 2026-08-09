"""Run N replications of the model-backed outcome pipeline and aggregate.

A "replication" is one complete invocation of the model runner against
the same context sets. The output is:

  - N per-run outcome files: ``outcomes_model_<runner>_<version>_run<i:03d>.jsonl``
  - One summary file: ``replications_summary_<version>.jsonl`` (v1 contract)

The summary carries per-strategy bootstrap CIs across runs, so callers
can answer "is the +0.04 delta we observed in r1 reliable?" without
rereading the per-run rows.

Optional paired comparison: pass ``--compare-right`` to also point at a
second pool source (e.g. ``data/processed/auto``) and the summary will
include a per-strategy paired bootstrap CI on the per-query difference
(left minus right).

This is a thin wrapper around ``generate_model_outcomes.py``: it shells
out per replication so each run is an independent, resumable artifact.
The aggregation step is then deterministic (pure stdlib bootstrap).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from context_engine.env import load_dotenv
from context_engine.artifacts import ContextSet, Outcome
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.io import load_jsonl, write_jsonl
from context_engine.replications import (
    paired_summary,
    summarize_replications,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_replications")
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
        "--context-sets",
        default=None,
        help="Explicit context-sets filename (under dataset_dir). Defaults to context_sets_<version>.jsonl.",
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
        "--n-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for the CI computation. Default: 1000.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for the bootstrap CI. Default: 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the bootstrap resampling. Default: 0.",
    )
    parser.add_argument(
        "--compare-right-dataset-dir",
        default=None,
        help="Optional second dataset_dir for a paired comparison. "
        "When set, the summary also includes a per-strategy paired "
        "bootstrap CI on (left - right) per-query overall scores.",
    )
    parser.add_argument(
        "--compare-right-context-sets",
        default=None,
        help="Explicit context-sets filename for the right pool (under --compare-right-dataset-dir). "
        "Defaults to context_sets_<right-version>.jsonl.",
    )
    parser.add_argument(
        "--compare-right-outcomes",
        default=None,
        help="Explicit outcomes filename for the right pool (under --compare-right-dataset-dir). "
        "Defaults to outcomes_model_<runner>_<right-version>.jsonl.",
    )
    parser.add_argument(
        "--compare-right-artifact-version",
        default=None,
        help="Artifact version for the right pool (used to default "
        "context_sets/outcomes filenames). Defaults to the same as --artifact-version.",
    )
    parser.add_argument(
        "--compare-left-label",
        default="left",
        help="Label for the left pool source in the paired summary.",
    )
    parser.add_argument(
        "--compare-right-label",
        default="right",
        help="Label for the right pool source in the paired summary.",
    )
    return parser


def _run_outcome_generation(
    *,
    python_exec: str,
    config_path: str | None,
    runner: str,
    model: str | None,
    context_sets: str | None,
    dataset_dir: str,
    artifact_version: str,
    output_path: Path,
    no_resume: bool,
) -> None:
    """Shell out to generate_model_outcomes.py for one replication."""
    cmd: list[str] = [
        python_exec,
        "scripts/generate_model_outcomes.py",
        "--runner", runner,
        "--dataset-dir", dataset_dir,
        "--artifact-version", artifact_version,
        "--output", str(output_path),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if model:
        cmd.extend(["--model", model])
    if context_sets:
        cmd.extend(["--context-sets", context_sets])
    if no_resume:
        cmd.append("--no-resume")

    result = subprocess.run(cmd, env={"PYTHONPATH": "src", **__import__("os").environ})
    if result.returncode != 0:
        raise RuntimeError(f"generate_model_outcomes failed for {output_path}")


def _load_run(
    context_sets_dir: Path,
    outcomes_dir: Path,
    *,
    context_sets_filename: str,
    outcomes_filename: str,
) -> tuple[list[ContextSet], list[Outcome]]:
    """Load one run.

    ``context_sets_dir`` is the canonical dataset_dir (context sets are
    versioned and shared across replications). ``outcomes_dir`` is where
    the per-run outcome files live (typically config.dataset_dir for
    the main run, or --out-dir when set).
    """
    context_sets = [
        ContextSet.from_dict(row)
        for row in load_jsonl(context_sets_dir / context_sets_filename)
    ]
    outcomes = [
        Outcome.from_dict(row)
        for row in load_jsonl(outcomes_dir / outcomes_filename)
    ]
    return context_sets, outcomes


def _outcome_filename_for_run(runner: str, version: str, run_index: int) -> str:
    return f"outcomes_model_{runner}_{version}_run{run_index:03d}.jsonl"


def _summary_filename(version: str) -> str:
    return f"replications_summary_{version}.jsonl"


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    config = config_from_args(args)
    out_dir = Path(args.out_dir) if args.out_dir else config.dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2

    # 1. Run N replications.
    for run_index in range(args.n):
        output_path = out_dir / _outcome_filename_for_run(args.runner, config.artifact_version, run_index)
        if output_path.exists() and not args.no_resume:
            print(f"run {run_index:03d}: existing, skipping -> {output_path}", flush=True)
            continue
        if output_path.exists() and args.no_resume:
            output_path.unlink()
        print(f"run {run_index:03d}: starting -> {output_path}", flush=True)
        _run_outcome_generation(
            python_exec=sys.executable,
            config_path=args.config,
            runner=args.runner,
            model=args.model,
            context_sets=args.context_sets,
            dataset_dir=str(config.dataset_dir),
            artifact_version=config.artifact_version,
            output_path=output_path,
            no_resume=args.no_resume,
        )

    # 2. Load all runs back into memory.
    runs: list[tuple[list[ContextSet], list[Outcome]]] = []
    context_sets_filename = args.context_sets or f"context_sets_{config.artifact_version}.jsonl"
    for run_index in range(args.n):
        outcomes_filename = _outcome_filename_for_run(args.runner, config.artifact_version, run_index)
        runs.append(
            _load_run(
                context_sets_dir=config.dataset_dir,
                outcomes_dir=out_dir,
                context_sets_filename=context_sets_filename,
                outcomes_filename=outcomes_filename,
            )
        )

    # 3. Aggregate into a deterministic summary.
    summary = summarize_replications(
        runs,
        experiment_name=config.experiment_name,
        runner=args.runner,
        model_name=args.model or "_default_",
        artifact_version=config.artifact_version,
        n_resamples=args.n_resamples,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    # 4. Optional paired comparison against a second pool source.
    if args.compare_right_dataset_dir:
        right_dataset_dir = Path(args.compare_right_dataset_dir)
        right_version = args.compare_right_artifact_version or config.artifact_version
        right_context_sets_filename = (
            args.compare_right_context_sets
            or f"context_sets_{right_version}.jsonl"
        )
        right_outcomes_filename = (
            args.compare_right_outcomes
            or f"outcomes_model_{args.runner}_{right_version}.jsonl"
        )
        right_run = _load_run(
            context_sets_dir=right_dataset_dir,
            outcomes_dir=right_dataset_dir,
            context_sets_filename=right_context_sets_filename,
            outcomes_filename=right_outcomes_filename,
        )
        pairs = paired_summary(
            [runs[0]],
            [right_run],
            left_pool_source=args.compare_left_label,
            right_pool_source=args.compare_right_label,
            n_resamples=args.n_resamples,
            ci_level=args.ci_level,
            seed=args.seed,
        )
        summary = summary.with_paired(pairs)

    # 5. Write the summary.
    summary_path = out_dir / _summary_filename(config.artifact_version)
    write_jsonl(summary_path, [summary.to_dict()])
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
