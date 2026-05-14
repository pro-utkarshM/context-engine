from __future__ import annotations

import argparse

from context_engine.analysis import render_text_report
from context_engine.artifacts import ContextSet, Outcome
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.io import load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_results")
    add_config_args(parser)
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Outcome artifact filename under dataset_dir. Defaults to outcomes_<version>.jsonl.",
    )
    parser.add_argument(
        "--context-sets",
        default=None,
        help="Context set artifact filename under dataset_dir. Defaults to context_sets_<version>.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)

    context_sets_path = (
        config.dataset_dir / args.context_sets
        if args.context_sets
        else resolved_artifact_path(config, "context_sets")
    )
    outcomes_path = (
        config.dataset_dir / args.outcomes
        if args.outcomes
        else resolved_artifact_path(config, "outcomes")
    )

    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(context_sets_path)]
    outcomes = [Outcome.from_dict(row) for row in load_jsonl(outcomes_path)]
    print(render_text_report(context_sets, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
