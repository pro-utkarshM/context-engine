from __future__ import annotations

import argparse
from pathlib import Path

from context_engine.artifacts import ContextSet, Outcome
from context_engine.analysis import render_text_report
from context_engine.io import load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_results")
    parser.add_argument(
        "--outcomes",
        default="outcomes_v1.jsonl",
        help="Outcome artifact filename under data/processed/ to analyze.",
    )
    parser.add_argument(
        "--context-sets",
        default="context_sets_v1.jsonl",
        help="Context set artifact filename under data/processed/ to join against.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = Path("data/processed")
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(base / args.context_sets)]
    outcomes = [Outcome.from_dict(row) for row in load_jsonl(base / args.outcomes)]
    print(render_text_report(context_sets, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
