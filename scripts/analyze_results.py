from __future__ import annotations

from pathlib import Path

from context_engine.artifacts import ContextSet, Outcome
from context_engine.analysis import render_text_report
from context_engine.io import load_jsonl


def main() -> int:
    base = Path("data/processed")
    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(base / "context_sets_v1.jsonl")]
    outcomes = [Outcome.from_dict(row) for row in load_jsonl(base / "outcomes_v1.jsonl")]
    print(render_text_report(context_sets, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
