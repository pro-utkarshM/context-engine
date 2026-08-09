"""Reproduce the r2 gold_plus_distractors investigation (closes #10).

This script regenerates the paired bootstrap CI table from the
5 canonical replications and 5 auto replications stored under
``data/processed/``.

Outputs the per-strategy CI table and per-query deltas that underpin
``.planning/R2_INVESTIGATION.md``.

Run with:
    .venv/bin/python scripts/investigate_regression.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_engine.stats import summarize_paired_delta


CANON_RUN_DIR = Path("data/processed/replications_minimax_v1")
AUTO_RUN_DIR = Path("data/processed/auto")
N_RUNS = 5


def _load_runs(run_dir: Path, glob: str) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for i in range(N_RUNS):
        path = run_dir / glob.format(i=i)
        if not path.exists():
            raise FileNotFoundError(f"missing run: {path}")
        with path.open() as handle:
            runs.append([json.loads(line) for line in handle if line.strip()])
    return runs


def _per_query_per_strategy(runs: list[list[dict]]) -> dict[str, dict[str, list[float]]]:
    table: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for run in runs:
        for row in run:
            set_id = row["set_id"]
            strategy = set_id.split("_", 2)[2]
            query_id = row["query_id"]
            table[strategy].setdefault(query_id, []).append(row["scores"]["overall"])
    return table


def main() -> int:
    canon_runs = _load_runs(CANON_RUN_DIR, "outcomes_model_minimax_v1_run{i:03d}.jsonl")
    auto_runs = _load_runs(AUTO_RUN_DIR, "outcomes_model_minimax_auto_v1_run{i:03d}.jsonl")
    canon_table = _per_query_per_strategy(canon_runs)
    auto_table = _per_query_per_strategy(auto_runs)

    print("Paired bootstrap CI: canonical - auto (positive = canonical better)")
    print(f"strategy                    n  delta   ci_low  ci_high p     verdict")
    print("-" * 78)

    for strategy in sorted(canon_table):
        if strategy not in auto_table:
            continue
        canon_data = canon_table[strategy]
        auto_data = auto_table[strategy]
        common_q = sorted(set(canon_data) & set(auto_data))
        canon_per_q = [sum(canon_data[q]) / len(canon_data[q]) for q in common_q]
        auto_per_q = [sum(auto_data[q]) / len(auto_data[q]) for q in common_q]
        delta = summarize_paired_delta(canon_per_q, auto_per_q, n_resamples=2000, seed=0)
        exclude_zero = not (delta.ci_low <= 0 <= delta.ci_high)
        verdict = "reliable" if exclude_zero else "provisional"
        print(
            f"{strategy:<26} {delta.n:>3} {delta.mean_delta:>+7.4f} "
            f"{delta.ci_low:>+7.4f} {delta.ci_high:>+7.4f} "
            f"{delta.p_value_two_sided:>5.3f} {verdict}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
