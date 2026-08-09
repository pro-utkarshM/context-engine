"""Statistical audit script for the r4 prompt-ablated comparisons.

Reads the per-run outcome files and produces the audit-grade paired-query
statistics. The independent experimental unit is the query: we average
within-query reps, then bootstrap the per-query deltas.

Inputs:
  - data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run{000..004}.jsonl
  - data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run{000..004}.jsonl

Output:
  - .planning/STATISTICAL_AUDIT_RESULTS.md  (human-readable table)
  - Selections of the per-query breakdown are echoed to stdout.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def load_per_query_scores(outcome_dir: Path, *, set_id_suffix: str = None):
    """Load outcome files and return {query_id: [score, score, ...]} per query.

    Only loads files matching the outcome pattern ``outcomes_model_*.jsonl``.
    Non-outcome files (corpus_chunks, queries, etc.) are skipped.

    If set_id_suffix is given, include only rows whose set_id ENDS WITH that suffix.
    """
    per_q: dict[str, list[float]] = defaultdict(list)
    files = sorted(outcome_dir.glob("outcomes_model_*.jsonl"))
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = r["set_id"]
                if set_id_suffix is not None and not sid.endswith(set_id_suffix):
                    continue
                per_q[r["query_id"]].append(r["scores"]["overall"])
    return per_q


def format_report(result, *, sign_convention: str) -> str:
    lines = []
    lines.append(f"== {result.left_label} vs {result.right_label} ==")
    lines.append(f"  Sign convention: {sign_convention}")
    lines.append(f"  N queries: {result.n_queries}, reps per query: {result.reps_per_query}")
    lines.append(f"  Mean left:  {result.mean_left:.4f}")
    lines.append(f"  Mean right: {result.mean_right:.4f}")
    lines.append(f"  Mean delta: {result.delta_summary.mean_delta:+.4f}")
    lines.append(f"  95% bootstrap CI: [{result.delta_summary.ci_low:+.4f}, {result.delta_summary.ci_high:+.4f}]")
    lines.append(f"  p-value: {result.delta_summary.p_value_two_sided:.4f}")
    lines.append(f"  Bootstrap: {result.n_resamples} samples, seed={result.seed}")
    lines.append(f"  Per-query deltas:")
    for qid in sorted(result.per_query_deltas):
        d = result.per_query_deltas[qid]
        lines.append(f"    {qid}: {d:+.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-dir", default="data/processed/learned_v3_context_first")
    parser.add_argument("--canon-dir", default="data/processed/canon_r4_context_first")
    parser.add_argument("--out", default=".planning/STATISTICAL_AUDIT_RESULTS.md")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    learned_dir = Path(args.learned_dir)
    canon_dir = Path(args.canon_dir)
    if not learned_dir.is_dir():
        raise SystemExit(f"learned dir not found: {learned_dir}")
    if not canon_dir.is_dir():
        raise SystemExit(f"canon dir not found: {canon_dir}")

    learned = load_per_query_scores(learned_dir)
    print(f"Loaded learned outcomes: {len(learned)} queries, {sum(len(v) for v in learned.values())} reps total")

    strategies = [
        ("topk_pool_order", "topk_pool_order"),
        ("gold_plus_distractors", "gold_plus_distractors"),
        ("gold_only", "gold_only"),
        ("minimal_support", "minimal_support"),
        ("shuffled_order", "shuffled_order"),
    ]

    results = []
    for canon_strategy, suffix in strategies:
        canon = load_per_query_scores(canon_dir, set_id_suffix=f"_{suffix}")
        print(f"Loaded {canon_strategy} outcomes: {len(canon)} queries, {sum(len(v) for v in canon.values())} reps total")
        if not canon:
            print(f"  Skipping {canon_strategy} (no data)")
            continue
        result = paired_query_summary(
            learned, canon,
            left_label="learned_v3_context_first",
            right_label=canon_strategy,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        results.append((canon_strategy, result))

    # Write report
    out_lines = []
    out_lines.append("# Statistical Audit Results — r4 prompt-ablated comparisons")
    out_lines.append("")
    out_lines.append("**Methodology**:")
    out_lines.append(f"- Independent experimental unit: query (n = 10)")
    out_lines.append(f"- Repetitions per query: 5 (model stochasticity)")
    out_lines.append(f"- Bootstrap unit: per-query paired delta (10 observations)")
    out_lines.append(f"- Bootstrap: percentile method, {args.n_resamples} resamples, seed = {args.seed}")
    out_lines.append(f"- p-value: percentile bootstrap (share of bootstrap means with sign disagreeing with observed, floor at 1/n_resamples)")
    out_lines.append("")
    out_lines.append("**Sign convention**: `delta = mean(left) - mean(right)`. Positive = left wins.")
    out_lines.append("")
    out_lines.append("**Inputs**:")
    out_lines.append(f"- `learned_v3_context_first`: {learned_dir}")
    out_lines.append(f"- `canon_r4_context_first`: {canon_dir}")
    out_lines.append("")
    for canon_strategy, result in results:
        out_lines.append(format_report(result, sign_convention="left - right; positive = learned wins"))
        out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"\nWrote {out_path}")

    print()
    print("=" * 70)
    print("Summary (learned_v3_context_first vs canonical):")
    print("=" * 70)
    for canon_strategy, result in results:
        ls = result.delta_summary
        excl = "EXCLUDES 0" if (ls.ci_low > 0 or ls.ci_high < 0) else "INCLUDES 0"
        print(f"  vs {canon_strategy:<25}: mean_delta={ls.mean_delta:+.4f}, CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f}  [{excl}]")


if __name__ == "__main__":
    main()
