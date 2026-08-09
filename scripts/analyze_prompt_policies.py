"""Compare prompt-policy effects across strategies.

For each strategy, averages within-query reps for each policy, then
compares per-query deltas across policies. The analysis is the audit-
grade per-query paired delta with bootstrap CI.

Policies:
- always_question_first: rebuilt from data/processed/policy_q_first
- always_context_first: rebuilt from data/processed/canon_r4_context_first (r4 baseline)
- adaptive_by_chunk_count: reuses always_question_first for gold_only
  (since chunk_count=1 <= 2), and always_context_first for everything
  else (chunk_count >= 3).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def load_per_query_scores(outcome_dir: Path, *, set_id_suffix: str = None):
    """Load outcomes: {query_id: [score, score, ...]}."""
    per_q: dict[str, list[float]] = defaultdict(list)
    for path in sorted(outcome_dir.glob("outcomes_model_*.jsonl")):
        # Filter out adaptive-only files (we handle those separately)
        if "adaptive" in path.name:
            continue
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


def load_per_query_scores_for_strategy(outcome_dir: Path, strategy: str):
    """Load outcomes filtered to a specific strategy."""
    expected_suffix = f"_{strategy}" if strategy != "learned_v3" else "_learned"
    return load_per_query_scores(outcome_dir, set_id_suffix=expected_suffix)


def load_adaptive_gold_only(outcome_dir: Path):
    """For gold_only, adaptive == question_first (chunk_count=1 <= 2).

    We use the question_first data here because the adaptive policy
    produces the same prompt for 1-chunk contexts. The two "separate"
    runs (question_first and adaptive_by_chunk_count) are independent
    rep samples, but the prompts are identical, so the comparisons are
    only meaningful when we use the SAME rep samples.
    """
    return load_per_query_scores_for_strategy(Path(outcome_dir), "gold_only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-first-dir", default="data/processed/policy_q_first")
    parser.add_argument("--context-first-dir", default="data/processed/canon_r4_context_first")
    parser.add_argument("--learned-context-first-dir", default="data/processed/learned_v3_context_first")
    parser.add_argument("--out", default=".planning/PROMPT_POLICY_ABLATION.md")
    args = parser.parse_args()

    # Strategy -> (q_first_dir, context_first_dir, both?)
    # gold_only, gpd, topk_pool_order: canonical strategies
    # learned_v3: separate learned_v3 dir
    strategies = [
        ("gold_only", "canonical", "gold_only"),
        ("gold_plus_distractors", "canonical", "gold_plus_distractors"),
        ("topk_pool_order", "canonical", "topk_pool_order"),
        ("learned_v3", "learned", "learned"),
    ]

    out_lines = []
    out_lines.append("# Prompt Policy Ablation")
    out_lines.append("")
    out_lines.append("Three policies tested on the same context sets:")
    out_lines.append("- **always_question_first**: Q before C (the original prompt order)")
    out_lines.append("- **always_context_first**: C before Q (the r4 default)")
    out_lines.append("- **adaptive_by_chunk_count**: Q-first for ≤2 chunks, C-first otherwise")
    out_lines.append("")
    out_lines.append("Per-strategy means (5 reps × 10 queries).")
    out_lines.append("")

    # Compute per-strategy per-policy mean
    print(f"{'strategy':<25} {'q_first':>10} {'c_first':>10} {'adaptive':>10}")
    print("-" * 60)
    for strategy_name, dir_kind, suffix in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        # Adaptive: gold_only uses question_first (chunk_count=1), others use context_first
        if strategy_name == "gold_only":
            adaptive = load_adaptive_gold_only(Path(args.q_first_dir))
        else:
            adaptive = c_first  # chunk_count >= 3, so adaptive == context_first

        # Compute means
        q_first_means = [sum(v)/len(v) for v in q_first.values()]
        c_first_means = [sum(v)/len(v) for v in c_first.values()]
        adaptive_means = [sum(v)/len(v) for v in adaptive.values()]

        q_first_mean = sum(q_first_means) / len(q_first_means)
        c_first_mean = sum(c_first_means) / len(c_first_means)
        adaptive_mean = sum(adaptive_means) / len(adaptive_means)

        print(f"{strategy_name:<25} {q_first_mean:>10.4f} {c_first_mean:>10.4f} {adaptive_mean:>10.4f}")

        out_lines.append(f"## {strategy_name}")
        out_lines.append(f"  - question_first: {q_first_mean:.4f}")
        out_lines.append(f"  - context_first: {c_first_mean:.4f}")
        out_lines.append(f"  - adaptive: {adaptive_mean:.4f}")

        # Per-query breakdown
        out_lines.append(f"  - Per-query breakdown:")
        common_q = sorted(q_first.keys() & c_first.keys() & adaptive.keys())
        for qid in common_q:
            q_q = sum(q_first[qid]) / len(q_first[qid])
            c_q = sum(c_first[qid]) / len(c_first[qid])
            a_q = sum(adaptive[qid]) / len(adaptive[qid])
            out_lines.append(f"    - {qid}: q_first={q_q:.4f}, c_first={c_q:.4f}, adaptive={a_q:.4f}")
        out_lines.append("")

    # Paired comparisons: context_first vs adaptive (which is identical for all but gold_only)
    # for each strategy:
    out_lines.append("## Paired Comparisons (Context-first vs Question-first)")
    out_lines.append("")
    out_lines.append("For each strategy, the per-query delta (context_first - question_first) is computed and bootstrapped.")
    out_lines.append("Positive delta: context_first improves score.")
    out_lines.append("")

    print()
    print("=" * 70)
    print("Paired: context_first - question_first")
    print("=" * 70)
    out_lines.append("### context_first - question_first")
    for strategy_name, dir_kind, suffix in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        # Rename so c_first is left, q_first is right
        result = paired_query_summary(
            c_first, q_first,
            left_label="context_first",
            right_label="question_first",
            n_resamples=2000, seed=0,
        )
        ls = result.delta_summary
        excl = "EXCLUDES 0" if (ls.ci_low > 0 or ls.ci_high < 0) else "INCLUDES 0"
        print(f"  {strategy_name:<25}: delta={ls.mean_delta:+.4f} CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f}  [{excl}]")
        out_lines.append(f"  - **{strategy_name}**: delta={ls.mean_delta:+.4f}, CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f} [{excl}]")

    # Now: adaptive vs each of the two extremes
    out_lines.append("")
    out_lines.append("### adaptive - question_first")
    print()
    print("=" * 70)
    print("Paired: adaptive - question_first")
    print("=" * 70)
    for strategy_name, dir_kind, suffix in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")

        if strategy_name == "gold_only":
            adaptive = load_adaptive_gold_only(Path(args.q_first_dir))
        else:
            # Adaptive == context_first
            if dir_kind == "canonical":
                adaptive = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
            else:
                adaptive = load_per_query_scores(Path(args.learned_context_first_dir))

        result = paired_query_summary(
            adaptive, q_first,
            left_label="adaptive",
            right_label="question_first",
            n_resamples=2000, seed=0,
        )
        ls = result.delta_summary
        excl = "EXCLUDES 0" if (ls.ci_low > 0 or ls.ci_high < 0) else "INCLUDES 0"
        print(f"  {strategy_name:<25}: delta={ls.mean_delta:+.4f} CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f}  [{excl}]")
        out_lines.append(f"  - **{strategy_name}**: delta={ls.mean_delta:+.4f}, CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f} [{excl}]")

    out_lines.append("")
    out_lines.append("### adaptive - context_first")
    print()
    print("=" * 70)
    print("Paired: adaptive - context_first")
    print("=" * 70)
    for strategy_name, dir_kind, suffix in strategies:
        if dir_kind == "canonical":
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        if strategy_name == "gold_only":
            adaptive = load_adaptive_gold_only(Path(args.q_first_dir))
        else:
            adaptive = c_first  # tan to context_first

        # Skip - adaptive == context_first for these
        if strategy_name != "gold_only":
            out_lines.append(f"  - **{strategy_name}**: adaptive == context_first (chunk_count={3 if strategy_name == 'gold_plus_distractors' else 5})")
            print(f"  {strategy_name:<25}: adaptive == context_first (skip)")
            continue

        result = paired_query_summary(
            adaptive, c_first,
            left_label="adaptive",
            right_label="context_first",
            n_resamples=2000, seed=0,
        )
        ls = result.delta_summary
        excl = "EXCLUDES 0" if (ls.ci_low > 0 or ls.ci_high < 0) else "INCLUDES 0"
        print(f"  {strategy_name:<25}: delta={ls.mean_delta:+.4f} CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f}  [{excl}]")
        out_lines.append(f"  - **{strategy_name}**: delta={ls.mean_delta:+.4f}, CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f} [{excl}]")

    # Chunk-count analysis: results grouped by chunk count
    out_lines.append("")
    out_lines.append("## Chunk-count Analysis")
    out_lines.append("")
    out_lines.append("Selected-chunk count per strategy:")
    out_lines.append("- gold_only: 1 chunk per query")
    out_lines.append("- gold_plus_distractors: 3 chunks per query")
    out_lines.append("- topk_pool_order: 5 chunks per query")
    out_lines.append("- learned_v3: 5 chunks per query")
    out_lines.append("")
    out_lines.append("**Group A (1-2 chunks)**: gold_only → adaptive == question_first")
    out_lines.append("")
    out_lines.append("**Group B (3+ chunks)**: gold_plus_distractors, topk_pool_order, learned_v3 → adaptive == context_first")
    out_lines.append("")

    # Compute and report Group A and Group B aggregate
    out_lines.append("### Group A (1-2 chunks): gold_only")
    out_lines.append("")
    out_lines.append("For chunk_count <= 2, adaptive == question_first. The full comparison is the gold_only row above.")
    out_lines.append("")

    out_lines.append("### Group B (3+ chunks): aggregated")
    out_lines.append("")
    out_lines.append("For chunk_count >= 3, adaptive == context_first. Aggregate the per-query deltas across gold_plus_distractors, topk_pool_order, and learned_v3.")
    out_lines.append("")

    # Aggregate deltas across all 3 strategies × 10 queries = 30 paired observations
    all_c_first = defaultdict(list)
    all_q_first = defaultdict(list)
    for strategy_name, dir_kind, suffix in strategies:
        if strategy_name == "gold_only":
            continue  # skip - in Group A
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        for qid in q_first:
            # Tag the queries with strategy to keep them distinct
            all_q_first[f"{strategy_name}_{qid}"] = q_first[qid]
            all_c_first[f"{strategy_name}_{qid}"] = c_first[qid]

    result = paired_query_summary(
        all_c_first, all_q_first,
        left_label="context_first",
        right_label="question_first",
        n_resamples=2000, seed=0,
    )
    ls = result.delta_summary
    excl = "EXCLUDES 0" if (ls.ci_low > 0 or ls.ci_high < 0) else "INCLUDES 0"
    print()
    print("=" * 70)
    print(f"Group B (3+ chunks, aggregated): delta={ls.mean_delta:+.4f} CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f}  [{excl}]")
    print(f"  n_queries: {ls.n}")
    out_lines.append(f"  - delta={ls.mean_delta:+.4f}, CI=[{ls.ci_low:+.4f}, {ls.ci_high:+.4f}], p={ls.p_value_two_sided:.4f} [{excl}]")
    out_lines.append(f"  - n_queries (pooled across strategies): {ls.n}")
    out_lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
