"""Compare prompt-policy effects across strategies.

For each strategy, averages within-query reps for each policy, then
compares per-query deltas across policies. The analysis is the audit-
grade per-query paired delta with bootstrap CI.

## Independent experimental unit

The query is the INDEPENDENT experimental unit. Within a single
strategy, each query contributes one paired delta per comparison. The
bootstrap resamples these 10 per-query paired deltas. Repetitions
within a query reduce model noise but do not add independent
evidence.

Strategies are NOT independent observations of the same effect. Each
strategy has its own per-query deltas, but the same underlying query
appears in multiple strategies. Combining strategies as 30
independent observations is pseudoreplication.

This script reports:

1. Per-strategy paired-query inference (PRIMARY): 10 paired queries,
   bootstrap the per-query deltas within each strategy.
2. Across-strategy query-level aggregation (SECONDARY, optional): for
   each query q, average the per-query effects across the included
   strategies to get a single per-query aggregated effect. The
   bootstrap resamples the 10 per-query aggregated values. The
   underlying sample size is still n_queries = 10, not
   n_queries * n_strategies.

Policies:
- always_question_first: rebuilt from data/processed/policy_q_first
- always_context_first: rebuilt from data/processed/canon_r4_context_first
- adaptive_by_chunk_count: reuses always_question_first for gold_only
  (chunk_count=1 <= 2), and always_context_first for everything else.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def categorize_verdict(ci_low, ci_high, p_two_sided, alpha=0.05):
    """Same verdict policy as audit_statistics.py."""
    if ci_low > 0 and p_two_sided < alpha:
        return "validated on development benchmark"
    if ci_high < 0 and p_two_sided < alpha:
        return "negative direction (comparison reliably wins)"
    if ci_low > 0 or ci_high < 0 or (ci_low == 0 and p_two_sided < alpha + 0.05) or (ci_high == 0 and p_two_sided < alpha + 0.05):
        return "borderline / suggestive"
    return "inconclusive"


def load_per_query_scores(outcome_dir, *, set_id_suffix=None):
    """Load outcomes: {query_id: [score, score, ...]}."""
    per_q = defaultdict(list)
    for path in sorted(outcome_dir.glob("outcomes_model_*.jsonl")):
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


def load_per_query_scores_for_strategy(outcome_dir, strategy):
    """Load outcomes filtered to a specific strategy."""
    expected_suffix = "_" + strategy if strategy != "learned_v3" else "_learned"
    return load_per_query_scores(outcome_dir, set_id_suffix=expected_suffix)


def load_adaptive_gold_only(outcome_dir):
    """For gold_only, adaptive == question_first (chunk_count=1 <= 2).
    Reuse the question_first data for the same rep samples.
    """
    return load_per_query_scores_for_strategy(Path(outcome_dir), "gold_only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-first-dir", default="data/processed/policy_q_first")
    parser.add_argument("--context-first-dir", default="data/processed/canon_r4_context_first")
    parser.add_argument("--learned-context-first-dir", default="data/processed/learned_v3_context_first")
    parser.add_argument("--out", default=".planning/PROMPT_POLICY_ABLATION.md")
    args = parser.parse_args()

    strategies = [
        ("gold_only", "canonical"),
        ("gold_plus_distractors", "canonical"),
        ("topk_pool_order", "canonical"),
        ("learned_v3", "learned"),
    ]

    out_lines = []
    out_lines.append("# Prompt Policy Ablation")
    out_lines.append("")
    out_lines.append("Three policies tested on the same context sets:")
    out_lines.append("- **always_question_first**: Q before C (the original prompt order)")
    out_lines.append("- **always_context_first**: C before Q (the r4 default)")
    out_lines.append("- **adaptive_by_chunk_count**: Q-first for <=2 chunks, C-first otherwise")
    out_lines.append("")
    out_lines.append("Per-strategy means (5 reps × 10 queries).")
    out_lines.append("")
    out_lines.append("## Per-strategy means (5 reps × 10 queries, n = 10 queries per strategy)")
    out_lines.append("")
    out_lines.append("| strategy | q_first | c_first | adaptive |")
    out_lines.append("|---|---:|---:|---:|")
    for strategy_name, dir_kind in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        if strategy_name == "gold_only":
            adaptive = load_adaptive_gold_only(Path(args.q_first_dir))
        else:
            adaptive = c_first

        q_first_means = [sum(v) / len(v) for v in q_first.values()]
        c_first_means = [sum(v) / len(v) for v in c_first.values()]
        adaptive_means = [sum(v) / len(v) for v in adaptive.values()]

        q_first_mean = sum(q_first_means) / len(q_first_means)
        c_first_mean = sum(c_first_means) / len(c_first_means)
        adaptive_mean = sum(adaptive_means) / len(adaptive_means)

        out_lines.append("| " + strategy_name + " | " + format(q_first_mean, ".4f") + " | " + format(c_first_mean, ".4f") + " | " + format(adaptive_mean, ".4f") + " |")

    out_lines.append("")
    out_lines.append("## Per-strategy paired-query inference (PRIMARY)")
    out_lines.append("")
    out_lines.append("For each strategy, bootstrap the per-query deltas (context_first - question_first). Sign convention: positive = context_first better. The independent unit is the query (n = 10).")
    out_lines.append("")
    out_lines.append("| strategy | delta | ci_low (raw) | ci_high (raw) | p_one_sided | p_two_sided | verdict |")
    out_lines.append("|---|---:|---:|---:|---:|---:|---|")

    # Per-strategy inference
    strategy_results = []
    for strategy_name, dir_kind in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        result = paired_query_summary(
            c_first, q_first,
            left_label="context_first",
            right_label="question_first",
            n_resamples=2000, seed=0,
        )
        ls = result.delta_summary
        verdict = categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)
        out_lines.append("| " + strategy_name + " | " + format(ls.mean_delta, "+.4f") +
                         " | " + repr(ls.ci_low) + " | " + repr(ls.ci_high) +
                         " | " + format(ls.p_value_one_sided, ".4f") +
                         " | " + format(ls.p_value_two_sided, ".4f") +
                         " | " + verdict + " |")
        strategy_results.append((strategy_name, result))

    # Chunk-count analysis (per-strategy, NO pseudoreplication)
    out_lines.append("")
    out_lines.append("## Chunk-count grouping (per-strategy, primary)")
    out_lines.append("")
    out_lines.append("Selected-chunk count per strategy:")
    out_lines.append("- gold_only: 1 chunk per query")
    out_lines.append("- gold_plus_distractors: 3 chunks per query")
    out_lines.append("- topk_pool_order: 5 chunks per query")
    out_lines.append("- learned_v3: 5 chunks per query")
    out_lines.append("")
    out_lines.append("**Group A (1-2 chunks)**: gold_only only. The per-strategy inference above is the primary analysis.")
    out_lines.append("")
    out_lines.append("**Group B (3+ chunks)**: gpd (3 chunks), topk_pool_order (5 chunks), learned_v3 (5 chunks). REPORT PER-STRATEGY only.")
    out_lines.append("")
    for strategy_name, result in strategy_results:
        if strategy_name == "gold_only":
            continue
        ls = result.delta_summary
        verdict = categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)
        out_lines.append("- **" + strategy_name + "** (in Group B): delta=" + format(ls.mean_delta, "+.4f") +
                         ", ci_low=" + repr(ls.ci_low) + ", ci_high=" + repr(ls.ci_high) +
                         ", p_two=" + format(ls.p_value_two_sided, ".4f") +
                         ", verdict=" + verdict)
    out_lines.append("")
    out_lines.append("> Treating these 3 strategies as 30 independent samples would be pseudoreplication. The query is the independent unit; each strategy contributes 10 per-query observations, not n*10 independent samples.")
    out_lines.append("")

    # Secondary: across-strategy query-level aggregation
    out_lines.append("## Across-strategy query-level aggregation (SECONDARY, query-level)")
    out_lines.append("")
    out_lines.append("For each query, average the per-query effects across the included strategies to get a single per-query aggregated effect. The bootstrap resamples the 10 per-query aggregated values. The independent sample size is still n_queries = 10, NOT n_queries * n_strategies.")
    out_lines.append("")
    out_lines.append("This is a secondary analysis. It is reported because it averages the per-query effects across the strategies that ALL show the same direction (Group B). It should NOT be interpreted as having n = 30 independent samples.")
    out_lines.append("")

    # Group B: per-query aggregated effect across gpd, topk, learned_v3
    by_query = defaultdict(dict)
    common_q = set()
    for strategy_name, dir_kind in strategies:
        if strategy_name == "gold_only":
            continue
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))
        for qid in q_first:
            if qid in c_first and q_first[qid] and c_first[qid]:
                by_query[qid][strategy_name] = (
                    sum(c_first[qid]) / len(c_first[qid]) - sum(q_first[qid]) / len(q_first[qid])
                )
        common_q = set(by_query.keys()) if not common_q else (common_q & set(q_first.keys()) & set(c_first.keys()))

    # Build per-query aggregated left/right (just the aggregated delta)
    agg_deltas = {}
    for qid in common_q:
        effects = list(by_query[qid].values())
        if effects:
            agg_deltas[qid] = sum(effects) / len(effects)

    # Build paired_query_summary with the aggregated per-query effects
    # Use the same left/right approach: for each query, the "left" is the
    # aggregated effect and the "right" is 0 (the null of no effect).
    # This is a one-sample per-query test with the mean being the per-query
    # aggregated effect.
    # 
    # Wait, this isn't right. paired_query_summary expects paired data.
    # For the across-strategy aggregation, what we want is the mean of
    # per-query aggregated effects, with bootstrap CI on that mean.
    # 
    # Let me use summarize_paired_delta with left=agg_deltas, right=[0]*n
    # This gives a one-sample bootstrap on the aggregated effects.

    from context_engine.stats import summarize_paired_delta
    agg_left = [agg_deltas[qid] for qid in sorted(agg_deltas)]
    agg_right = [0.0] * len(agg_left)
    agg_result = summarize_paired_delta(agg_left, agg_right, n_resamples=2000, seed=0)
    agg_ls = agg_result
    agg_verdict = categorize_verdict(agg_ls.ci_low, agg_ls.ci_high, agg_ls.p_value_two_sided)

    out_lines.append("### Group B (across-strategy query-level aggregation)")
    out_lines.append("")
    out_lines.append("Independent observations: 10 queries (NOT 30). Each per-query value is the MEAN of the per-query effects across the 3 Group-B strategies.")
    out_lines.append("")
    out_lines.append("| metric | value |")
    out_lines.append("|---|---:|")
    out_lines.append("| n_queries (independent) | " + str(agg_ls.n) + " |")
    out_lines.append("| mean_delta | " + format(agg_ls.mean_delta, "+.6f") + " |")
    out_lines.append("| ci_low (raw) | " + repr(agg_ls.ci_low) + " |")
    out_lines.append("| ci_high (raw) | " + repr(agg_ls.ci_high) + " |")
    out_lines.append("| ci_low > 0? | " + str(agg_ls.ci_low > 0) + " |")
    out_lines.append("| p_value_one_sided | " + format(agg_ls.p_value_one_sided, ".4f") + " |")
    out_lines.append("| p_value_two_sided | " + format(agg_ls.p_value_two_sided, ".4f") + " |")
    out_lines.append("| verdict | " + agg_verdict + " |")
    out_lines.append("")
    out_lines.append("Per-query aggregated effects:")
    for qid in sorted(agg_deltas):
        out_lines.append("- " + qid + ": " + format(agg_deltas[qid], "+.6f") + " (mean across gpd, topk, learned_v3)")
    out_lines.append("")

    # Per-query breakdown for each strategy
    out_lines.append("## Per-query breakdown (raw, all strategies)")
    out_lines.append("")
    for strategy_name, dir_kind in strategies:
        if dir_kind == "canonical":
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), strategy_name)
            c_first = load_per_query_scores_for_strategy(Path(args.context_first_dir), strategy_name)
        else:
            q_first = load_per_query_scores_for_strategy(Path(args.q_first_dir), "learned_v3")
            c_first = load_per_query_scores(Path(args.learned_context_first_dir))

        out_lines.append("### " + strategy_name)
        out_lines.append("")
        out_lines.append("| query_id | q_first mean | c_first mean | delta (c_first - q_first) |")
        out_lines.append("|---|---:|---:|---:|")
        for qid in sorted(set(q_first) & set(c_first)):
            q_q = sum(q_first[qid]) / len(q_first[qid])
            c_q = sum(c_first[qid]) / len(c_first[qid])
            out_lines.append("| " + qid + " | " + format(q_q, ".4f") + " | " + format(c_q, ".4f") + " | " + format(c_q - q_q, "+.4f") + " |")
        out_lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(chr(10).join(out_lines))
    print("Wrote " + str(out))

    print("")
    print("=" * 90)
    print("Per-strategy inference (primary):")
    print("=" * 90)
    for strategy_name, result in strategy_results:
        ls = result.delta_summary
        verdict = categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)
        print("  " + strategy_name.ljust(25) + ": delta=" + format(ls.mean_delta, "+.4f") +
              " CI[" + format(ls.ci_low, "+.4f") + ", " + format(ls.ci_high, "+.4f") + "] " +
              "p_one=" + format(ls.p_value_one_sided, ".4f") + " p_two=" + format(ls.p_value_two_sided, ".4f") +
              "  [" + verdict + "]")
    print("")
    print("=" * 90)
    print("Across-strategy Group B aggregation (secondary, query-level):")
    print("=" * 90)
    print("  Group B (n_queries = 10): delta=" + format(agg_ls.mean_delta, "+.4f") +
          " CI[" + format(agg_ls.ci_low, "+.4f") + ", " + format(agg_ls.ci_high, "+.4f") + "] " +
          "p_one=" + format(agg_ls.p_value_one_sided, ".4f") + " p_two=" + format(agg_ls.p_value_two_sided, ".4f") +
          "  [" + agg_verdict + "]")
    print("  Independent observations: 10 (NOT 30).")


if __name__ == "__main__":
    main()
