"""Statistical audit script for the r4 prompt-ablated comparisons.

Reads the per-run outcome files and produces the audit-grade paired-query
statistics. The independent experimental unit is the query: we average
within-query reps, then bootstrap the per-query deltas.

Statistical procedure (pinned in this script):

- Independent experimental unit: query (n = 10).
- Repetitions per query: 5 (model stochasticity — reduces noise, not
  independent evidence).
- Bootstrap unit: per-query paired delta (10 observations).
- Bootstrap: percentile method, 2000 resamples, seed = 0.
- p-values:
  - ``p_value_one_sided`` = share of bootstrap means whose sign disagrees
    with the observed mean (one-sided tail probability). Floored at
    ``1/n_resamples``.
  - ``p_value_two_sided`` = ``min(1.0, 2 * min(p_lower, p_upper))``.
- CI: 95% central mass of the bootstrap distribution. The raw float
  values are preserved (no rounding) so the boundary can be inspected.

Inputs:
  - data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run{000..004}.jsonl
  - data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run{000..004}.jsonl

Output:
  - .planning/STATISTICAL_AUDIT_RESULTS.md  (human-readable table)
  - Selections of the per-query breakdown are echoed to stdout.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def load_per_query_scores(outcome_dir, *, set_id_suffix=None):
    """Load outcome files and return {query_id: [score, score, ...]} per query.

    Only loads files matching the outcome pattern ``outcomes_model_*.jsonl``.
    Non-outcome files (corpus_chunks, queries, etc.) are skipped.

    If set_id_suffix is given, include only rows whose set_id ENDS WITH that suffix.
    """
    per_q = defaultdict(list)
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


def categorize_verdict(ci_low, ci_high, p_two_sided, alpha=0.05):
    """Return a verdict category based on the CI bounds and p-value.

    Rules:
      - ``ci_low > 0`` AND ``p_value_two_sided < alpha``:
        "validated on development benchmark (positive direction)"
      - ``ci_high < 0`` AND ``p_value_two_sided < alpha``:
        "negative direction (negative direction reliably established)"
      - ``ci_low == 0`` OR ``ci_high == 0`` OR ``p_value_two_sided is on the boundary``:
        "borderline / suggestive"
      - else: "inconclusive"
    """
    if ci_low > 0 and p_two_sided < alpha:
        return "validated on development benchmark"
    if ci_high < 0 and p_two_sided < alpha:
        return "negative direction (comparison reliably wins)"
    if ci_low > 0 or ci_high < 0 or (ci_low == 0 and p_two_sided < alpha + 0.05) or (ci_high == 0 and p_two_sided < alpha + 0.05):
        return "borderline / suggestive"
    return "inconclusive"


def format_report(result, *, sign_convention):
    """Format a single comparison result. CI bounds are reported at full
    float precision (no rounding) so the boundary is honest.

    The displayed ``+0.0000`` of a 4-decimal rounding is NOT sufficient
    to determine exclusion; the raw value is reported alongside.
    """
    ls = result.delta_summary
    lines = []
    lines.append("== " + result.left_label + " vs " + result.right_label + " ==")
    lines.append("  Sign convention: " + sign_convention)
    lines.append("  N queries: " + str(result.n_queries) + ", reps per query: " + str(result.reps_per_query))
    lines.append("  Mean left:  " + format(result.mean_left, ".6f"))
    lines.append("  Mean right: " + format(result.mean_right, ".6f"))
    lines.append("  Mean delta: " + format(ls.mean_delta, "+.6f"))
    lines.append("  95% bootstrap CI (raw floats):")
    lines.append("    ci_low  = " + repr(ls.ci_low))
    lines.append("    ci_high = " + repr(ls.ci_high))
    lines.append("    ci_low > 0? " + str(ls.ci_low > 0))
    lines.append("    ci_low <= 0? " + str(ls.ci_low <= 0))
    lines.append("  95% bootstrap CI (display): [" + format(ls.ci_low, "+.4f") + ", " + format(ls.ci_high, "+.4f") + "]")
    lines.append("  One-sided p-value: " + format(ls.p_value_one_sided, ".6f"))
    lines.append("  Two-sided p-value: " + format(ls.p_value_two_sided, ".6f"))
    lines.append("  Bootstrap: " + str(result.n_resamples) + " samples, seed=" + str(result.seed))
    lines.append("  Verdict: " + categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided))
    lines.append("  Per-query deltas:")
    for qid in sorted(result.per_query_deltas):
        d = result.per_query_deltas[qid]
        lines.append("    " + qid + ": " + format(d, "+.6f"))
    return chr(10).join(lines)


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
        raise SystemExit("learned dir not found: " + str(learned_dir))
    if not canon_dir.is_dir():
        raise SystemExit("canon dir not found: " + str(canon_dir))

    learned = load_per_query_scores(learned_dir)
    print("Loaded learned outcomes: " + str(len(learned)) + " queries, " + str(sum(len(v) for v in learned.values())) + " reps total")

    strategies = [
        ("topk_pool_order", "topk_pool_order"),
        ("gold_plus_distractors", "gold_plus_distractors"),
        ("gold_only", "gold_only"),
        ("minimal_support", "minimal_support"),
        ("shuffled_order", "shuffled_order"),
    ]

    results = []
    for canon_strategy, suffix in strategies:
        canon = load_per_query_scores(canon_dir, set_id_suffix="_" + suffix)
        print("Loaded " + canon_strategy + " outcomes: " + str(len(canon)) + " queries, " + str(sum(len(v) for v in canon.values())) + " reps total")
        if not canon:
            print("  Skipping " + canon_strategy + " (no data)")
            continue
        result = paired_query_summary(
            learned, canon,
            left_label="learned_v3_context_first",
            right_label=canon_strategy,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        results.append((canon_strategy, result))

    out_lines = []
    out_lines.append("# Statistical Audit Results — r4 prompt-ablated comparisons")
    out_lines.append("")
    out_lines.append("**Methodology**:")
    out_lines.append("- Independent experimental unit: query (n_queries = 10)")
    out_lines.append("- Repetitions per query: 5 (model stochasticity)")
    out_lines.append("- Bootstrap unit: per-query paired delta (10 observations)")
    out_lines.append("- Bootstrap: percentile method, " + str(args.n_resamples) + " resamples, seed = " + str(args.seed))
    out_lines.append("- p_value_one_sided: share of bootstrap means with sign disagreeing with observed, floored at 1/n_resamples")
    out_lines.append("- p_value_two_sided: min(1.0, 2 * min(p_lower, p_upper))")
    out_lines.append("")
    out_lines.append("**Sign convention**: `delta = mean(left) - mean(right)`. Positive = left wins.")
    out_lines.append("")
    out_lines.append("**CI rounding policy**: CI bounds are reported at full float precision (no rounding). A 4-decimal display is shown alongside for human readability, but the verdict is keyed off the raw value.")
    out_lines.append("")
    out_lines.append("**Verdict language**:")
    out_lines.append("- `validated on development benchmark`: CI lower bound strictly > 0 AND p_value_two_sided < 0.05")
    out_lines.append("- `borderline / suggestive`: CI lower bound is exactly 0 (or the boundary case), significance on the boundary")
    out_lines.append("- `inconclusive`: CI includes 0 materially, no sufficient evidence")
    out_lines.append("")
    out_lines.append("**Inputs**:")
    out_lines.append("- `learned_v3_context_first`: " + str(learned_dir))
    out_lines.append("- `canon_r4_context_first`: " + str(canon_dir))
    out_lines.append("")
    for canon_strategy, result in results:
        out_lines.append(format_report(result, sign_convention="left - right; positive = learned wins"))
        out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(chr(10).join(out_lines))
    print("")
    print("Wrote " + str(out_path))

    print("")
    print("=" * 90)
    print("Summary (learned_v3_context_first vs canonical):")
    print("=" * 90)
    for canon_strategy, result in results:
        ls = result.delta_summary
        verdict = categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)
        print("  vs " + canon_strategy.ljust(25) + ": mean_delta=" + format(ls.mean_delta, "+.4f") +
              ", CI[" + format(ls.ci_low, "+.4f") + ", " + format(ls.ci_high, "+.4f") + "], " +
              "p_one=" + format(ls.p_value_one_sided, ".4f") + ", p_two=" + format(ls.p_value_two_sided, ".4f") +
              "  [" + verdict + "]")


if __name__ == "__main__":
    main()
