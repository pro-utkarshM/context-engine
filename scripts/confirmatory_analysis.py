"""Confirmatory analysis for the Phase N benchmark.

Performs three paired-query analyses:
1. learned_v3 vs topk_pool_order on confirmatory queries only (PRIMARY)
2. learned_v3 vs gold_plus_distractors on confirmatory queries (secondary, oracle-informed)
3. learned_v3 vs topk_pool_order on the full 30 queries (secondary)

Uses the same methodology as .planning/STATISTICAL_AUDIT.md:
- Bootstrap unit: per-query paired delta
- Bootstrap: percentile, 2000 resamples, seed 0
- p_value_one_sided: opposite-sign share, floored at 1/n_resamples
- p_value_two_sided: min(1.0, 2 * min(p_lower, p_upper))
- Sign convention: delta = learned - comparison; positive = learned wins
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def load_canonical_outcomes(dir):
    """Load canonical (non-learned) outcomes into {query_id: {strategy: [scores]}}."""
    per_q = defaultdict(lambda: defaultdict(list))
    for path in sorted(dir.glob("outcomes_model_minimax_v1_run*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = r["set_id"]
                # Format: q_XXXX_strategy
                parts = sid.split("_", 2)
                if len(parts) >= 3:
                    strategy = parts[2]
                else:
                    strategy = "unknown"
                per_q[r["query_id"]][strategy].append(r["scores"]["overall"])
    return per_q


def load_learned_outcomes(dir):
    """Load learned_v3 outcomes. Returns {query_id: [scores]}."""
    per_q = defaultdict(list)
    for path in sorted(dir.glob("outcomes_model_minimax_learned_v3_v1_run*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                per_q[r["query_id"]].append(r["scores"]["overall"])
    return per_q


def categorize_verdict(ci_low, ci_high, p_two_sided, alpha=0.05):
    if ci_low > 0 and p_two_sided < alpha:
        return "validated"
    if ci_high < 0 and p_two_sided < alpha:
        return "negative direction"
    if ci_low > 0 or ci_high < 0 or (ci_low == 0 and p_two_sided < alpha + 0.05) or (ci_high == 0 and p_two_sided < alpha + 0.05):
        return "borderline / suggestive"
    return "inconclusive"


def get_split(queries, split_name):
    """Return the query_ids for the given split."""
    return [q["query_id"] for q in queries if q["metadata"]["split"] == split_name]


def main():
    # Load all queries for split metadata
    with open("data/processed/queries_v1.jsonl") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    # Load canonical outcomes
    canon = load_canonical_outcomes(Path("data/processed/canon_r4_confirmatory"))
    # Load learned v3 outcomes
    learned_per_q = load_learned_outcomes(Path("data/processed/learned_v3_confirmatory"))

    # Get split query IDs
    confirmatory_qids = get_split(queries, "confirmatory")
    development_qids = get_split(queries, "development")

    # PRIMARY: learned_v3 vs topk_pool_order on confirmatory queries
    conf_learned = {q: learned_per_q.get(q, []) for q in confirmatory_qids}
    conf_topk = {q: canon.get(q, {}).get("topk_pool_order", []) for q in confirmatory_qids}
    conf_gpd = {q: canon.get(q, {}).get("gold_plus_distractors", []) for q in confirmatory_qids}

    # Drop queries with no data on either side
    common = sorted([q for q in confirmatory_qids if conf_learned.get(q) and conf_topk.get(q)])
    conf_learned = {q: conf_learned[q] for q in common}
    conf_topk = {q: conf_topk[q] for q in common}
    conf_gpd = {q: conf_gpd[q] for q in common if conf_gpd.get(q)}

    print("=" * 70)
    print("PRIMARY: learned_v3 vs topk_pool_order on confirmatory queries")
    print("=" * 70)
    print(f"  n_queries: {len(conf_learned)}")
    result = paired_query_summary(
        conf_learned, conf_topk,
        left_label="learned_v3",
        right_label="topk_pool_order",
        n_resamples=2000, seed=0,
    )
    ls = result.delta_summary
    wins = sum(1 for d in result.per_query_deltas.values() if d > 0)
    losses = sum(1 for d in result.per_query_deltas.values() if d < 0)
    ties = sum(1 for d in result.per_query_deltas.values() if d == 0)
    print(f"  learned mean: {result.mean_left:.4f}")
    print(f"  topk mean: {result.mean_right:.4f}")
    print(f"  mean_delta: {ls.mean_delta:+.4f}")
    print(f"  ci_low (raw): {ls.ci_low!r}")
    print(f"  ci_high (raw): {ls.ci_high!r}")
    print(f"  ci_low > 0? {ls.ci_low > 0}")
    print(f"  p_value_one_sided: {ls.p_value_one_sided:.4f}")
    print(f"  p_value_two_sided: {ls.p_value_two_sided:.4f}")
    print(f"  reps_per_query: {result.reps_per_query}")
    print(f"  Win/loss/tie: {wins}/{losses}/{ties}")
    print(f"  Verdict: {categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)}")
    print()
    print("  Per-query deltas:")
    for qid in sorted(result.per_query_deltas):
        d = result.per_query_deltas[qid]
        print(f"    {qid}: {d:+.4f}")

    primary_result = {
        "n_queries": len(conf_learned),
        "reps_per_query": result.reps_per_query,
        "learned_mean": result.mean_left,
        "topk_mean": result.mean_right,
        "mean_delta": ls.mean_delta,
        "ci_low_raw": ls.ci_low,
        "ci_high_raw": ls.ci_high,
        "ci_low_display": round(ls.ci_low, 4),
        "ci_high_display": round(ls.ci_high, 4),
        "p_one_sided": ls.p_value_one_sided,
        "p_two_sided": ls.p_value_two_sided,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "verdict": categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided),
        "per_query_deltas": result.per_query_deltas,
    }

    # SECONDARY (oracle-informed): learned_v3 vs gold_plus_distractors on confirmatory
    print()
    print("=" * 70)
    print("SECONDARY (oracle-informed): learned_v3 vs gold_plus_distractors on confirmatory")
    print("=" * 70)
    result_sec = paired_query_summary(
        conf_learned, conf_gpd,
        left_label="learned_v3",
        right_label="gold_plus_distractors",
        n_resamples=2000, seed=0,
    )
    ls2 = result_sec.delta_summary
    print(f"  n_queries: {len(conf_learned)}")
    print(f"  learned mean: {result_sec.mean_left:.4f}")
    print(f"  gpd mean: {result_sec.mean_right:.4f}")
    print(f"  mean_delta: {ls2.mean_delta:+.4f}")
    print(f"  ci_low (raw): {ls2.ci_low!r}")
    print(f"  ci_high (raw): {ls2.ci_high!r}")
    print(f"  p_value_one_sided: {ls2.p_value_one_sided:.4f}")
    print(f"  p_value_two_sided: {ls2.p_value_two_sided:.4f}")
    print(f"  Verdict: {categorize_verdict(ls2.ci_low, ls2.ci_high, ls2.p_value_two_sided)}")

    secondary_result = {
        "n_queries": len(conf_learned),
        "learned_mean": result_sec.mean_left,
        "gpd_mean": result_sec.mean_right,
        "mean_delta": ls2.mean_delta,
        "ci_low_raw": ls2.ci_low,
        "ci_high_raw": ls2.ci_high,
        "p_one_sided": ls2.p_value_one_sided,
        "p_two_sided": ls2.p_value_two_sided,
        "verdict": categorize_verdict(ls2.ci_low, ls2.ci_high, ls2.p_value_two_sided),
    }

    # DEVELOPMENT: learned_v3 vs topk_pool_order on development queries
    dev_learned = {q: learned_per_q.get(q, []) for q in development_qids}
    dev_topk = {q: canon.get(q, {}).get("topk_pool_order", []) for q in development_qids}
    dev_common = sorted([q for q in development_qids if dev_learned.get(q) and dev_topk.get(q)])
    dev_learned = {q: dev_learned[q] for q in dev_common}
    dev_topk = {q: dev_topk[q] for q in dev_common}

    print()
    print("=" * 70)
    print("DEVELOPMENT: learned_v3 vs topk_pool_order on development queries")
    print("=" * 70)
    result_dev = paired_query_summary(
        dev_learned, dev_topk,
        left_label="learned_v3",
        right_label="topk_pool_order",
        n_resamples=2000, seed=0,
    )
    lsd = result_dev.delta_summary
    print(f"  n_queries: {len(dev_learned)}")
    print(f"  learned mean: {result_dev.mean_left:.4f}")
    print(f"  topk mean: {result_dev.mean_right:.4f}")
    print(f"  mean_delta: {lsd.mean_delta:+.4f}")
    print(f"  ci_low (raw): {lsd.ci_low!r}")
    print(f"  ci_high (raw): {lsd.ci_high!r}")
    print(f"  p_value_one_sided: {lsd.p_value_one_sided:.4f}")
    print(f"  p_value_two_sided: {lsd.p_value_two_sided:.4f}")
    print(f"  Verdict: {categorize_verdict(lsd.ci_low, lsd.ci_high, lsd.p_value_two_sided)}")

    dev_result = {
        "n_queries": len(dev_learned),
        "learned_mean": result_dev.mean_left,
        "topk_mean": result_dev.mean_right,
        "mean_delta": lsd.mean_delta,
        "ci_low_raw": lsd.ci_low,
        "ci_high_raw": lsd.ci_high,
        "p_one_sided": lsd.p_value_one_sided,
        "p_two_sided": lsd.p_value_two_sided,
        "verdict": categorize_verdict(lsd.ci_low, lsd.ci_high, lsd.p_value_two_sided),
    }

    # FULL: learned_v3 vs topk_pool_order on all 30 queries
    full_learned = {q: learned_per_q.get(q, []) for q in learned_per_q}
    full_topk = {q: canon.get(q, {}).get("topk_pool_order", []) for q in learned_per_q}
    full_common = sorted([q for q in full_learned if full_topk.get(q)])
    full_learned = {q: full_learned[q] for q in full_common}
    full_topk = {q: full_topk[q] for q in full_common}

    print()
    print("=" * 70)
    print("FULL: learned_v3 vs topk_pool_order on all 30 queries")
    print("=" * 70)
    result_full = paired_query_summary(
        full_learned, full_topk,
        left_label="learned_v3",
        right_label="topk_pool_order",
        n_resamples=2000, seed=0,
    )
    lsf = result_full.delta_summary
    print(f"  n_queries: {len(full_learned)}")
    print(f"  learned mean: {result_full.mean_left:.4f}")
    print(f"  topk mean: {result_full.mean_right:.4f}")
    print(f"  mean_delta: {lsf.mean_delta:+.4f}")
    print(f"  ci_low (raw): {lsf.ci_low!r}")
    print(f"  ci_high (raw): {lsf.ci_high!r}")
    print(f"  p_value_one_sided: {lsf.p_value_one_sided:.4f}")
    print(f"  p_value_two_sided: {lsf.p_value_two_sided:.4f}")
    print(f"  Verdict: {categorize_verdict(lsf.ci_low, lsf.ci_high, lsf.p_value_two_sided)}")

    full_result = {
        "n_queries": len(full_learned),
        "learned_mean": result_full.mean_left,
        "topk_mean": result_full.mean_right,
        "mean_delta": lsf.mean_delta,
        "ci_low_raw": lsf.ci_low,
        "ci_high_raw": lsf.ci_high,
        "p_one_sided": lsf.p_value_one_sided,
        "p_two_sided": lsf.p_value_two_sided,
        "verdict": categorize_verdict(lsf.ci_low, lsf.ci_high, lsf.p_value_two_sided),
    }

    # Save the report
    report = {
        "primary": primary_result,
        "secondary_oracle_informed": secondary_result,
        "development": dev_result,
        "full": full_result,
    }
    with open("/tmp/confirmatory_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print()
    print(f"Saved report to /tmp/confirmatory_analysis.json")


if __name__ == "__main__":
    main()
