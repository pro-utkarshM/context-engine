"""Unified confirmatory analysis — single shared path for all three views.

Computes the three views (development, confirmatory, full benchmark) using
ONE shared analysis path. The sign convention is locked:
delta = learned_v3 - comparison; positive = learned wins.

Frozen statistical procedure:
- Bootstrap unit: per-query paired delta
- Bootstrap: percentile, 2000 resamples, seed 0
- p_value_one_sided: opposite-sign share, floored at 1/n_resamples
- p_value_two_sided: min(1.0, 2 * min(p_lower, p_upper))
- CI: 95% central mass of the bootstrap distribution (raw float)

Verdict language:
- "validated on development benchmark": ci_low > 0 AND p_two_sided < 0.05
- "borderline / suggestive": boundary case
- "negative direction (comparison reliably wins)": ci_high < 0 AND p_two_sided < 0.05
- "inconclusive": CI includes 0 materially
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def categorize_verdict(ci_low: float, ci_high: float, p_two_sided: float, alpha: float = 0.05) -> str:
    if ci_low > 0 and p_two_sided < alpha:
        return "validated on development benchmark"
    if ci_high < 0 and p_two_sided < alpha:
        return "negative direction (comparison reliably wins)"
    if ci_low > 0 or ci_high < 0 or (ci_low == 0 and p_two_sided < alpha + 0.05) or (ci_high == 0 and p_two_sided < alpha + 0.05):
        return "borderline / suggestive"
    return "inconclusive"


def load_learned_outcomes(dir_path: Path) -> dict[str, list[float]]:
    """Load learned_v3 outcomes: {query_id: [score, score, ...]}."""
    per_q: dict[str, list[float]] = defaultdict(list)
    for path in sorted(dir_path.glob("outcomes_model_minimax_learned_v3_v1_run*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                per_q[r["query_id"]].append(r["scores"]["overall"])
    return dict(per_q)


def load_canonical_outcomes(dir_path: Path) -> dict[str, dict[str, list[float]]]:
    """Load canonical outcomes: {query_id: {strategy: [score, ...]}}."""
    per_q: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(dir_path.glob("outcomes_model_minimax_v1_run*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = r["set_id"]
                parts = sid.split("_", 2)
                if len(parts) >= 3:
                    strategy = parts[2]
                else:
                    strategy = "unknown"
                per_q[r["query_id"]][strategy].append(r["scores"]["overall"])
    return dict(per_q)


def get_split_query_ids(queries_path: Path) -> dict[str, list[str]]:
    """Read the queries file and split by `metadata.split`."""
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    out: dict[str, list[str]] = {"development": [], "confirmatory": [], "full": []}
    for q in queries:
        split = q["metadata"]["split"]
        if split in out:
            out[split].append(q["query_id"])
        out["full"].append(q["query_id"])
    for k in out:
        out[k] = sorted(out[k])
    return out


def run_paired(
    learned_per_q: dict[str, list[float]],
    comparison_per_q: dict[str, list[float]],
    comparison_label: str,
    qids: list[str],
) -> dict:
    """Run a single paired-query analysis on the given query subset.

    Sign convention: delta = learned - comparison; positive = learned wins.
    """
    # Restrict to qids and to those with data on both sides.
    left = {q: learned_per_q.get(q, []) for q in qids}
    right = {q: comparison_per_q.get(q, []) for q in qids}
    common = sorted([q for q in qids if left.get(q) and right.get(q)])
    left = {q: left[q] for q in common}
    right = {q: right[q] for q in common}

    result = paired_query_summary(
        left, right,
        left_label="learned_v3",
        right_label=comparison_label,
        n_resamples=2000, seed=0,
    )
    ls = result.delta_summary
    wins = sum(1 for d in result.per_query_deltas.values() if d > 0)
    losses = sum(1 for d in result.per_query_deltas.values() if d < 0)
    ties = sum(1 for d in result.per_query_deltas.values() if d == 0)
    verdict = categorize_verdict(ls.ci_low, ls.ci_high, ls.p_value_two_sided)
    return {
        "n_queries": len(common),
        "reps_per_query": result.reps_per_query,
        "learned_mean": result.mean_left,
        "comparison_mean": result.mean_right,
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
        "verdict": verdict,
        "per_query_deltas": result.per_query_deltas,
    }


def main():
    queries_path = Path("data/processed/queries_v1.jsonl")
    splits = get_split_query_ids(queries_path)

    learned = load_learned_outcomes(Path("data/processed/learned_v3_confirmatory"))
    canon = load_canonical_outcomes(Path("data/processed/canon_r4_confirmatory"))
    gpd_per_q = {qid: scores["gold_plus_distractors"] for qid, scores in canon.items() if "gold_plus_distractors" in scores}
    topk_per_q = {qid: scores["topk_pool_order"] for qid, scores in canon.items() if "topk_pool_order" in scores}

    results = {}
    for split_name, qids in splits.items():
        if not qids:
            continue
        # PRIMARY: learned_v3 vs topk_pool_order
        primary = run_paired(learned, topk_per_q, "topk_pool_order", qids)
        # SECONDARY (oracle-informed): learned_v3 vs gold_plus_distractors
        oracle = run_paired(learned, gpd_per_q, "gold_plus_distractors", qids)

        results[split_name] = {
            "primary_learned_v3_vs_topk_pool_order": primary,
            "secondary_learned_v3_vs_gold_plus_distractors": oracle,
            "n_queries": len(qids),
        }

    return results


if __name__ == "__main__":
    import json
    results = main()
    print(json.dumps(results, indent=2, default=str))
