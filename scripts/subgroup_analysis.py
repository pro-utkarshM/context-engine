"""Subgroup descriptive analyses for the confirmatory results.

Per-topic and per-difficulty breakdowns. These are DESCRIPTIVE only.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from context_engine.paired_query import paired_query_summary


def load_canonical_outcomes(dir):
    per_q = defaultdict(lambda: defaultdict(list))
    for path in sorted(dir.glob("outcomes_model_minimax_v1_run*.jsonl")):
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
    return per_q


def load_learned_outcomes(dir):
    per_q = defaultdict(list)
    for path in sorted(dir.glob("outcomes_model_minimax_learned_v3_v1_run*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                per_q[r["query_id"]].append(r["scores"]["overall"])
    return per_q


def main():
    with open("data/processed/queries_v1.jsonl") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    canon = load_canonical_outcomes(Path("data/processed/canon_r4_confirmatory"))
    learned = load_learned_outcomes(Path("data/processed/learned_v3_confirmatory"))

    confirmatory = [q for q in queries if q["metadata"]["split"] == "confirmatory"]

    # By topic
    print("=" * 70)
    print("Per-topic descriptive (confirmatory only, learned_v3 vs topk_pool_order)")
    print("=" * 70)
    by_topic = defaultdict(list)
    for q in confirmatory:
        qid = q["query_id"]
        if qid in learned and qid in canon and "topk_pool_order" in canon[qid]:
            l = sum(learned[qid]) / len(learned[qid])
            t = sum(canon[qid]["topk_pool_order"]) / len(canon[qid]["topk_pool_order"])
            by_topic[q["metadata"]["topic"]].append((qid, l - t, l, t))

    for topic, results in sorted(by_topic.items()):
        deltas = [d for _, d, _, _ in results]
        learned_means = [l for _, _, l, _ in results]
        topk_means = [t for _, _, _, t in results]
        print(f"  {topic}:")
        print(f"    n_queries: {len(results)}")
        print(f"    learned mean: {sum(learned_means)/len(learned_means):.4f}")
        print(f"    topk mean: {sum(topk_means)/len(topk_means):.4f}")
        print(f"    mean delta: {sum(deltas)/len(deltas):+.4f}")
        for qid, d, l, t in results:
            pass  # printing per-query is verbose

    # By difficulty
    print()
    print("=" * 70)
    print("Per-difficulty descriptive (confirmatory only)")
    print("=" * 70)
    by_difficulty = defaultdict(list)
    for q in confirmatory:
        qid = q["query_id"]
        if qid in learned and qid in canon and "topk_pool_order" in canon[qid]:
            l = sum(learned[qid]) / len(learned[qid])
            t = sum(canon[qid]["topk_pool_order"]) / len(canon[qid]["topk_pool_order"])
            by_difficulty[q["difficulty"]].append((qid, l - t))

    for diff, results in sorted(by_difficulty.items()):
        deltas = [d for _, d in results]
        print(f"  {diff}: n_queries={len(results)}, mean_delta={sum(deltas)/len(deltas):+.4f}")

    # By multi-hop
    print()
    print("=" * 70)
    print("Per-multi-hop descriptive (confirmatory only)")
    print("=" * 70)
    by_multihop = defaultdict(list)
    for q in confirmatory:
        qid = q["query_id"]
        if qid in learned and qid in canon and "topk_pool_order" in canon[qid]:
            l = sum(learned[qid]) / len(learned[qid])
            t = sum(canon[qid]["topk_pool_order"]) / len(canon[qid]["topk_pool_order"])
            by_multihop[q["metadata"]["requires_multi_hop"]].append((qid, l - t))

    for mh, results in sorted(by_multihop.items()):
        deltas = [d for _, d in results]
        print(f"  multihop={mh}: n_queries={len(results)}, mean_delta={sum(deltas)/len(deltas):+.4f}")


if __name__ == "__main__":
    main()
