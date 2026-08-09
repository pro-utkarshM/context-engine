
"""Run MiniMax-M3 on a single query with a custom prompt, N times.

Outputs an outcomes-style JSONL file with the model's responses for
each run. Used by the prompt ablation experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from context_engine.env import load_dotenv
from context_engine.evaluation import score_correctness
from context_engine.prompting import PromptPayload
from context_engine.runner import MiniMaxResponsesRunner


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, help="Path to a prompt.txt file")
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--gold-answer", required=True)
    parser.add_argument("--query-text", required=True, help="Full query text, used by score_correctness for context-aware scoring.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--delay", type=float, default=0.0, help="Sleep between runs (seconds)")
    args = parser.parse_args()

    model_name = args.model or os.environ.get("MINIMAX_MODEL", "MiniMax-M3")
    runner = MiniMaxResponsesRunner()
    prompt = open(args.prompt_file).read()

    with open(args.out, "w") as f:
        for run_idx in range(args.n):
            print(f"run {run_idx+1}/{args.n}...", flush=True)
            payload = PromptPayload(
                query_id=args.query_id,
                context_set_id=args.set_id,
                prompt=prompt,
                estimated_prompt_tokens=0,
            )
            response = runner.run(payload, model_name=model_name)
            # Build a minimal Query object for scoring; needs query text + gold_answer.
            from context_engine.artifacts import Query
            _q = Query.from_dict({
                "query_id": args.query_id,
                "query": args.query_text,
                "task_type": "doc_qa",
                "difficulty": "easy",
                "gold_answer": args.gold_answer,
                "gold_support_ids": [],
                "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
            })
            correctness = score_correctness(_q, response.answer)
            row = {
                "query_id": args.query_id,
                "set_id": args.set_id,
                "run_index": run_idx,
                "answer": response.answer,
                "scores": {
                    "correctness": correctness,
                    "support": 1.0,
                    "overall": correctness * 0.6 + 1.0 * 0.3 + (1.0 - 0) * 0.1,
                },
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
                "evaluator_version": "eval_v1_model_runner",
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if args.delay > 0:
                time.sleep(args.delay)
    print(f"wrote {args.n} outcomes -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
