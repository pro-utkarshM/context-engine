from pathlib import Path

from context_engine.dataset import BenchmarkDataset
from context_engine.io import write_jsonl


def test_dataset_loads_versioned_artifacts(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "corpus_chunks_v1.jsonl",
        [
            {
                "chunk_id": "pg16_auth_001",
                "doc_version": "16",
                "doc_path": "client-authentication.md",
                "section_path": ["Client Authentication", "The pg_hba.conf File"],
                "source_type": "doc",
                "text": "The pg_hba.conf file controls client authentication...",
                "token_count": 148,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": "pg16_auth_002",
                "metadata": {"topic": "authentication", "subtopic": "pg_hba.conf"},
            }
        ],
    )
    write_jsonl(
        tmp_path / "queries_v1.jsonl",
        [
            {
                "query_id": "q_0001",
                "query": "Which file controls client authentication rules in PostgreSQL?",
                "task_type": "doc_qa",
                "difficulty": "easy",
                "gold_answer": "pg_hba.conf controls client authentication rules.",
                "gold_support_ids": ["pg16_auth_001"],
                "metadata": {
                    "topic": "authentication",
                    "requires_multi_hop": False,
                    "question_family": "fact_lookup",
                },
            }
        ],
    )
    write_jsonl(
        tmp_path / "candidate_pools_v1.jsonl",
        [
            {
                "query_id": "q_0001",
                "candidate_pool_id": "pool_q_0001_v1",
                "candidate_ids": ["pg16_auth_001", "pg15_auth_003", "pg16_roles_011"],
                "composition": {
                    "gold_count": 1,
                    "plausible_count": 10,
                    "distractor_count": 6,
                    "neutral_count": 3,
                },
                "gold_in_pool": True,
            }
        ],
    )
    write_jsonl(
        tmp_path / "context_sets_v1.jsonl",
        [
            {
                "set_id": "q_0001_set_03",
                "query_id": "q_0001",
                "candidate_pool_id": "pool_q_0001_v1",
                "strategy": "gold_plus_distractors",
                "selected_ids": ["pg16_auth_001", "pg15_auth_003", "pg16_conf_004"],
                "ordering_type": "best_first",
                "token_count": 501,
                "metadata": {
                    "contains_all_gold": True,
                    "missing_gold_count": 0,
                    "distractor_types": ["stale", "topical_wrong"],
                },
            }
        ],
    )
    write_jsonl(
        tmp_path / "outcomes_v1.jsonl",
        [
            {
                "set_id": "q_0001_set_03",
                "query_id": "q_0001",
                "answer": "The file is pg_hba.conf.",
                "scores": {"correctness": 1.0, "support": 1.0, "overall": 0.9},
                "prompt_tokens": 721,
                "completion_tokens": 39,
                "latency_ms": 2410,
                "evaluator_version": "eval_v1",
            }
        ],
    )
    write_jsonl(
        tmp_path / "marginal_impact_v1.jsonl",
        [
            {
                "query_id": "q_0001",
                "base_set_id": "q_0001_set_03",
                "chunk_id": "pg15_auth_003",
                "operation": "remove",
                "base_score": 0.90,
                "new_score": 0.95,
                "delta": 0.05,
            }
        ],
    )

    dataset = BenchmarkDataset.from_directory(tmp_path)

    assert len(dataset.corpus_chunks) == 1
    assert dataset.query_by_id()["q_0001"].metadata.question_family == "fact_lookup"
    assert dataset.context_set_by_id()["q_0001_set_03"].metadata.distractor_types == ["stale", "topical_wrong"]
