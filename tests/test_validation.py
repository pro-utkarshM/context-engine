from pathlib import Path

from context_engine.io import write_jsonl
from context_engine.validation import infer_artifact_name, validate_jsonl_file


def test_infer_artifact_name_from_filename() -> None:
    assert infer_artifact_name("queries_v1.jsonl") == "queries"


def test_validate_jsonl_file_reads_contract_rows(tmp_path: Path) -> None:
    target = tmp_path / "queries_v1.jsonl"
    write_jsonl(
        target,
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

    summary = validate_jsonl_file(target)

    assert summary.artifact_name == "queries"
    assert summary.row_count == 1
