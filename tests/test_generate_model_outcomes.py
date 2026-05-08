from pathlib import Path

from context_engine.io import write_jsonl
from scripts.generate_model_outcomes import _load_existing_outcomes


def test_load_existing_outcomes_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert _load_existing_outcomes(tmp_path / "missing.jsonl") == []


def test_load_existing_outcomes_reads_rows(tmp_path: Path) -> None:
    target = tmp_path / "outcomes.jsonl"
    write_jsonl(target, [{"set_id": "s1", "query_id": "q1"}])
    rows = _load_existing_outcomes(target)
    assert rows == [{"query_id": "q1", "set_id": "s1"}]
