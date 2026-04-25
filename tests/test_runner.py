import json
from unittest.mock import patch

from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.model_outcomes import evaluate_with_runner
from context_engine.runner import OpenAIResponsesRunner, StubModelRunner


def test_evaluate_with_runner_uses_model_response() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0001",
            "query": "Which configuration file controls client authentication rules in PostgreSQL?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "stub_answer",
            "gold_support_ids": ["c1"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "fact_lookup",
            },
        }
    )
    context_set = ContextSet.from_dict(
        {
            "set_id": "q_0001_gold_only",
            "query_id": "q_0001",
            "candidate_pool_id": "pool_q_0001_v1",
            "strategy": "gold_only",
            "selected_ids": ["c1"],
            "ordering_type": "best_first",
            "token_count": 100,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": [],
            },
        }
    )
    chunks_by_id = {
        "c1": CorpusChunk.from_dict(
            {
                "chunk_id": "c1",
                "doc_version": "16",
                "doc_path": "auth-pg-hba-conf.html",
                "section_path": ["Client Authentication"],
                "source_type": "doc",
                "text": "Chunk one text.",
                "token_count": 100,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "file"},
            }
        )
    }

    outcome = evaluate_with_runner(
        query=query,
        context_set=context_set,
        chunks_by_id=chunks_by_id,
        runner=StubModelRunner(default_answer="stub_answer"),
        model_name="stub-model",
    )

    assert outcome.answer == "stub_answer"
    assert outcome.scores.correctness == 1.0
    assert outcome.evaluator_version == "eval_v1_model_runner"


def test_openai_runner_parses_output_text_response() -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": "The file is pg_hba.conf.",
                    "usage": {"input_tokens": 42, "output_tokens": 5},
                }
            ).encode("utf-8")

    with patch("context_engine.runner.request.urlopen", return_value=FakeResponse()):
        runner = OpenAIResponsesRunner(api_key="test-key")
        response = runner.run(
            type("Payload", (), {"prompt": "prompt", "estimated_prompt_tokens": 10})(),
            model_name="gpt-5",
        )

    assert response.answer == "The file is pg_hba.conf."
    assert response.prompt_tokens == 42
    assert response.completion_tokens == 5
