from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.prompting import assemble_prompt


def test_assemble_prompt_includes_query_and_ordered_chunks() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0001",
            "query": "Which configuration file controls client authentication rules in PostgreSQL?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "The file is pg_hba.conf.",
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
            "selected_ids": ["c1", "c2"],
            "ordering_type": "best_first",
            "token_count": 123,
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
                "token_count": 50,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "file"},
            }
        ),
        "c2": CorpusChunk.from_dict(
            {
                "chunk_id": "c2",
                "doc_version": "16",
                "doc_path": "auth-pg-hba-conf.html",
                "section_path": ["Client Authentication"],
                "source_type": "doc",
                "text": "Chunk two text.",
                "token_count": 73,
                "chunk_index": 2,
                "prev_chunk_id": "c1",
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "rules"},
            }
        ),
    }

    payload = assemble_prompt(query=query, context_set=context_set, chunks_by_id=chunks_by_id)

    assert "Question:" in payload.prompt
    assert "Chunk one text." in payload.prompt
    assert "Chunk two text." in payload.prompt
    assert payload.estimated_prompt_tokens == 123
