from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.evaluation import evaluate_context_set, generate_baseline_answer, score_correctness


def _chunk(chunk_id: str, text: str, token_count: int) -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "auth-pg-hba-conf.html",
            "section_path": ["Client Authentication", "The pg_hba.conf File"],
            "source_type": "doc",
            "text": text,
            "token_count": token_count,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": "authentication", "subtopic": "test"},
        }
    )


def _query() -> Query:
    return Query.from_dict(
        {
            "query_id": "q_0001",
            "query": "Which configuration file controls client authentication rules in PostgreSQL?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "The file is pg_hba.conf.",
            "gold_support_ids": ["gold_a"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "fact_lookup",
            },
        }
    )


def _context_set() -> ContextSet:
    return ContextSet.from_dict(
        {
            "set_id": "q_0001_gold_only",
            "query_id": "q_0001",
            "candidate_pool_id": "pool_q_0001_v1",
            "strategy": "gold_only",
            "selected_ids": ["gold_a"],
            "ordering_type": "best_first",
            "token_count": 100,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": [],
            },
        }
    )


def test_generate_baseline_answer_uses_first_chunk_sentence() -> None:
    answer = generate_baseline_answer(
        _query(),
        _context_set(),
        {"gold_a": _chunk("gold_a", "The file is pg_hba.conf. More detail follows.", 100)},
    )
    assert answer == "The file is pg_hba.conf."


def test_evaluate_context_set_scores_exact_match() -> None:
    outcome = evaluate_context_set(
        query=_query(),
        context_set=_context_set(),
        chunks_by_id={"gold_a": _chunk("gold_a", "The file is pg_hba.conf. More detail follows.", 100)},
    )
    assert outcome.scores.correctness == 1.0
    assert outcome.scores.support == 1.0
    assert outcome.prompt_tokens == 100


def test_generate_baseline_answer_prefers_better_matching_later_chunk() -> None:
    answer = generate_baseline_answer(
        Query.from_dict(
            {
                "query_id": "q_0002",
                "query": "What additional permission is required after pg_hba.conf checks succeed?",
                "task_type": "doc_qa",
                "difficulty": "easy",
                "gold_answer": "The user still needs CONNECT privilege on the target database.",
                "gold_support_ids": ["gold_b"],
                "metadata": {
                    "topic": "authentication",
                    "requires_multi_hop": False,
                    "question_family": "fact_lookup",
                },
            }
        ),
        ContextSet.from_dict(
            {
                "set_id": "q_0002_topk_pool_order",
                "query_id": "q_0002",
                "candidate_pool_id": "pool_q_0002_v1",
                "strategy": "topk_pool_order",
                "selected_ids": ["distractor_a", "gold_b"],
                "ordering_type": "pool_order",
                "token_count": 150,
                "metadata": {
                    "contains_all_gold": True,
                    "missing_gold_count": 0,
                    "distractor_types": ["unknown"],
                },
            }
        ),
        {
            "distractor_a": _chunk("distractor_a", "This section discusses reloading configuration files.", 70),
            "gold_b": _chunk(
                "gold_b",
                "After pg_hba.conf checks succeed, the user still needs CONNECT privilege on the target database.",
                80,
            ),
        },
    )
    assert answer == "The user still needs CONNECT privilege on the target database."


def test_generate_baseline_answer_extracts_file_span_from_explanatory_sentence() -> None:
    answer = generate_baseline_answer(
        _query(),
        _context_set(),
        {
            "gold_a": _chunk(
                "gold_a",
                "PostgreSQL 16 says client authentication is controlled by the pg_hba.conf file in the cluster data directory unless hba_file points elsewhere.",
                100,
            )
        },
    )
    assert answer == "The file is pg_hba.conf."


def test_generate_baseline_answer_extracts_setting_name() -> None:
    answer = generate_baseline_answer(
        Query.from_dict(
            {
                "query_id": "q_0005",
                "query": "What server setting can prevent remote TCP/IP connections even if host rules exist in pg_hba.conf?",
                "task_type": "doc_qa",
                "difficulty": "medium",
                "gold_answer": "The required server setting is listen_addresses.",
                "gold_support_ids": ["gold_a"],
                "metadata": {
                    "topic": "authentication",
                    "requires_multi_hop": False,
                    "question_family": "troubleshooting",
                },
            }
        ),
        _context_set(),
        {
            "gold_a": _chunk(
                "gold_a",
                "Remote TCP/IP access also depends on listen_addresses, because the default server behavior only listens on the loopback address localhost.",
                100,
            )
        },
    )
    assert answer == "The required server setting is listen_addresses."


def test_score_correctness_accepts_exact_file_span() -> None:
    assert score_correctness(_query(), "pg_hba.conf") == 0.7


def test_score_correctness_accepts_exact_setting_span() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0005",
            "query": "What server setting can prevent remote TCP/IP connections even if host rules exist in pg_hba.conf?",
            "task_type": "doc_qa",
            "difficulty": "medium",
            "gold_answer": "Remote TCP/IP connections require an appropriate listen_addresses setting, because the default server behavior listens only on localhost.",
            "gold_support_ids": ["gold_a"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "troubleshooting",
            },
        }
    )
    assert score_correctness(query, "listen_addresses") == 0.7


def test_score_correctness_accepts_privilege_span() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0010",
            "query": "What additional permission is required after pg_hba.conf checks succeed?",
            "task_type": "doc_qa",
            "difficulty": "medium",
            "gold_answer": "Even after a connection passes pg_hba.conf checks, the user still needs CONNECT privilege on the target database.",
            "gold_support_ids": ["gold_a"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "troubleshooting",
            },
        }
    )
    assert score_correctness(query, "CONNECT privilege on the target database") == 0.7


def test_score_correctness_accepts_peer_ident_paraphrase() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0008",
            "query": "What is the difference between peer and ident authentication for local versus TCP/IP connections?",
            "task_type": "doc_qa",
            "difficulty": "medium",
            "gold_answer": "peer is only for local connections, while ident is for TCP/IP and if ident is specified for a local connection PostgreSQL uses peer instead.",
            "gold_support_ids": ["gold_a"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "comparison",
            },
        }
    )
    answer = "peer is available only for local connections; ident is only for TCP/IP and if ident is specified for a local connection PostgreSQL uses peer."
    assert score_correctness(query, answer) == 0.7
