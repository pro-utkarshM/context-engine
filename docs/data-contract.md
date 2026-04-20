# Data Contract

Version: `v1`

This document freezes the on-disk data formats for `PG-Context-Select-v1`. All benchmark artifacts are JSON Lines files (`.jsonl`), one JSON object per line, UTF-8 encoded.

## Naming And Versioning

- File naming convention: `<artifact_name>_v<major>.jsonl`
- Initial files:
  - `corpus_chunks_v1.jsonl`
  - `queries_v1.jsonl`
  - `candidate_pools_v1.jsonl`
  - `context_sets_v1.jsonl`
  - `outcomes_v1.jsonl`
  - `marginal_impact_v1.jsonl`
- Backward-incompatible schema changes require a new major version.
- Additive optional fields may be introduced within the same major version.

## Global Conventions

- IDs are strings and must be globally unique within their artifact type.
- Required fields must always be present.
- Optional fields may be omitted or set to `null` only where explicitly allowed.
- `gold_support_ids` is nullable for future task-success-only benchmarks, but must be non-null in `v1`.
- Scores are floating-point values in `[0.0, 1.0]` unless otherwise noted.

## 1. `corpus_chunks_v1.jsonl`

Represents the chunked corpus.

Required fields:

- `chunk_id: string`
- `doc_version: string`
- `doc_path: string`
- `section_path: string[]`
- `source_type: string`
- `text: string`
- `token_count: integer`
- `chunk_index: integer`
- `prev_chunk_id: string | null`
- `next_chunk_id: string | null`
- `metadata: object`

Optional fields:

- none in `v1`

Example:

```json
{
  "chunk_id": "pg16_auth_001",
  "doc_version": "16",
  "doc_path": "client-authentication.md",
  "section_path": ["Client Authentication", "The pg_hba.conf File"],
  "source_type": "doc",
  "text": "The pg_hba.conf file controls client authentication...",
  "token_count": 148,
  "chunk_index": 1,
  "prev_chunk_id": null,
  "next_chunk_id": "pg16_auth_002",
  "metadata": {
    "topic": "authentication",
    "subtopic": "pg_hba.conf"
  }
}
```

## 2. `queries_v1.jsonl`

Represents benchmark questions and answer annotations.

Required fields:

- `query_id: string`
- `query: string`
- `task_type: string`
- `difficulty: string`
- `gold_answer: string`
- `gold_support_ids: string[]`
- `metadata: object`

Optional fields:

- `gold_support_ids: null` is reserved for future versions but should not be used in `v1`

Example:

```json
{
  "query_id": "q_0001",
  "query": "Which file controls client authentication rules in PostgreSQL?",
  "task_type": "doc_qa",
  "difficulty": "easy",
  "gold_answer": "pg_hba.conf controls client authentication rules.",
  "gold_support_ids": ["pg16_auth_001"],
  "metadata": {
    "topic": "authentication",
    "requires_multi_hop": false,
    "question_family": "fact_lookup"
  }
}
```

## 3. `candidate_pools_v1.jsonl`

Represents fixed candidate pools for selector evaluation.

Required fields:

- `query_id: string`
- `candidate_pool_id: string`
- `candidate_ids: string[]`
- `composition: object`
- `gold_in_pool: boolean`

Required `composition` fields:

- `gold_count: integer`
- `plausible_count: integer`
- `distractor_count: integer`
- `neutral_count: integer`

Optional fields:

- none in `v1`

Example:

```json
{
  "query_id": "q_0001",
  "candidate_pool_id": "pool_q_0001_v1",
  "candidate_ids": [
    "pg16_auth_001",
    "pg15_auth_003",
    "pg16_roles_011",
    "pg16_conf_004",
    "pg16_auth_008"
  ],
  "composition": {
    "gold_count": 1,
    "plausible_count": 10,
    "distractor_count": 6,
    "neutral_count": 3
  },
  "gold_in_pool": true
}
```

## 4. `context_sets_v1.jsonl`

Represents concrete context selections produced from a fixed candidate pool.

Required fields:

- `set_id: string`
- `query_id: string`
- `candidate_pool_id: string`
- `strategy: string`
- `selected_ids: string[]`
- `ordering_type: string`
- `token_count: integer`
- `metadata: object`

Required `metadata` fields:

- `contains_all_gold: boolean`
- `missing_gold_count: integer`
- `distractor_types: string[]`

Optional fields:

- none in `v1`

Example:

```json
{
  "set_id": "q_0001_set_03",
  "query_id": "q_0001",
  "candidate_pool_id": "pool_q_0001_v1",
  "strategy": "gold_plus_distractors",
  "selected_ids": ["pg16_auth_001", "pg15_auth_003", "pg16_conf_004"],
  "ordering_type": "best_first",
  "token_count": 501,
  "metadata": {
    "contains_all_gold": true,
    "missing_gold_count": 0,
    "distractor_types": ["stale", "topical_wrong"]
  }
}
```

## 5. `outcomes_v1.jsonl`

Represents model outputs and evaluation results for a context set.

Required fields:

- `set_id: string`
- `query_id: string`
- `answer: string`
- `scores: object`
- `prompt_tokens: integer`
- `completion_tokens: integer`
- `latency_ms: integer`
- `evaluator_version: string`

Required `scores` fields:

- `correctness: number`
- `support: number`
- `overall: number`

Optional fields:

- none in `v1`

Example:

```json
{
  "set_id": "q_0001_set_03",
  "query_id": "q_0001",
  "answer": "The file is pg_hba.conf.",
  "scores": {
    "correctness": 1.0,
    "support": 1.0,
    "overall": 0.9
  },
  "prompt_tokens": 721,
  "completion_tokens": 39,
  "latency_ms": 2410,
  "evaluator_version": "eval_v1"
}
```

## 6. `marginal_impact_v1.jsonl`

Represents signed utility deltas for adding or removing a chunk from a context set.

Required fields:

- `query_id: string`
- `base_set_id: string`
- `chunk_id: string`
- `operation: string`
- `base_score: number`
- `new_score: number`
- `delta: number`

Constraints:

- `operation` must be one of `add` or `remove`
- `delta` is signed and may be positive or negative
- `delta` should equal `new_score - base_score`

Example:

```json
{
  "query_id": "q_0001",
  "base_set_id": "q_0001_set_03",
  "chunk_id": "pg15_auth_003",
  "operation": "remove",
  "base_score": 0.90,
  "new_score": 0.95,
  "delta": 0.05
}
```
