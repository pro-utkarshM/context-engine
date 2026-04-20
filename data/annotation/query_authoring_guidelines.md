# Query Authoring Guidelines

Write manual benchmark queries before attempting automation.

## Goal

Produce high-quality `queries_v1.jsonl` rows for `PG-Context-Select-v1` with:

- a clear question
- a single gold answer
- correct `gold_support_ids`
- consistent difficulty and question-family labels

## Query Families

Use this target mix for the first batch:

- `fact_lookup`
- `procedural`
- `comparison`
- `troubleshooting`
- `multi_hop`

## Rules

1. Write questions that can be answered from the selected PostgreSQL docs subset.
2. Prefer precise wording over open-ended phrasing.
3. Avoid questions with multiple equally valid answers unless the benchmark explicitly supports that.
4. Every query must map to at least one gold-support chunk.
5. Multi-hop queries must require at least two gold-support chunks.
6. If a chunk is helpful but not sufficient, do not mark it as gold support.

## Difficulty Labels

- `easy`: answer supported by one obvious chunk
- `medium`: answer supported by one non-obvious chunk or two easy chunks
- `hard`: requires synthesis, disambiguation, or multi-hop support

## Examples

`fact_lookup`
- Query: `Which file controls client authentication rules in PostgreSQL?`
- Gold answer: `pg_hba.conf controls client authentication rules.`

`procedural`
- Query: `How do you enable WAL archiving in PostgreSQL?`
- Gold answer: short, declarative summary of the required steps

`comparison`
- Query: `What is the difference between a role attribute and a database privilege?`

`multi_hop`
- Query: `Which configuration change enables feature X, and what matching server-side condition must also be satisfied?`

## Authoring Workflow

1. Pick one doc section.
2. Draft 2-3 questions from different families.
3. Write the gold answer.
4. Annotate `gold_support_ids`.
5. Validate that the answer is fully defensible from the gold-support chunks alone.

## Minimum Fields Per Query

- `query_id`
- `query`
- `task_type`
- `difficulty`
- `gold_answer`
- `gold_support_ids`
- `metadata.topic`
- `metadata.requires_multi_hop`
- `metadata.question_family`
