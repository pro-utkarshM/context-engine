# Component Interface Spec

Version: `v1`

This document defines module boundaries and I/O contracts. It intentionally excludes implementation details.

## Shared Types

- `Query`: benchmark query object from `queries_v1.jsonl`
- `Chunk`: corpus chunk object from `corpus_chunks_v1.jsonl`
- `CandidatePool`: pool object from `candidate_pools_v1.jsonl`
- `ContextSet`: set object from `context_sets_v1.jsonl`
- `Outcome`: result object from `outcomes_v1.jsonl`

## Corpus Loader

Input:
- path to `corpus_chunks_v1.jsonl`

Output:
- iterable of `Chunk`
- random access by `chunk_id`

## Query Loader

Input:
- path to `queries_v1.jsonl`

Output:
- iterable of `Query`
- random access by `query_id`

## Retriever

Input:
- `query: string`
- `pool_size: integer`
- optional corpus filter metadata

Output:
- ordered list of retrieval results

Retrieval result item:
- `chunk_id: string`
- `score: number`
- `retriever_name: string`

Contract:
- Retriever does not enforce token budget.
- Retriever does not mutate corpus artifacts.

## Candidate Pool Builder

Input:
- `query: Query`
- retrieval results from one or more retrievers
- chunk lookup by `chunk_id`
- optional injected distractor candidates

Output:
- `CandidatePool`

Contract:
- must produce a fixed pool for downstream selector evaluation
- must ensure `gold_in_pool=true` for `v1`

## Selector

Input:
- `query: Query`
- `candidate_pool: CandidatePool`
- resolved candidate chunks with metadata
- `token_budget: integer`
- `strategy: string`

Output:
- ordered list of selected chunk records

Selected chunk record:
- `chunk_id: string`
- `rank: integer`
- optional `selector_score: number`

Contract:
- output ordering is significant
- total selected token count must not exceed token budget
- selector may use metadata, but must only select from the provided fixed pool

## Context Assembler

Input:
- `query: Query`
- ordered selected chunks
- prompt template identifier

Output:
- assembled prompt payload for model execution

Contract:
- preserves selector ordering
- computes prompt token estimate

## Answer Generator

Input:
- assembled prompt payload
- `model_name: string`

Output:
- `answer: string`
- usage metadata:
  - `prompt_tokens: integer`
  - `completion_tokens: integer`
  - `latency_ms: integer`

## Evaluator

Input:
- `query: Query`
- `answer: string`
- `selected_chunk_ids: string[]`
- optional selected chunk texts
- scoring weights

Output:
- evaluation result

Evaluation result fields:
- `correctness: number`
- `support: number`
- `overall: number`

Contract:
- evaluator must be deterministic for a fixed evaluator version and model configuration
- evaluator must not alter benchmark source data

## Marginal Impact Analyzer

Input:
- `query: Query`
- `base_context_set: ContextSet`
- `chunk_id: string`
- `operation: add | remove`
- same evaluator configuration used for base run

Output:
- marginal impact record from `marginal_impact_v1.jsonl`

Contract:
- delta is signed
- add/remove may either improve or degrade score

## Experiment Runner

Input:
- experiment config
- dataset split identifier

Output:
- generated `ContextSet` rows
- generated `Outcome` rows
- optional `marginal_impact` rows

Contract:
- one config file must be sufficient to reproduce a run
- all outputs must record versioned artifact references
