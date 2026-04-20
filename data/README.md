# Data Layout

This directory holds benchmark data and intermediate artifacts for `PG-Context-Select-v1`.

Expected layout:

```text
data/
  raw/
    postgres_v15/
    postgres_v16/
  processed/
    corpus_chunks_v1.jsonl
    queries_v1.jsonl
    candidate_pools_v1.jsonl
    context_sets_v1.jsonl
    outcomes_v1.jsonl
    marginal_impact_v1.jsonl
  splits/
    train_ids.json
    dev_ids.json
    test_ids.json
  annotation/
    distractor_guidelines.md
    query_authoring_guidelines.md
```

Notes:

- `raw/` contains source documentation snapshots.
- `processed/` contains versioned benchmark artifacts that match `docs/data-contract.md`.
- `splits/` contains document-level or section-level split assignments.
- `annotation/` contains compact instructions for manual query writing and distractor labeling.

The current repository only scaffolds this layout. Benchmark files should be created through explicit data-prep scripts or manual authoring steps, not ad hoc edits.
