# Experiment Config Format

Every experiment must be reproducible from one config file.

Format: YAML or JSON with the following required keys:

- `experiment_name`
- `model_name`
- `dataset_name`
- `dataset_split`
- `candidate_pool_version`
- `selector_strategy`
- `token_budget`
- `scoring_weights`
- `evaluator_version`

Required scoring weights for `v1`:

- `correctness: 0.6`
- `support: 0.3`
- `efficiency: 0.1`

Example:

```yaml
experiment_name: pg_v1_hybrid_selector_budget_1500
model_name: gpt-4.1-mini
dataset_name: PG-Context-Select-v1
dataset_split: dev
candidate_pool_version: candidate_pools_v1.jsonl
selector_strategy: topk_hybrid
token_budget: 1500
evaluator_version: eval_v1
scoring_weights:
  correctness: 0.6
  support: 0.3
  efficiency: 0.1
```

Do not adjust scoring weights after results are observed.
