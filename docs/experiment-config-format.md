# Experiment Config Format

Every experiment must be reproducible from one config file.

Format: **JSON** (no YAML dependency in this repo).

## Required Keys

- `experiment_name`
- `model_name`
- `selector_strategy`
- `evaluator_version`

## Optional Keys (with defaults)

- `dataset_dir` — directory containing versioned JSONL artifacts. Default: `data/processed`.
- `artifact_version` — major version suffix used to resolve `<artifact>_<version>.jsonl`. Default: `v1`.
- `token_budget` — maximum prompt tokens enforced by selector and used by the efficiency score. Default: `1500`.
- `scoring_weights` — object with `correctness`, `support`, `efficiency` floats. Default: `0.6 / 0.3 / 0.1`.
- `seed` — RNG seed for any non-deterministic generation. Default: `1337`.

Required scoring weights for `v1` (frozen — **do not change after results are observed**):

- `correctness: 0.6`
- `support: 0.3`
- `efficiency: 0.1`

Any additional keys are kept under `ExperimentConfig.extra` for downstream phases (e.g., `learned_model_path`, `selector_training_artifact`).

## Example

```json
{
  "experiment_name": "pg_v1_baseline_strategies",
  "model_name": "gpt-5",
  "selector_strategy": "default",
  "evaluator_version": "eval_v1_model_runner",
  "dataset_dir": "data/processed",
  "artifact_version": "v1",
  "token_budget": 1500,
  "scoring_weights": {
    "correctness": 0.6,
    "support": 0.3,
    "efficiency": 0.1
  },
  "seed": 1337
}
```

## CLI Integration

All generation and analysis scripts accept the shared flags:

- `--config <path>` — load defaults from a JSON config file.
- `--dataset-dir <path>` — override `dataset_dir` (or set it when no config is provided).
- `--artifact-version <ver>` — override `artifact_version`.

Per-script flags such as `--output`, `--context-sets`, `--outcomes` continue to accept explicit paths and take precedence over the resolved defaults.

Example:

```bash
PYTHONPATH=src python scripts/generate_context_sets.py --config configs/experiment_v1_baseline.json
PYTHONPATH=src python scripts/generate_outcomes.py    --config configs/experiment_v1_baseline.json
PYTHONPATH=src python scripts/generate_model_outcomes.py --config configs/experiment_v1_baseline.json --runner openai
PYTHONPATH=src python scripts/analyze_results.py     --config configs/experiment_v1_baseline.json
```
