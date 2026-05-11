# Run Model-Backed Outcomes

This document explains how to run the model-backed outcome pipeline on the `model-backed-runner` branch.

## What This Path Does

The model-backed runner takes:

- `queries_v1.jsonl`
- `corpus_chunks_v1.jsonl`
- `context_sets_v1.jsonl`

and produces:

- `outcomes_model_stub_v1.jsonl` when using the stub runner
- `outcomes_model_openai_v1.jsonl` when using the OpenAI runner

## Prerequisites

- Python 3.11+
- repository checked out on the `model-backed-runner` branch
- benchmark artifacts already present in `data/processed/`

Optional but recommended:

- a local `.env` file based on `.env.example`

## Environment Setup

Create a `.env` file in the repository root:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
OPENAI_BASE_URL=https://api.openai.com/v1
```

Notes:

- `OPENAI_API_KEY` is required only for the real OpenAI runner.
- `OPENAI_MODEL` is optional. If omitted, the script defaults to `gpt-5`.
- `OPENAI_BASE_URL` is optional unless you need a custom OpenAI-compatible endpoint.

## Run The Stub Runner

This path does not call the network. It is useful for testing the model-backed pipeline shape.

```bash
PYTHONPATH=src python scripts/generate_model_outcomes.py --runner stub
```

Output:

```text
data/processed/outcomes_model_stub_v1.jsonl
```

Validate the generated artifact:

```bash
PYTHONPATH=src python -m context_engine.cli data/processed/outcomes_model_stub_v1.jsonl --artifact outcomes
```

## Run The OpenAI Runner

This path calls the OpenAI Responses API.

```bash
PYTHONPATH=src python scripts/generate_model_outcomes.py --runner openai
```

If you want to override the model explicitly:

```bash
PYTHONPATH=src python scripts/generate_model_outcomes.py --runner openai --model gpt-5
```

Output:

```text
data/processed/outcomes_model_openai_v1.jsonl
```

Validate the generated artifact:

```bash
PYTHONPATH=src python -m context_engine.cli data/processed/outcomes_model_openai_v1.jsonl --artifact outcomes
```

## Run Tests

```bash
python -m pytest -q
```

## Common Failure Cases

`OPENAI_API_KEY is required`
- Your `.env` is missing `OPENAI_API_KEY`, or the script is being run from a different working directory.

`OpenAI API request failed`
- Check network access, API key validity, and `OPENAI_BASE_URL`.

`FileNotFoundError` for benchmark artifacts
- Make sure these exist in `data/processed/`:
  - `corpus_chunks_v1.jsonl`
  - `queries_v1.jsonl`
  - `context_sets_v1.jsonl`

## Current Limitations

- The OpenAI runner is implemented with plain HTTP and minimal response parsing.
- Latency is currently reported as `0`; real timing instrumentation has not been added yet.
- The benchmark evaluator is still rule-based.
