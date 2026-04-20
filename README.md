# Context Engine

Context Engine is a research-driven project for building a utility-aware context selection system for LLMs.

The core idea is simple:

- standard retrieval optimizes for similarity
- useful context should optimize for downstream task success

This project is built around that distinction. The goal is to evaluate whether a learned selector can choose better context than static retrieval baselines when both are given the same fixed candidate pool and token budget.

## Project Thesis

LLM systems often fail not because the model lacks capability, but because the wrong supporting context is selected and injected into the prompt.

This repository is building a benchmark and experimental stack to test the following claim:

`Given the same candidate pool, a learned selector can outperform static retrieval or heuristic reranking on answer quality and distractor robustness.`

## Current Scope

The first benchmark is:

- `PG-Context-Select-v1`
- corpus: PostgreSQL documentation
- task: technical documentation QA
- focus: fixed candidate pools, typed distractors, and signed marginal-impact analysis

The benchmark is designed to measure:

- answer correctness
- support grounding
- token efficiency
- robustness under distractor context

## Repository Status

The repository currently contains the frozen design contracts that code will be built against:

- [docs/data-contract.md](docs/data-contract.md)
- [docs/component-interface-spec.md](docs/component-interface-spec.md)
- [docs/experiment-config-format.md](docs/experiment-config-format.md)
- [docs/contribution-guide.md](docs/contribution-guide.md)

These docs define:

- benchmark artifact schemas
- module boundaries and I/O contracts
- experiment reproducibility format
- contribution expectations

## Planned System

The implementation will be built in stages:

1. Corpus ingestion and chunking
2. Candidate retrieval and fixed pool generation
3. Context-set generation with distractor-aware variants
4. Prompt assembly and answer generation
5. Evaluation and marginal-impact analysis
6. Learned selector training and comparison against baselines

The first version is intentionally narrow. It does not begin with RL, graphs, or a custom transformer. It starts with a clean benchmark, a fixed retrieval/selection split, and a measurable selector improvement target.

## Design Principles

- Freeze the benchmark contract before writing code.
- Separate retrieval from selection during evaluation.
- Manufacture training signal through counterfactual context sets.
- Treat marginal impact as signed utility, not assumed benefit.
- Prefer controlled experiments over broad architecture claims.

## Immediate Next Steps

- finalize PostgreSQL corpus versions and section subset
- hand-chunk one section and verify chunking rules
- write and annotate a small manual query set
- implement artifact models and dataset loaders from the data contract

## One-Line Summary

Context Engine is a benchmark and system for learning what context an LLM should actually see, instead of guessing with similarity search.
