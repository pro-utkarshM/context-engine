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


## Project Status

The v1 benchmark has been built and validated. The current state is:

- **10 PostgreSQL documentation queries** (development set; confirmatory follow-up planned).
- **5 model replications** per (query, strategy) combination.
- **5 canonical strategies** (gold_only, gold_plus_distractors, minimal_support, shuffled_order, topk_pool_order) plus the **learned_v3** estimator.
- **3 prompt policies** (always_question_first, always_context_first, adaptive_by_chunk_count).

### Thesis-validation status (audited)

The audit-grade paired query comparison (per-query mean, n=10, bootstrap 2000 rep, seed=0):

| Comparison | delta | 95% CI | p | Verdict |
|---|---:|---|---:|---|
| learned_v3_context_first vs topk_pool_order | +0.0588 | [+0.0000, +0.1260] | 0.0255 | **provisional validated** |
| learned_v3_context_first vs gold_plus_distractors | +0.0108 | [-0.0314, +0.0703] | 0.3515 | provisional in trend |
| learned_v3_context_first vs gold_only | +0.0474 | [-0.0124, +0.1402] | 0.1260 | provisional in trend |
| learned_v3_context_first vs minimal_support | -0.0030 | [-0.0288, +0.0312] | 0.3575 | tied |
| learned_v3_context_first vs shuffled_order | +0.0168 | [-0.0168, +0.0588] | 0.2865 | provisional in trend |

**Reading**:

- The learned selector **provisional validated** (5% level) against
  the static retrieval baseline (`topk_pool_order`). The CI lower bound
  is exactly zero (to 4 decimals), so the verdict is "provisional
  validated at the borderline of 95% significance".
- The learned selector is **competitive** with the oracle-informed
  reference (`gold_plus_distractors`) — the CI crosses 0, so we
  cannot reject the null of no difference.
- The thesis is **NOT yet formally validated**; the 10-query
  development corpus is too small to make a broad claim. The
  confirmatory test is the ~30-query expansion planned for after the
  policy is frozen.

### Prompt-policy choice

The three-policy ablation showed:

- For **single-chunk contexts**, Question-first is significantly
  better than Context-first (delta = +0.084 favoring Q-first, p < 0.001).
- For **multi-chunk contexts**, Context-first is marginally better
  than Question-first (delta = +0.011 favoring C-first, p = 0.20, not
  significant).
- The **adaptive policy** (Q-first for `chunk_count <= 2`, C-first
  otherwise) is dominated: it picks the best of both worlds on the
  development set.

The adaptive policy is **NOT yet the default** — it is the candidate
for the prompt policy freeze, pending the confirmatory test on the
~30-query held-out corpus.

### Development-vs-confirmatory framing

The 10-query corpus has been used to develop the per-query marginal
impact estimator, identify ordering sensitivity, identify prompt-order
sensitivity, and motivate the adaptive prompt policy. These 10 queries
are now treated as **development-only**. The confirmatory thesis test
requires the ~30-query expansion; the current numbers are provisional
only.

See `.planning/STATISTICAL_AUDIT.md` for the audit methodology,
`.planning/THESIS_VALIDATION.md` for the formal framing, and
`.planning/R4_PROMPT_ABLATION.md` for the three-policy ablation details.
