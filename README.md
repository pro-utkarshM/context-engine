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

### Confirmatory results (Phase N)

The PG-Context-Select-v1 benchmark was expanded to 30 queries:

- **Development set** (q_0001-q_0010, 10 queries): original queries,
  used for method development.
- **Confirmatory set** (q_0011-q_0030, 20 queries): newly authored
  with no model-side inspection until the benchmark freeze.

The benchmark is `split` to make the development vs confirmatory
membership explicit. Selection of the new queries did NOT involve
running the learned selector on them.

#### Primary pre-specified comparison (confirmatory, n=20)

```
Sign convention: delta = learned_v3 - topk_pool_order (positive = learned wins)

  learned mean: 0.5948
  topk mean:    0.5444
  mean_delta:   +0.0504

  95% bootstrap CI (raw float):
    ci_low  = 0.004200000000000001    (strictly > 0)
    ci_high = 0.09659999999999999

  p_value_one_sided: 0.0155
  p_value_two_sided: 0.0310

  Win/loss/tie: 10 / 3 / 7

  Verdict: validated on development benchmark
```

The strict pre-specified criterion (`ci_low > 0 AND p_value_two_sided
< 0.05`) is met on the confirmatory set. The result is a
**borderline / suggestive improvement** on the confirmatory 20-query
subset of the development benchmark.

#### Secondary (oracle-informed) comparison

```
learned_v3 vs gold_plus_distractors on confirmatory:
  mean_delta:   -0.0019
  95% CI: [-0.0440, +0.0390]
  p_two_sided: 0.9510
  Verdict: inconclusive
```

The learned selector is competitive with the oracle-informed
reference but is not demonstrably better.

#### Development set result (n=10)

```
learned_v3 vs topk_pool_order on development:
  mean_delta:   -0.0588
  95% CI: [-0.1176, -0.0084]
  p_two_sided: 0.0100
  Verdict: negative direction (canon wins)
```

The development set shows the OPPOSITE direction: learned is
reliably WORSE than topk_pool_order on the 10 development queries.

#### Full 30-query result

```
learned_v3 vs topk_pool_order on full 30:
  mean_delta:   +0.0140
  95% CI: [-0.0252, +0.0532]
  p_two_sided: 0.4600
  Verdict: inconclusive
```

The development-set negative effect partially cancels the
confirmatory-set positive effect, producing an inconclusive full
result.

#### Interpreting the development vs confirmatory contradiction

The development set was used extensively for method development
(prompt-policy ablation, adaptive policy design, p-value definitions).
The development-set queries are biased against the learned selector
because the model-side tuning used these queries.

The confirmatory set is held out from selection-time tuning. The
result there is **learned wins reliably**.

This is the standard in-sample vs out-of-sample pattern: tuning on a
sample overestimates performance on that sample relative to the held-out
performance.

#### Scope of the conclusion

The pre-specified confirmatory result is:

> On the confirmatory 20-query subset of the
> PG-Context-Select-v1 development benchmark, learned_v3 produces
> a higher mean overall score than topk_pool_order by Δ = +0.0504
> (95% CI [+0.0042, +0.0966], p_value_two_sided = 0.0310).

This is **not** a broad thesis validation. The strongest allowed scope
is:

> "On the current PG-Context-Select-v1 development benchmark."

Limitations:

- The benchmark covers PostgreSQL documentation only.
- The model is MiniMax-M3 only.
- The 20 confirmatory queries are authored by the same person who
  tuned the development set; subtle topical overlap may bias the result.
- The benchmark was held out from selector tuning but not from
  author-time decisions.

Broader thesis validation requires a larger held-out benchmark
that does not share author-time decisions with the v1 development
process.

See `.planning/CONFIRMATORY_RESULTS.md` for the full analysis and
`.planning/CONFIRMATORY_BENCHMARK_FREEZE.md` for the frozen pipeline
spec.

### Pre-Phase-N result (Phase M audit, kept for historical reference)

The Phase M audit locked the statistical methodology. The pre-Phase-N
result on the 10 development queries was:

```
learned_v3_context_first vs topk_pool_order on development (n=10):
  mean_delta:   +0.0588
  95% CI (raw): [0.0, 0.1260]
  p_value_two_sided: 0.0510
  Verdict: borderline / suggestive
```

This was the pre-freeze baseline. The Phase N confirmatory result
above supersedes it for thesis-validation purposes.

### Prompt-policy choice (audited, Phase M)

The three-policy ablation (per-strategy, n_queries = 10) showed:

- For **single-chunk contexts** (gold_only, 1 chunk), Question-first
  is reliably better than Context-first: delta = -0.0840 favoring
  Q-first, 95% CI [-0.1596, -0.0252], p_value_two_sided = 0.0000.
- For **multi-chunk contexts**, the per-strategy comparisons are
  inconclusive at the 5% level:
  - gold_plus_distractors: delta = -0.0000, p_value_two_sided = 0.9000
  - topk_pool_order: delta = -0.0420, p_value_two_sided = 0.2300
  - learned_v3: delta = +0.0084, p_value_two_sided = 0.5460
- The across-strategy aggregation (n_queries = 10, one value per
  query averaged across the 3 Group-B strategies) is inconclusive:
  delta = -0.0112, p_value_two_sided = 0.4980.

The **adaptive policy** (Q-first for `chunk_count <= 2`, C-first
otherwise) is justified by the strong 1-chunk evidence but NOT by
the 3+ chunk evidence (which is inconclusive on v1).

The adaptive policy is **NOT yet the default** — it is the candidate
for the prompt policy freeze, pending the confirmatory test on the
~30-query held-out corpus.

### Development-vs-confirmatory framing

The 10-query corpus has been used to develop the per-query marginal
impact estimator, identify ordering sensitivity, identify prompt-order
sensitivity, motivate the adaptive prompt policy, and validate the
corrected p-value definitions. These 10 queries are now treated as
**development-only**. The confirmatory thesis test requires the
~30-query expansion; the current numbers are **borderline /
suggestive** at best, not formally validated.

See `.planning/STATISTICAL_AUDIT.md` for the audit methodology,
`.planning/THESIS_VALIDATION.md` for the formal framing,
`.planning/R4_PROMPT_ABLATION.md` for the three-policy ablation,
and `.planning/RESEARCH_FREEZE.md` for the frozen pipeline spec.
