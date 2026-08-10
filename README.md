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

- **Development subset** (q_0001-q_0010, 10 queries): original
  queries, used for method development.
- **Confirmatory subset** (q_0011-q_0030, 20 queries): newly
  authored with no model-side inspection until the benchmark
  freeze.

Each query's `metadata.split` field marks its membership. Selection
of the new queries did NOT involve running the learned selector on
them.

#### Primary pre-specified comparison (confirmatory subset, n=20)

```
Sign convention: delta = learned_v3 - topk_pool_order
(positive = learned wins)

  learned mean: 0.5948
  topk mean:    0.5444
  mean_delta:   +0.0504

  95% bootstrap CI (raw float):
    ci_low  = 0.004200000000000001    (strictly > 0)
    ci_high = 0.09659999999999999

  95% CI (4-decimal display): [+0.0042, +0.0966]

  p_value_one_sided: 0.0155
  p_value_two_sided: 0.0310

  Win/loss/tie: 10 / 3 / 7

  Verdict: validated on development benchmark
```

The strict pre-specified criterion (`ci_low > 0 AND p_value_two_sided
< 0.05`) is met on the confirmatory subset.

#### Other subsets

| Subset | n | learned mean | topk mean | mean_delta | 95% CI | p_two_sided | Win/Loss/Tie | Verdict |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Development (q_0001-q_0010) | 10 | 0.6681 | 0.7269 | -0.0588 | [-0.1176, -0.0084] | 0.0100 | 0/4/6 | negative direction |
| Confirmatory (q_0011-q_0030) | 20 | 0.5948 | 0.5444 | +0.0504 | [+0.0042, +0.0966] | 0.0310 | 10/3/7 | validated |
| Full (q_0001-q_0030) | 30 | 0.6192 | 0.6052 | +0.0140 | [-0.0252, +0.0532] | 0.4600 | 10/7/13 | inconclusive |

The development subset and the confirmatory subset show opposite
mean deltas. This is a descriptive observation, not a causal
explanation. Both numbers are computed correctly from their
respective (independent) frozen artifacts. The sign difference
reflects the corpus expansion: the BM25 retriever returns different
candidates against the 38-chunk corpus (vs. the original 8-chunk
corpus), which changes both the topk_pool_order and learned_v3
selections and therefore the per-query outcomes. See
`.planning/CONFIRMATORY_INTEGRITY_AUDIT.md` for the full trace.

#### Oracle-informed comparison (secondary)

```
learned_v3 vs gold_plus_distractors (oracle-informed):
  confirmatory:    delta=-0.0019, CI [-0.0440, +0.0390], p=0.9510 (inconclusive)
  development:     delta=-0.0390, CI [-0.0808, +0.0024], p=0.0590 (inconclusive)
  full:            delta=-0.0143, CI [-0.0448, +0.0184], p=0.3880 (inconclusive)
```

The learned selector is competitive with the oracle-informed
reference on all subsets but is not demonstrably better.

#### Multi-hop distribution

| Subset | n_queries | n_multi_hop | percentage |
|---|---:|---:|---:|
| Development | 10 | 0 | 0.0% |
| Confirmatory | 20 | 8 | 40.0% |
| Full | 30 | 8 | 26.7% |

The confirmatory subset meets the planning target (>= 30%
multi-hop in the new queries). The full benchmark rate is below
30% because the development subset has 0 multi-hop queries.

#### Scope of the conclusion

The pre-specified confirmatory result is:

> On the pre-frozen held-out 20-query confirmatory subset of
> PG-Context-Select-v1, learned_v3 achieved a +0.0504 higher mean
> overall score than the canonical static baseline topk_pool_order
> (95% CI [+0.0042, +0.0966], two-sided bootstrap p = 0.0310).

This result is specific to the current PostgreSQL benchmark and
MiniMax-M3; broader generalization has not yet been established.

Limitations:

- The benchmark covers PostgreSQL documentation only.
- The model is MiniMax-M3 only.
- The 20 confirmatory queries are authored by the same person who
  tuned the development set.
- The 5 reps per query are stochastic.
- Multiple-comparison correction across the 5 strategy comparisons
  is not applied.

See `.planning/CONFIRMATORY_RESULTS.md` for the full numerical
analysis and `.planning/CONFIRMATORY_INTEGRITY_AUDIT.md` for the
trace of the development vs confirmatory difference.

### Pre-Phase-N result (Phase M audit, kept for historical reference)

The Phase M audit locked the statistical methodology. On the
PR #21 frozen artifacts (8-chunk corpus, no confirmatory queries):

```
learned_v3_context_first vs topk_pool_order on development (n=10):
  mean_delta:   +0.0588
  95% CI (raw): [0.0, 0.1260]
  p_value_two_sided: 0.0510
  Verdict: borderline / suggestive
```

This was the pre-freeze baseline. The Phase N confirmatory result
above supersedes it for thesis-validation purposes. The numerical
value differs because Phase N expanded the corpus (8 → 38 chunks)
and regenerated all artifacts.

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
