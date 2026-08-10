# Confirmatory Results — Phase N

Written 2026-08-09, updated 2026-08-09 after Phase O integrity audit.

This document reports the confirmatory analysis results. The analysis
path is shared with development and full-benchmark views via
`scripts/confirmatory_analysis_unified.py`.

## Method

The pre-specified primary comparison is `learned_v3` vs
`topk_pool_order` on the confirmatory subset (q_0011-q_0030, n=20).
The frozen statistical procedure (from `.planning/STATISTICAL_AUDIT.md`):

- Independent experimental unit: query.
- Repetitions per query: 5.
- Bootstrap unit: per-query paired delta.
- Bootstrap: percentile, 2000 resamples, seed 0.
- p_value_one_sided: opposite-sign share, floored at 1/n_resamples.
- p_value_two_sided: min(1.0, 2 * min(p_lower, p_upper)).
- Sign convention: delta = learned_v3 - comparison; positive = learned wins.

Pre-specified verdict criterion: `ci_low > 0 AND p_value_two_sided < 0.05`
= "validated on development benchmark".

## Primary pre-specified comparison

### Confirmatory subset (q_0011-q_0030, n=20)

```
n_queries: 20 (independent)
reps_per_query: 5

learned mean: 0.5948
topk mean:    0.5444
mean_delta:   +0.0504

95% bootstrap CI (raw float):
  ci_low  = 0.004200000000000001    (strictly > 0)
  ci_high = 0.09659999999999999

95% bootstrap CI (display): [+0.0042, +0.0966]

p_value_one_sided: 0.0155
p_value_two_sided: 0.0310   (two-sided < 0.05)

Win/loss/tie: 10 / 3 / 7

Verdict: validated on development benchmark
```

### Per-query deltas (confirmatory subset)

```
q_0011 ( WIN): delta=+0.0840, replication, medium, single-hop
q_0012 ( WIN): delta=+0.0840, replication, medium, single-hop
q_0013 ( TIE): delta=+0.0000, replication, hard, multi-hop
q_0014 ( TIE): delta=+0.0000, replication, medium, single-hop
q_0015 ( TIE): delta=+0.0000, replication, medium, single-hop
q_0016 ( WIN): delta=+0.0840, partitioning, medium, multi-hop
q_0017 ( TIE): delta=+0.0000, partitioning, medium, single-hop
q_0018 (LOSS): delta=-0.0840, partitioning, medium, single-hop
q_0019 ( WIN): delta=+0.0840, partitioning, hard, single-hop
q_0020 ( TIE): delta=+0.0000, configuration, medium, multi-hop
q_0021 (LOSS): delta=-0.1680, configuration, medium, multi-hop
q_0022 ( WIN): delta=+0.1680, configuration, hard, multi-hop
q_0023 ( WIN): delta=+0.1680, configuration, easy, single-hop
q_0024 ( WIN): delta=+0.2520, authentication, medium, multi-hop
q_0025 ( WIN): delta=+0.0840, authentication, medium, multi-hop
q_0026 (LOSS): delta=-0.0840, authentication, hard, single-hop
q_0027 ( TIE): delta=+0.0000, roles, medium, single-hop
q_0028 ( TIE): delta=+0.0000, roles, hard, single-hop
q_0029 ( WIN): delta=+0.2520, backup, medium, multi-hop
q_0030 ( WIN): delta=+0.0840, logging, medium, single-hop
```

## Development subset (q_0001-q_0010, n=10)

For comparison with the confirmatory result, the same analysis on the
development subset:

```
learned_v3 vs topk_pool_order on development subset:
  learned mean: 0.6681
  topk mean:    0.7269
  mean_delta:   -0.0588
  95% CI (raw): [-0.1176, -0.0084]
  p_value_one_sided: 0.0005
  p_value_two_sided: 0.0100
  Win/loss/tie: 0 / 4 / 6
  Verdict: negative direction (canon wins)
```

The development subset shows the OPPOSITE direction. The exact same
magnitude (+/- 0.0588) is a consequence of the corpus expansion in
Phase N: the BM25 retriever returns different candidates against the
expanded corpus, which changes both the topk_pool_order selection and
the learned_v3 selection. See `.planning/CONFIRMATORY_INTEGRITY_AUDIT.md`
for the full trace.

## Full PG-Context-Select-v1 benchmark (q_0001-q_0030, n=30)

```
learned_v3 vs topk_pool_order on full benchmark:
  learned mean: 0.6192
  topk mean:    0.6052
  mean_delta:   +0.0140
  95% CI (raw): [-0.0252, +0.0532]
  p_value_one_sided: 0.2300
  p_value_two_sided: 0.4600
  Win/loss/tie: 10 / 7 / 13
  Verdict: inconclusive
```

The development subset's negative effect and the confirmatory subset's
positive effect partially cancel, producing an inconclusive full result.

## Oracle-informed comparisons (secondary)

The oracle-informed reference (gold_plus_distractors) is the closest
heuristic that has access to gold annotation information at construction
time. It is a secondary comparison.

### Confirmatory subset

```
learned_v3 vs gold_plus_distractors (n=20):
  learned mean: 0.5948
  gpd mean:    0.5967
  mean_delta:   -0.0019
  95% CI (raw): [-0.0440, +0.0390]
  p_value_two_sided: 0.9510
  Win/loss/tie: 7 / 13 / 0
  Verdict: inconclusive
```

### Development subset

```
learned_v3 vs gold_plus_distractors (n=10):
  learned mean: 0.6681
  gpd mean:    0.7070
  mean_delta:   -0.0390
  95% CI (raw): [-0.0808, +0.0024]
  p_value_two_sided: 0.0590
  Win/loss/tie: 1 / 9 / 0
  Verdict: inconclusive
```

### Full benchmark

```
learned_v3 vs gold_plus_distractors (n=30):
  learned mean: 0.6192
  gpd mean:    0.6335
  mean_delta:   -0.0143
  95% CI (raw): [-0.0448, +0.0184]
  p_value_two_sided: 0.3880
  Win/loss/tie: 8 / 22 / 0
  Verdict: inconclusive
```

Across all three views, the learned selector is competitive with the
oracle-informed reference but is not demonstrably better.

## Subgroup descriptive analyses (NOT pre-specified)

These are exploratory subgroup observations from the confirmatory subset.
NOT formal hypothesis tests; do not over-interpret.

### Per-topic (confirmatory only)

| Topic | n_queries | learned mean | topk mean | mean delta |
|---|---|---:|---:|---:|
| authentication | 3 | 0.6181 | 0.5341 | +0.0840 |
| backup | 1 | 0.7872 | 0.5352 | +0.2520 |
| configuration | 4 | 0.6623 | 0.6203 | +0.0420 |
| logging | 1 | 0.4523 | 0.3683 | +0.0840 |
| partitioning | 4 | 0.5782 | 0.5572 | +0.0210 |
| replication | 5 | 0.6209 | 0.5873 | +0.0336 |
| roles | 2 | 0.3674 | 0.3674 | +0.0000 |

All seven topics show non-negative mean deltas. The pattern is broadly
consistent across categories, but per-topic sample sizes are small.

### Per-difficulty (confirmatory only)

| Difficulty | n_queries | mean_delta |
|---|---:|---:|
| easy | 1 | +0.1680 |
| medium | 14 | +0.0480 |
| hard | 5 | +0.0336 |

The pattern is consistent across all difficulty levels.

### Per-multi-hop (confirmatory only)

| Multi-hop | n_queries | mean_delta |
|---|---:|---:|
| True | 8 | +0.0840 |
| False | 12 | +0.0280 |

The learned selector's advantage is stronger on multi-hop queries
(where it has more chances to win by reordering within the 5-chunk
candidate pool).

## Multi-hop distribution (corrected)

The planning target was >= 30% multi-hop in the new (confirmatory)
queries:

| Split | n_queries | n_multihop | percentage |
|---|---:|---:|---:|
| Development subset | 10 | 0 | 0.0% |
| Confirmatory subset | 20 | 8 | 40.0% |
| Full PG-Context-Select-v1 | 30 | 8 | 26.7% |

The confirmatory subset meets the >= 30% planning target (40%). The
full benchmark rate (26.7%) is below 30% because the development
subset has 0 multi-hop queries.

## Frozen pipeline recap

The confirmatory experiment was run with the frozen pipeline
recorded in `.planning/RESEARCH_FREEZE.md`:

- Selector: learned v3 (per-query marginal impact + negative-utility tiebreak)
- Estimator version: v3
- Scoring weights: correctness 0.6, support 0.3, efficiency 0.1
- Evaluator version: eval_v1_model_runner
- Prompt regime: always_context_first
- Model: MiniMax-M3
- 5 reps per (query, strategy)
- Statistical procedure: percentile bootstrap, 2000 resamples, seed 0
- p_value_two_sided: min(1.0, 2 * min(p_lower, p_upper))

No pipeline parameters were tuned after the benchmark freeze.

## Anti-leakage verification

During the authoring of q_0011-q_0030, the following were NOT done:

- Running learned_v3 on the new queries.
- Inspecting per-query learned_v3 scores.
- Running model outcomes on the new queries.
- Tuning the prompt policy based on new-query outcomes.
- Tuning the adaptive prompt threshold.
- Inspecting selector comparisons on the new queries.

The only model invocations during authoring were:

- Structural validation (chunk references, gold references, schema).
- Building candidate pools via the existing BM25 retriever.

The confirmatory queries were authored and frozen BEFORE the
confirmatory outcomes were generated.

## Scope of the conclusion

The pre-specified confirmatory result is:

> On the pre-frozen held-out 20-query confirmatory subset of
> PG-Context-Select-v1, learned_v3 outperformed topk_pool_order by
> Δ = +0.0504 (95% CI [+0.0042, +0.0966], two-sided bootstrap
> p = 0.0310).

This is **not** a broad thesis validation. The strongest allowed scope
is:

> "On the current PG-Context-Select-v1 benchmark (PostgreSQL docs,
> MiniMax-M3, 30 queries)."

Limitations:

1. The benchmark covers PostgreSQL documentation only.
2. The model is MiniMax-M3 only.
3. The 20 confirmatory queries are authored by the same person who
   tuned the development set; subtle topical overlap may bias the result.
4. The 5 reps per query are stochastic and do not average out over a
   single run.
5. Multiple-comparison correction across the 5 strategy comparisons
   is not applied.

## Verification

```bash
.venv/bin/python scripts/confirmatory_analysis_unified.py
.venv/bin/python -m pytest -q
```

The unified script uses one shared analysis path for all three views.
