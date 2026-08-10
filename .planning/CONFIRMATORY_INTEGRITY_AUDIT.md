# Confirmatory Integrity Audit — Phase O

Written 2026-08-09. Reproduces and cross-checks the development vs
confirmatory result reported in PR #22. Verifies that the frozen
pipeline was used, that all hashes match the freeze document, and
that no benchmark content was modified after the freeze.

## 1. Reason for the audit

PR #22 reported two seemingly contradictory findings:

- Confirmatory (q_0011-q_0030, n=20): learned_v3 vs topk_pool_order
  delta = +0.0504, p_two_sided = 0.0310, verdict: validated.

- Development (q_0001-q_0010, n=10): learned_v3 vs topk_pool_order
  delta = -0.0588, p_two_sided = 0.0100, verdict: negative direction.

The exact same magnitude (+/- 0.0588) with opposite signs was flagged as
suspicious. This audit traces the cause, verifies the confirmatory
result independently, and corrects the documentation.

## 2. PR #21 development result (frozen reference)

The Phase M freeze reported for the development set, on the PR #21
artifacts (frozen 8-chunk corpus):

```
PR #21 REPRODUCTION (learned_v3_context_first vs topk_pool_order):
  n_queries: 10
  learned mean: 0.7257
  topk mean:    0.6669
  mean_delta:   +0.0588   (learned wins)
  ci_low (raw): 0.0
  ci_high (raw): 0.126
  p_value_one_sided: 0.0255
  p_value_two_sided: 0.0510
  reps_per_query: 5
  Per-query deltas (10 queries):
    q_0001: +0.0000
    q_0002: +0.0000
    q_0003: +0.0000
    q_0004: +0.0000
    q_0005: +0.0000
    q_0006: +0.2520
    q_0007: +0.0000
    q_0008: +0.2520
    q_0009: +0.0000
    q_0010: +0.0840
```

Source artifacts (PR #21 frozen):

- `data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run000.jsonl`
- `data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run001.jsonl`
- `data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run002.jsonl`
- `data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run003.jsonl`
- `data/processed/learned_v3_context_first/outcomes_model_minimax_learned_v3_v1_run004.jsonl`
- `data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run000.jsonl`
- `data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run001.jsonl`
- `data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run002.jsonl`
- `data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run003.jsonl`
- `data/processed/canon_r4_context_first/outcomes_model_minimax_v1_run004.jsonl`

## 3. PR #22 development result (new artifacts)

Phase N expanded the corpus from 8 chunks to 38 chunks (replication,
partitioning, configuration, etc.). All artifacts were regenerated:

- candidate_pools_v1.jsonl regenerated against the 38-chunk corpus
- context_sets_v1.jsonl regenerated
- marginal_impact_minimax_v1.jsonl regenerated
- learned_v3_confirmatory/ regenerated
- canon_r4_confirmatory/ regenerated

The PR #22 development result, computed from the regenerated artifacts:

```
PR #22 REPRODUCTION (learned_v3 vs topk_pool_order on development):
  n_queries: 10
  learned mean: 0.6681
  topk mean:    0.7269
  mean_delta:   -0.0588   (canon wins)
  ci_low (raw): -0.1176
  ci_high (raw): -0.0084
  p_value_one_sided: 0.0005
  p_value_two_sided: 0.0100
  reps_per_query: 5
  Per-query deltas (10 queries):
    q_0001: +0.0000
    q_0002: +0.0000
    q_0003: -0.0840
    q_0004: +0.0000
    q_0005: +0.0000
    q_0006: -0.2520
    q_0007: -0.0840
    q_0008: -0.1680
    q_0009: +0.0000
    q_0010: +0.0000
```

## 4. Root cause of the discrepancy

The discrepancy is NOT a sign error or artifact-selection error. It is a
real consequence of the corpus expansion:

1. The PR #21 corpus had 8 chunks (all pg_hba.conf authentication).
2. Phase N expanded to 38 chunks covering replication, partitioning,
   configuration, logging, backup, roles.
3. The BM25 retriever, with the expanded corpus, now returns different
   candidate chunks for each query.
4. The topk_pool_order strategy now includes chunks from new topics
   (e.g., for q_0006: the pool now includes pg16_auth_007, pg16_auth_002,
   pg16_auth_001, pg16_auth_004, pg15_auth_001 — vs. PR #21's pool of
   pg16_auth_004, pg15_auth_001, pg16_auth_002, pg16_auth_006, pg16_auth_001).
5. The learned_v3 selector, retrained on the new marginal_impact data,
   picks different chunks. For q_0006 it now selects
   [pg16_auth_004, pg16_auth_002, pg15_auth_001, pg16_auth_007, pg16_auth_001]
   (vs. PR #21's [pg16_auth_004, pg16_auth_002, pg16_auth_001, pg16_auth_006,
   pg15_auth_001]).
6. The model produces different outcomes on these different contexts.

The per-query deltas change qualitatively:
- PR #21: 3 wins (q_0006, q_0008, q_0010), 7 ties.
- PR #22: 4 losses (q_0003, q_0006, q_0007, q_0008), 6 ties.

The mean_delta sign flips because the per-query sign distribution changed.

This is NOT a sign or order error. The source-of-truth development
result (under the new frozen artifacts) is -0.0588, not +0.0588.

## 5. Source-of-truth artifacts

The frozen confirmatory experiment uses these artifacts (all hashes
match the freeze document):

| Artifact | SHA-256 |
|---|---|
| queries_v1.jsonl | 588c3fb1858092e7434496cae4a884e65ef3d482c1d2092d90982f66d75b8f35 |
| corpus_chunks_v1.jsonl | 1e1c37bca3eba0b0d2902e9d661b752af9ccdd96e99bb728bd5040fbcb9186b2 |
| candidate_pools_v1.jsonl | 9b9d15a5b925e36df6d77d85d6f11930f9f7b6b744926881e76d4d81c86ee028 |
| context_sets_v1.jsonl | 1e311e6203c8ee927b2867de055aaee426dfe4009a0a719261883cbf7cdc96de |
| marginal_impact_minimax_v1.jsonl | a489fe4cdce9185c1e40414feaeba453eb1e06b3d29660c8abd52c800b679420 |

All hashes verified against `git show c878368:data/processed/<file>` for
PR #22, and against `git show a18fb1d:data/processed/<file>` for PR #21.

The outputs of the confirmatory experiment (regenerated during PR #22):

| Artifact | SHA-256 |
|---|---|
| canon_r4_confirmatory/replications_summary_v1.jsonl | 9ca944e3d7c359a763600122dcee3bb0f9a59a5e3c2965fa5ddc43b3a924b9bb |
| canon_r4_confirmatory/outcomes_model_minimax_v1_run000.jsonl | 9ea45fa4a37e82993bddd441cadc16ace6443d2bc8d5b9f6364bfd2d951dbc3b |
| canon_r4_confirmatory/outcomes_model_minimax_v1_run001.jsonl | 678519450af61aeb... |
| canon_r4_confirmatory/outcomes_model_minimax_v1_run002.jsonl | 70ec15af43f5cede... |
| canon_r4_confirmatory/outcomes_model_minimax_v1_run003.jsonl | a2e2b00a433ca630... |
| canon_r4_confirmatory/outcomes_model_minimax_v1_run004.jsonl | 481e7b78f4d42876... |
| learned_v3_confirmatory/context_sets_learned_v3_v1.jsonl | 24d57cda007da7b5... |
| learned_v3_confirmatory/outcomes_model_minimax_learned_v3_v1_run000.jsonl | ddc988c08d449d66... |
| learned_v3_confirmatory/outcomes_model_minimax_learned_v3_v1_run001.jsonl | 0ec99403c7075479... |
| learned_v3_confirmatory/outcomes_model_minimax_learned_v3_v1_run002.jsonl | c105c61c7aa2a011... |
| learned_v3_confirmatory/outcomes_model_minimax_learned_v3_v1_run003.jsonl | c79772cab35452f8... |
| learned_v3_confirmatory/outcomes_model_minimax_learned_v3_v1_run004.jsonl | a961339e3bbd67cd... |

All output hashes match PR #22 commit exactly.

## 6. Sign convention (locked)

Everywhere:

```text
delta = learned_v3 - comparison
positive means learned_v3 wins.
```

The unified analysis path `scripts/confirmatory_analysis_unified.py`
applies this sign convention consistently across all three views.

## 7. Corrected three-view results (one shared analysis path)

### Development subset (q_0001-q_0010, n=10)

```
learned_v3 vs topk_pool_order:
  learned mean: 0.6681
  topk mean:    0.7269
  mean_delta:   -0.0588
  95% CI (raw): [-0.1176, -0.0084]
  p_value_one_sided: 0.0005
  p_value_two_sided: 0.0100
  Win/loss/tie: 0 / 4 / 6
  Verdict: negative direction (canon wins)
```

### Confirmatory subset (q_0011-q_0030, n=20)

```
learned_v3 vs topk_pool_order:
  learned mean: 0.5948
  topk mean:    0.5444
  mean_delta:   +0.0504
  95% CI (raw): [0.0042, 0.0966]
  p_value_one_sided: 0.0155
  p_value_two_sided: 0.0310
  Win/loss/tie: 10 / 3 / 7
  Verdict: validated on development benchmark
```

### Full PG-Context-Select-v1 (q_0001-q_0030, n=30)

```
learned_v3 vs topk_pool_order:
  learned mean: 0.6192
  topk mean:    0.6052
  mean_delta:   +0.0140
  95% CI (raw): [-0.0252, +0.0532]
  p_value_one_sided: 0.2300
  p_value_two_sided: 0.4600
  Win/loss/tie: 10 / 7 / 13
  Verdict: inconclusive
```

## 8. Oracle-informed comparisons

### Development subset (q_0001-q_0010)

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

### Confirmatory subset (q_0011-q_0030)

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

### Full benchmark (q_0001-q_0030)

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

## 9. Artifact hash verification

All 5 frozen input artifact hashes match the freeze document. All
12 confirmatory output artifact hashes match the PR #22 commit. No
drift detected.

## 10. Freeze integrity verification

The frozen benchmark is preserved:

- No confirmatory query was modified after the freeze (verified by
  hash comparison with PR #22 commit).
- No gold answer was modified (queries_v1.jsonl hash matches).
- No gold_support_ids were modified (queries_v1.jsonl hash matches).
- No candidate pool was altered based on observed outcomes (the pools
  were generated before any confirmatory outcome was inspected).
- No confirmatory query was deleted (n=20 confirmed).
- No new confirmatory query was added after seeing results.

The full set of 30 queries has the correct split metadata:
- 10 queries with `metadata.split = "development"`
- 20 queries with `metadata.split = "confirmatory"`
- 0 queries with no split

## 11. Multi-hop percentage correction

The previous report stated:

> 8/30 (27%) — at least 30% threshold met

This is incorrect:
- 8/30 = 26.666...% ≈ 26.7% (not 27%)
- The 30% target was for the **new** queries (not the full benchmark)

The correct statement:

- Full benchmark: 8/30 = 26.7% multi-hop
- Development subset: 0/10 = 0% multi-hop
- Confirmatory subset: 8/20 = 40% multi-hop
- The planning target was >= 30% multi-hop in the new queries (achieved:
  40%). The full-benchmark rate is below 30% because the development
  set has 0 multi-hop queries.

## 12. Terminology corrections

The previous report used the phrase:

> "confirmatory 20-query subset of the PG-Context-Select-v1 development benchmark"

This terminology is incorrect because:

1. The PG-Context-Select-v1 benchmark as a whole is NOT a development
   benchmark — it is a benchmark that contains both development and
   confirmatory subsets.
2. The "development" label applies only to q_0001-q_0010.

Correct terminology:

- `development subset` (q_0001-q_0010)
- `confirmatory subset` (q_0011-q_0030)
- `full PG-Context-Select-v1 benchmark` (q_0001-q_0030)

## 13. Removed false narrative

The previous report claimed:

> "The development set shows the OPPOSITE direction... This is the standard in-sample vs out-of-sample pattern."

After the audit, this narrative is preserved as a descriptive observation
(development subset shows -0.0588, confirmatory subset shows +0.0504), but
NOT promoted to a causal explanation. The audit verified that:

1. Both numbers are computed correctly from their respective artifacts.
2. The sign difference reflects the corpus expansion (real change in data).
3. No causal mechanism (in-sample bias, model tuning, etc.) has been
   independently established.

The descriptive observation is preserved; the causal interpretation is
removed.

## 14. Final claims justified

After the audit:

1. On the pre-frozen held-out 20-query confirmatory subset of
   PG-Context-Select-v1, learned_v3 outperformed topk_pool_order by
   Δ = +0.0504 (95% CI [+0.0042, +0.0966], two-sided bootstrap
   p = 0.0310).

2. On the 10-query development subset, learned_v3 was reliably WORSE
   than topk_pool_order (Δ = -0.0588, CI [-0.1176, -0.0084],
   p = 0.0100).

3. On the full 30-query benchmark, the result is inconclusive
   (Δ = +0.0140, CI [-0.0252, +0.0532], p = 0.4600).

4. Against the oracle-informed reference (gold_plus_distractors), the
   learned selector is competitive but not demonstrably better on any
   subset (all p_two_sided >= 0.05).

5. The strict pre-specified validation criterion (ci_low > 0 AND
   p_value_two_sided < 0.05) is met on the confirmatory subset only.

This result is specific to the current PG-Context-Select-v1 benchmark
and MiniMax-M3. Broader generalization has not been established.

## 15. Remaining concerns

- The confirmatory queries are authored by the same person who tuned
  the development set. Subtle topical overlap may bias the result.
- The benchmark covers PostgreSQL documentation only.
- The model is MiniMax-M3 only.
- Multiple-comparison correction across the 5 strategy comparisons
  has not been applied.
- The corpus is still small (38 chunks). Generalization to a larger
  corpus is not established.
