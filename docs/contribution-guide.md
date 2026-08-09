# Contribution Guide

## Add A New Retriever Backend

The Retriever Protocol in `docs/component-interface-spec.md` lets you
add new retrieval algorithms without touching the candidate-pool
builder, the selector, or the evaluator. Two paths are supported:

### Path A: register a built-in (sibling to BM25Retriever)

Use this when you want to ship the retriever as part of the package
(e.g. it has tests, docs, and is generally useful).

1. Implement the class in `src/context_engine/retrieval.py` as a sibling
   of `BM25Retriever`. Required shape: `name: str`, `index(chunks)`,
   `retrieve(query, *, pool_size, filter_metadata=None) -> list[RetrievalResult]`.
   Pure stdlib is the project standard; check `docs/component-interface-spec.md`
   for the contract.
2. Add a dispatch branch in `scripts/build_candidate_pools.py`:
   - add the name to the `--retriever` choices tuple
   - add an `elif args.retriever == "your_name"` branch in `main()`
3. Re-export from `src/context_engine/__init__.py`.
4. Add tests in `tests/test_retrieval.py` (or a sibling file): ordering,
   monotonicity, Protocol conformance, no-index / empty-query /
   zero-pool guards, and a v1-corpus regression test (gold-in-top-8).
5. Update `docs/data-contract.md` if the retriever changes the
   candidate-pool composition (the contract treats pools as opaque;
   usually no change is needed).

### Path B: register at runtime via `--retriever-module`

Use this for one-off experiments or when you don't want to merge a new
class into the package.

1. Write a Python module that exposes a class implementing the
   Protocol:
   ```python
   from context_engine.retrieval import BM25Retriever

   class MyRetriever:
       name = "my_retriever"
       def __init__(self):
           self._bm25 = BM25Retriever()
       def index(self, chunks):
           self._bm25.index(chunks)
       def retrieve(self, query, *, pool_size, filter_metadata=None):
           return self._bm25.retrieve(query, pool_size=pool_size, filter_metadata=filter_metadata)
   ```
2. Make sure the module is importable (add its parent directory to
   `PYTHONPATH` or install the package).
3. Invoke the script with `--retriever-module my_module:MyRetriever`.
   The script validates Protocol conformance, attempts
   no-argument instantiation, and runs the same pipeline as the
   built-in dispatch.

The two paths are mutually exclusive: passing both `--retriever` and
`--retriever-module` is an error. The built-in `random` retriever is
shipped as a no-signal baseline; use it to verify that any new
retriever is doing meaningful work.

Compare against `topk_pool_order` per the strategy-add path below.
A new retriever should not regress the per-strategy mean by more than
the run-to-run CI half-width on any single strategy. If it does,
file a follow-up issue before merging.

## Add A New Selector Strategy

1. Implement the strategy behind the existing selector interface.
2. Add an experiment config using the new strategy name.
3. Run the eval harness on at least one split.
4. Include results versus `topk_pool_order` in the PR.

`topk_pool_order` is the canonical static baseline (top-K of the candidate
pool in pool order, no learned scoring). All strategies — hand-authored,
heuristic, or learned — should be compared against it. New strategies
should match or exceed its `overall` mean without regressing `support`
on the same query slice.

## Add A New Distractor Type

1. Update the distractor decision tree and examples.
2. Define a deterministic generation recipe.
3. Add the label to any validation logic that expects known distractor types.
4. Include at least one benchmark example in the PR description.

## Run The Eval Harness

Use a single experiment config file and produce versioned outputs for:
- `context_sets`
- `outcomes`
- optional `marginal_impact`

## PR Requirements

- reference the exact config file used
- list affected artifact versions
- include evaluation summary
- do not change frozen data-contract fields without a version bump
