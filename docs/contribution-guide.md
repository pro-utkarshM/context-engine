# Contribution Guide

## Add A New Selector Strategy

1. Implement the strategy behind the existing selector interface.
2. Add an experiment config using the new strategy name.
3. Run the eval harness on at least one split.
4. Include results versus `topk_hybrid` in the PR.

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
