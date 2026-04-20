# Distractor Annotation Guide

Use this guide to label distractor chunks consistently.

## Core Rule

A distractor is a chunk that is plausible enough to compete for retrieval or selection but should not be treated as gold support for the query.

## Decision Tree

1. Is the chunk correct for an older version but wrong for the target version?
- yes: `stale`
- no: continue

2. Does the chunk directly conflict with the gold answer?
- yes: `contradictory`
- no: continue

3. Does the chunk answer a different valid question in the same topic or section?
- yes: `topical_wrong`
- no: continue

4. Does the chunk answer part of this exact question, but not enough to justify the final answer?
- yes: `partial_truth`
- no: continue

5. Is the chunk the step before or after the needed step in the same workflow?
- yes: `adjacent_procedure`
- no: continue

6. Is the chunk highly similar to the gold-support chunk in wording or structure, but missing or altering the decisive fact?
- yes: `near_duplicate`
- no: continue

7. Is the chunk long, term-overlapping, and plausibly relevant but low-utility for answering?
- yes: `verbose_noise`
- no: label `topical_wrong`

## Tiebreakers

`partial_truth` vs `topical_wrong`
- `partial_truth`: helps answer this exact question, but incompletely
- `topical_wrong`: answers a different question

`adjacent_procedure` vs `topical_wrong`
- `adjacent_procedure`: same workflow, wrong step
- `topical_wrong`: same topic, different task

`near_duplicate` vs `partial_truth`
- `near_duplicate`: near-copy with altered decisive fact
- `partial_truth`: one component of the answer, not a near-copy

`stale` vs `contradictory`
- `stale`: wrong because version changed
- `contradictory`: wrong within the target-version context

## Gold Support Reminder

If a chunk is helpful but not sufficient, it is not gold support.

If the answer requires two chunks together, both may be gold support.

## Required Annotation Output

For each distractor chunk:

- `distractor_type`
- optional note: why it is not gold support

## Examples To Add During Dataset Creation

For each distractor type, keep:

- 2 positive examples
- 2 negative examples

Store those examples beside the first annotated query batch so the labeling rules stay grounded in real corpus cases.
