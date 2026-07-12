"""Marginal-impact computation for chunk-level utility validation.

Implements the contract defined in ``docs/component-interface-spec.md`` under
``Marginal Impact Analyzer``. Given a base context set and a target chunk, the
helper produces a ``MarginalImpact`` row recording the signed score delta for
either adding the chunk (if absent) or removing it (if present).

Design choices:

* The score axis is parameterized via ``score_key``. By default it uses
  ``Outcome.scores.overall`` — the same weighted score the selector is
  optimising — but correctness-only deltas are also useful for isolating
  signal from support/efficiency noise.
* The helper is pure: it takes a scorer callable and yields ``MarginalImpact``
  rows. The driving script owns runner wiring and persistence.
* Add/remove are symmetric in the contract: ``delta == new_score - base_score``.
  A negative delta means the chunk hurt; a positive delta means it helped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .artifacts import ContextSet, MarginalImpact, Outcome
from .authoring import make_marginal_impact


Operation = Literal["add", "remove"]
ScoreKey = Literal["correctness", "support", "efficiency", "overall"]


class MarginalImpactError(ValueError):
    """Raised when a marginal-impact request violates an obvious precondition."""


def _select_score(outcome: Outcome, score_key: ScoreKey) -> float:
    return getattr(outcome.scores, score_key)


def _variant_context_set(
    base: ContextSet,
    chunk_id: str,
    operation: Operation,
    chunks_by_id: dict[str, int] | None = None,
) -> ContextSet:
    # ``chunks_by_id`` is accepted for forward compatibility (token-count
    # adjustment when adding); today we preserve the base token_count and
    # rely on the scorer/evaluator to derive its own estimate.
    del chunks_by_id

    if operation == "add":
        if chunk_id in base.selected_ids:
            raise MarginalImpactError(
                f"chunk {chunk_id!r} already in base set {base.set_id!r}; nothing to add"
            )
        new_selected = list(base.selected_ids) + [chunk_id]
    else:  # remove
        if chunk_id not in base.selected_ids:
            raise MarginalImpactError(
                f"chunk {chunk_id!r} not in base set {base.set_id!r}; nothing to remove"
            )
        new_selected = [cid for cid in base.selected_ids if cid != chunk_id]
        if not new_selected:
            raise MarginalImpactError(
                f"cannot remove the last chunk from base set {base.set_id!r}"
            )

    return ContextSet(
        set_id=base.set_id,
        query_id=base.query_id,
        candidate_pool_id=base.candidate_pool_id,
        strategy=base.strategy,
        selected_ids=new_selected,
        ordering_type=base.ordering_type,
        token_count=base.token_count,
        metadata=base.metadata,
    )


def compute_marginal_impact(
    *,
    base_set: ContextSet,
    chunk_id: str,
    operation: Operation,
    base_score: float,
    new_score: float,
) -> MarginalImpact:
    """Build a ``MarginalImpact`` row from pre-computed base and new scores.

    Use this when the caller has already evaluated both context sets and only
    needs the artifact row. Delta is signed (``new_score - base_score``).
    """
    return make_marginal_impact(
        query_id=base_set.query_id,
        base_set_id=base_set.set_id,
        chunk_id=chunk_id,
        operation=operation,
        base_score=base_score,
        new_score=new_score,
    )


def evaluate_marginal_impact(
    *,
    base_set: ContextSet,
    chunk_id: str,
    operation: Operation,
    scorer: Callable[[ContextSet], Outcome],
    score_key: ScoreKey = "overall",
) -> MarginalImpact:
    """Evaluate a marginal-impact row end-to-end.

    ``scorer`` must accept a ``ContextSet`` and return an ``Outcome``. The base
    outcome is supplied via ``base_outcome`` when it is already known (the
    common case: it was computed during the main outcome pass and is sitting
    in the outcomes artifact).
    """
    base_outcome = scorer(base_set)
    variant_set = _variant_context_set(base_set, chunk_id, operation)
    new_outcome = scorer(variant_set)

    return compute_marginal_impact(
        base_set=base_set,
        chunk_id=chunk_id,
        operation=operation,
        base_score=_select_score(base_outcome, score_key),
        new_score=_select_score(new_outcome, score_key),
    )