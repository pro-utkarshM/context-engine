"""Tests for the audited multi-run paired-query statistics."""

from __future__ import annotations

import pytest

from context_engine.paired_query import PairedQueryDeltaSummary, paired_query_summary
from context_engine.stats import PairedDeltaSummary


def _norm_dict_to_per_query(scores: dict[str, float]) -> dict[str, list[float]]:
    """Helper: convert {qid: scalar} -> {qid: [scalar]} for the API."""
    return {qid: [value] for qid, value in scores.items()}


# ---- Synthesis fixtures ------------------------------------------------------

# Synthetic test with known per-query deltas. ``delays`` are the
# (left - right) per-query deltas. With 5 queries and a single rep
# each, the bootstrap CI should be roughly centered on the mean.
_KNOWN_LEFT = {"q1": [0.7], "q2": [0.8], "q3": [0.9], "q4": [0.6], "q5": [0.5]}
_KNOWN_RIGHT = {"q1": [0.5], "q2": [0.6], "q3": [0.7], "q4": [0.4], "q5": [0.3]}


# ---- Core behavior -----------------------------------------------------------

def test_paired_query_summary_reproduces_known_mean_delta() -> None:
    """Mean of per-query deltas equals the expected value."""

    summary = paired_query_summary(
        _KNOWN_LEFT, _KNOWN_RIGHT,
        left_label="A", right_label="B",
        n_resamples=200, seed=0,
    )
    # Each deltas is 0.2 (0.7-0.5, 0.8-0.6, etc.), so mean delta must be 0.2.
    assert summary.n_queries == 5
    assert summary.reps_per_query == 1
    assert summary.mean_left == pytest.approx(0.7)
    assert summary.mean_right == pytest.approx(0.5)
    assert summary.delta_summary.mean_delta == pytest.approx(0.2)
    assert summary.delta_summary.p_value_two_sided < 0.01


def test_paired_query_summary_strategies_appear_in_per_query_breakdown() -> None:
    """Every common query appears in the per-query block."""

    summary = paired_query_summary(
        _KNOWN_LEFT, _KNOWN_RIGHT,
        left_label="A", right_label="B",
        n_resamples=200, seed=0,
    )
    assert set(summary.per_query_left) == {"q1", "q2", "q3", "q4", "q5"}
    assert set(summary.per_query_right) == {"q1", "q2", "q3", "q4", "q5"}
    assert set(summary.per_query_deltas) == {"q1", "q2", "q3", "q4", "q5"}


def test_paired_query_summary_stratifies_reps_per_query() -> None:
    """``reps_per_query`` reports the matched rep count."""

    left = {"q1": [0.7, 0.8, 0.9], "q2": [0.5, 0.5]}
    right = {"q1": [0.5, 0.6, 0.7], "q2": [0.5, 0.5]}
    summary = paired_query_summary(left, right, left_label="A", right_label="B", n_resamples=200, seed=0)
    # 3 reps for q1, 2 for q2. The summary uses the max so the call is
    # honest about the available coverage.
    assert summary.reps_per_query == 3


def test_paired_query_summary_per_query_means_average_reps() -> None:
    """Per-query means are the average of the per-query reps."""

    left = {"q1": [0.7, 0.8, 0.9], "q2": [0.5, 0.5]}
    right = {"q1": [0.5, 0.6, 0.7], "q2": [0.5, 0.5]}
    summary = paired_query_summary(left, right, left_label="A", right_label="B", n_resamples=200, seed=0)
    assert summary.per_query_left["q1"] == pytest.approx(0.8)
    assert summary.per_query_left["q2"] == pytest.approx(0.5)
    assert summary.per_query_right["q1"] == pytest.approx(0.6)
    assert summary.per_query_right["q2"] == pytest.approx(0.5)


# ---- Sign convention ---------------------------------------------------------

def test_paired_query_summary_sign_convention_reversal() -> None:
    """Reversing left/right flips the sign of the mean delta."""

    forward = paired_query_summary(
        _KNOWN_LEFT, _KNOWN_RIGHT,
        left_label="A", right_label="B",
        n_resamples=200, seed=0,
    )
    backward = paired_query_summary(
        _KNOWN_RIGHT, _KNOWN_LEFT,
        left_label="B", right_label="A",
        n_resamples=200, seed=0,
    )
    assert forward.delta_summary.mean_delta == pytest.approx(-backward.delta_summary.mean_delta)
    assert forward.delta_summary.ci_low == pytest.approx(-backward.delta_summary.ci_high)
    assert forward.delta_summary.ci_high == pytest.approx(-backward.delta_summary.ci_low)


def test_paired_query_summary_zero_delta_under_identical_inputs() -> None:
    """Identical inputs produce a zero delta with p-value 1.0."""

    same = {"q1": [0.5], "q2": [0.5], "q3": [0.5]}
    summary = paired_query_summary(
        same, dict(same),
        left_label="A", right_label="B",
        n_resamples=200, seed=0,
    )
    assert summary.delta_summary.mean_delta == pytest.approx(0.0)
    assert summary.delta_summary.p_value_two_sided == pytest.approx(1.0)


# ---- Determinism -------------------------------------------------------------

def test_paired_query_summary_is_deterministic_for_same_seed() -> None:
    """Same seed → same summary."""

    a = paired_query_summary(_KNOWN_LEFT, _KNOWN_RIGHT, left_label="A", right_label="B", n_resamples=200, seed=99)
    b = paired_query_summary(_KNOWN_LEFT, _KNOWN_RIGHT, left_label="A", right_label="B", n_resamples=200, seed=99)
    assert a == b


def test_paired_query_summary_varies_with_seed() -> None:
    """Different seeds → different bootstrap CIs (with high probability)."""

    a = paired_query_summary(_KNOWN_LEFT, _KNOWN_RIGHT, left_label="A", right_label="B", n_resamples=200, seed=1)
    b = paired_query_summary(_KNOWN_LEFT, _KNOWN_RIGHT, left_label="A", right_label="B", n_resamples=200, seed=99999)
    # Bootstrap CI is a stochastic estimator; different seeds should
    # give different bounds (1000 resamples is enough to make this stable).
    assert (a.delta_summary.ci_low, a.delta_summary.ci_high) != (b.delta_summary.ci_low, b.delta_summary.ci_high)


# ---- Coverage / edge cases ---------------------------------------------------

def test_paired_query_summary_drops_queries_with_no_reps() -> None:
    """A query with empty rep lists on either side is dropped."""

    left = {"q1": [0.7], "q2": []}
    right = {"q1": [0.5], "q2": [0.5]}
    summary = paired_query_summary(left, right, left_label="A", right_label="B", n_resamples=200, seed=0)
    assert summary.n_queries == 1
    assert "q1" in summary.per_query_left
    assert "q2" not in summary.per_query_left


def test_paired_query_summary_only_keeps_common_queries() -> None:
    """Queries present on only one side are dropped."""

    left = {"q1": [0.7], "q2": [0.8]}
    right = {"q1": [0.5]}
    summary = paired_query_summary(left, right, left_label="A", right_label="B", n_resamples=200, seed=0)
    assert summary.n_queries == 1
    assert "q1" in summary.per_query_left
    assert "q2" not in summary.per_query_left


def test_paired_query_summary_handles_missing_query_ids() -> None:
    """Empty intersection raises a clear error."""

    with pytest.raises(ValueError, match="no overlapping queries"):
        paired_query_summary(
            {"q1": [0.7]}, {"q2": [0.5]},
            left_label="A", right_label="B",
        )


def test_paired_query_summary_handles_empty_inputs() -> None:
    """Either side empty raises a clear error."""

    with pytest.raises(ValueError, match="non-empty"):
        paired_query_summary({}, {"q1": [0.5]}, left_label="A", right_label="B")
    with pytest.raises(ValueError, match="non-empty"):
        paired_query_summary({"q1": [0.5]}, {}, left_label="A", right_label="B")


# ---- Output shape ------------------------------------------------------------

def test_paired_query_summary_to_dict_is_serializable() -> None:
    """``to_dict`` round-trips into a plain JSON-safe dict."""

    import json

    summary = paired_query_summary(_KNOWN_LEFT, _KNOWN_RIGHT, left_label="A", right_label="B", n_resamples=200, seed=0)
    d = summary.to_dict()
    # The dict must be JSON-serializable.
    json.dumps(d)
    assert d["left_label"] == "A"
    assert d["right_label"] == "B"
    assert d["n_queries"] == 5
    assert d["reps_per_query"] == 1
    assert "delta_summary" in d
    assert "per_query" in d
    # The per-query block must include the per-query deltas.
    assert "q1" in d["per_query"]
    assert "delta" in d["per_query"]["q1"]


def test_paired_query_summary_bootstraps_per_query_deltas_not_raw_reps() -> None:
    """The bootstrap unit is the per-query delta (10 obs), not the raw
    per-rep observation (50 obs). This matters because the old
    outcome-level bootstrap gives a different (and wrong) CI for the
    multi-run benchmark.
    """
    # Same setup as the r4 audit: 5 reps per query, 10 queries.
    left = {f"q{i:04d}": [0.7, 0.7, 0.7, 0.7, 0.7] for i in range(1, 11)}
    right = {f"q{i:04d}": [0.5, 0.5, 0.5, 0.5, 0.5] for i in range(1, 11)}
    # Add some variance to one query on the right side so the bootstrap
    # isn't a degenerate point mass.
    right["q0006"] = [0.3, 0.3, 0.3, 0.3, 0.3]

    summary = paired_query_summary(left, right, left_label="L", right_label="R", n_resamples=2000, seed=0)
    assert summary.n_queries == 10
    assert summary.reps_per_query == 5
    # All per-query deltas are 0.2 except q0006 which is 0.4.
    expected = {f"q{i:04d}": 0.2 for i in range(1, 11)}
    expected["q0006"] = 0.4
    for qid, value in expected.items():
        assert summary.per_query_deltas[qid] == pytest.approx(value)
    # Mean delta is (9 * 0.2 + 0.4) / 10 = 0.22.
    assert summary.delta_summary.mean_delta == pytest.approx(0.22)
